"""Grasp selection and arm IK: picks a reachable grasp and solves the poses for it."""
from __future__ import annotations

import mink
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

from molmospaces.components.robot import JOINT_NAMES as _JOINTS

HEIGHT_MIN, HEIGHT_MAX = 0.35, 0.793

_WAIST = ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]

def _hand(side):
    """Joint/site name bundle for one hand side."""
    return dict(
        arm=[f"{side}_{j}" for j in "shoulder_pitch_joint shoulder_roll_joint shoulder_yaw_joint elbow_joint wrist_roll_joint wrist_pitch_joint wrist_yaw_joint".split()],
        gripper=f"{side}_Joint1_1", site=f"{side}_grasp")
# The action space exposes only the right arm, so there is one hand.
_HANDS = {"right": _hand("right")}


PHASE_IDLE = "idle"
PHASE_APPROACH = "approach"
PHASE_REALIGN = "realign"
PHASE_DESCEND = "descend"
PHASE_OPEN_HOLD = "open_hold"
PHASE_CLOSE = "close"
PHASE_POST_CLOSE = "post_close"
PHASE_LIFT = "lift"
PHASE_DONE = "done"
_PHASE_ORDER = [PHASE_APPROACH, PHASE_DESCEND, PHASE_OPEN_HOLD, PHASE_CLOSE, PHASE_POST_CLOSE, PHASE_LIFT, PHASE_DONE]


class GraspPolicy:
    PREGRASP_OFFSET = (0.05, 0.125)
    LIFT = 0.15
    IK_DT = 1e-2
    IK_THRESH = 0.06
    HEIGHT_DAMPING = 5e5

    def __init__(self):
        """Grasp planner state; phases start at IDLE until plan() is called."""
        self._standalone = None
        self._phase = PHASE_IDLE
        self._hand = "right"
        self._ik_error_fn = None
        # Injected by G1Controller.set_env / .setup, or filled in by plan().
        # Declared here so the object's full state is visible in one place.
        self._env = None
        self._use_height = True          # G1Controller.setup sets False
        self._skip_pregrasp_ik = False   # G1Controller.setup sets True
        self._fast_precheck = False      # set around precheck_grasp only
        self._cached = None              # grasp reused from a successful precheck
        self._grasp_local = None
        self._target_body_id = -1
        self._cached_target_body_id = None
        self._cached_bids_for_model_id = None
        self._cached_probe_bids = set()
        self._cached_robot_bids = set()
        self._cached_target_bids = set()
        # Per-plan state, filled by plan() / setup(); listed so the full shape
        # of the object is visible here rather than scattered across methods.
        self._ik_mdl = None
        self._ik_fj_qa = None
        self._grasp_pos = None
        self._grasp_rot = None
        self._lift_pos = None
        self._pregrasp = None
        self._pregrasp_joints = None

    def setup(self, model, data, prefix):
        """Cache model handles, joint qpos addresses, and grasp-site ids for both hands."""
        self._model, self._data, self._prefix = model, data, prefix
        self._standalone = model
        self._stj = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)]
        self._qposadr = {}
        for jn in _JOINTS:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, prefix + jn)
            if jid >= 0: self._qposadr[jn] = model.jnt_qposadr[jid]
        self._freejoint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, prefix + "floating_base_joint")
        self._fj_scene = model.jnt_qposadr[self._freejoint_id]
        self._sites = {h: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, prefix + c["site"]) for h, c in _HANDS.items()}
        self._ik_cfg = None
        self._col_limit = None

    @property
    def done(self):
        """True once the grasp sequence has finished."""
        return self._phase == PHASE_DONE

    def _build_collision_limit(self, target_pos, radius=1.0):
        """Build a mink collision limit between the arms and nearby scene geoms around the target."""
        m, d = self._model, self._data
        mujoco.mj_fwdPosition(m, d)
        robot_bids = {i for i in range(m.nbody)
                      if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) or "").startswith(self._prefix)}
        right_arm_gids, left_arm_gids, body_gids, obs_gids = [], [], [], []
        for gid in range(m.ngeom):
            bid = m.geom_bodyid[gid]
            bname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
            if bid in robot_bids:
                if any(s in bname for s in ("shoulder", "elbow", "wrist", "gripper")):
                    if "right" in bname:
                        right_arm_gids.append(gid)
                    elif "left" in bname:
                        left_arm_gids.append(gid)
                elif any(s in bname for s in ("pelvis", "torso", "hip", "waist")):
                    body_gids.append(gid)
                continue
            gtype = m.geom_type[gid]
            if gtype == mujoco.mjtGeom.mjGEOM_PLANE: continue
            if gtype == mujoco.mjtGeom.mjGEOM_MESH: continue
            gname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
            if gname == "floor": continue
            if np.linalg.norm(d.geom_xpos[gid, :2] - target_pos[:2]) > radius: continue
            obs_gids.append(gid)
        arm_gids = right_arm_gids + left_arm_gids
        pairs = []
        if arm_gids and obs_gids: pairs.append((arm_gids, obs_gids))
        if arm_gids and body_gids: pairs.append((arm_gids, body_gids))
        if right_arm_gids and left_arm_gids:
            pairs.append((right_arm_gids, left_arm_gids))
        self._col_limit = (
            mink.CollisionAvoidanceLimit(
                model=m, geom_pairs=pairs,
                minimum_distance_from_collisions=0.03, collision_detection_distance=0.1)
            if pairs else None)

    def _sync(self):
        """Snapshot the current sim qpos for seeding the IK configuration."""
        return self._data.qpos.copy()

    def _ensure_ik_config(self):
        """Build the mink configuration over the standalone robot model on first use."""
        if self._ik_cfg is not None:
            return
        self._ik_cfg = mink.Configuration(self._standalone)
        self._ik_mdl = self._ik_cfg.model
        self._ik_fj_qa = self._ik_mdl.jnt_qposadr[
            mujoco.mj_name2id(self._ik_mdl, mujoco.mjtObj.mjOBJ_JOINT, self._prefix + "floating_base_joint")]

    def _ik_dof_mask(self, ik_joints):
        """1.0 on the dofs IK may move (arm, waist, and pelvis z when height is enabled)."""
        mask = np.zeros(self._ik_mdl.nv)
        for jn in ik_joints:
            jid = mujoco.mj_name2id(self._ik_mdl, mujoco.mjtObj.mjOBJ_JOINT, self._prefix + jn)
            if jid >= 0:
                mask[self._ik_mdl.jnt_dofadr[jid]] = 1.0
        if self._use_height:
            fj_dof = self._ik_mdl.jnt_dofadr[
                mujoco.mj_name2id(self._ik_mdl, mujoco.mjtObj.mjOBJ_JOINT, self._prefix + "floating_base_joint")]
            mask[fj_dof + 2] = 1.0
        return mask

    def _ik_tasks(self, config, site, pos, rot):
        """Frame task on the wrist site plus a weak posture regularizer."""
        ht = mink.FrameTask(frame_name=self._prefix + site, frame_type="site",
                            position_cost=100, orientation_cost=1, lm_damping=1)
        r = mink.SO3.from_matrix(rot) if rot is not None else mink.SO3.identity()
        ht.set_target(mink.SE3.from_rotation_and_translation(r, np.asarray(pos, dtype=np.float64)))
        post = mink.PostureTask(self._ik_mdl, cost=1e-2)
        post.set_target_from_configuration(config)
        return ht, post

    def _run_ik(self, config, ht, post, mask):
        """Iterate the damped-least-squares solver until convergence or stall."""
        limits = [mink.ConfigurationLimit(self._ik_mdl)]
        if self._col_limit is not None:
            limits.append(self._col_limit)
        prev = float("inf")
        for step in range(300):
            try:
                vel = mink.solve_ik(config, [ht, post], self.IK_DT, "daqp", damping=1e-1, limits=limits)
            except Exception:
                break
            vel *= mask
            config.integrate_inplace(vel, self.IK_DT)
            q = config.q.copy()
            q[self._ik_fj_qa + 2] = np.clip(q[self._ik_fj_qa + 2], HEIGHT_MIN, HEIGHT_MAX)
            config.update(q)
            err = np.linalg.norm(ht.compute_error(config)[:3])
            if err < 0.001:
                break
            if step == 100 and err > 0.1:
                break
            if step > 100 and err > prev - 1e-5:
                break
            prev = err

    def _apply_ik_height(self, config):
        """Push the solved pelvis height into the scene model's spring target."""
        ik_h = np.clip(config.q[self._ik_fj_qa + 2], HEIGHT_MIN, HEIGHT_MAX)
        self._model.qpos_spring[self._fj_scene + 2] = ik_h
        self._model.dof_damping[self._model.jnt_dofadr[self._freejoint_id] + 2] = self.HEIGHT_DAMPING

    def _ik_joint_targets(self, config):
        """Read the solved configuration back out as a name to angle map."""
        out = {}
        for jn in self._stj:
            if jn == self._prefix + "floating_base_joint":
                continue
            jid = mujoco.mj_name2id(self._ik_mdl, mujoco.mjtObj.mjOBJ_JOINT, jn)
            if jid < 0:
                continue
            key = jn[len(self._prefix):] if jn.startswith(self._prefix) else jn
            out[key] = config.q[self._ik_mdl.jnt_qposadr[jid]]
        return out

    def _solve_ik(self, pos, rot=None, hand=None):
        """Solve arm+waist IK for a wrist pose on the standalone robot model; returns joint targets."""
        hcfg = _HANDS[hand or self._hand]
        self._ensure_ik_config()
        config = self._ik_cfg
        config.update(self._sync())
        mask = self._ik_dof_mask(set(hcfg["arm"] + _WAIST))
        ht, post = self._ik_tasks(config, hcfg["site"], pos, rot)
        self._run_ik(config, ht, post, mask)
        if self._use_height:
            self._apply_ik_height(config)
        return self._ik_joint_targets(config)

    def _refresh_body_caches(self, m):
        """Recompute probe/robot body-id sets when the scene model changes (O(N^2), so cached)."""
        if self._cached_bids_for_model_id == id(m):
            return
        probe_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "gripper_probe")
        pb = set()
        if probe_bid >= 0:
            for bid in range(m.nbody):
                b = bid
                while b > 0:
                    if b == probe_bid:
                        pb.add(bid)
                        break
                    b = m.body_parentid[b]
        self._cached_probe_bids = pb
        self._cached_robot_bids = {
            i for i in range(m.nbody)
            if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) or "").startswith(self._prefix)}
        self._cached_bids_for_model_id = id(m)
        self._cached_target_body_id = None   # force a target recompute below

    def _target_body_ids(self, m):
        """Body ids of the target and everything parented under it, cached per target."""
        tbid = self._target_body_id
        if self._cached_target_body_id == tbid:
            return self._cached_target_bids
        tb = set()
        if tbid >= 0:
            stack = [tbid]
            while stack:
                b = stack.pop()
                tb.add(b)
                for cb in range(m.nbody):
                    if m.body_parentid[cb] == b and cb != b:
                        stack.append(cb)
        self._cached_target_bids = tb
        self._cached_target_body_id = tbid
        return tb

    def _park_probe(self, m, sd, qa):
        """Move the probe body out of the scene at z=10 and refresh derived state."""
        sd.qpos[qa:qa + 3] = [0, 0, 10]
        mujoco.mj_forward(m, sd)

    def _probe_collides(self, m, sd, qa, pt, quat, probe_bids, ignore):
        """Teleport the probe to one checkpoint and report a contact with anything not ignored."""
        sd.qpos[qa:qa + 3] = pt
        sd.qpos[qa + 3:qa + 7] = quat
        mujoco.mj_forward(m, sd)
        mujoco.mj_collision(m, sd)
        for k in range(sd.ncon):
            c = sd.contact[k]
            b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
            if b1 in probe_bids or b2 in probe_bids:
                other = b2 if b1 in probe_bids else b1
                if other not in ignore:
                    return True
        return False

    def _path_is_clear(self, grasp_T):
        """Probe-body sweep from pregrasp to grasp to reject grasps that collide with the scene."""
        grasp_pos = grasp_T[:3, 3]
        offset = np.random.uniform(0.10, 0.15)
        pregrasp_pos = grasp_pos - offset * grasp_T[:3, 2]
        env = self._env
        if env is None:
            return True
        m, sd = env.scene.model, env.scene.data
        self._refresh_body_caches(m)
        probe_bids = self._cached_probe_bids
        if not probe_bids:
            return True
        ignore = self._cached_robot_bids | self._target_body_ids(m) | probe_bids

        qa = m.joint("gripper_probe_joint").qposadr[0]
        for jname, val in (("gripper_probe_joint_a", 0.04), ("gripper_probe_joint_b", -0.04)):
            jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jname)
            if jid >= 0:
                sd.qpos[m.jnt_qposadr[jid]] = val
        quat = R.from_matrix(grasp_T[:3, :3]).as_quat(scalar_first=True)

        # The precheck only needs a smell test (is the grasp pose inside a wall?).
        # Once committed to a grasp, sweep pregrasp -> mid -> grasp so a blocked
        # approach is caught too.
        checkpoints = ([grasp_pos] if self._fast_precheck
                       else [pregrasp_pos, 0.5 * pregrasp_pos + 0.5 * grasp_pos, grasp_pos])
        for pt in checkpoints:
            if self._probe_collides(m, sd, qa, pt, quat, probe_bids, ignore):
                self._park_probe(m, sd, qa)
                return False
        self._park_probe(m, sd, qa)
        return True

    def _ik_error(self, T, hand):
        """Position error (m) of the IK solution for wrist pose T."""
        if self._ik_error_fn is not None:
            return self._ik_error_fn(T, hand)
        sid = self._sites[hand]
        ik = self._solve_ik(T[:3,3], T[:3,:3], hand)
        dc = mujoco.MjData(self._model); dc.qpos[:] = self._data.qpos
        for jn, v in ik.items():
            if jn in self._qposadr: dc.qpos[self._qposadr[jn]] = v
        if self._ik_cfg is not None:
            dc.qpos[self._fj_scene + 2] = np.clip(self._ik_cfg.q[self._ik_fj_qa + 2], HEIGHT_MIN, HEIGHT_MAX)
        mujoco.mj_forward(self._model, dc)
        return float(np.linalg.norm(dc.site_xpos[sid] - T[:3,3]))

    def _grasp_candidates(self, Tw, vg, dc):
        """World grasp frames from the asset's valid grasps, yaw-flipped toward the current hand."""
        vg = np.asarray(vg, dtype=np.float64)
        flip = R.from_euler("z", np.pi)
        cands = []
        for hand in _HANDS:
            hr = R.from_matrix(dc.site_xmat[self._sites[hand]].reshape(3, 3))
            for go in vg:
                c = Tw @ go
                ro = R.from_matrix(c[:3, :3])
                rf = ro * flip
                # Pick the yaw=0 or yaw=pi flip closer to the current hand, which keeps IK well-conditioned.
                if (hr.inv() * rf).magnitude() < (hr.inv() * ro).magnitude():
                    c = c.copy()
                    c[:3, :3] = rf.as_matrix()
                cands.append((c, hand))
        np.random.shuffle(cands)
        return cands

    def _filter_by_reach(self, cands, dc):
        """Drop candidates outside arm reach; kills most of them before any IK runs."""
        MAX_REACH, MIN_REACH = 0.85, 0.10     # G1 arm fully extended, generous
        shoulder_world = {}
        for hand in _HANDS:
            sb = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY,
                                   f"{self._prefix}{hand}_shoulder_pitch_link")
            if sb >= 0:
                shoulder_world[hand] = dc.xpos[sb].copy()
        if not shoulder_world:
            return cands
        return [(c, h) for (c, h) in cands
                if h not in shoulder_world
                or MIN_REACH <= float(np.linalg.norm(c[:3, 3] - shoulder_world[h])) <= MAX_REACH]

    def _prepend_cached_grasp(self, cands, Tw):
        """Put the grasp found during precheck first, so it is tried before anything else."""
        cached = self._cached
        self._cached = None
        if cached is None or cached["hand"] != "right":
            return cands
        if "grasp_local" in cached:
            grasp_T = Tw @ cached["grasp_local"]
        else:
            grasp_T = np.eye(4)
            grasp_T[:3, 3] = cached["grasp_pos"]
            grasp_T[:3, :3] = cached["grasp_rot"]
        return [(grasp_T, cached["hand"])] + cands

    def _pick_reachable_grasp(self, cands, dc):
        """Best candidate by IK error whose approach path is clear, or None."""
        ik_results = []
        # cands is shuffled, so 30 is representative; any cached grasp sits at index 0.
        try:
            for c, h in cands[:30]:
                ik_results.append((self._ik_error(c, h), c, h))
        except KeyboardInterrupt:
            return None, None
        if not ik_results:
            return None, None
        ik_results.sort(key=lambda x: x[0])
        # Path-check only the top K: K=1 is a cheap smell test during precheck,
        # K=5 at runtime once we are committed to grasping.
        for e, c, h in ik_results[:1 if self._fast_precheck else 5]:
            if e >= 0.1:
                break                      # even the best is out of reach
            if self._path_is_clear(c):
                return c, h
        return None, None

    def _fallback_grasp(self, dc, target_pos):
        """Axis-aligned grasp in front of the hand, used when the asset has no grasp file."""
        cr = dc.site_xmat[self._sites["right"]].reshape(3, 3)
        fd = cr[:, 0].copy()
        fd[2] = 0
        if np.linalg.norm(fd) < 1e-6:
            fd = np.array([1., 0., 0.])
        fd /= np.linalg.norm(fd)
        up = np.array([0., 0., 1.])
        rt = np.cross(up, fd)
        rt /= np.linalg.norm(rt) + 1e-8
        fd = np.cross(rt, up)
        grasp_T = np.eye(4)
        grasp_T[:3, :3] = np.column_stack([fd, rt, up])
        grasp_T[:3, 3] = target_pos
        return grasp_T

    def _commit_grasp(self, grasp_T, Tw):
        """Store the grasp, pregrasp and lift poses and prepare the approach interpolation."""
        off = np.random.uniform(*self.PREGRASP_OFFSET)
        self._pregrasp = (grasp_T[:3, 3] - off * grasp_T[:3, 2]).copy()
        self._grasp_pos = grasp_T[:3, 3].copy()
        self._grasp_rot = grasp_T[:3, :3].copy()
        self._lift_pos = grasp_T[:3, 3].copy()
        self._lift_pos[2] += self.LIFT
        self._grasp_local = np.linalg.inv(Tw) @ grasp_T

        # Fast mode skips the pregrasp IK and the collision-limit scan; both are
        # only needed for runtime grasp execution, not for the feasibility probe.
        if self._fast_precheck:
            self._pregrasp_joints = None
            return
        if self._skip_pregrasp_ik:
            # The WBC controller re-solves IK per step, so this result is unused.
            self._pregrasp_joints = None
        else:
            try:
                self._pregrasp_joints = self._solve_ik(self._pregrasp, self._grasp_rot, self._hand)
            except Exception:
                self._pregrasp_joints = None

        self._build_collision_limit(self._grasp_pos)
        self._ik_cfg = None

    def plan(self, info):
        """Score candidate grasps (IK error, clearance, approach) and lock in grasp/pregrasp/lift poses."""
        self._phase = PHASE_IDLE
        self._hand = "right"
        tp = info.get("target_object_pose")
        vg = info.get("valid_grasps")
        if tp is None:
            self._phase = PHASE_DONE
            return
        # _path_is_clear needs this to exclude the target from collision rejection.
        env = self._env
        self._target_body_id = (env.task.target.body_id
                                if env is not None and env.task.target is not None else -1)

        tp = np.asarray(tp, dtype=np.float64)
        Tw = np.eye(4)
        Tw[:3, :3] = R.from_quat(tp[3:], scalar_first=True).as_matrix()
        Tw[:3, 3] = tp[:3]
        dc = mujoco.MjData(self._model)
        dc.qpos[:] = self._data.qpos
        dc.qvel[:] = self._data.qvel
        dc.ctrl[:] = self._data.ctrl
        mujoco.mj_forward(self._model, dc)

        if vg is not None and len(vg) > 0:
            cands = self._prepend_cached_grasp(
                self._filter_by_reach(self._grasp_candidates(Tw, vg, dc), dc), Tw)
            grasp_T, hand = self._pick_reachable_grasp(cands, dc)
            if grasp_T is None:
                self._phase = PHASE_DONE
                return
            self._hand = hand
        else:
            self._hand = "right"
            grasp_T = self._fallback_grasp(dc, tp[:3])

        self._commit_grasp(grasp_T, Tw)
        self._phase = PHASE_APPROACH

    def refresh_grasp_for_current_object(self):
        """Recompute world grasp/lift poses from the stored object-local grasp after the object moved."""
        env = self._env
        if env is None or self._grasp_local is None:
            return False
        tgt = env.target
        if tgt is None:
            return False
        pos = tgt.position(env.scene.data)
        quat = tgt.quat(env.scene.data)
        Tw = np.eye(4)
        Tw[:3,:3] = R.from_quat(quat, scalar_first=True).as_matrix()
        Tw[:3, 3] = pos
        grasp_T = Tw @ self._grasp_local
        self._grasp_pos = grasp_T[:3, 3].copy()
        self._grasp_rot = grasp_T[:3, :3].copy()
        self._lift_pos = self._grasp_pos.copy(); self._lift_pos[2] += self.LIFT
        return True
