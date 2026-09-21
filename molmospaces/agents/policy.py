"""Scripted walk-and-grasp controller: navigates to the target, then runs the grasp phases."""
from __future__ import annotations

from pathlib import Path

import mink
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp

from molmospaces.agents.grasp_planner import (
    GraspPolicy, HEIGHT_MIN, HEIGHT_MAX, _HANDS, _PHASE_ORDER, _WAIST,
    PHASE_APPROACH, PHASE_CLOSE, PHASE_DESCEND, PHASE_DONE, PHASE_IDLE,
    PHASE_LIFT, PHASE_OPEN_HOLD, PHASE_POST_CLOSE, PHASE_REALIGN)
from molmospaces.agents.nav import _astar, _simplify_path
from molmospaces.components.controller import G1Controller as _BaseController
from molmospaces.components.robot import (PELVIS_FORWARD_OFFSET as _PELVIS_FWD,
                                          XML_PATH as _ROBOT_XML_PATH)

ACTION_DIM = 15  # cmd(3) + height(1) + waist(3) + right_arm(7) + right_grip(1)


class G1Controller(_BaseController):
    WAYPOINT_REACH = 0.10
    FINAL_REACH = 0.05
    SPEED = 0.3
    TURN_KP = 2.0
    MAX_TURN = 1.0
    FACE_TURN = 1.2
    FACE_TOL = 0.1
    FACE_WP_TOL = 0.25
    GRIPPER_OPEN = -0.0222
    GRIPPER_CLOSED = 0.0245

    NEAR_GOAL_DIST = 0.05
    REALIGN_MAX = 0.30
    REALIGN_REACH = 0.020
    REALIGN_YAW_REACH = 0.04
    REALIGN_SPEED_X = 0.25
    REALIGN_SPEED_X_MIN = 0.07
    REALIGN_RAMP_STEPS = 12
    REALIGN_RAMP_DOWN_DIST = 0.08
    REALIGN_INITIAL_HOLD = 20
    REALIGN_SPEED_Y = 0.045
    REALIGN_YAW_SPEED = 0.40
    REALIGN_DRIVE_STEPS = 150
    REALIGN_SETTLE = 80

    def __init__(self, walk_timeout_s=20.0, walk_fail_dist=0.2, face_yaw_offset=0.0,
                 grasp_retry_closer=False):
        """Scripted walk+grasp controller wrapping the ONNX locomotion policies."""
        super().__init__(models_dir=Path(__file__).resolve().parent / "models")
        self._walk_timeout_s = float(walk_timeout_s)
        self._walk_fail_dist = float(walk_fail_dist)
        self._face_yaw_offset_max = float(face_yaw_offset)
        self._grasp_retry_closer = bool(grasp_retry_closer)
        self._env = None
        self._speed = self.SPEED
        self._min_speed = 0.3
        self._grasp_speed_scale = 1.25
        self._nav_noise_scale = 0.0
        self._upper_cmd = None
        # Per-episode state. reset() re-initialises these; declared here so the
        # object's full shape is visible without reading every method.
        self._face_yaw_offset = 0.0
        self._quick_approach = False
        self._stall_min_dist = float("inf")
        self._stall_last_progress_t = 0.0
        self._cmd_smoothed = None
        self._ik_cache = None
        self._last_arm_joints = None
        self._last_waist_joints = None
        self._init_upper = None
        self._info = None
        self._height_cmd = None
        self._gripper_intent = None
        self._goal_xy_target = None
        self._goal_yaw_target = None
        self._init_arm_at_pregrasp = False
        self._start_at_pregrasp = False
        self._pregrasp = None
        self._pregrasp_rot = None
        self._pregrasp_joints = None
        self._lift_pos = None
        self._approach_start = None
        self._approach_slerp = None
        self._descend_start = None
        self._descend_slerp = None
        self._realign_axis = None
        self._realign_offset = 0.0
        self._realign_target_xy = None
        self._realign_arrived_at = None

    @classmethod
    def create(cls, config):
        """Build from a ConfigDict (factory used by the agents registry)."""
        return cls(
            walk_timeout_s=config.get("walk_timeout_s", 20.0),
            walk_fail_dist=config.get("walk_fail_dist", 0.2),
            face_yaw_offset=config.get("face_yaw_offset", 0.0),
            grasp_retry_closer=config.get("grasp_retry_closer", False),
        )

    def set_env(self, env):
        """Attach the env to the controller and its grasp planner."""
        super().set_env(env)
        self._env = env
        self._grasp_planner._env = env

    def _bind_scene_ids(self, model, prefix):
        """Cache the grasp-site ids and the right gripper's qpos address for this scene."""
        self._sites = {h: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{prefix}{c['site']}")
                       for h, c in _HANDS.items()}
        r_grip_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{prefix}right_Joint1_1")
        self._right_grip_qa = model.jnt_qposadr[r_grip_jid] if r_grip_jid >= 0 else None

    def _build_ik_model(self, model, prefix):
        """Load the standalone IK model and map scene qpos slices onto it."""
        self._ik_cfg = mink.Configuration(mujoco.MjModel.from_xml_path(_ROBOT_XML_PATH))
        ik_model = self._ik_cfg.model
        ik_fj = mujoco.mj_name2id(ik_model, mujoco.mjtObj.mjOBJ_JOINT, "floating_base_joint")
        self._ik_fj_dof = ik_model.jnt_dofadr[ik_fj]
        self._ik_fj_qa = ik_model.jnt_qposadr[ik_fj]
        self._scene_to_ik_qpos = []
        for jid in range(ik_model.njnt):
            jname = mujoco.mj_id2name(ik_model, mujoco.mjtObj.mjOBJ_JOINT, jid)
            if not jname:
                continue
            scene_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{prefix}{jname}")
            if scene_jid < 0:
                continue
            qsz = 7 if ik_model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE else 1
            self._scene_to_ik_qpos.append(
                (model.jnt_qposadr[scene_jid], ik_model.jnt_qposadr[jid], qsz))
        return ik_model

    def _build_hand_tasks(self, ik_model):
        """Per-hand mink frame task, DOF mask and arm qpos addresses."""
        self._ik_hand_cfg = {}
        for hand in _HANDS:
            arm_joints = list(_HANDS[hand]["arm"])
            mask = np.zeros(ik_model.nv)
            for jn in arm_joints + _WAIST:
                jid = mujoco.mj_name2id(ik_model, mujoco.mjtObj.mjOBJ_JOINT, jn)
                if jid >= 0:
                    mask[ik_model.jnt_dofadr[jid]] = 1.0
            mask_h = mask.copy()
            mask_h[self._ik_fj_dof + 2] = 1.0
            arm_qa = np.array(
                [ik_model.jnt_qposadr[mujoco.mj_name2id(ik_model, mujoco.mjtObj.mjOBJ_JOINT, jn)]
                 for jn in arm_joints], dtype=np.int32)
            self._ik_hand_cfg[hand] = {
                "mask": mask, "mask_h": mask_h, "arm_joints": arm_joints, "arm_qa": arm_qa,
                "task": mink.FrameTask(frame_name=_HANDS[hand]["site"], frame_type="site",
                                       position_cost=100, orientation_cost=1, lm_damping=1),
            }

    def _build_ik_tasks(self, ik_model):
        """Posture and pelvis tasks, the waist tilt envelope, and the IK limits."""
        posture_cost = np.full(ik_model.nv, 0.1)
        # Waist cost below the pelvis-z cost so reaching prefers tilting the torso
        # over crouching, while still springing back toward upright.
        for jn in _WAIST:
            jid = mujoco.mj_name2id(ik_model, mujoco.mjtObj.mjOBJ_JOINT, jn)
            if jid >= 0:
                posture_cost[ik_model.jnt_dofadr[jid]] = 0.2
        posture_cost[self._ik_fj_dof + 2] = 0.1
        self._waist_qa = np.array(
            [ik_model.jnt_qposadr[mujoco.mj_name2id(ik_model, mujoco.mjtObj.mjOBJ_JOINT, jn)]
             for jn in _WAIST], dtype=np.int32)
        self._ik_posture = mink.PostureTask(ik_model, cost=posture_cost)
        # xy strongly anchored so walking is unaffected, z loose so the wrist task
        # can pull the base down for low handles.
        self._ik_pelvis_task = mink.FrameTask(
            frame_name="pelvis", frame_type="body",
            position_cost=[5.0, 5.0, 0.3], orientation_cost=0, lm_damping=1)
        # Waist torso-tilt envelope (keep in sync with g1_server deploy clips).
        for jn, lo, hi in [("waist_yaw_joint",   -0.5, 0.5),
                           ("waist_roll_joint",  -0.4, 0.4),
                           ("waist_pitch_joint", -0.1, 0.4)]:   # pitch asymmetric
            jid = mujoco.mj_name2id(ik_model, mujoco.mjtObj.mjOBJ_JOINT, jn)
            if jid >= 0:
                ik_model.jnt_range[jid] = [lo, hi]
        self._ik_limits = [mink.ConfigurationLimit(ik_model)]
        # Applied only during grasp execution, see _solve_ik_wbc.
        self._self_collision_limit = self._build_self_collision_limit(ik_model)
        self._active_hand = "right"

    def _build_grasp_planner(self, model, data, prefix):
        """Attach the grasp planner, wired to score candidates with the whole-body IK."""
        self._grasp_planner = GraspPolicy()
        self._grasp_planner.setup(model, data, prefix)
        self._grasp_planner._use_height = False
        self._grasp_planner._ik_error_fn = self._wbc_ik_error
        # The WBC re-solves IK per step and never reads pregrasp_joints.
        self._grasp_planner._skip_pregrasp_ik = True

    def setup(self, model, data, prefix="robot_0/"):
        """Bind to the scene model: site ids, gripper address, and the mink whole-body IK stack."""
        super().setup(model, data, prefix)
        self._bind_scene_ids(model, prefix)
        ik_model = self._build_ik_model(model, prefix)
        self._build_hand_tasks(ik_model)
        self._build_ik_tasks(ik_model)

        self._waypoints = []
        self._wp_idx = 0
        self._arrived = self._facing = False
        self._target_xy = self._object_xy = None
        self._has_path = False

        self._grasp_phase = PHASE_IDLE
        self._grasp_step = 0
        self._grasp_pos = None
        self._grasp_rot = None
        self._hand = "right"
        self._grasp_retries_used = 0

        self._build_grasp_planner(model, data, prefix)

    @property
    def has_path(self):
        """True if the last reset produced a navigable path."""
        return self._has_path

    def in_precision_phase(self):
        """True during grasp phases where action noise must be suppressed."""
        return self._grasp_phase in (PHASE_REALIGN, PHASE_DESCEND, PHASE_OPEN_HOLD,
                                     PHASE_CLOSE, PHASE_POST_CLOSE, PHASE_LIFT)

    @property
    def done(self):
        """True once the grasp sequence has finished."""
        return self._grasp_phase == PHASE_DONE

    def precheck_grasp(self, info):
        """Teleport to the goal pose and dry-run the grasp planner; caches the found grasp for reuse."""
        env = self._env
        if env is None:
            return True
        goal_xy = info.get("goal_xy"); tgt = info.get("target_object_position")
        if goal_xy is None or tgt is None:
            return False
        goal_yaw = info.get("goal_yaw")
        if goal_yaw is None:
            goal_yaw = float(np.arctan2(tgt[1]-goal_xy[1], tgt[0]-goal_xy[0]))
        saved_qpos = env.scene.data.qpos.copy()
        saved_ctrl = env.scene.data.ctrl.copy()
        try:
            env.robot.set_pose(goal_xy, goal_yaw)
            self._set_groot_defaults()
            mujoco.mj_forward(env.scene.model, env.scene.data)
            self._grasp_planner._fast_precheck = True
            try:
                self._grasp_planner.plan(info)
            finally:
                self._grasp_planner._fast_precheck = False
            found = self._grasp_planner._phase != PHASE_DONE
            if found:
                p = self._grasp_planner
                self._grasp_planner._cached = dict(
                    grasp_local=p._grasp_local.copy(),
                    grasp_pos=p._grasp_pos.copy(),
                    grasp_rot=p._grasp_rot.copy(),
                    pregrasp=p._pregrasp.copy(),
                    pregrasp_joints=dict(p._pregrasp_joints) if getattr(p, "_pregrasp_joints", None) else None,
                    hand=p._hand)
            return found
        finally:
            env.scene.data.qpos[:] = saved_qpos
            env.scene.data.ctrl[:] = saved_ctrl
            mujoco.mj_forward(env.scene.model, env.scene.data)

    def _xy(self):
        """Robot ground position: pelvis xy pushed forward by the pelvis offset."""
        px, py = self._data.xpos[self._pelvis, :2]
        yaw = self._yaw()
        c, s = np.cos(yaw), np.sin(yaw)
        return np.array([px + c * _PELVIS_FWD, py + s * _PELVIS_FWD])

    def _yaw(self):
        """Pelvis heading angle in the world frame."""
        m = self._data.xmat[self._pelvis].reshape(3, 3)
        return np.arctan2(m[1, 0], m[0, 0])

    def _plan_path(self, info):
        """A* plan from the current position to the goal, storing simplified world waypoints."""
        occ = info.get("nav_occupancy_map") or info.get("occupancy_map")
        tgt = info.get("target_object_position")
        if occ is None or tgt is None:
            return
        self._object_xy = np.asarray(tgt[:2], dtype=np.float64)
        goal_xy = info.get("goal_xy")
        self._target_xy = np.asarray(goal_xy[:2] if goal_xy is not None else tgt[:2], dtype=np.float64)
        path = _astar(occ.occupancy, occ._world_to_px(self._xy()), occ._world_to_px(self._target_xy))
        if path:
            path = _simplify_path(path, occ.occupancy)
            self._waypoints = [(occ.map_to_world @ np.array([r, c, 1.0]))[:2] for r, c in path]
            if goal_xy is not None and self._waypoints:
                self._waypoints[-1] = np.asarray(goal_xy[:2], dtype=np.float64)
            self._has_path = True

    def _update_nav_command(self):
        """Compute the (vx, vy, vyaw) walk command: face, drive waypoints, brake into the goal."""
        if self._arrived:
            self._cmd[:] = 0
            return
        if not self._waypoints:
            self._cmd[:] = 0
            return
        xy, yaw = self._xy(), self._yaw()
        if self._facing:
            face = self._object_xy if self._object_xy is not None else self._target_xy
            if face is not None:
                desired = (np.arctan2(face[1]-xy[1], face[0]-xy[0])
                           + self._face_yaw_offset)
                ye = (desired - yaw + np.pi) % (2*np.pi) - np.pi
                if abs(ye) > self.FACE_TOL:
                    self._cmd[:] = [0, 0, np.clip(self.TURN_KP*ye, -self.FACE_TURN, self.FACE_TURN)]
                    return
            self._arrived = True; self._cmd[:] = 0
            return
        wp = self._waypoints[self._wp_idx]
        if np.linalg.norm(xy - wp) < self.WAYPOINT_REACH and self._wp_idx < len(self._waypoints)-1:
            self._wp_idx += 1
        wp = self._waypoints[self._wp_idx]
        delta = wp - xy; dist = np.linalg.norm(delta)
        final = self._wp_idx >= len(self._waypoints) - 1
        # Smoothstep brake — zero derivative at both ends so we "roll" into a stop
        # instead of stepping. Longer runway (0.70 m) gives ~2.3s of decel at
        # 0.3 m/s — feels natural in real-life walking.
        BRAKE_DIST = 0.70
        STOP_PAD = 0.04  # extra margin to absorb walking inertia
        stop_dist = self.FINAL_REACH + STOP_PAD
        # Arrive at stop_dist (where the brake zeros speed), else robot hangs in the
        # dead zone short of FINAL_REACH.
        if final and dist <= stop_dist:
            self._facing = True; self._cmd[:] = 0
            return
        ye = (np.arctan2(delta[1], delta[0]) - yaw + np.pi) % (2*np.pi) - np.pi
        if abs(ye) > self.FACE_WP_TOL:
            self._cmd[:] = [0, 0, np.clip(self.TURN_KP*ye, -self.MAX_TURN, self.MAX_TURN)]
            return
        if final:
            if dist <= stop_dist:
                spd = 0.0
            elif dist >= BRAKE_DIST:
                spd = self._speed
            else:
                t = (dist - stop_dist) / (BRAKE_DIST - stop_dist)
                spd = self._speed * (3 * t * t - 2 * t * t * t)
        else:
            spd = np.clip(dist, self._min_speed, self._speed)
        c, s = np.cos(yaw), np.sin(yaw)
        lx, ly = c*delta[0]+s*delta[1], -s*delta[0]+c*delta[1]
        ln = max(np.sqrt(lx**2+ly**2), 1e-6)
        ang = np.sign(self.TURN_KP*ye) * np.clip(abs(self.TURN_KP*ye), 0.05, self.MAX_TURN)
        self._cmd[:] = [spd*lx/ln, np.clip(spd*ly/ln, -0.5, 0.5), ang]

    def perturb_action_for_rollout(self, action):
        """Add clipped gaussian nav noise to the base command (idle nav only)."""
        out = np.asarray(action, dtype=np.float32).copy()
        if self._nav_noise_scale <= 0 or self._arrived or self._grasp_phase != PHASE_IDLE:
            return out
        if not self._waypoints or np.linalg.norm(out[0:3]) < 1e-6:
            return out
        out[0:2] += np.random.normal(0.0, 0.02 * self._nav_noise_scale, size=2)
        out[2] += np.random.normal(0.0, 0.05 * self._nav_noise_scale)
        out[0] = np.clip(out[0], -self._speed, self._speed)
        out[1] = np.clip(out[1], -0.5, 0.5)
        out[2] = np.clip(out[2], -self.MAX_TURN, self.MAX_TURN)
        return out

    def _wbc_ik_error(self, T, hand):
        """IK error for pose T using the whole-body solver with `hand` active."""
        saved_hand = self._active_hand
        self._active_hand = hand
        _, _, _, err = self._solve_ik_wbc(T[:3, 3], T[:3, :3])
        self._active_hand = saved_hand
        return err

    def _sync_ik_qpos(self):
        """Copy the scene qpos into the standalone IK model's layout."""
        ik_q = np.zeros(self._ik_cfg.model.nq, dtype=np.float64)
        for scene_qa, ik_qa, qsz in self._scene_to_ik_qpos:
            ik_q[ik_qa:ik_qa+qsz] = self._data.qpos[scene_qa:scene_qa+qsz]
        return ik_q

    def _build_self_collision_limit(self, ik_model):
        """Self-collision limit over arm/torso and arm/arm geom pairs, or None if there are none."""
        right_arm, left_arm, body = [], [], []
        for gid in range(ik_model.ngeom):
            # Only geoms that actually collide (skip visual-only meshes).
            if ik_model.geom_contype[gid] == 0 and ik_model.geom_conaffinity[gid] == 0:
                continue
            bid = ik_model.geom_bodyid[gid]
            bname = mujoco.mj_id2name(ik_model, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
            if any(s in bname for s in ("shoulder", "elbow", "wrist", "gripper")):
                (right_arm if "right" in bname else left_arm if "left" in bname else []).append(gid)
            elif any(s in bname for s in ("pelvis", "torso", "hip", "waist")):
                body.append(gid)
        arm = right_arm + left_arm
        pairs = []
        if arm and body:
            pairs.append((arm, body))
        if right_arm and left_arm:
            pairs.append((right_arm, left_arm))
        if not pairs:
            return None
        return mink.CollisionAvoidanceLimit(
            model=ik_model, geom_pairs=pairs,
            minimum_distance_from_collisions=0.02, collision_detection_distance=0.08)

    def _solve_ik_wbc(self, target_pos, target_rot=None, avoid_self_collision=False):
        """Whole-body IK: track the wrist target while keeping posture and standing height; returns (arm, waist, height, err)."""
        hcfg = self._ik_hand_cfg[self._active_hand]
        self._ik_cfg.update(self._sync_ik_qpos())
        q_post = self._ik_cfg.q.copy()
        q_post[self._waist_qa] = 0.0
        self._ik_posture.set_target(q_post)
        # Anchor pelvis xy to current (walking unaffected) but z to standing height
        # so the IK actively wants to stand back up when the wrist target allows it.
        pelvis_T = self._ik_cfg.get_transform_frame_to_world("pelvis", "body")
        pelvis_pos = pelvis_T.translation().copy()
        pelvis_pos[2] = HEIGHT_MAX
        self._ik_pelvis_task.set_target(
            mink.SE3.from_rotation_and_translation(pelvis_T.rotation(), pelvis_pos))
        rot = mink.SO3.from_matrix(target_rot) if target_rot is not None else mink.SO3.identity()
        hcfg["task"].set_target(mink.SE3.from_rotation_and_translation(rot, np.asarray(target_pos, dtype=np.float64)))
        # Tighter IK during precision phases — wrist has to actually land on the grasp/close target.
        precision = self._grasp_phase in (PHASE_DESCEND, PHASE_CLOSE, PHASE_POST_CLOSE, PHASE_LIFT)
        max_iters = 60 if precision else 20
        conv_thresh = 0.0015 if precision else 0.005
        prev_err = float("inf")
        err = float("inf")
        limits = self._ik_limits
        if avoid_self_collision and self._self_collision_limit is not None:
            limits = self._ik_limits + [self._self_collision_limit]
        for step in range(max_iters):
            try:
                vel = mink.solve_ik(
                    self._ik_cfg,
                    [hcfg["task"], self._ik_posture, self._ik_pelvis_task],
                    1e-2, "daqp", damping=1e-1, limits=limits)
            except Exception:
                break
            vel *= hcfg["mask_h"]
            self._ik_cfg.integrate_inplace(vel, 1e-2)
            q_tmp = self._ik_cfg.q.copy()
            q_tmp[self._ik_fj_qa + 2] = np.clip(q_tmp[self._ik_fj_qa + 2], HEIGHT_MIN, HEIGHT_MAX)
            self._ik_cfg.update(q_tmp)
            err = float(np.linalg.norm(hcfg["task"].compute_error(self._ik_cfg)[:3]))
            if err < conv_thresh:
                break
            if step > 10 and err > prev_err - 1e-5:
                break
            prev_err = err
        q = self._ik_cfg.q
        arm = q[hcfg["arm_qa"]].astype(np.float32)
        waist = q[self._waist_qa].astype(np.float32)
        ik_h = float(np.clip(q[self._ik_fj_qa + 2], HEIGHT_MIN, HEIGHT_MAX))
        return arm, waist, ik_h, err

    def _start_grasp(self, info):
        """Enter the grasp sequence, reusing the precheck-cached grasp when available or planning fresh."""
        self._grasp_planner._env = self._env
        cached = self._grasp_planner._cached
        if self._start_at_pregrasp and cached is not None and "pregrasp" in cached:
            self._grasp_pos = cached["grasp_pos"].copy()
            self._grasp_rot = cached["grasp_rot"].copy()
            self._pregrasp = cached["pregrasp"].copy()
            self._grasp_planner._grasp_local = cached["grasp_local"].copy()
            self._grasp_planner._grasp_pos = cached["grasp_pos"].copy()
            self._grasp_planner._grasp_rot = cached["grasp_rot"].copy()
            self._grasp_planner._pregrasp = cached["pregrasp"].copy()
            self._grasp_planner._hand = cached["hand"]
            self._grasp_planner._phase = PHASE_APPROACH
            self._pregrasp_rot = None
            self._lift_pos = self._grasp_pos.copy()
            self._lift_pos[2] += GraspPolicy.LIFT
            self._hand = cached["hand"]
            self._active_hand = self._hand
            self._grasp_phase = PHASE_REALIGN
            self._grasp_step = 0
            self._setup_axis_realign()
            return
        self._refresh_info_target_pose(info)
        self._grasp_planner.plan(info)
        if self._grasp_planner._phase == PHASE_DONE:
            # Optional retry: inch closer to the object once and re-walk before giving up.
            if self._grasp_retry_closer and self._grasp_retries_used < 1 \
                    and self._install_closer_waypoint(info):
                self._grasp_retries_used += 1
                self._grasp_planner._phase = PHASE_IDLE
                return
            self._grasp_phase = PHASE_DONE
            return
        self._grasp_pos = self._grasp_planner._grasp_pos.copy()
        self._grasp_rot = self._grasp_planner._grasp_rot.copy()
        self._pregrasp = self._grasp_planner._pregrasp.copy()
        self._pregrasp_rot = None
        offset = info.get("pregrasp_offset")
        if offset is not None:
            xyz_off, rot_off = offset
            self._pregrasp = self._pregrasp + xyz_off
            self._pregrasp_rot = rot_off @ self._grasp_rot
        self._lift_pos = self._grasp_pos.copy()
        self._lift_pos[2] += GraspPolicy.LIFT
        self._hand = self._grasp_planner._hand
        self._active_hand = self._hand
        self._grasp_phase = PHASE_APPROACH
        self._grasp_step = 0

    def _install_closer_waypoint(self, info):
        """Retry the grasp from a closer standoff; False if already at the minimum."""
        env = self._env
        if env is None:
            return False
        obj_xy = info.get("target_object_position")
        if obj_xy is None:
            return False
        obj_xy = np.asarray(obj_xy, dtype=np.float64)[:2]
        cur = self._xy().astype(np.float64)
        delta = obj_xy - cur
        cur_standoff = float(np.linalg.norm(delta))
        r_min = float(getattr(env, "_grasp_spawn_radius_min", 0.25))
        if cur_standoff <= r_min + 1e-3:
            return False
        # Sample in the middle 50% of [r_min, cur_standoff] — avoid the bottom 25%
        # (riskiest, closest to object) and the top 25% (no meaningful progress).
        span = cur_standoff - r_min
        lo, hi = r_min + 0.25 * span, r_min + 0.75 * span
        new_standoff = float(np.random.uniform(lo, hi))
        direction = delta / cur_standoff
        new_wp = obj_xy - new_standoff * direction
        self._waypoints = [np.asarray(new_wp, dtype=np.float64)]
        self._wp_idx = 0
        self._arrived = False
        self._facing = False
        self._has_path = True
        self._grasp_phase = PHASE_IDLE
        self._grasp_step = 0
        self._ik_cache = None
        self._upper_cmd = None
        self._cmd_smoothed = None
        return True

    def _setup_axis_realign(self):
        """Set the realign target to the planned goal pose and reset arrival tracking."""
        self._realign_target_xy = (self._goal_xy_target.copy()
                                   if self._goal_xy_target is not None else self._xy())
        self._realign_arrived_at = None

    def _refresh_info_target_pose(self, info):
        """Sync info's target_object_* with live sim pose; object may have shifted during nav."""
        if self._env is None or info is None:
            return
        tgt = self._env.target
        if tgt is None:
            return
        try:
            pos = tgt.position(self._env.scene.data)
            quat = tgt.quat(self._env.scene.data)
        except Exception:
            return
        info["target_object_position"] = np.asarray(pos, dtype=np.float64)
        info["target_object_pose"] = np.concatenate([pos, quat]).astype(np.float64)

    def _grasp_target(self):
        """Interpolated wrist (pos, rot) target for the current grasp phase."""
        t = min(self._grasp_step / max(1, self._grasp_steps()), 1.0)
        p = self._grasp_phase
        if p == PHASE_APPROACH:
            sid = self._sites[self._hand]
            if self._grasp_step == 0:
                self._approach_start = self._data.site_xpos[sid].copy()
                target_rot = self._pregrasp_rot if self._pregrasp_rot is not None else self._grasp_rot
                self._approach_slerp = Slerp([0, 1], R.concatenate([
                    R.from_matrix(self._data.site_xmat[sid].reshape(3, 3)),
                    R.from_matrix(target_rot)]))
            pos = self._approach_start*(1-t) + self._pregrasp*t
            rot = self._approach_slerp(t).as_matrix()
            return pos, rot
        if p == PHASE_DESCEND:
            sid = self._sites[self._hand]
            if self._grasp_step == 0:
                self._descend_start = self._data.site_xpos[sid].copy()
                start_rot = self._data.site_xmat[sid].reshape(3, 3).copy()
                self._descend_slerp = Slerp([0, 1], R.concatenate([
                    R.from_matrix(start_rot),
                    R.from_matrix(self._grasp_rot)]))
            rot = self._descend_slerp(t).as_matrix()
            return self._descend_start*(1-t) + self._grasp_pos*t, rot
        if p == PHASE_REALIGN:
            return self._pregrasp, self._pregrasp_rot if self._pregrasp_rot is not None else self._grasp_rot
        if p in (PHASE_OPEN_HOLD, PHASE_CLOSE, PHASE_POST_CLOSE): return self._grasp_pos, self._grasp_rot
        if p == PHASE_LIFT: return self._grasp_pos*(1-t) + self._lift_pos*t, self._grasp_rot
        return None, None

    def _grasp_steps(self):
        """Step budget for the current grasp phase, scaled by grasp speed."""
        steps = {
            "approach": 700,
            "realign": 500,
            "descend": 600,
            "open_hold": 160,
            "close": 800,
            "post_close": 400,
            "lift": 600,
        }.get(self._grasp_phase, 1)
        if self._grasp_phase in (PHASE_APPROACH, PHASE_DESCEND, PHASE_LIFT):
            steps = int(round(steps / self._grasp_speed_scale))
        return max(1, steps)

    def _grasp_min_steps(self):
        """Minimum steps before a phase may be declared done."""
        if self._grasp_phase == PHASE_APPROACH and self._quick_approach:
            return 5
        return {"approach": 40, "realign": 20, "descend": 40, "open_hold": 10, "close": 240, "post_close": 80, "lift": 40}.get(self._grasp_phase, 1)

    def _grasp_phase_goal(self):
        """Terminal wrist (pos, rot) the current phase is converging to."""
        p = self._grasp_phase
        if p == PHASE_APPROACH or p == PHASE_REALIGN:
            return self._pregrasp, self._grasp_rot
        if p in (PHASE_DESCEND, PHASE_OPEN_HOLD, PHASE_CLOSE, PHASE_POST_CLOSE):
            return self._grasp_pos, self._grasp_rot
        if p == PHASE_LIFT:
            return self._lift_pos, self._grasp_rot
        return None, None

    def _grasp_phase_done(self):
        """Phase completion test: gripper closed for CLOSE, wrist-on-target for motion phases."""
        if self._grasp_step < self._grasp_min_steps():
            return False
        if self._grasp_phase == PHASE_CLOSE:
            if self._right_grip_qa is None:
                return False
            return self._data.qpos[self._right_grip_qa] > self.GRIPPER_CLOSED - 0.004
        if self._grasp_phase in (PHASE_OPEN_HOLD, PHASE_POST_CLOSE):
            return True
        if self._grasp_phase == PHASE_REALIGN:
            if self._start_at_pregrasp:
                if self._realign_arrived_at is not None:
                    return (self._grasp_step - self._realign_arrived_at) >= self.REALIGN_SETTLE
                return False
            close = float(np.linalg.norm(self._xy() - self._realign_target_xy)) < self.REALIGN_REACH
            if not close:
                self._realign_arrived_at = None
                return False
            if self._realign_arrived_at is None:
                self._realign_arrived_at = self._grasp_step
            return (self._grasp_step - self._realign_arrived_at) >= self.REALIGN_SETTLE
        goal_pos, goal_rot = self._grasp_phase_goal()
        if goal_pos is None:
            return False
        sid = self._sites[self._hand]
        pos_err = float(np.linalg.norm(self._data.site_xpos[sid] - goal_pos))
        # angle(R1^T @ R2) = acos((trace(R1^T @ R2) - 1) / 2) — no scipy.Rotation overhead.
        R1 = self._data.site_xmat[sid].reshape(3, 3)
        rel_trace = float(np.dot(R1.ravel(), goal_rot.ravel()))
        cos = max(-1.0, min(1.0, (rel_trace - 1.0) * 0.5))
        rot_err = float(np.arccos(cos))
        return pos_err < 0.035 and rot_err < 0.45

    def _advance_grasp(self):
        """Advance the grasp phase machine, refreshing grasp poses after approach/realign."""
        prev = self._grasp_phase
        if prev == PHASE_REALIGN:
            self._grasp_phase = PHASE_DESCEND
            self._grasp_step = 0
            self._ik_cache = None
            self._upper_cmd = None
            self._realign_arrived_at = None
            self._quick_approach = True
            if self._grasp_planner.refresh_grasp_for_current_object():
                self._grasp_pos = self._grasp_planner._grasp_pos.copy()
                self._grasp_rot = self._grasp_planner._grasp_rot.copy()
                self._lift_pos = self._grasp_planner._lift_pos.copy()
            return
        try:
            idx = _PHASE_ORDER.index(self._grasp_phase)
            self._grasp_phase = _PHASE_ORDER[idx + 1]
        except (ValueError, IndexError):
            self._grasp_phase = PHASE_DONE
        self._grasp_step = 0
        self._ik_cache = None
        self._upper_cmd = None
        self._quick_approach = False
        if prev == PHASE_APPROACH and self._grasp_phase == PHASE_DESCEND:
            if self._grasp_planner.refresh_grasp_for_current_object():
                self._grasp_pos = self._grasp_planner._grasp_pos.copy()
                self._grasp_rot = self._grasp_planner._grasp_rot.copy()
                self._lift_pos = self._grasp_planner._lift_pos.copy()

    def _realign_cmd(self):
        """Base velocity command that walks the residual pregrasp error to zero (per realign axis)."""
        if self._start_at_pregrasp and self._realign_axis == "x":
            if self._grasp_step < self.REALIGN_INITIAL_HOLD:
                return np.zeros(3, dtype=np.float32)
            delta_world = self._realign_target_xy - self._xy()
            yaw = self._yaw()
            lx = np.cos(yaw) * delta_world[0] + np.sin(yaw) * delta_world[1]
            direction = -float(np.sign(self._realign_offset))
            signed_progress_remaining = direction * lx
            if signed_progress_remaining <= self.REALIGN_REACH:
                self._realign_arrived_at = self._realign_arrived_at or self._grasp_step
                return np.zeros(3, dtype=np.float32)
            spd_up = self.REALIGN_SPEED_X_MIN + (self.REALIGN_SPEED_X - self.REALIGN_SPEED_X_MIN) * \
                min(1.0, self._grasp_step / max(1.0, self.REALIGN_RAMP_STEPS))
            decel_t = min(1.0, signed_progress_remaining / max(1e-6, self.REALIGN_RAMP_DOWN_DIST))
            spd_down = self.REALIGN_SPEED_X_MIN + (self.REALIGN_SPEED_X - self.REALIGN_SPEED_X_MIN) * decel_t
            speed = min(spd_up, spd_down)
            return np.array([direction * speed, 0.0, 0.0], dtype=np.float32)
        if self._start_at_pregrasp and self._realign_axis == "yaw":
            yaw_err = (self._goal_yaw_target - self._yaw() + np.pi) % (2 * np.pi) - np.pi
            direction = -float(np.sign(self._realign_offset))
            signed_progress_remaining = direction * (-yaw_err)
            if abs(yaw_err) <= self.REALIGN_YAW_REACH or signed_progress_remaining <= 0:
                self._realign_arrived_at = self._realign_arrived_at or self._grasp_step
                return np.zeros(3, dtype=np.float32)
            return np.array([0.0, 0.0, direction * self.REALIGN_YAW_SPEED], dtype=np.float32)
        if self._start_at_pregrasp and self._realign_axis == "y":
            vy = -float(np.sign(self._realign_offset)) * self.REALIGN_SPEED_Y
            return np.array([0.0, vy, 0.0], dtype=np.float32)
        delta_world = self._realign_target_xy - self._xy()
        yaw = self._yaw()
        c, s = np.cos(yaw), np.sin(yaw)
        lx = c * delta_world[0] + s * delta_world[1]
        ly = -s * delta_world[0] + c * delta_world[1]
        dist = float(np.linalg.norm(delta_world))
        if dist < self.REALIGN_REACH:
            return np.zeros(3, dtype=np.float32)
        ln = max(np.sqrt(lx * lx + ly * ly), 1e-6)
        speed = min(self.REALIGN_SPEED_X, dist * 2.0)
        return np.array([speed * lx / ln, speed * ly / ln, 0.0], dtype=np.float32)

    def _clear_episode_state(self):
        """Reset all per-episode nav, grasp and smoothing state."""
        self._arrived = self._facing = False
        self._waypoints = []
        self._wp_idx = 0
        self._has_path = False
        self._grasp_phase = PHASE_IDLE
        self._grasp_step = 0
        self._ik_cache = None
        self._grasp_retries_used = 0
        self._upper_cmd = None
        self._realign_target_xy = None
        self._realign_arrived_at = None
        self._quick_approach = False
        self._last_arm_joints = None
        self._last_waist_joints = None
        self._cmd_smoothed = None
        self._stall_min_dist = float("inf")
        self._stall_last_progress_t = 0.0

    def _read_reset_info(self, info):
        """Pull this episode's start pose and goal out of the env's reset info."""
        self._init_arm_at_pregrasp = bool(info and info.get("init_arm_at_pregrasp"))
        self._start_at_pregrasp = bool(info and info.get("start_at_pregrasp"))
        self._realign_axis = info.get("realign_axis") if info else None
        self._realign_offset = float(info.get("realign_offset", 0.0)) if info else 0.0
        self._pregrasp_joints = (dict(info["pregrasp_joints"])
                                 if info and info.get("pregrasp_joints") else None)
        self._goal_xy_target = (np.asarray(info["goal_xy"], dtype=np.float64).copy()
                                if info and info.get("goal_xy") is not None else None)
        self._goal_yaw_target = float(info["goal_yaw"]) if info and info.get("goal_yaw") is not None else None

    def _sample_episode_style(self, info):
        """Draw the per-episode motion style (walk speed, facing offset, grasp pace, nav noise)."""
        self._speed = float(np.random.uniform(0.3, 0.5))
        self._min_speed = 0.05
        r = float(info.get("face_yaw_offset_max", self._face_yaw_offset_max)) if info else self._face_yaw_offset_max
        self._face_yaw_offset = float(np.random.uniform(-r, r)) if r > 0 else 0.0
        self._grasp_speed_scale = float(np.random.uniform(0.85, 1.15))
        self._nav_noise_scale = float(np.random.choice([0.0, 1.0], p=[0.7, 0.3]))

    def _begin_nav_or_grasp(self, info):
        """Grasp straight away when already at the goal, otherwise plan a path to it."""
        goal_xy = info.get("goal_xy")
        already_there = (self._start_at_pregrasp or
                         (goal_xy is not None and
                          np.linalg.norm(self._xy() - np.asarray(goal_xy)) < self.NEAR_GOAL_DIST))
        if already_there:
            self._arrived = True
            self._has_path = True
            self._start_grasp(info)
        else:
            self._plan_path(info)

    def _apply_initial_pose(self, info):
        """Teleport the upper body to the episode's requested start pose, if any."""
        init_height = info.get("init_height") if info else None
        if init_height is not None:
            self._height_cmd = float(init_height)

        if self._start_at_pregrasp and self._pregrasp_joints is not None and self._env is not None:
            self._env.robot.apply_arm_pose(self._pregrasp_joints)
            self._sync_upper_targets()
            self._init_upper = None
            return

        init_upper = info.get("init_upper_pose")
        if init_upper is not None and self._env is not None:
            self._env.robot.apply_upper_pose(init_upper)
            self._sync_upper_targets()
        self._init_upper = np.asarray(init_upper, dtype=np.float32) if init_upper is not None else None

    def _sync_upper_targets(self):
        """Re-latch the upper-body PD targets onto the pose just written into qpos."""
        self._target_q[15:] = self._data.qpos[self._jqpos[15:]].astype(np.float32)
        mujoco.mj_forward(self._model, self._data)

    def reset(self, info):
        """Clear all nav/grasp state and plan the path for a new episode."""
        self._clear_episode_state()
        self._read_reset_info(info)
        self._sample_episode_style(info)
        super()._reset_wbc_state()
        self._info = info
        self._begin_nav_or_grasp(info)
        super()._write_default_pose()
        self._apply_initial_pose(info)

    def set_step_info(self, info):
        """Store the latest env info dict for the next sample_actions call."""
        self._info = info

    def _recover_if_stalled(self):
        """Fast-forward to arrived when a pregrasp-posed robot stops making walking progress."""
        if not (self._init_arm_at_pregrasp and not self._arrived
                and self._grasp_phase == PHASE_IDLE and self._waypoints and self._env is not None):
            return
        cur_dist = float(np.linalg.norm(self._xy() - self._waypoints[-1]))
        if cur_dist < self._stall_min_dist - 0.02:      # 2cm counts as progress
            self._stall_min_dist = cur_dist
            self._stall_last_progress_t = self._env.time
        elif self._env.time - self._stall_last_progress_t > 0.5:
            self._arrived = True
            self._facing = True
            self._stall_min_dist = float("inf")
            self._stall_last_progress_t = self._env.time

    def _give_up_if_walk_timed_out(self):
        """End the episode if the robot is still far from the goal past the walk timeout."""
        if (not self._arrived and self._grasp_phase == PHASE_IDLE
                and self._env is not None
                and self._env.time > self._walk_timeout_s
                and self._target_xy is not None
                and float(np.linalg.norm(self._xy() - self._target_xy)) > self._walk_fail_dist):
            self._grasp_phase = PHASE_DONE

    def _update_base_command(self):
        """Set the base velocity command: realign drive, or nav with low-pass smoothing."""
        self._update_nav_command()
        if self._grasp_phase == PHASE_REALIGN:
            self._cmd[:] = self._realign_cmd()
            return
        target = self._cmd.copy()
        prev = self._cmd_smoothed
        if prev is None:
            prev = np.zeros(3, dtype=np.float32)
        if self._arrived or self._facing:
            self._cmd_smoothed = target.astype(np.float32)
        else:
            alpha = 0.15
            self._cmd_smoothed = (1.0 - alpha) * prev + alpha * target
            self._cmd_smoothed[2] = float(target[2])
        self._cmd[:] = self._cmd_smoothed

    def _step_grasp_phase(self):
        """Advance the grasp phase machine one step; returns its wrist target and gripper fraction."""
        pos, rot, grip = None, None, 0.0
        self._gripper_intent = 0.0
        if self._grasp_phase in (PHASE_IDLE, PHASE_DONE):
            return pos, rot, grip
        if self._grasp_phase_done():
            self._advance_grasp()
        pos, rot = self._grasp_target()
        if self._grasp_phase in (PHASE_CLOSE, PHASE_POST_CLOSE, PHASE_LIFT):
            grip = 1.0
        # main.py records this binary intent so the saved label flips at PHASE_CLOSE.
        self._gripper_intent = float(grip)
        if grip > 0 and self._grasp_phase == PHASE_CLOSE:
            grip = min(self._grasp_step / max(1, self._grasp_steps()), 1.0)
        self._grasp_step += 1
        if self._grasp_step >= self._grasp_steps():
            self._advance_grasp()
        return pos, rot, grip

    def _solve_upper_body(self, pos, rot):
        """Arm, waist and height targets for the wrist pose, from the pregrasp cache or decimated IK."""
        arm_joints = np.zeros(7, dtype=np.float32)
        waist_joints = np.zeros(3, dtype=np.float32)
        height_cmd = self._height_cmd

        if (self._grasp_phase == PHASE_REALIGN and self._start_at_pregrasp
                and self._pregrasp_joints is not None):
            hcfg = self._ik_hand_cfg[self._hand]
            arm_joints = np.array([self._pregrasp_joints.get(jn, 0.0)
                                   for jn in hcfg["arm_joints"]], dtype=np.float32)
            waist_joints = np.array([self._pregrasp_joints.get(jn, 0.0) for jn in _WAIST],
                                    dtype=np.float32)
            pos = self._pregrasp
        elif pos is not None:
            IK_DECIM = 3
            if self._ik_cache is None or (self._step_counter % IK_DECIM) == 0:
                # Self-collision avoidance on only during grasp execution.
                arm_joints, waist_joints, ik_h, _ = self._solve_ik_wbc(
                    pos, rot, avoid_self_collision=True)
                self._ik_cache = (arm_joints, waist_joints, ik_h)
            else:
                arm_joints, waist_joints, ik_h = self._ik_cache
            height_cmd = self._height_cmd + 0.1 * (ik_h - self._height_cmd)
            last_arm = self._last_arm_joints
            if last_arm is not None and self._quick_approach and self._grasp_step < 30:
                alpha = float(np.clip(self._grasp_step / 30.0, 0.0, 1.0))
                arm_joints = (1.0 - alpha) * last_arm + alpha * arm_joints
                last_waist = self._last_waist_joints
                if last_waist is not None:
                    waist_joints = (1.0 - alpha) * last_waist + alpha * waist_joints
        else:
            return pos, arm_joints, waist_joints, height_cmd

        self._last_arm_joints = arm_joints.copy()
        self._last_waist_joints = waist_joints.copy()
        return pos, arm_joints, waist_joints, height_cmd

    def _assemble_action(self, obs, pos, grip, arm_joints, waist_joints, height_cmd):
        """Pack the 15-dim action, holding the pregrasp pose when no wrist target is active."""
        action = np.zeros(ACTION_DIM, dtype=np.float32)
        action[0:3] = self._cmd
        action[3] = height_cmd
        action[4:7] = waist_joints
        action[14] = self.GRIPPER_OPEN
        if pos is not None:
            action[7:14] = arm_joints
            action[14] = self.GRIPPER_OPEN * (1 - grip) + self.GRIPPER_CLOSED * grip
            return action
        if self._init_upper is None:
            return action

        iu = self._init_upper
        # Only drive the waist for a pregrasp pose; for random arm_init_radius poses
        # the waist noise is tiny and a zero target is fine.
        if self._init_arm_at_pregrasp:
            action[4:7] = iu[0:3]
        action[7:14] = iu[3:10]
        action[14] = iu[10]
        if self._upper_cmd is None:
            self._upper_cmd = np.concatenate([
                obs["base_height"].astype(np.float32),
                obs["joint_pos"][12:15].astype(np.float32),
                obs["joint_pos"][22:29].astype(np.float32),
            ])
        target_upper = action[3:14].copy()
        alpha = 0.08 if self._grasp_step < 80 else 0.18
        self._upper_cmd = (1.0 - alpha) * self._upper_cmd + alpha * target_upper
        action[3:14] = self._upper_cmd
        return action

    def sample_actions(self, obs):
        """Main per-step controller: nav until arrived, then run the grasp phase machine; returns the 15-dim action."""
        self._step_counter += 1
        self._recover_if_stalled()
        self._give_up_if_walk_timed_out()
        self._update_base_command()
        if self._arrived and self._grasp_phase == PHASE_IDLE:
            self._start_grasp(self._info or {})
        pos, rot, grip = self._step_grasp_phase()
        pos, arm_joints, waist_joints, height_cmd = self._solve_upper_body(pos, rot)
        return self._assemble_action(obs, pos, grip, arm_joints, waist_joints, height_cmd)


def get_config():
    """Default agent ConfigDict for the scripted G1 controller."""
    import ml_collections
    return ml_collections.ConfigDict(dict(
        agent_name="g1",
        walk_timeout_s=20.0,
        walk_fail_dist=0.2,
        face_yaw_offset=0.0,
        grasp_retry_closer=True,
    ))
