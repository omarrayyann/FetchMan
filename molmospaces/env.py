import gc
import glob
from contextlib import nullcontext
from pathlib import Path

import gymnasium as gym
import mujoco
import mujoco.viewer
import numpy as np
from gymnasium import spaces

from molmospaces import ASSETS_DIR
from molmospaces.components import Scene
from molmospaces.agents.policy import (
    PHASE_IDLE, PHASE_REALIGN, PHASE_APPROACH, PHASE_DESCEND,
    PHASE_OPEN_HOLD, PHASE_CLOSE, PHASE_POST_CLOSE, PHASE_LIFT,
)
from molmospaces.components.robot import G1Robot, XML_PATH, JOINT_NAMES, PREFIX
from molmospaces.tasks import PickTask
from molmospaces.tasks.pick import get_config as get_task_config


def _resolve_scene_paths(pattern):
    """Expand a scene pattern (xml path, .txt scene list, or glob) into a list of scene XML paths, skipping ceiling variants."""
    cand = Path(pattern)
    repo_root = ASSETS_DIR.parent.parent
    for base in (None, repo_root, ASSETS_DIR):
        full = cand if cand.is_absolute() else (base / cand if base is not None else None)
        if full is None or not full.is_file():
            continue
        if full.suffix == ".txt":
            out = []
            for line in full.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                lp = Path(line) if Path(line).is_absolute() else ASSETS_DIR / line
                if lp.is_file() and not lp.name.endswith("_ceiling.xml"):
                    out.append(lp)
            return out
        break
    p = cand if cand.is_absolute() else ASSETS_DIR / cand
    return [Path(m) for m in sorted(f for f in glob.glob(str(p)) if not f.endswith("_ceiling.xml"))]


def _mj_id(model, obj_type, name):
    """Look up a MuJoCo object id by name under the robot prefix."""
    return mujoco.mj_name2id(model, obj_type, f"{PREFIX}{name}")


def _mat_to_quat(mat):
    """Convert a 3x3 rotation matrix to a wxyz quaternion."""
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, mat.reshape(-1))
    return q



class G1Env(gym.Env):

    cameras = {"head_image": "head_pov", "wrist_image": "right_wrist_camera"}

    @staticmethod
    def _normalize_skill_profiles(skill_profiles):
        """Validate the (name, weight, overrides) profiles and normalise their weights to 1."""
        if not skill_profiles:
            return []
        profiles = [(str(name), float(weight), dict(profile))
                    for name, weight, profile in skill_profiles]
        wsum = sum(w for _, w, _ in profiles)
        assert wsum > 0, "skill_profiles weights must sum > 0"
        return [(n, w / wsum, p) for n, w, p in profiles]

    def _init_texture_pools(self, scene_textures_glob):
        """Build the per-category texture pools, falling back to a flat glob, else disable."""
        self._texture_pools = self._build_thor_texture_pools()
        if not self._texture_pools:
            if Path(scene_textures_glob).is_absolute():
                flat = sorted(glob.glob(scene_textures_glob))
            else:
                flat = sorted(str(p) for p in ASSETS_DIR.glob(scene_textures_glob))
            if flat:
                from molmospaces.components.constants import SCENE_TEXTURE_CATEGORIES
                self._texture_pools = {c: list(flat)
                                       for c in set(SCENE_TEXTURE_CATEGORIES.values())}
        if not self._texture_pools:
            print("[env] texture randomization requested but no textures available; disabling")
            self._randomize_textures = False

    def _load_initial_scene(self):
        """Draw scenes until one compiles with a valid target; a bad scene must not kill the worker."""
        for _ in range(50):
            try:
                self._load_scene(self._scene_paths[self.np_random.integers(len(self._scene_paths))])
                return
            except (ValueError, IndexError, IOError, OSError) as e:
                print(f"[env] init: skipping scene (load failed: {type(e).__name__}: {e})")
            except Exception as e:
                print(f"[env] init: skipping scene (load error: {type(e).__name__}: {e})")
        raise RuntimeError(
            f"Could not load any valid initial scene from {len(self._scene_paths)} candidates after 50 tries")

    def __init__(self, scene, objects=".*", seed=None,
                 randomize_scene=False, randomize_scene_freq=1,
                 randomize_object=False,
                 randomize_placement=True,
                 spawn_radius_min=1.0, spawn_radius_max=8.0,
                 grasp_spawn_radius_min=0.25, grasp_spawn_radius_max=0.80,
                 spawn_visibility_check=False,
                 arm_init_radius=0.0,
                 pregrasp_xyz_noise=0.0, pregrasp_rot_noise=0.0,
                 action_noise_std=0.005, action_noise_stride=5,
                 spawn_at_grasp=False,
                 sample_spawn_first=False,    # nav: sample spawn first, then pick goal on closest ring point
                 ring_num_angles=32,          # # of candidate angles on the goal ring when sample_spawn_first
                 spawn_along_line=False,      # nav: place spawn on the line out from the goal, guaranteed clear
                 walk_dist_min=0.3, walk_dist_max=0.8,  # spawn_along_line: walk length from goal
                 spawn_reachability_check=True,
                 randomize_textures=False, scene_textures_glob="textures/randomization/*.png",
                 max_textures=5, randomize_textures_keep_prob=0.25,
                 randomize_textures_solid_color_prob=0.30,
                 deterministic_scene_textures=False,
                 randomize_lighting=False, randomize_lighting_keep_prob=0.25,
                 face_yaw_offset=0.0,  # consumed by G1Controller; ignored here
                 randomize_height=True, randomize_height_min=0.0, randomize_height_favored=0.95,
                 randomize_height_max=None,
                 randomize_robot_height=False,
                 randomize_robot_height_min=0.7, randomize_robot_height_max=0.793,
                 wrist_camera_pos_noise=0.0, wrist_camera_rot_noise=0.0,
                 wrist_camera_fovy_noise=0.0,
                 head_camera_pos_noise=0.0, head_camera_rot_noise=0.0,
                 head_camera_fovy_noise=0.0, head_camera_distortion_noise=0.0,
                 goal_offset_xy_noise=0.0, goal_offset_yaw_noise=0.0,
                 start_at_pregrasp_xy_noise=0.0, start_at_pregrasp_yaw_noise=0.0,
                 start_at_pregrasp_joint_noise=0.0,
                 init_arm_at_pregrasp=False,   # at reset, set robot arm to pregrasp pose so APPROACH is a no-op
                 skill_profiles=None,
                 object_noise=0.0,           # pick task: gaussian xy jitter on object spawn (m)
                 terminate_before_grasp_collision=True,  # if True, terminate when robot hits non-floor, non-target world during IDLE/REALIGN (pre-grasp navigation)
                 terminate_on_grasp_collision=True,      # if True, terminate when robot hits non-floor, non-target world during APPROACH/DESCEND/CLOSE/LIFT (grasp arm motion)
                 terminate_grasp_if_not_visible=True,    # if True, terminate once the controller starts reaching (APPROACH onward) if the target object is not visible in the head camera (so we don't collect grasp demos of an off-screen object)
                 reset_precheck_grasp=True,              # if True, env.reset rejects spawn poses whose grasp the scripted controller's planner can't solve. GRPO doesn't need this (we don't require reachability — failed eps just return 0 reward), and the planner's np.random.shuffle is a non-deterministic source that breaks group-mate bit-equality.
                 launch_viewer=False, camera_size=(224, 224)):
        """Build the env: resolve scenes, load the first one, and store all randomization knobs."""
        super().__init__()
        self._viewer = None
        self._launch_viewer = launch_viewer
        self._scene_paths = _resolve_scene_paths(scene)
        self._object_regex = objects or ".*"
        self.np_random = np.random.default_rng(seed)
        self._current_scene_path = None
        self._randomize_scene = randomize_scene
        self._randomize_scene_freq = max(1, int(randomize_scene_freq))
        self._reset_counter = 0
        self._frozen_full_state = None
        self._randomize_placement = randomize_placement
        self._spawn_radius_min = spawn_radius_min
        self._spawn_radius_max = spawn_radius_max
        self._grasp_spawn_radius_min = float(grasp_spawn_radius_min)
        self._grasp_spawn_radius_max = float(grasp_spawn_radius_max)
        self._spawn_visibility_check = spawn_visibility_check
        self._arm_init_radius = float(arm_init_radius)
        self._pregrasp_xyz_noise = float(pregrasp_xyz_noise)
        self._pregrasp_rot_noise = float(pregrasp_rot_noise)
        self._action_noise_std = float(action_noise_std)
        self._action_noise_stride = max(1, int(action_noise_stride))
        self._action_noise_offset = np.zeros(10, dtype=np.float64)
        # Last commanded base velocity (vx, vy, omega) — exposed as obs so the
        # policy has temporal awareness of its own velocity commands.
        self._last_base_vel_cmd = np.zeros(3, dtype=np.float32)
        self._action_noise_step = 0
        self._spawn_at_grasp = bool(spawn_at_grasp)
        self._terminate_before_grasp_collision = bool(terminate_before_grasp_collision)
        self._terminate_on_grasp_collision = bool(terminate_on_grasp_collision)
        self._terminate_grasp_if_not_visible = bool(terminate_grasp_if_not_visible)
        self._reset_precheck_grasp = bool(reset_precheck_grasp)
        self._sample_spawn_first = bool(sample_spawn_first)
        self._ring_num_angles = int(ring_num_angles)
        self._spawn_along_line = bool(spawn_along_line)
        self._walk_dist_min = float(walk_dist_min)
        self._walk_dist_max = float(walk_dist_max)
        self._spawn_reachability_check = bool(spawn_reachability_check)
        self._randomize_lighting = bool(randomize_lighting)
        self._randomize_lighting_keep_prob = float(randomize_lighting_keep_prob)
        self._randomize_height = bool(randomize_height)
        self._randomize_height_min = float(randomize_height_min)
        self._randomize_height_favored = float(randomize_height_favored)
        self._randomize_height_max = (None if randomize_height_max is None
                                 else float(randomize_height_max))
        self._randomize_robot_height = bool(randomize_robot_height)
        self._randomize_robot_height_min = float(randomize_robot_height_min)
        self._randomize_robot_height_max = float(randomize_robot_height_max)
        self._init_camera_noise(wrist_camera_pos_noise, wrist_camera_rot_noise, wrist_camera_fovy_noise,
                                head_camera_pos_noise, head_camera_rot_noise, head_camera_fovy_noise,
                                head_camera_distortion_noise)
        self._goal_offset_xy_noise = float(goal_offset_xy_noise)
        self._goal_offset_yaw_noise = float(goal_offset_yaw_noise)
        self._start_at_pregrasp_xy_noise = float(start_at_pregrasp_xy_noise)
        self._start_at_pregrasp_yaw_noise = float(start_at_pregrasp_yaw_noise)
        self._start_at_pregrasp_joint_noise = float(start_at_pregrasp_joint_noise)
        self._init_arm_at_pregrasp = bool(init_arm_at_pregrasp)
        self._face_yaw_offset = float(face_yaw_offset)
        self._skill_profiles = self._normalize_skill_profiles(skill_profiles)
        self._active_profile = None
        self._randomize_textures = bool(randomize_textures)
        self._deterministic_scene_textures = bool(deterministic_scene_textures)
        self._randomize_textures_keep_prob = float(randomize_textures_keep_prob)
        self._randomize_textures_solid_color_prob = float(randomize_textures_solid_color_prob)
        # max_textures is interpreted per category (5 walls, 5 floors, etc.).
        self._texture_pools: dict[str, list[str]] = {}
        self._max_textures = int(max_textures) if max_textures else 0
        self._scene_texture_paths: dict[str, list[str]] = {}
        if self._randomize_textures:
            self._init_texture_pools(scene_textures_glob)
        self._init_runtime_state(camera_size)

        cfg = get_task_config()
        cfg.randomize_object = randomize_object
        cfg.object_noise = float(object_noise)
        self.task = PickTask(config=cfg, object_regex=self._object_regex)

        self._load_initial_scene()
        self.observation_space = self._build_observation_space()
        self.action_space = spaces.Box(-np.inf, np.inf, shape=(15,))

    def _init_camera_noise(self, wrist_pos, wrist_rot, wrist_fovy,
                           head_pos, head_rot, head_fovy, head_distortion):
        """Store the camera jitter magnitudes and the baselines they perturb around."""
        self._wrist_camera_pos_noise = float(wrist_pos)
        self._wrist_camera_rot_noise = float(wrist_rot)
        self._wrist_camera_fovy_noise = float(wrist_fovy)
        self._head_camera_pos_noise = float(head_pos)
        self._head_camera_rot_noise = float(head_rot)
        self._head_camera_fovy_noise = float(head_fovy)
        self._head_camera_distortion_noise = float(head_distortion)
        self._wrist_cam_pos0 = None
        self._wrist_cam_quat0 = None
        self._wrist_cam_fovy0 = None
        self._head_cam_ids_all = []
        self._head_cam_pos0_all = []
        self._head_cam_quat0_all = []

    def _init_runtime_state(self, camera_size):
        """Handles and per-episode state that are populated once a scene is loaded."""
        self._skip_episode = False
        self._sim_time = 0.0
        self._prev_grasp_phase = None   # detects the demonstrator's reach-start transition
        self.robot = None
        self.agent = None
        self._cam_size = tuple(camera_size)
        self._renderer = None
        self._renderers: dict[tuple[int, int], "mujoco.Renderer"] = {}
        self._cam_ids = {}
        self._fisheye = None

    def _build_observation_space(self):
        """Observation schema; joint_pos is legs[0:12] waist[12:15] left_arm[15:22] right_arm[22:29] grip[29]."""
        return spaces.Dict({
            "base_position": spaces.Box(-np.inf, np.inf, shape=(2,)),
            "base_yaw": spaces.Box(-np.pi, np.pi, shape=(1,)),
            "base_rpy": spaces.Box(-np.pi, np.pi, shape=(3,)),
            "base_rp":  spaces.Box(-np.pi, np.pi, shape=(2,)),
            "last_base_vel_cmd": spaces.Box(-np.inf, np.inf, shape=(3,)),
            "base_height": spaces.Box(-np.inf, np.inf, shape=(1,)),
            "base_velocity": spaces.Box(-np.inf, np.inf, shape=(3,)),
            "base_angular_velocity": spaces.Box(-np.inf, np.inf, shape=(3,)),
            "joint_pos": spaces.Box(-np.inf, np.inf, shape=(len(JOINT_NAMES),)),
            "upper_joint_pos": spaces.Box(-np.inf, np.inf, shape=(11,)),
            "right_hand_pose": spaces.Box(-np.inf, np.inf, shape=(7,)),
            "right_gripper_pos": spaces.Box(-0.0222, 0.0245, shape=(1,)),
            "target_object_pose": spaces.Box(-np.inf, np.inf, shape=(7,)),
            "target_point": spaces.Box(-np.inf, np.inf, shape=(2,)),
        })

    @property
    def camera_shape(self):
        """(H, W, 3) shape of rendered camera images."""
        return (*self._cam_size, 3)

    def _scene_name(self):
        """Current scene path relative to the assets dir (for logging/export)."""
        if self._current_scene_path is None:
            return ""
        p = Path(self._current_scene_path)
        try:
            return str(p.resolve().relative_to(ASSETS_DIR.resolve()))
        except Exception:
            return str(p)

    def _sample_scene_textures(self):
        """Sample per-category texture subsets for this scene load (deterministic when configured for GRPO)."""
        if not self._randomize_textures or not self._texture_pools:
            return {}
        # GRPO needs identical Scene contents across group-mate workers and
        # identical np_random consumption regardless of whether _load_scene
        # short-circuits. Both require: pool subsampling uses a deterministic
        # local RNG, not self.np_random.
        if self._deterministic_scene_textures:
            rng = np.random.RandomState(12345)
            choice = rng.choice
        else:
            choice = self.np_random.choice
        out: dict[str, list[str]] = {}
        # Iterate in SORTED category order so deterministic-RNG consumption is
        # identical across workers (the dict's natural order comes from a set
        # iteration upstream, which is per-process random via PYTHONHASHSEED).
        for cat in sorted(self._texture_pools):
            pool = self._texture_pools[cat]
            if self._max_textures and self._max_textures < len(pool):
                idx = choice(len(pool), size=self._max_textures, replace=False)
                out[cat] = [pool[int(i)] for i in idx]
            else:
                out[cat] = list(pool)
        return out

    def _release_gpu_resources(self):
        """Close renderers and the robot's visibility renderer before dropping the old scene."""
        # The robot owns a separate renderer whose framebuffer leaks on the render
        # GPU every reload if it is not closed here.
        if self.robot is not None and hasattr(self.robot, "close"):
            self.robot.close()
        for r in self._renderers.values():
            try:
                r.close()
            except Exception:
                pass
        self._renderers.clear()
        self._renderer = None
        self._fisheye = None

    def _cache_obs_indices(self):
        """Cache the qpos/dof/site/body ids the observation builder reads every step."""
        m = self.scene.model
        JOINT, SITE, BODY, CAMERA = (mujoco.mjtObj.mjOBJ_JOINT, mujoco.mjtObj.mjOBJ_SITE,
                                     mujoco.mjtObj.mjOBJ_BODY, mujoco.mjtObj.mjOBJ_CAMERA)
        self._obs_qpos_ids = np.array([m.jnt_qposadr[_mj_id(m, JOINT, n)] for n in JOINT_NAMES])
        self._obs_fj_dadr = m.jnt_dofadr[_mj_id(m, JOINT, "floating_base_joint")]
        self._obs_r_sid = _mj_id(m, SITE, "right_grasp")
        self._obs_r_grip_qa = m.jnt_qposadr[_mj_id(m, JOINT, "right_Joint1_1")]
        self._obs_pelvis_bid = _mj_id(m, BODY, "pelvis")
        # Used by the pre-grasp robot-vs-world contact check.
        self._robot_body_set = {bid for bid in range(m.nbody)
                                if (mujoco.mj_id2name(m, BODY, bid) or "").startswith(PREFIX)}
        # Base rgba hardcoded so per-reset jitter never drifts.
        self._robot_white_mid = int(
            mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_MATERIAL, f"{PREFIX}white"))
        self._robot_white_rgba0 = np.array([0.75, 0.75, 0.78, 1.0], dtype=np.float32)
        self._cam_ids = {name: _mj_id(m, CAMERA, mj_name) for name, mj_name in self.cameras.items()}

    def _cache_camera_poses(self):
        """Snapshot the nominal wrist and head-rig camera poses that randomization perturbs."""
        m = self.scene.model
        wrist_id = self._cam_ids.get("wrist_image", -1)
        if wrist_id >= 0:
            self._wrist_cam_pos0 = m.cam_pos[wrist_id].copy()
            self._wrist_cam_quat0 = m.cam_quat[wrist_id].copy()
            self._wrist_cam_fovy0 = float(m.cam_fovy[wrist_id])
        else:
            self._wrist_cam_pos0 = None
            self._wrist_cam_quat0 = None
            self._wrist_cam_fovy0 = None

        # The head camera is a fisheye composited from 5 tile cameras sharing an
        # optical center; all of them move together so the noise reaches the output.
        head_tile_names = [f"{PREFIX}head_pov{suf}" for suf in
                           ("", "_tile_center", "_tile_up", "_tile_down",
                            "_tile_left", "_tile_right")]
        self._head_cam_ids_all = []
        self._head_cam_pos0_all = []
        self._head_cam_quat0_all = []
        for nm in head_tile_names:
            cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, nm)
            if cid >= 0:
                self._head_cam_ids_all.append(cid)
                self._head_cam_pos0_all.append(m.cam_pos[cid].copy())
                self._head_cam_quat0_all.append(m.cam_quat[cid].copy())

    def _relaunch_viewer(self):
        """The viewer binds to one MjModel/MjData, so it is relaunched per scene."""
        if self._viewer is not None:
            try:
                self._viewer.close()
            except Exception:
                pass
            self._viewer = None
        if not self._launch_viewer:
            return
        self._viewer = mujoco.viewer.launch_passive(
            self.scene.model, self.scene.data,
            key_callback=lambda k: setattr(self, '_skip_episode', True) if k == 32 else None,
            show_left_ui=False, show_right_ui=False)
        self._viewer.opt.geomgroup[5] = 1

    def _load_scene(self, xml_path, texture_override=None):
        """Load a scene XML into a fresh Scene/Robot/Task stack, reusing the current one when the path is unchanged."""
        xml_path = Path(xml_path)
        if xml_path == self._current_scene_path and texture_override is None:
            return
        # Clear up front so a failure below cannot leave a half-built scene that
        # the next _load_scene would early-return into.
        self._current_scene_path = None
        self._release_gpu_resources()

        self._scene_texture_paths = (texture_override if texture_override is not None
                                     else self._sample_scene_textures())
        self.scene = Scene(xml_path, robot_xml=XML_PATH, mobile_regex=self._object_regex,
                           scene_textures=self._scene_texture_paths)
        hl = self.scene.model.vis.headlight
        self._init_headlight = (hl.ambient.copy(), hl.diffuse.copy(), hl.specular.copy())
        self.occ = self.scene.occupancy_map(agent_radius=0.15)
        # Extra-inflated for A* path planning; goal sampling still uses self.occ.
        self.occ_safe = self.occ.dilated(0.125)
        self.robot = G1Robot(self.scene.model, self.scene.data)
        self.robot.set_defaults()
        self.task.set_objects(self.scene)

        # The 16k default shadow map dominates render time; clamp before renderers exist.
        if self.scene.model.vis.quality.shadowsize > 4096:
            self.scene.model.vis.quality.shadowsize = 4096
        self._cache_obs_indices()
        self._cache_camera_poses()
        self._relaunch_viewer()

        if getattr(self, "agent", None) is not None:
            self.agent.setup(self.scene.model, self.scene.data)
            self.agent.set_env(self)

        # Periodic gc sweep for pybind buffers whose free lagged; throttled so it
        # does not tax collection at randomize_scene_freq=1.
        self._reload_count = getattr(self, "_reload_count", 0) + 1
        if self._reload_count % 16 == 0:
            gc.collect()

        # Only now is the scene fully loaded; a mid-load failure leaves the cache
        # None so the next call rebuilds instead of early-returning.
        self._current_scene_path = xml_path

    def set_agent(self, agent):
        """Attach the scripted controller so reset/step can consult it."""
        self.agent = agent

    @property
    def target(self):
        """Current target object (task-owned)."""
        return self.task.target
    @property
    def time(self):
        """Sim time in seconds since reset."""
        return self._sim_time
    @property
    def viewer_running(self):
        """True while the interactive viewer window is open."""
        return self._viewer is not None and self._viewer.is_running()

    def _pelvis_frame(self):
        """Return a to_local(pos, quat) closure that maps world poses into the pelvis frame."""
        d = self.scene.data
        pos = d.xpos[self._obs_pelvis_bid].copy()
        mat = d.xmat[self._obs_pelvis_bid].reshape(3, 3).copy()
        quat_inv = np.zeros(4)
        mujoco.mju_negQuat(quat_inv, _mat_to_quat(mat))

        def to_local(pos_w, quat_w=None):
            """Map a world position (and optional quaternion) into the pelvis frame."""
            rel_pos = mat.T @ (pos_w - pos)
            if quat_w is None:
                return rel_pos
            q = np.zeros(4)
            mujoco.mju_mulQuat(q, quat_inv, quat_w)
            if q[0] < 0:
                q = -q
            return np.concatenate([rel_pos, q])
        return to_local

    def target_point_in_head(self):
        """(u, v) pixel of the target in the head fisheye, or None if behind the camera."""
        head_id = self._cam_ids.get("head_image", -1)
        tgt = self.target
        if head_id < 0 or tgt is None:
            return None
        if self._fisheye is None:
            self._ensure_fisheye("head_pov", self._cam_size[0], self._cam_size[1], 512)
        d = self.scene.data
        cam_pos = d.cam_xpos[head_id]
        cam_mat = d.cam_xmat[head_id].reshape(3, 3)
        p_cam = cam_mat.T @ (tgt.position(d) - cam_pos)
        return self._fisheye.project_camera_point(p_cam)

    def _target_visible_in_head(self):
        """True if the target projects inside the head fisheye frame."""
        pt = self.target_point_in_head()
        if pt is None:
            return False
        H, W = self._cam_size
        u, v = pt
        return 0.0 <= u <= W and 0.0 <= v <= H

    def _build_obs(self):
        """Assemble the observation dict (proprio, hand/target poses in pelvis frame, target point)."""
        d = self.scene.data
        tgt = self.task.target
        to_local = self._pelvis_frame()
        r_quat = _mat_to_quat(d.site_xmat[self._obs_r_sid])

        pm = d.xmat[self._obs_pelvis_bid]
        pitch = np.arcsin(-pm[6])
        roll = np.arctan2(pm[7], pm[8])
        yaw = np.arctan2(pm[3], pm[0])
        base_rpy = np.array([roll, pitch, yaw], dtype=np.float32)
        joint_pos = d.qpos[self._obs_qpos_ids].astype(np.float32)
        upper_joint_pos = np.concatenate([joint_pos[12:15], joint_pos[22:30]])

        # 2D point conditioning: target object projected into the head fisheye,
        # normalized to [0,1] (x=u/W, y=v/H). (-1,-1) when behind the camera or
        # unavailable. Values can fall slightly outside [0,1] when near the FoV edge.
        pt = self.target_point_in_head()
        if pt is None:
            target_point = np.array([-1.0, -1.0], dtype=np.float32)
        else:
            H, W = self._cam_size
            target_point = np.array([pt[0] / W, pt[1] / H], dtype=np.float32)

        return {
            "base_position": self.robot.get_xy().astype(np.float32),
            "base_yaw": np.array([base_rpy[2]], dtype=np.float32),
            "base_rpy": base_rpy,
            "base_rp":  base_rpy[:2].astype(np.float32),  # roll, pitch only (sim2real safe)
            "last_base_vel_cmd": self._last_base_vel_cmd.copy(),
            "base_height": np.array([self.robot.pelvis_height()], dtype=np.float32),
            "base_velocity": d.qvel[self._obs_fj_dadr:self._obs_fj_dadr+3].astype(np.float32),
            "base_angular_velocity": d.qvel[self._obs_fj_dadr+3:self._obs_fj_dadr+6].astype(np.float32),
            "joint_pos": joint_pos,
            "upper_joint_pos": upper_joint_pos,
            "right_hand_pose": to_local(d.site_xpos[self._obs_r_sid], r_quat).astype(np.float32),
            "right_gripper_pos": np.array([d.qpos[self._obs_r_grip_qa]], dtype=np.float32),
            "target_object_pose": to_local(tgt.position(d), tgt.quat(d)).astype(np.float32),
            "target_point": target_point,
        }

    def _ensure_renderer(self, h, w):
        """Get or lazily create the offscreen renderer for (h, w)."""
        key = (int(h), int(w))
        r = self._renderers.get(key)
        if r is None:
            r = mujoco.Renderer(self.scene.model, h, w)
            self._renderers[key] = r
        self._renderer = r
        return r

    def _default_render_opt(self):
        """Default MjvOption with the robot head/logo geom group enabled."""
        opt = mujoco.MjvOption()
        mujoco.mjv_defaultOption(opt)
        opt.geomgroup[5] = 1   # robot head/logo + wrist_mount — hidden on wrist cam only
        return opt

    def render_cameras(self):
        """Render all policy cameras: fisheye head plus pinhole wrist."""
        H, W = self._cam_size
        out = {}
        for name, mj_name in self.cameras.items():
            if mj_name == "head_pov":
                out[name] = self.render_fisheye(output_h=H, output_w=W).copy()
            else:
                r = self._ensure_renderer(H, W)
                opt = self._default_render_opt()
                if name == "wrist_image":
                    opt.geomgroup[1] = 0   # hide wrist_mount + camera body from own wrist obs
                r.update_scene(self.scene.data, self._cam_ids[name], opt)
                out[name] = r.render().copy()
        return out

    def render_fisheye(self, base_cam_name='head_pov', output_h=224, output_w=224,
                       tile_size=512):
        """Render the head camera through the 5-tile fisheye projection."""
        self._ensure_fisheye(base_cam_name, output_h, output_w, tile_size)
        r = self._ensure_renderer(self._fisheye.tile_size, self._fisheye.tile_size)
        opt = self._default_render_opt()
        opt.geomgroup[5] = 0
        return self._fisheye.render(self.scene.data, r, scene_option=opt)

    def _ensure_fisheye(self, base_cam_name, output_h, output_w, tile_size):
        """Get or lazily (re)build the FisheyeRenderer for the requested output size."""
        if (self._fisheye is None
                or self._fisheye.output_h != output_h
                or self._fisheye.output_w != output_w):
            from molmospaces.components.fisheye import FisheyeRenderer
            tile_names = [f"{PREFIX}{base_cam_name}_tile_{face}"
                          for face in ('center', 'up', 'down', 'left', 'right')]
            self._fisheye = FisheyeRenderer(
                self.scene.model, tile_cam_names=tile_names,
                tile_size=tile_size, output_h=output_h, output_w=output_w,
            )

    def render_debug_panel(self):
        """Side-by-side strip of all camera views with name overlays (viewer debugging)."""
        imgs = self.render_cameras()
        if not imgs:
            return None
        panels = []
        for name, img in imgs.items():
            img = img.copy()
            try:
                import cv2 as _cv2
                _cv2.putText(img, name, (8, 22), _cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                             (0, 0, 0), 3, _cv2.LINE_AA)
                _cv2.putText(img, name, (8, 22), _cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                             (255, 255, 255), 1, _cv2.LINE_AA)
            except ImportError:
                pass
            panels.append(img)
        return np.concatenate(panels, axis=1)

    def _robot_touches_world(self):
        """True if any robot geom contacts non-robot, non-floor, non-target geometry."""
        m, d = self.scene.model, self.scene.data
        rset = self._robot_body_set
        if not rset:
            return False
        # Target body set (for compound objects). May be absent on bare/open tasks.
        tset = getattr(self.task, "_target_body_set", None) or set()
        PLANE = mujoco.mjtGeom.mjGEOM_PLANE
        for i in range(d.ncon):
            c = d.contact[i]
            bid1 = int(m.geom_bodyid[c.geom1])
            bid2 = int(m.geom_bodyid[c.geom2])
            b1 = bid1 in rset
            b2 = bid2 in rset
            if b1 == b2:  # both robot (self) OR both non-robot (world-world)
                continue
            # Floor / world plane: don't count.
            if m.geom_type[c.geom1] == PLANE or m.geom_type[c.geom2] == PLANE:
                continue
            # Target object (bowl/etc): don't count — contact with the goal is fine.
            other = bid2 if b1 else bid1
            if other in tset:
                continue
            return True
        return False

    def _robot_has_scene_collision(self):
        """True if any robot geom touches non-floor world geometry (used during placement search)."""
        m, d = self.scene.model, self.scene.data
        mujoco.mj_collision(m, d)
        robot_bids = set()
        for bid in range(m.nbody):
            bname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
            if bname.startswith(PREFIX):
                robot_bids.add(bid)
        for i in range(d.ncon):
            c = d.contact[i]
            b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
            b1_robot = b1 in robot_bids
            b2_robot = b2 in robot_bids
            if b1_robot != b2_robot:
                other = b2 if b1_robot else b1
                bname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, other) or ""
                if m.geom_type[c.geom1] == mujoco.mjtGeom.mjGEOM_PLANE or m.geom_type[c.geom2] == mujoco.mjtGeom.mjGEOM_PLANE:
                    continue
                return True
        return False

    def _sample_goal_pose(self, target_pos, obj=None, attempts=25, sampled_upper=None):
        """Sample a collision-free standoff (xy, yaw) facing the target, with a visibility check."""
        # fast-fail: no free standoff cell around the target -> unservable
        if not self.occ.any_free_in_annulus(target_pos[:2], self._grasp_spawn_radius_min,
                                            self._grasp_spawn_radius_max):
            return None, None
        saved_qpos = self.scene.data.qpos.copy()
        saved_ctrl = self.scene.data.ctrl.copy()
        try:
            for _ in range(attempts):
                xy = self.occ.sample_near(target_pos[:2],
                                          radius_min=self._grasp_spawn_radius_min,
                                          radius_max=self._grasp_spawn_radius_max,
                                          np_random=self.np_random)
                if xy is None:
                    continue
                xy = np.asarray(xy, dtype=np.float64)
                yaw = float(np.arctan2(target_pos[1] - xy[1], target_pos[0] - xy[0]))
                if obj is None:
                    return xy, yaw
                self.robot.set_pose(xy, yaw)
                self.robot.set_defaults()
                if self.agent is not None:
                    self.agent._set_groot_defaults()
                if sampled_upper is not None:
                    self.robot.apply_upper_pose(sampled_upper)
                mujoco.mj_forward(self.scene.model, self.scene.data)
                # Gate visibility check on _reset_precheck_grasp (it's a GPU
                # render, so non-deterministic across worker EGL contexts).
                if self._reset_precheck_grasp and not self.robot.check_object_visibility(obj.body_id):
                    continue
                if self._robot_has_scene_collision():
                    continue
                return xy, yaw
        finally:
            self.scene.data.qpos[:] = saved_qpos
            self.scene.data.ctrl[:] = saved_ctrl
            mujoco.mj_forward(self.scene.model, self.scene.data)
        return None, None

    def _sample_spawn_then_goal(self, target_pos, obj, sampled_upper=None, spawn_attempts=50):
        """Sample the spawn first, then a goal on a ring near it; all-None on failure."""
        saved_qpos = self.scene.data.qpos.copy()
        saved_ctrl = self.scene.data.ctrl.copy()
        try:
            for _ in range(spawn_attempts):
                spawn_xy = self.occ.sample_near(target_pos[:2],
                                                radius_min=self._spawn_radius_min,
                                                radius_max=self._spawn_radius_max,
                                                np_random=self.np_random)
                if spawn_xy is None:
                    continue
                spawn_xy = np.asarray(spawn_xy, dtype=np.float64)
                bearing = float(np.arctan2(target_pos[1] - spawn_xy[1], target_pos[0] - spawn_xy[0]))
                spawn_yaw = bearing + self.np_random.uniform(-0.5, 0.5)
                if not self._place_robot_and_check(spawn_xy, spawn_yaw, obj, sampled_upper,
                                                   visibility=self._spawn_visibility_check):
                    continue
                if (self._spawn_reachability_check
                        and not self.occ_safe.same_free_component(spawn_xy, target_pos[:2])):
                    continue
                goal_xy, goal_yaw = self._sample_goal_on_ring(spawn_xy, target_pos, obj, sampled_upper)
                if goal_xy is None:
                    continue
                return spawn_xy, spawn_yaw, goal_xy, goal_yaw
        finally:
            self.scene.data.qpos[:] = saved_qpos
            self.scene.data.ctrl[:] = saved_ctrl
            mujoco.mj_forward(self.scene.model, self.scene.data)
        return None, None, None, None

    def _sample_goal_on_ring(self, spawn_xy, target_pos, obj, sampled_upper=None):
        """Pick a goal on one standoff ring, nearest-to-spawn first."""
        r = float(self.np_random.uniform(self._grasp_spawn_radius_min, self._grasp_spawn_radius_max))
        thetas = np.linspace(0.0, 2 * np.pi, self._ring_num_angles, endpoint=False)
        cands = np.stack([target_pos[0] + r * np.cos(thetas),
                          target_pos[1] + r * np.sin(thetas)], axis=1)
        order = np.argsort(np.linalg.norm(cands - spawn_xy, axis=1))
        for idx in order:
            xy = cands[idx]
            if not self.occ.is_free(xy):
                continue
            yaw = float(np.arctan2(target_pos[1] - xy[1], target_pos[0] - xy[0]))
            # The visibility check renders, so it is per-EGL-context and can differ
            # between workers with identical sim state; gated so it can be turned off.
            if self._place_robot_and_check(xy, yaw, obj, sampled_upper,
                                           visibility=self._reset_precheck_grasp):
                return xy, yaw
        return None, None

    def _sample_spawn_along_line(self, goal_xy, target_pos, obj, sampled_upper, march_step=0.05):
        """Place the spawn on the clear line out from the goal, so the path is collision-free by construction."""
        obj_xy = np.asarray(target_pos[:2], dtype=np.float64)
        gxy = np.asarray(goal_xy, dtype=np.float64)
        delta = gxy - obj_xy
        d_goal = float(np.linalg.norm(delta))
        if d_goal < 1e-6:
            return None, None
        direction = delta / d_goal  # away from object, through goal, outward
        # March outward from the goal until blocked or capped. Walk length is measured
        # from the goal, not the object, so it does not depend on where the goal lands.
        walk_min = self._walk_dist_min
        walk_max = self._walk_dist_max
        cap = walk_max + 0.1
        r_max = 0.0
        # Use occ (15cm robot radius), same as goal sampling. occ_safe (extra 12.5cm
        # dilation for A*) would reject too many goals that landed in tight corners.
        while r_max + march_step <= cap:
            probe = gxy + (r_max + march_step) * direction
            if not self.occ.is_free(probe):
                break
            r_max += march_step
        # Offset of spawn behind goal, along the outward direction.
        min_offset = walk_min
        max_offset = min(r_max, walk_max)
        if max_offset < min_offset + 1e-3:
            return None, None
        offset = float(self.np_random.uniform(min_offset, max_offset))
        spawn_xy = gxy + offset * direction
        bearing = float(np.arctan2(target_pos[1] - spawn_xy[1], target_pos[0] - spawn_xy[0]))
        spawn_yaw = bearing + float(self.np_random.uniform(-0.8, 0.8))
        saved_qpos = self.scene.data.qpos.copy()
        saved_ctrl = self.scene.data.ctrl.copy()
        try:
            if self._place_robot_and_check(spawn_xy, spawn_yaw, obj, sampled_upper,
                                           visibility=self._spawn_visibility_check):
                if (not self._spawn_reachability_check
                        or self.occ_safe.same_free_component(spawn_xy, target_pos[:2])):
                    return spawn_xy, spawn_yaw
        finally:
            self.scene.data.qpos[:] = saved_qpos
            self.scene.data.ctrl[:] = saved_ctrl
            mujoco.mj_forward(self.scene.model, self.scene.data)
        return None, None

    def _place_robot_and_check(self, xy, yaw, obj, sampled_upper, visibility):
        """Set robot pose + run mj_forward, then collision/visibility checks. Returns bool."""
        self.robot.set_pose(xy, yaw)
        self.robot.set_defaults()
        if self.agent is not None:
            self.agent._set_groot_defaults()
        if sampled_upper is not None:
            self.robot.apply_upper_pose(sampled_upper)
        mujoco.mj_forward(self.scene.model, self.scene.data)
        if self._robot_has_scene_collision():
            return False
        if visibility and not self.robot.check_object_visibility(obj.body_id):
            return False
        return True

    def _compute_pregrasp_upper(self, goal_xy, goal_yaw):
        """Plan the grasp at the goal pose and return its 11-element upper-body pose vector."""
        if self.agent is None:
            return None
        info_preview = self.task.make_info(self.scene, self.np_random)
        info_preview["goal_xy"] = goal_xy
        info_preview["goal_yaw"] = goal_yaw
        saved_qpos = self.scene.data.qpos.copy()
        saved_ctrl = self.scene.data.ctrl.copy()
        try:
            self.robot.set_pose(goal_xy, goal_yaw)
            self.agent._set_groot_defaults()
            mujoco.mj_forward(self.scene.model, self.scene.data)
            self.agent._grasp_planner.plan(info_preview)
            pregrasp_joints = getattr(self.agent._grasp_planner, "_pregrasp_joints", None)
        finally:
            self.scene.data.qpos[:] = saved_qpos
            self.scene.data.ctrl[:] = saved_ctrl
            mujoco.mj_forward(self.scene.model, self.scene.data)
        if not pregrasp_joints:
            return None
        arr = np.zeros(11, dtype=np.float32)
        upper_names = [
            "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
            "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint", "right_elbow_joint",
            "right_wrist_roll_joint", "right_wrist_pitch_joint",
            "right_wrist_yaw_joint",
        ]
        for i, jn in enumerate(upper_names):
            if jn in pregrasp_joints:
                arr[i] = float(pregrasp_joints[jn])
        arr[10] = -0.0222   # GRIPPER_OPEN
        return arr

    def _sample_pregrasp_offset(self):
        """Sample the per-episode (xyz, rotation) noise applied to the pregrasp pose, or None when disabled."""
        if self._pregrasp_xyz_noise <= 0 and self._pregrasp_rot_noise <= 0:
            return None
        xyz = (self.np_random.uniform(-self._pregrasp_xyz_noise, self._pregrasp_xyz_noise, size=3)
               if self._pregrasp_xyz_noise > 0 else np.zeros(3))
        if self._pregrasp_rot_noise > 0:
            from scipy.spatial.transform import Rotation as _R
            axis = self.np_random.normal(size=3)
            axis /= max(np.linalg.norm(axis), 1e-9)
            angle = self.np_random.uniform(-self._pregrasp_rot_noise, self._pregrasp_rot_noise)
            rot = _R.from_rotvec(axis * angle).as_matrix()
        else:
            rot = np.eye(3)
        return (xyz, rot)

    def _snapshot_full_state(self, info):
        """Capture everything a freeze=True replay needs to reproduce this reset bit-exactly."""
        m, d = self.scene.model, self.scene.data
        return {
            "qpos": d.qpos.copy(), "qvel": d.qvel.copy(),
            "act": d.act.copy() if d.act.size else None,
            "ctrl": d.ctrl.copy(),
            "mocap_pos": d.mocap_pos.copy(), "mocap_quat": d.mocap_quat.copy(),
            "eq_active": d.eq_active.copy(),
            "body_pos": m.body_pos.copy(),
            "body_simple": m.body_simple.copy(),
            "body_sameframe": m.body_sameframe.copy(),
            "geom_matid": m.geom_matid.copy(),
            "mat_rgba": m.mat_rgba.copy(),
            "light_pos": m.light_pos.copy() if m.nlight else None,
            "light_dir": m.light_dir.copy() if m.nlight else None,
            "light_specular": m.light_specular.copy() if m.nlight else None,
            "light_ambient": m.light_ambient.copy() if m.nlight else None,
            "light_diffuse": m.light_diffuse.copy() if m.nlight else None,
            "light_active": m.light_active.copy() if m.nlight else None,
            "cam_pos": m.cam_pos.copy(), "cam_quat": m.cam_quat.copy(),
            "cam_fovy": m.cam_fovy.copy(),
            "np_random": self.np_random.bit_generator.state,
            "reset_counter": self._reset_counter,
            "obj_idx": getattr(self.task, "_obj_idx", None),
            "target": self.task.target,
            "target_z0": float(getattr(self.task, "_target_z0", 0.0)),
            "target_body_set": set(getattr(self.task, "_target_body_set", set())),
            "info": dict(info),
        }

    def _restore_full_state(self, fs):
        """Blit a snapshot back over model and data, skipping randomization entirely."""
        m, d = self.scene.model, self.scene.data
        d.qpos[:] = fs["qpos"]
        d.qvel[:] = fs["qvel"]
        if fs["act"] is not None and d.act.size:
            d.act[:] = fs["act"]
        d.ctrl[:] = fs["ctrl"]
        d.mocap_pos[:] = fs["mocap_pos"]
        d.mocap_quat[:] = fs["mocap_quat"]
        d.eq_active[:] = fs["eq_active"]
        m.body_pos[:] = fs["body_pos"]
        m.body_simple[:] = fs["body_simple"]
        m.body_sameframe[:] = fs["body_sameframe"]
        m.geom_matid[:] = fs["geom_matid"]
        m.mat_rgba[:] = fs["mat_rgba"]
        if m.nlight:
            m.light_pos[:] = fs["light_pos"]
            m.light_dir[:] = fs["light_dir"]
            m.light_specular[:] = fs["light_specular"]
            m.light_ambient[:] = fs["light_ambient"]
            m.light_diffuse[:] = fs["light_diffuse"]
            m.light_active[:] = fs["light_active"]
        m.cam_pos[:] = fs["cam_pos"]
        m.cam_quat[:] = fs["cam_quat"]
        m.cam_fovy[:] = fs["cam_fovy"]
        self.np_random.bit_generator.state = fs["np_random"]
        self._reset_counter = fs["reset_counter"]
        if hasattr(self.task, "_obj_idx") and fs["obj_idx"] is not None:
            self.task._obj_idx = fs["obj_idx"]
        self.task.target = fs["target"]
        self.task._target_z0 = fs["target_z0"]
        self.task._target_body_set = fs["target_body_set"]
        mujoco.mj_forward(m, d)

    def _clear_episode_counters(self):
        """Zero the per-episode sim clock, action noise and grasp-phase trackers."""
        self._sim_time = 0.0
        self._action_noise_step = 0
        self._prev_grasp_phase = None
        self._action_noise_offset.fill(0.0)
        self._skip_episode = False
        self._last_base_vel_cmd = np.zeros(3, dtype=np.float32)

    def reset(self, *, seed=None, options=None, freeze=False):
        """Reset to a new episode; freeze=True bit-exactly replays the last non-frozen reset (GRPO group-mates)."""
        if freeze:
            if self._frozen_full_state is None:
                raise RuntimeError(
                    "reset(freeze=True) called before any non-frozen reset; nothing to repeat")
            self._restore_full_state(self._frozen_full_state)
            self._clear_episode_counters()
            return self._build_obs(), dict(self._frozen_full_state["info"])

        super().reset(seed=seed)
        self._skip_episode = False
        self._reset_counter += 1
        self._last_base_vel_cmd = np.zeros(3, dtype=np.float32)
        for _ in range(12):
            try:
                r = self._try_reset()
            except mujoco.FatalError as e:
                print(f"[env] _try_reset MuJoCo error, retrying: {e}")
                continue
            if r is not None:
                info = r[1] if isinstance(r, tuple) and len(r) >= 2 else {}
                self._frozen_full_state = self._snapshot_full_state(info)
                return r
        raise RuntimeError("Could not find valid placement after 12 retries")

    # ---- explicit episode export/restore (fixed eval sets) ----

    def export_reset_state(self, info=None):
        """JSON-able dict capturing everything reset randomized, for exact replay."""
        m, d = self.scene.model, self.scene.data
        def rel(p):
            """Make an asset path relative to the assets dir when possible."""
            p = str(p)
            for base in (ASSETS_DIR, ASSETS_DIR.resolve()):
                try: return str(Path(p).relative_to(base))
                except ValueError: pass
            try: return str(Path(p).resolve().relative_to(ASSETS_DIR.resolve()))
            except ValueError: return p
        st = {
            "scene": rel(self._current_scene_path),
            "scene_textures": {c: [rel(p) for p in v]
                               for c, v in (self.scene._scene_texture_paths or {}).items()},
            "target_name": self.task.target.name,
            "target_z0": float(self.task._target_z0),
            "robot_xy": [float(x) for x in self.robot.get_xy()],
            "robot_yaw": float(self.robot.get_yaw()),
            "object_name": (info or {}).get("object_name", ""),
            "prompt": (info or {}).get("prompt", ""),
            "init_height": (info or {}).get("init_height"),
            # Structural: the regex that decides which bodies get freejoints,
            # i.e. the qpos layout. Restore re-applies it so the eval env config
            # (--env.objects) doesn't have to match the generation config.
            "object_regex": self._object_regex,
            "qpos": d.qpos.tolist(), "qvel": d.qvel.tolist(), "ctrl": d.ctrl.tolist(),
            "mocap_pos": d.mocap_pos.tolist(), "mocap_quat": d.mocap_quat.tolist(),
            "eq_active": d.eq_active.tolist(),
            "body_pos": m.body_pos.tolist(),
            "body_simple": m.body_simple.tolist(), "body_sameframe": m.body_sameframe.tolist(),
            "geom_matid": m.geom_matid.tolist(), "mat_rgba": m.mat_rgba.tolist(),
            "cam_pos": m.cam_pos.tolist(), "cam_quat": m.cam_quat.tolist(),
            "cam_fovy": m.cam_fovy.tolist(),
            "headlight": [m.vis.headlight.ambient.tolist(), m.vis.headlight.diffuse.tolist(),
                          m.vis.headlight.specular.tolist()],
        }
        if m.nlight:
            st.update(light_pos=m.light_pos.tolist(), light_dir=m.light_dir.tolist(),
                      light_specular=m.light_specular.tolist(), light_ambient=m.light_ambient.tolist(),
                      light_diffuse=m.light_diffuse.tolist(), light_active=m.light_active.tolist())
            if hasattr(m, "light_castshadow"):
                st["light_castshadow"] = m.light_castshadow.tolist()
        if self._fisheye is not None:
            st["fisheye_K"] = self._fisheye.K.tolist()
            st["fisheye_D"] = self._fisheye.D.tolist()
        return st

    @staticmethod
    def _abs_scene_path(p):
        """Resolve an exported scene path back to an absolute path under the assets dir."""
        p = Path(p)
        return (p if p.is_absolute() else ASSETS_DIR / p).resolve()

    def _adopt_export_regexes(self, st):
        """Adopt the generation-time object regexes; returns True if a recompile is needed."""
        # The freejoint regex sets the qpos layout, so a saved state only fits a
        # scene compiled with the same regex, whatever --env.objects says now.
        changed = False
        if "object_regex" in st and st["object_regex"] != self._object_regex:
            self._object_regex = st["object_regex"]
            self.task._object_regex = st["object_regex"]
            changed = True
        return changed

    def _reload_scene_for_restore(self, st):
        """Reload the exported scene unless the current one already matches it exactly."""
        textures = {c: [str(ASSETS_DIR / p) if not Path(p).is_absolute() else p for p in v]
                    for c, v in st.get("scene_textures", {}).items()}
        regex_changed = self._adopt_export_regexes(st)
        same_scene = (not regex_changed
                      and self._current_scene_path is not None
                      and self._abs_scene_path(st["scene"]) == self._abs_scene_path(self._current_scene_path))
        same_tex = (textures == {c: [str(p) for p in v]
                                 for c, v in (self.scene._scene_texture_paths or {}).items()}
                    if self.scene else False)
        if not (same_scene and same_tex):
            self._current_scene_path = None
            self._load_scene(st["scene"], texture_override=textures)

    def _write_sim_state(self, st):
        """Blit the exported qpos/qvel/ctrl/mocap arrays into data."""
        d = self.scene.data
        d.qpos[:] = st["qpos"]
        d.qvel[:] = st["qvel"]
        d.ctrl[:] = st["ctrl"]
        if d.mocap_pos.size:
            d.mocap_pos[:] = st["mocap_pos"]
            d.mocap_quat[:] = st["mocap_quat"]
        if d.eq_active.size:
            d.eq_active[:] = st["eq_active"]

    def _write_appearance(self, st):
        """Blit the exported body poses, materials, cameras and lighting into the model."""
        m = self.scene.model
        m.body_pos[:] = st["body_pos"]
        m.body_simple[:] = st["body_simple"]
        m.body_sameframe[:] = st["body_sameframe"]
        m.geom_matid[:] = st["geom_matid"]
        m.mat_rgba[:] = st["mat_rgba"]
        m.cam_pos[:] = st["cam_pos"]
        m.cam_quat[:] = st["cam_quat"]
        m.cam_fovy[:] = st["cam_fovy"]
        hl = m.vis.headlight
        hl.ambient[:] = st["headlight"][0]
        hl.diffuse[:] = st["headlight"][1]
        hl.specular[:] = st["headlight"][2]
        if m.nlight and "light_pos" in st:
            m.light_pos[:] = st["light_pos"]
            m.light_dir[:] = st["light_dir"]
            m.light_specular[:] = st["light_specular"]
            m.light_ambient[:] = st["light_ambient"]
            m.light_diffuse[:] = st["light_diffuse"]
            m.light_active[:] = st["light_active"]
            if "light_castshadow" in st and hasattr(m, "light_castshadow"):
                m.light_castshadow[:] = st["light_castshadow"]
        if "fisheye_K" in st:
            if self._fisheye is None:
                self._ensure_fisheye("head_pov", self._cam_size[0], self._cam_size[1], 512)
            self._fisheye.set_intrinsics(np.asarray(st["fisheye_K"]), np.asarray(st["fisheye_D"]))

    def _bind_restored_target(self, st):
        """Point the task at the exported target object and reload its grasps."""
        target = next((o for o in self.scene.pickable if o.name == st["target_name"]), None)
        if target is None:
            raise RuntimeError(f"restore: target {st['target_name']!r} not in scene {st['scene']}")
        self.task.target = target
        self.task._target_z0 = float(st["target_z0"])
        self.task._target_body_set = self.scene.get_body_descendants(target.body_id)
        self.task._target_grasps = self.task._load_grasps(getattr(target, "asset_id", ""))

    def _restored_info(self, st):
        """Build the reset info for a restored episode, including the exported upper-body pose."""
        info = self.task.make_info(self.scene, self.np_random)
        info.update(scene=st["scene"], object_name=st.get("object_name", ""))
        if st.get("prompt"):
            info["prompt"] = st["prompt"]
        # The exported qpos was captured AFTER agent.reset applied its sampled
        # upper-body pose. Hand that pose back so agent.reset re-applies the same
        # values (a no-op on qpos) instead of clobbering them with defaults.
        idx = self.robot._UPPER_RAND_IDX
        info["init_upper_pose"] = self.scene.data.qpos[self.robot._qpos_ids[idx]].copy()
        if st.get("init_height") is not None:
            info["init_height"] = float(st["init_height"])
        return info

    def restore_reset_state(self, st):
        """Exact replay of an export_reset_state() dict. Returns (obs, info)."""
        self._reload_scene_for_restore(st)
        m, d = self.scene.model, self.scene.data
        self._write_sim_state(st)
        self._write_appearance(st)
        self._bind_restored_target(st)
        self._clear_episode_counters()
        # Exact replay is noise-free by definition: without this a benchmark episode
        # inherits whatever action_noise_std the previous episode's skill profile
        # left behind. Training is unaffected, since env.reset() re-applies it.
        self._action_noise_std = 0.0
        # Reset to a blank slate after body_pos is set: _load_scene forwarded at the
        # default pose, and a later mj_forward will not recompute those cached frames.
        mujoco.mj_resetData(m, d)
        self._write_sim_state(st)
        mujoco.mj_forward(m, d)
        return self._build_obs(), self._restored_info(st)

    def _randomize_lights(self):
        """Jitter light positions/colors and toggle a random subset of lights."""
        m = self.scene.model
        rng = self.np_random
        n = m.nlight
        m.light_pos[:] = self.scene._init_light_pos + rng.uniform(-1.0, 1.0, (n, 3))
        for i in range(n):
            axis = rng.uniform(-1, 1, 3)
            axis /= max(float(np.linalg.norm(axis)), 1e-6)
            ang = float(rng.uniform(-0.8, 0.8))
            cos, sin = float(np.cos(ang)), float(np.sin(ang))
            d = self.scene._init_light_dir[i]
            new_d = (d * cos + np.cross(axis, d) * sin
                     + axis * float(np.dot(axis, d)) * (1.0 - cos))
            norm = float(np.linalg.norm(new_d))
            if norm > 1e-6:
                m.light_dir[i] = new_d / norm * float(np.linalg.norm(d))
        scene_bright = float(rng.uniform(0.25, 1.7))
        bright = scene_bright * rng.uniform(0.5, 1.4, (n, 1))
        WARM = np.array([1.0, 0.78, 0.55]); COOL = np.array([0.65, 0.82, 1.0])
        scene_t = float(rng.uniform(-1.0, 1.0))
        ts = np.clip(scene_t + rng.uniform(-0.35, 0.35, n), -1.0, 1.0)
        tint = np.where(ts[:, None] >= 0,
                        1.0 + ts[:, None] * (WARM[None] - 1.0),
                        1.0 + (-ts[:, None]) * (COOL[None] - 1.0))
        m.light_specular[:] = np.clip(
            self.scene._init_light_specular + rng.uniform(-0.3, 0.3, (n, 3)), 0, 1)
        m.light_ambient[:] = np.clip(
            (self.scene._init_light_ambient + rng.uniform(-0.2, 0.2, (n, 3))) * bright * tint, 0, 1)
        m.light_diffuse[:] = np.clip(
            (self.scene._init_light_diffuse + rng.uniform(-0.2, 0.2, (n, 3))) * bright * tint, 0, 1)
        p_on = float(rng.uniform(0.25, 0.95))
        m.light_active[:] = (rng.uniform(0, 1, n) < p_on).astype(m.light_active.dtype)
        if not m.light_active.any():
            m.light_active[0] = 1
        if hasattr(m, "light_castshadow"):
            m.light_castshadow[:] = (rng.uniform(0, 1, n) < 0.5).astype(m.light_castshadow.dtype)
        if getattr(self, "_init_headlight", None) is not None:
            hl = m.vis.headlight
            hl_t = np.clip(scene_t + float(rng.uniform(-0.2, 0.2)), -1, 1)
            hl_tint = (1.0 + hl_t * (WARM - 1.0)) if hl_t >= 0 else (1.0 + (-hl_t) * (COOL - 1.0))
            hl_scale = float(np.clip(scene_bright * rng.uniform(0.6, 1.3), 0.3, 1.6))
            amb0, dif0, spec0 = self._init_headlight
            hl.ambient[:] = np.clip(amb0 * hl_scale * hl_tint, 0, 1)
            hl.diffuse[:] = np.clip(dif0 * hl_scale * hl_tint, 0, 1)
            hl.specular[:] = np.clip(spec0 * float(rng.uniform(0.5, 1.2)), 0, 1)
        if self._robot_white_mid >= 0:
            rgba = self._robot_white_rgba0.copy()
            rgba[:3] = np.clip(rgba[:3] + rng.uniform(-0.1, 0.1, 3).astype(rgba.dtype), 0, 1)
            m.mat_rgba[self._robot_white_mid] = rgba

    def _perturb_camera(self, cam_id, pos0, quat0, fovy0, pn, rn, fn):
        """Apply position/rotation/fovy noise to one camera around its nominal pose."""
        if cam_id < 0 or pos0 is None:
            return
        if pn <= 0 and rn <= 0 and fn <= 0:
            return
        m = self.scene.model
        rng = self.np_random
        m.cam_pos[cam_id] = pos0 + rng.uniform(-pn, pn, 3)
        if rn > 0:
            axis = rng.uniform(-1, 1, 3)
            axis /= max(float(np.linalg.norm(axis)), 1e-6)
            ang = float(rng.uniform(-rn, rn))
            half = 0.5 * ang
            dq = np.array([float(np.cos(half)), *(axis * float(np.sin(half)))])
            w0, x0, y0, z0 = quat0
            w1, x1, y1, z1 = dq
            m.cam_quat[cam_id] = [
                w1*w0 - x1*x0 - y1*y0 - z1*z0,
                w1*x0 + x1*w0 + y1*z0 - z1*y0,
                w1*y0 - x1*z0 + y1*w0 + z1*x0,
                w1*z0 + x1*y0 - y1*x0 + z1*w0,
            ]
        else:
            m.cam_quat[cam_id] = quat0
        if fn > 0:
            m.cam_fovy[cam_id] = fovy0 + float(rng.uniform(-fn, fn))
        else:
            m.cam_fovy[cam_id] = fovy0

    def _randomize_wrist_camera(self):
        """Apply the configured perturbation to the wrist camera."""
        self._perturb_camera(
            self._cam_ids.get("wrist_image", -1),
            self._wrist_cam_pos0, self._wrist_cam_quat0, self._wrist_cam_fovy0,
            self._wrist_camera_pos_noise, self._wrist_camera_rot_noise,
            self._wrist_camera_fovy_noise,
        )

    def _randomize_head_camera(self):
        """Perturb all fisheye tile cameras with one shared delta; fovy noise rescales the fisheye intrinsics."""
        # Head = fisheye. pos/rot perturb all 5 tile cameras (+ head_pov) with
        # the SAME delta so they stay co-located. fovy scales the fisheye K
        # (fx, fy) — this triggers a LUT rebuild via set_intrinsics, ~50ms.
        if not self._head_cam_ids_all:
            return
        pn = self._head_camera_pos_noise
        rn = self._head_camera_rot_noise
        fn = self._head_camera_fovy_noise
        dn = self._head_camera_distortion_noise
        if pn <= 0 and rn <= 0 and fn <= 0 and dn <= 0:
            return
        m = self.scene.model
        rng = self.np_random
        dp = rng.uniform(-pn, pn, 3) if pn > 0 else np.zeros(3)
        if rn > 0:
            axis = rng.uniform(-1, 1, 3)
            axis /= max(float(np.linalg.norm(axis)), 1e-6)
            ang = float(rng.uniform(-rn, rn))
            half = 0.5 * ang
            dq = np.array([float(np.cos(half)), *(axis * float(np.sin(half)))])
        else:
            dq = np.array([1.0, 0.0, 0.0, 0.0])
        for cid, pos0, quat0 in zip(self._head_cam_ids_all,
                                     self._head_cam_pos0_all,
                                     self._head_cam_quat0_all):
            m.cam_pos[cid] = pos0 + dp
            w0, x0, y0, z0 = quat0
            w1, x1, y1, z1 = dq
            m.cam_quat[cid] = [
                w1*w0 - x1*x0 - y1*y0 - z1*z0,
                w1*x0 + x1*w0 + y1*z0 - z1*y0,
                w1*y0 - x1*z0 + y1*w0 + z1*x0,
                w1*z0 + x1*y0 - y1*x0 + z1*w0,
            ]
        if (fn > 0 or dn > 0) and self._fisheye is not None:
            from molmospaces.components.fisheye import HEAD_FISHEYE_K, HEAD_FISHEYE_D
            K = HEAD_FISHEYE_K.copy()
            D = HEAD_FISHEYE_D.copy()
            if fn > 0:
                # fovy noise (degrees) → focal-length scale around the calibrated f.
                scale = 1.0 + float(rng.uniform(-fn, fn)) / 90.0
                K[0, 0] *= scale
                K[1, 1] *= scale
            if dn > 0:
                # Proportional noise: each coef shifts by ±dn fraction of its own
                # magnitude. Keeps the distortion model self-consistent — k1
                # changes ~20× more than k4 in absolute terms, same as the
                # calibrated ratio. Avoids edge scrambling from over-perturbed k3/k4.
                D = D * (1.0 + rng.uniform(-dn, dn, 4))
            self._fisheye.set_intrinsics(K=K, D=D)

    @staticmethod
    def _build_thor_texture_pools():
        """Build category -> texture-file pools from repo-local textures or the THOR material database."""
        import json as _json
        from molmospaces.components.constants import SCENE_TEXTURE_CATEGORIES
        # repo-local assets/textures/<Category>/ takes precedence over the THOR db
        canonical = set(SCENE_TEXTURE_CATEGORIES.values())
        local_root = ASSETS_DIR / "textures"
        if local_root.is_dir():
            local_pools = {}
            for cat in sorted(canonical):
                files = sorted(str(p) for p in (local_root / cat).glob("*.png"))
                if files:
                    local_pools[cat] = files
            if local_pools:
                return local_pools
        db_path = ASSETS_DIR / "objects" / "thor" / "material-database.json"
        mt_path = ASSETS_DIR / "objects" / "thor" / "material_to_textures.json"
        if not db_path.exists() or not mt_path.exists():
            return {}
        with open(db_path) as f:
            cat_db = _json.load(f)
        with open(mt_path) as f:
            mat_tex = _json.load(f)
        canonical = set(SCENE_TEXTURE_CATEGORIES.values())
        canon_by_lc = {c.lower(): c for c in canonical}

        pools: dict[str, set[str]] = {c: set() for c in canonical}
        for db_cat, mats in cat_db.items():
            target = canon_by_lc.get(db_cat.lower())
            if target is None:
                continue
            for m in mats:
                rec = mat_tex.get(m) or {}
                tex = rec.get("_MainTex")
                if not tex:
                    continue
                rel = tex.split("Assets/ThorAssets/Textures/", 1)[-1]
                rel = rel.split("Textures/", 1)[-1]
                abs_path = ASSETS_DIR / "objects" / "thor" / "Textures" / rel
                if not abs_path.exists():
                    continue
                # MuJoCo's loader only accepts PNG; fall back to sibling .png.
                if abs_path.suffix.lower() != ".png":
                    png_alt = abs_path.with_suffix(".png")
                    if png_alt.exists():
                        abs_path = png_alt
                    else:
                        continue
                pools[target].add(str(abs_path))
        return {c: sorted(v) for c, v in pools.items() if v}

    def _is_descendant(self, child, ancestor):
        """True if `child` body is `ancestor` or below it in the kinematic tree."""
        cur = int(child)
        while cur > 0:
            if cur == ancestor:
                return True
            cur = int(self.scene.model.body_parentid[cur])
        return False

    def _has_freejoint(self, bid):
        """True if the body's first joint is a freejoint."""
        if bid <= 0:
            return False
        m = self.scene.model
        jadr = int(m.body(bid).jntadr[0])
        return jadr >= 0 and m.jnt_type[jadr] == mujoco.mjtJoint.mjJNT_FREE

    def _raycast_down_skip_self(self, from_bid):
        """Returns (hit_top_z, hit_body_id), (None, None) on miss, or (None, 'floor')."""
        m, d = self.scene.model, self.scene.data
        pnt = d.xpos[from_bid].copy().astype(np.float64)
        pnt[2] += 1e-3
        vec = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        for _ in range(20):
            gid_arr = np.zeros(1, dtype=np.int32)
            dist = mujoco.mj_ray(m, d, pnt, vec, None, 1, -1, gid_arr)
            gid = int(gid_arr[0])
            if dist < 0 or gid < 0:
                return None, None
            hit_bid = int(m.geom_bodyid[gid])
            if self._is_descendant(hit_bid, from_bid):
                pnt[2] = pnt[2] - dist - 1e-3
                continue
            if int(m.geom_type[gid]) == int(mujoco.mjtGeom.mjGEOM_PLANE):
                return None, "floor"
            return float(pnt[2] - dist), hit_bid
        return None, None

    def _trace_support_chain(self, body_id, max_depth=10):
        """Raycast downward to find the static body ultimately supporting `body_id`, collecting free bodies passed through."""
        current = int(body_id)
        chain = []
        for _ in range(max_depth):
            top_z, hit_bid = self._raycast_down_skip_self(current)
            if hit_bid is None or hit_bid == "floor":
                return 0, None, chain
            node = hit_bid
            while node > 0:
                if self._has_freejoint(node):
                    chain.append(node)
                    current = node
                    break
                if int(self.scene.model.body_parentid[node]) == 0:
                    return node, top_z, chain
                node = int(self.scene.model.body_parentid[node])
            else:
                return 0, None, chain
        return 0, None, chain

    def _support_group_via_contacts(self, support_bid):
        """All free bodies transitively in contact with the support body (they must move together)."""
        m, d = self.scene.model, self.scene.data

        def owner(bid):
            """Map a body to its owning free body or top-level ancestor."""
            cur = int(bid)
            while cur > 0:
                if self._has_freejoint(cur):
                    return cur
                if int(m.body_parentid[cur]) == 0:
                    return cur
                cur = int(m.body_parentid[cur])
            return 0

        adj = {}
        for i in range(d.ncon):
            c = d.contact[i]
            o1 = owner(int(m.geom_bodyid[int(c.geom1)]))
            o2 = owner(int(m.geom_bodyid[int(c.geom2)]))
            if o1 == o2 or o1 <= 0 or o2 <= 0:
                continue
            adj.setdefault(o1, set()).add(o2)
            adj.setdefault(o2, set()).add(o1)
        visited = {support_bid}
        queue = [support_bid]
        while queue:
            cur = queue.pop()
            for nb in adj.get(cur, ()):
                if nb not in visited:
                    visited.add(nb)
                    queue.append(nb)
        return [b for b in visited if self._has_freejoint(b)]

    def _bodies_on_support(self, sup_root, target_bid, target_chain):
        """Every body resting on the support surface: raycast chains plus the contact graph."""
        m = self.scene.model
        grouped = {target_bid, *target_chain}
        for bid in range(m.nbody):
            if bid in grouped or bid == sup_root or bid == 0:
                continue
            if int(m.body_parentid[bid]) != 0:
                continue
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
            if name.startswith(self.scene._robot_prefix) or name in ("grasp_probe", "gripper_probe"):
                continue
            sr, _, chain = self._trace_support_chain(bid)
            if sr == sup_root:
                grouped.add(bid)
                grouped.update(chain)
        # The contact graph catches edge-perched objects the raycast misses.
        for bid in self._support_group_via_contacts(sup_root):
            if bid in grouped or bid == sup_root or bid == 0:
                continue
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
            if name.startswith(self.scene._robot_prefix) or name in ("grasp_probe", "gripper_probe"):
                continue
            grouped.add(bid)
        return grouped

    def _bodies_under_support(self, sup_root, sup_top_z, grouped_bids):
        """Furniture standing under the support, which must ride down with it."""
        m, d = self.scene.model, self.scene.data
        sup_geoms = [g for g in range(m.ngeom) if int(m.geom_bodyid[g]) == sup_root]
        if not sup_geoms:
            return set()
        sup_xy = d.geom_xpos[sup_geoms][:, :2]
        lo = sup_xy.min(axis=0) - 0.05
        hi = sup_xy.max(axis=0) + 0.05
        _SKIP = ("wall", "floor", "ceiling", "door", "window")
        under = set()
        for bid in range(1, m.nbody):
            if bid in grouped_bids or bid == sup_root or int(m.body_parentid[bid]) != 0:
                continue
            c = d.xipos[bid]
            if not (lo[0] <= c[0] <= hi[0] and lo[1] <= c[1] <= hi[1] and c[2] < sup_top_z):
                continue
            name = (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, bid) or "").lower()
            if (name.startswith(self.scene._robot_prefix.lower())
                    or "probe" in name or any(k in name for k in _SKIP)):
                continue
            under.add(bid)
        if not under:
            return under
        for j in range(m.njnt):
            if m.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE:
                continue
            bid = int(m.jnt_bodyid[j])
            if bid in grouped_bids or bid in under or int(m.body_parentid[bid]) != 0:
                continue
            sr, _, chain = self._trace_support_chain(bid)
            if sr in under:
                under.add(bid)
                under.update(chain)
        return under

    def _shift_bodies_z(self, sup_root, grouped_bids, dz):
        """Shift the support and everything grouped with it by dz, then refresh kinematics."""
        m, d = self.scene.model, self.scene.data
        # Freejoints move via qpos; static bodies via body_pos (clear simple/sameframe).
        grouped_qadrs, static_bids = [], []
        for b in grouped_bids:
            jadr = int(m.body(b).jntadr[0])
            if jadr >= 0 and m.jnt_type[jadr] == mujoco.mjtJoint.mjJNT_FREE:
                grouped_qadrs.append(int(m.jnt_qposadr[jadr]))
            elif int(m.body_parentid[b]) == 0:
                static_bids.append(int(b))

        for b in [sup_root, *static_bids]:
            m.body_simple[b] = 0
            m.body_sameframe[b] = 0
            m.body_pos[b, 2] += dz
        for qa in grouped_qadrs:
            d.qpos[qa + 2] += dz

        # Sleeping bodies stay frozen, so toggle sleep off while kinematics refresh.
        sleep_bit = int(mujoco.mjtEnableBit.mjENBL_SLEEP)
        sleep_was_on = bool(m.opt.enableflags & sleep_bit)
        m.opt.enableflags &= ~sleep_bit
        mujoco.mj_forward(m, d)
        if sleep_was_on:
            m.opt.enableflags |= sleep_bit

    def _sample_support_height(self, sup_top_z):
        """Triangular-sampled new top height for the support, or None if it cannot move."""
        upper = sup_top_z
        if self._randomize_height_max is not None:
            upper = min(upper, self._randomize_height_max)
        if upper <= self._randomize_height_min:
            return None
        mode = float(np.clip(self._randomize_height_favored, self._randomize_height_min, upper))
        return float(self.np_random.triangular(self._randomize_height_min, mode, upper))

    def _randomize_target_support_height(self, target_obj):
        """Raise/lower the target's support surface (and everything on it) to a sampled height."""
        if not self._randomize_height or target_obj is None:
            return
        target_bid = int(target_obj.body_id)
        sup_root, sup_top_z, target_chain = self._trace_support_chain(target_bid)
        if sup_root == 0 or sup_top_z is None:
            return

        new_top = self._sample_support_height(sup_top_z)
        if new_top is None:
            return
        dz = new_top - sup_top_z
        if abs(dz) < 1e-3:
            return

        grouped = self._bodies_on_support(sup_root, target_bid, target_chain)
        if dz < 0:
            grouped |= self._bodies_under_support(sup_root, sup_top_z, grouped)
        self._shift_bodies_z(sup_root, grouped, dz)

    def _sample_skill_profile(self):
        """Pick a skill profile by weight and overwrite the affected spawn/noise knobs for this episode."""
        if not self._skill_profiles:
            self._active_profile = None
            return
        names = [n for n, _, _ in self._skill_profiles]
        weights = [w for _, w, _ in self._skill_profiles]
        idx = int(self.np_random.choice(len(names), p=weights))
        name, _, profile = self._skill_profiles[idx]
        self._active_profile = name
        if "spawn_at_grasp" in profile: self._spawn_at_grasp = bool(profile["spawn_at_grasp"])
        if "sample_spawn_first" in profile: self._sample_spawn_first = bool(profile["sample_spawn_first"])
        if "ring_num_angles" in profile: self._ring_num_angles = int(profile["ring_num_angles"])
        if "spawn_along_line" in profile: self._spawn_along_line = bool(profile["spawn_along_line"])
        if "spawn_reachability_check" in profile: self._spawn_reachability_check = bool(profile["spawn_reachability_check"])
        if "walk_dist_min" in profile: self._walk_dist_min = float(profile["walk_dist_min"])
        if "walk_dist_max" in profile: self._walk_dist_max = float(profile["walk_dist_max"])
        if "spawn_radius_min" in profile: self._spawn_radius_min = float(profile["spawn_radius_min"])
        if "spawn_radius_max" in profile: self._spawn_radius_max = float(profile["spawn_radius_max"])
        if "spawn_visibility_check" in profile:
            self._spawn_visibility_check = bool(profile["spawn_visibility_check"])
        if "arm_init_radius" in profile: self._arm_init_radius = float(profile["arm_init_radius"])
        if "pregrasp_xyz_noise" in profile: self._pregrasp_xyz_noise = float(profile["pregrasp_xyz_noise"])
        if "pregrasp_rot_noise" in profile: self._pregrasp_rot_noise = float(profile["pregrasp_rot_noise"])
        if "action_noise_std" in profile: self._action_noise_std = float(profile["action_noise_std"])
        if "face_yaw_offset" in profile: self._face_yaw_offset = float(profile["face_yaw_offset"])
        if "randomize_height" in profile: self._randomize_height = bool(profile["randomize_height"])
        if "randomize_height_min" in profile: self._randomize_height_min = float(profile["randomize_height_min"])
        if "randomize_height_favored" in profile:
            self._randomize_height_favored = float(profile["randomize_height_favored"])
        if "randomize_height_max" in profile:
            self._randomize_height_max = (None if profile["randomize_height_max"] is None
                                     else float(profile["randomize_height_max"]))
        if "start_at_pregrasp_xy_noise" in profile:
            self._start_at_pregrasp_xy_noise = float(profile["start_at_pregrasp_xy_noise"])
        if "start_at_pregrasp_yaw_noise" in profile:
            self._start_at_pregrasp_yaw_noise = float(profile["start_at_pregrasp_yaw_noise"])
        if "start_at_pregrasp_joint_noise" in profile:
            self._start_at_pregrasp_joint_noise = float(profile["start_at_pregrasp_joint_noise"])
        if "init_arm_at_pregrasp" in profile:
            self._init_arm_at_pregrasp = bool(profile["init_arm_at_pregrasp"])

    def _maybe_reload_scene(self):
        """Load a new scene when randomizing or when the current one has no targets; False on failure."""
        _need_objects = not getattr(self.task, "objects", None)
        if not (len(self._scene_paths) > 1
                and (_need_objects or (self._randomize_scene
                     and (self._reset_counter % self._randomize_scene_freq) == 0))):
            return True
        try:
            self._load_scene(self._scene_paths[self.np_random.integers(len(self._scene_paths))])
        except (ValueError, IndexError, IOError, OSError) as e:
            print(f"[env] skipping scene (load failed: {type(e).__name__}: {e})")
            return False
        except Exception as e:
            # Catch-all: a dead worker cascades to the whole vec_env, so any
            # scene-load error is swallowed and retried instead.
            print(f"[env] skipping scene (load error: {type(e).__name__}: {e})")
            return False
        return True

    def _randomize_appearance(self):
        """Re-texture and re-light the scene and jitter the cameras; must run after scene.reset()."""
        if (self._randomize_textures and self.scene.scene_matids and self.scene.scene_geom_ids
                and self.np_random.random() >= self._randomize_textures_keep_prob):
            color_mids = self.scene.scene_color_matids
            if color_mids:
                rand_rgba = self.np_random.uniform(0.0, 1.0, size=(len(color_mids), 4)).astype(np.float32)
                rand_rgba[:, 3] = 1.0
                for i, mid in enumerate(color_mids):
                    self.scene.model.mat_rgba[mid] = rand_rgba[i]
            scp = self._randomize_textures_solid_color_prob
            for cat, gids in self.scene.scene_geom_ids.items():
                if not gids:
                    continue
                tex_mids = self.scene.scene_matids.get(cat)
                if not tex_mids:
                    continue
                tex_picks = self.np_random.choice(tex_mids, size=len(gids))
                use_solid = (self.np_random.random(size=len(gids)) < scp) if color_mids else None
                solid_picks = (self.np_random.choice(color_mids, size=len(gids))
                               if color_mids else None)
                for k, gid in enumerate(gids):
                    if use_solid is not None and use_solid[k]:
                        self.scene.model.geom_matid[gid] = int(solid_picks[k])
                    else:
                        self.scene.model.geom_matid[gid] = int(tex_picks[k])
        if (self._randomize_lighting and self.scene.model.nlight > 0
                and self.np_random.random() >= self._randomize_lighting_keep_prob):
            self._randomize_lights()
        self._randomize_wrist_camera()
        self._randomize_head_camera()

    def _prepare_target(self):
        """Pick the target, jitter it, set its support height, and return (obj, grasp-frame xy)."""
        obj = self.task.select_target(self.np_random)
        # Jitter object xy before reading pos so goal sampling sees the perturbed location.
        self.task.perturb_objects(self.scene, self.np_random)
        # Must run before goal sampling so the goal lands at the new z.
        self._randomize_target_support_height(obj)
        self.task.init_target_tracking(self.scene)
        tgt = obj.position(self.scene.data)
        return obj, tgt

    def _sample_goal_and_spawn_first(self, tgt, obj, sampled_upper):
        """Sample the goal pose, and the spawn too when sample_spawn_first is on."""
        if self._sample_spawn_first and not self._spawn_at_grasp:
            sxy, syaw, goal_xy, goal_yaw = self._sample_spawn_then_goal(
                tgt, obj, sampled_upper=sampled_upper)
            return goal_xy, goal_yaw, sxy, syaw
        goal_xy, goal_yaw = self._sample_goal_pose(tgt, obj, sampled_upper=sampled_upper)
        return goal_xy, goal_yaw, None, None

    def _jitter_goal(self, goal_xy, goal_yaw):
        """Misalign the goal so the controller has to realign; yields base-motion grasp demos."""
        if self._goal_offset_xy_noise > 0:
            goal_xy = goal_xy + self.np_random.uniform(
                -self._goal_offset_xy_noise, self._goal_offset_xy_noise, size=2)
        if self._goal_offset_yaw_noise > 0:
            goal_yaw = float(goal_yaw + self.np_random.uniform(
                -self._goal_offset_yaw_noise, self._goal_offset_yaw_noise))
        return goal_xy, goal_yaw

    def _build_realign_info(self, goal_xy, goal_yaw):
        """Spawn behind the goal holding the pregrasp pose, so the episode starts mid-realign."""
        spp_xy_noise = self._start_at_pregrasp_xy_noise
        spp_yaw_noise = self._start_at_pregrasp_yaw_noise
        if not ((spp_xy_noise > 0 or spp_yaw_noise > 0) and self.agent is not None):
            return None
        pregrasp_joints = getattr(self.agent._grasp_planner, "_pregrasp_joints", None)
        if not pregrasp_joints:
            return None
        # The WBC walks forward well and sideways/yaw poorly, so offset in body-x only.
        axis = "x"
        offset = -float(self.np_random.uniform(spp_xy_noise * 0.5, spp_xy_noise))
        cy, sy = float(np.cos(goal_yaw)), float(np.sin(goal_yaw))
        spawn_xy = np.array(goal_xy, dtype=np.float64).copy()
        spawn_xy[0] += offset * cy
        spawn_xy[1] += offset * sy
        pj = dict(pregrasp_joints)
        jn = self._start_at_pregrasp_joint_noise
        if jn > 0:
            perturb = {
                "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
                "right_shoulder_yaw_joint", "right_elbow_joint",
                "right_wrist_roll_joint", "right_wrist_pitch_joint",
                "right_wrist_yaw_joint",
                "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
            }
            m = self.scene.model
            for name in list(pj.keys()):
                if name not in perturb:
                    continue
                jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{PREFIX}{name}")
                if jid < 0:
                    continue
                lo, hi = m.jnt_range[jid]
                pj[name] = float(np.clip(pj[name] + float(self.np_random.uniform(-jn, jn)), lo, hi))
        return dict(start_at_pregrasp=True, realign_axis=axis, realign_offset=offset,
                    pregrasp_joints=pj, spawn_xy=spawn_xy, spawn_yaw=float(goal_yaw))

    def _sample_random_spawn(self, tgt, obj, sampled_upper):
        """Rejection-sample a spawn in the annulus that is collision-free, visible and reachable."""
        saved_qpos = self.scene.data.qpos.copy()
        saved_ctrl = self.scene.data.ctrl.copy()
        xy, yaw = None, None
        for _ in range(50):
            cand_xy = self.occ.sample_near(tgt[:2],
                                           radius_min=self._spawn_radius_min,
                                           radius_max=self._spawn_radius_max,
                                           np_random=self.np_random)
            if cand_xy is None:
                continue
            cand_xy = np.asarray(cand_xy, dtype=np.float64)
            if self._spawn_visibility_check:
                bearing = float(np.arctan2(tgt[1] - cand_xy[1], tgt[0] - cand_xy[0]))
                cand_yaw = bearing + self.np_random.uniform(-0.5, 0.5)
            else:
                cand_yaw = self.np_random.uniform(-np.pi, np.pi)
            self.robot.set_pose(cand_xy, cand_yaw)
            self.robot.set_defaults()
            if self.agent is not None:
                self.agent._set_groot_defaults()
            if sampled_upper is not None:
                self.robot.apply_upper_pose(sampled_upper)
            mujoco.mj_forward(self.scene.model, self.scene.data)
            if self._robot_has_scene_collision():
                continue
            if self._spawn_visibility_check and not self.robot.check_object_visibility(obj.body_id):
                continue
            if (self._spawn_reachability_check
                    and not self.occ_safe.same_free_component(cand_xy, tgt[:2])):
                continue
            xy, yaw = cand_xy, cand_yaw
            break
        self.scene.data.qpos[:] = saved_qpos
        self.scene.data.ctrl[:] = saved_ctrl
        mujoco.mj_forward(self.scene.model, self.scene.data)
        return xy, yaw

    def _choose_spawn(self, realign_info, spawn_first, goal_xy, goal_yaw, tgt, obj, sampled_upper):
        """Resolve the spawn pose for whichever spawn mode this episode uses."""
        if realign_info is not None:
            return realign_info["spawn_xy"], realign_info["spawn_yaw"]
        if spawn_first[0] is not None:
            return spawn_first
        if self._spawn_at_grasp:
            return goal_xy, goal_yaw
        if self._spawn_along_line:
            return self._sample_spawn_along_line(goal_xy, tgt, obj, sampled_upper)
        if self._randomize_placement:
            return self._sample_random_spawn(tgt, obj, sampled_upper)
        return self.robot.get_xy(), self.robot.get_yaw()

    def _spawn_is_valid(self, tgt, realign_info):
        """Reject spawns off the mapped floor, or with no navigable route to the target."""
        if not self.occ.is_free(self.robot.get_xy()):
            return False
        if (self._spawn_reachability_check and realign_info is None
                and not self._spawn_at_grasp
                and not self.occ_safe.same_free_component(self.robot.get_xy(), tgt[:2])):
            return False
        return True

    def _make_reset_info(self, tgt, goal_xy, goal_yaw, sampled_upper, pregrasp_offset,
                         used_pregrasp_upper, realign_info):
        """Assemble the info dict handed back to the agent for this episode."""
        info = self.task.make_info(self.scene, self.np_random)
        info.update(distance=float(np.linalg.norm(self.robot.get_xy() - tgt[:2])),
                    occupancy_map=self.occ, nav_occupancy_map=self.occ_safe,
                    goal_xy=goal_xy, goal_yaw=goal_yaw,
                    init_upper_pose=sampled_upper, pregrasp_offset=pregrasp_offset,
                    init_arm_at_pregrasp=used_pregrasp_upper,
                    skill_profile=(self._active_profile or "default"),
                    face_yaw_offset_max=self._face_yaw_offset,
                    scene=self._scene_name())
        if realign_info is not None:
            info.update(start_at_pregrasp=True,
                        realign_axis=realign_info["realign_axis"],
                        realign_offset=realign_info["realign_offset"],
                        pregrasp_joints=realign_info["pregrasp_joints"])
        if self._randomize_robot_height and (realign_info is not None or self._spawn_at_grasp):
            info["init_height"] = float(self.np_random.uniform(
                self._randomize_robot_height_min, self._randomize_robot_height_max))
        return info

    def _maybe_start_at_pregrasp(self, sampled_upper, goal_xy, goal_yaw):
        """Swap in the pregrasp arm pose so APPROACH is a no-op; the flag lets the agent tell them apart."""
        if not (self._init_arm_at_pregrasp and self.agent is not None):
            return sampled_upper, False
        pregrasp_upper = self._compute_pregrasp_upper(goal_xy, goal_yaw)
        if pregrasp_upper is None:
            return sampled_upper, False
        return pregrasp_upper, True

    def _precheck_grasp_ok(self, goal_xy, goal_yaw):
        """Ask the scripted planner whether it can solve this grasp before committing to the spawn."""
        if self.agent is None or not self._reset_precheck_grasp:
            return True
        info = self.task.make_info(self.scene, self.np_random)
        info["goal_xy"] = goal_xy
        info["goal_yaw"] = goal_yaw
        return self.agent.precheck_grasp(info)

    def _place_robot(self, xy, yaw, realign_info, sampled_upper):
        """Teleport the robot to the chosen spawn with the right upper-body pose, at rest."""
        self.robot.set_pose(xy, yaw)
        self.robot.set_defaults()
        if realign_info is not None:
            self.robot.apply_arm_pose(realign_info["pregrasp_joints"])
        elif sampled_upper is not None:
            self.robot.apply_upper_pose(sampled_upper)
        self.robot.zero_velocities()
        mujoco.mj_forward(self.scene.model, self.scene.data)

    def _try_reset(self):
        """One placement attempt: pick scene/object/spawn, settle physics, and return (obs, info) or None."""
        self._sim_time = 0.0
        self._action_noise_step = 0
        self._action_noise_offset.fill(0.0)
        self._sample_skill_profile()
        if not self._maybe_reload_scene():
            return None

        self.scene.reset()
        self._randomize_appearance()
        self.robot.zero_velocities()
        self.robot.set_defaults()

        sampled_upper = (self.robot.sample_upper_pose(self.np_random, self._arm_init_radius)
                         if self._arm_init_radius > 0 else None)
        if sampled_upper is not None:
            self.robot.apply_upper_pose(sampled_upper)

        obj, tgt = self._prepare_target()
        goal_xy, goal_yaw, sf_xy, sf_yaw = self._sample_goal_and_spawn_first(
            tgt, obj, sampled_upper)
        if goal_xy is None:
            self.task.target = None
            return None

        sampled_upper, used_pregrasp_upper = self._maybe_start_at_pregrasp(
            sampled_upper, goal_xy, goal_yaw)
        goal_xy, goal_yaw = self._jitter_goal(goal_xy, goal_yaw)
        if not self._precheck_grasp_ok(goal_xy, goal_yaw):
            self.task.target = None
            return None

        realign_info = self._build_realign_info(goal_xy, goal_yaw)
        xy, yaw = self._choose_spawn(realign_info, (sf_xy, sf_yaw),
                                     goal_xy, goal_yaw, tgt, obj, sampled_upper)
        if xy is None:
            return None
        self._place_robot(xy, yaw, realign_info, sampled_upper)
        if not self._spawn_is_valid(tgt, realign_info):
            return None

        weld = mujoco.mj_name2id(self.scene.model, mujoco.mjtObj.mjOBJ_EQUALITY, "pelvis_weld")
        if weld >= 0:
            self.scene.data.eq_active[weld] = 0

        pregrasp_offset = self._sample_pregrasp_offset()
        info = self._make_reset_info(tgt, goal_xy, goal_yaw, sampled_upper, pregrasp_offset,
                                     used_pregrasp_upper, realign_info)
        return self._build_obs(), info

    def _apply_action_noise(self, action):
        """Add exploration noise to the arm/waist channels, except during precision phases."""
        if not (self._action_noise_std > 0 and len(action) == 15):
            return action
        in_precision = (self.agent is not None
                        and hasattr(self.agent, 'in_precision_phase')
                        and self.agent.in_precision_phase())
        if in_precision:
            self._action_noise_offset.fill(0.0)
            self._action_noise_step = 0
            return action
        # Resample once per record stride so saved frames stay consistent.
        if self._action_noise_step % self._action_noise_stride == 0:
            self._action_noise_offset = self.np_random.normal(
                0.0, self._action_noise_std, size=10)
        self._action_noise_step += 1
        action = action.copy()
        action[4:14] += self._action_noise_offset
        return action

    def _grasp_phase(self):
        """The demonstrator's current grasp phase, or None when there is no agent."""
        return getattr(self.agent, "_grasp_phase", None) if self.agent is not None else None

    def _check_termination(self):
        """Collision and visibility guards; returns (terminated, sim_error)."""
        if not self.robot.state_is_finite():
            return True, "non-finite state"
        if self.robot.pelvis_height() < 0.15:
            return True, f"fell (z={self.robot.pelvis_height():.3f})"

        # Walking/realigning: any world contact is a crash. Turn off for RL, where
        # collisions are part of exploration.
        if self._terminate_before_grasp_collision:
            if self._grasp_phase() in (None, PHASE_IDLE, PHASE_REALIGN) and self._robot_touches_world():
                return True, "robot hit world before grasp"
        # Reaching/closing/lifting: the target itself is excluded inside
        # _robot_touches_world, so touching the bowl does not count.
        if self._terminate_on_grasp_collision:
            in_grasp = self._grasp_phase() in (PHASE_APPROACH, PHASE_DESCEND, PHASE_OPEN_HOLD,
                                               PHASE_CLOSE, PHASE_POST_CLOSE, PHASE_LIFT)
            if in_grasp and self._robot_touches_world():
                return True, "robot hit world during grasp"
        # Checked once, at the step the controller starts reaching: a demo that
        # grasps something off-camera would teach the policy to do the same.
        if self._terminate_grasp_if_not_visible:
            phase = self._grasp_phase()
            starts_reaching = phase == PHASE_APPROACH and self._prev_grasp_phase != PHASE_APPROACH
            self._prev_grasp_phase = phase
            if starts_reaching and not self._target_visible_in_head():
                return True, "target not visible at grasp"
        return False, None

    def _step_physics(self, action):
        """Drive the controller (or raw ctrl) for one control step and check termination."""
        try:
            with (self._viewer.lock() if self._viewer else nullcontext()):
                if self.agent is not None and hasattr(self.agent, 'execute_action') and len(action) == 15:
                    self.agent.execute_action(action)
                else:
                    mask = ~np.isnan(action)
                    if np.any(mask):
                        self.scene.data.ctrl[self.robot.act_ids[mask]] = action[mask]
                mujoco.mj_step(self.scene.model, self.scene.data, nstep=self.robot.n_substeps)
                return self._check_termination()
        except mujoco.FatalError as e:
            return True, str(e)

    def _step_info(self, dist, success, sim_error):
        """Per-step info dict: target pose, distance, success and any sim error."""
        tgt = self.target
        info = self.task.step_info()
        info.update(target_object_position=tgt.position(self.scene.data),
                    target_object_pose=np.concatenate(
                        [tgt.position(self.scene.data), tgt.quat(self.scene.data)]),
                    distance=dist, success=success, sim_error=sim_error)
        self.task.attach_grasps(info)
        return info

    def step(self, action):
        """Advance one control step: add exploration noise outside precision phases, run the low-level controller, and report task success."""
        action = np.asarray(action, dtype=np.float64)
        # Record the pre-noise command; it is fed back as an observation next step.
        if len(action) >= 3:
            self._last_base_vel_cmd = action[:3].astype(np.float32).copy()
        action = self._apply_action_noise(action)

        terminated, sim_error = self._step_physics(action)

        self._sim_time += self.robot.n_substeps * self.scene.model.opt.timestep
        if self._viewer:
            self.sync_viewer()

        obs = self._build_obs()
        tgt = self.target
        dist = float(np.linalg.norm(self.robot.get_xy() - tgt.position(self.scene.data)[:2]))
        reward = self.task.compute_reward(self.scene)
        terminated = terminated or reward > 0.1
        success = reward > 0.04
        return obs, reward, terminated, False, self._step_info(dist, success, sim_error)

    def consume_skip_episode(self):
        """Return and clear the skip-episode flag set during reset."""
        s = self._skip_episode
        self._skip_episode = False
        return s

    def sync_viewer(self):
        """Push the current sim state (and debug markers) to the interactive viewer."""
        if not self._viewer:
            return
        with self._viewer.lock():
            if getattr(self, "debug", False):
                self._draw_debug_markers()
            self._viewer.sync()

    def _draw_debug_markers(self):
        """Redraw the viewer overlay: nav waypoints and the planned grasp pose."""
        scn = self._viewer.user_scn
        scn.ngeom = 0
        if self.agent is None:
            return
        self._draw_waypoints(scn)
        self._draw_grasp_probe(scn)

    def _draw_waypoints(self, scn):
        """Flat discs along the nav path, green behind the robot and red ahead."""
        wps = getattr(self.agent, "_waypoints", None)
        if not wps:
            return
        cur_idx = int(getattr(self.agent, "_wp_idx", 0))
        eye = np.eye(3).flatten()
        size = np.array([0.20, 0.003, 0.0], dtype=np.float32)
        for i, wp in enumerate(wps):
            if scn.ngeom >= len(scn.geoms):
                return
            rgba = np.array([0.0, 1.0, 0.0, 0.7] if i < cur_idx else [1.0, 0.1, 0.1, 0.7],
                            dtype=np.float32)
            mujoco.mjv_initGeom(scn.geoms[scn.ngeom],
                                type=int(mujoco.mjtGeom.mjGEOM_CYLINDER),
                                size=size,
                                pos=np.array([float(wp[0]), float(wp[1]), 0.003], dtype=np.float32),
                                mat=eye, rgba=rgba)
            scn.ngeom += 1

    def _draw_grasp_probe(self, scn):
        """Ghost gripper at the planned grasp; the real probe stays parked at z=10."""
        planner = getattr(self.agent, "_grasp_planner", None)
        if planner is None:
            return
        grasp_pos = getattr(planner, "_grasp_pos", None)
        grasp_rot = getattr(planner, "_grasp_rot", None)
        if grasp_pos is None or grasp_rot is None:
            return
        if not hasattr(self, "_probe_local_geoms"):
            self._probe_local_geoms = self._cache_probe_local_geoms()
        xpos = np.asarray(grasp_pos, dtype=np.float64)
        xmat = np.asarray(grasp_rot, dtype=np.float64)
        rgba = np.array([0.7, 0.9, 1.0, 0.45], dtype=np.float32)
        for type_int, size, local_pos, local_mat in self._probe_local_geoms:
            if scn.ngeom >= len(scn.geoms):
                return
            mujoco.mjv_initGeom(
                scn.geoms[scn.ngeom],
                type=type_int,
                size=np.asarray(size, dtype=np.float32),
                pos=(xpos + xmat @ local_pos).astype(np.float32),
                mat=(xmat @ local_mat).flatten().astype(np.float32),
                rgba=rgba,
            )
            scn.ngeom += 1

    def _cache_probe_local_geoms(self):
        """Snapshot the open-finger probe geoms in root-body frame for debug overlays."""
        m = self.scene.model
        d = self.scene.data
        gj = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "gripper_probe_joint")
        if gj < 0:
            return []
        saved = d.qpos.copy()
        qa = int(m.jnt_qposadr[gj])
        d.qpos[qa:qa + 3] = [0, 0, 0]
        d.qpos[qa + 3:qa + 7] = [1, 0, 0, 0]
        for jname, val in (("gripper_probe_joint_a", 0.04),
                            ("gripper_probe_joint_b", -0.04)):
            jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jname)
            if jid >= 0:
                d.qpos[m.jnt_qposadr[jid]] = val
        mujoco.mj_forward(m, d)
        probe_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "gripper_probe")
        descendants = self.scene.get_body_descendants(probe_bid) if probe_bid >= 0 else set()
        out = []
        for gid in range(m.ngeom):
            if int(m.geom_bodyid[gid]) not in descendants:
                continue
            out.append((
                int(m.geom_type[gid]),
                np.array(m.geom_size[gid], dtype=np.float64).copy(),
                np.array(d.geom_xpos[gid], dtype=np.float64).copy(),
                np.array(d.geom_xmat[gid], dtype=np.float64).reshape(3, 3).copy(),
            ))
        d.qpos[:] = saved
        mujoco.mj_forward(m, d)
        return out

    def close(self):
        """Close the viewer and every GL context, so nothing is torn down at interpreter exit."""
        if self._viewer:
            self._viewer.close()
            self._viewer = None
        self._release_gpu_resources()
def get_config():
    """Default env ConfigDict mirroring G1Env.__init__ arguments."""
    import ml_collections
    return ml_collections.ConfigDict(dict(
        scene="", objects=".*", seed=0,
        randomize_scene=False, randomize_scene_freq=1,
        randomize_object=False, randomize_placement=True,
        object_noise=0.0,
        spawn_radius_min=1.0, spawn_radius_max=8.0,
        grasp_spawn_radius_min=0.25, grasp_spawn_radius_max=0.80,
        spawn_visibility_check=False,
        arm_init_radius=0.0,
        pregrasp_xyz_noise=0.0, pregrasp_rot_noise=0.0,
        action_noise_std=0.005, action_noise_stride=5,
        spawn_at_grasp=False,
        sample_spawn_first=False, ring_num_angles=32,
        spawn_along_line=False, walk_dist_min=0.3, walk_dist_max=0.8,
        randomize_textures=False, scene_textures_glob="textures/randomization/*.png",
        max_textures=5,
        randomize_lighting=False,
        face_yaw_offset=0.0,
        randomize_height=True, randomize_height_min=0.0, randomize_height_favored=0.95,
        randomize_height_max=None,
        randomize_robot_height=False,
        randomize_robot_height_min=0.7, randomize_robot_height_max=0.793,
        wrist_camera_pos_noise=0.0, wrist_camera_rot_noise=0.0,
        wrist_camera_fovy_noise=0.0,
        head_camera_pos_noise=0.0, head_camera_rot_noise=0.0,
        head_camera_fovy_noise=0.0, head_camera_distortion_noise=0.0,
        goal_offset_xy_noise=0.0, goal_offset_yaw_noise=0.0,
        start_at_pregrasp_xy_noise=0.0, start_at_pregrasp_yaw_noise=0.0,
        start_at_pregrasp_joint_noise=0.0,
        init_arm_at_pregrasp=False,
        launch_viewer=False,
        camera_size=(224, 224)))


def make_env(config):
    """Instantiate G1Env from a ConfigDict/dict of kwargs."""
    kwargs = dict(config)
    if "camera_size" in kwargs:
        kwargs["camera_size"] = tuple(kwargs["camera_size"])
    return G1Env(**kwargs)
