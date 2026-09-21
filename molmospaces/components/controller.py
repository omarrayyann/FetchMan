from __future__ import annotations

import collections
from pathlib import Path

import mujoco
import numpy as np
import onnxruntime as ort

from molmospaces.components.robot import JOINT_NAMES as _JOINTS, DEFAULT_QPOS as _ROBOT_DEFAULT_QPOS

_WBC_NUM_ACTIONS = 15
_WBC_DEFAULT = np.array([-0.1,0,0,0.3,-0.2,0, -0.1,0,0,0.3,-0.2,0, 0,0,0], dtype=np.float32)
_WBC_CMD_SCALE = np.array([2.0, 2.0, 0.5], dtype=np.float32)
_WBC_ACTION_SCALE = 0.25
_WBC_ANG_VEL_SCALE = 0.5
_WBC_DOF_POS_SCALE = 1.0
_WBC_DOF_VEL_SCALE = 0.05
_WBC_HEIGHT_CMD = 0.74
_WBC_OBS_HISTORY = 6
_WBC_CONTROL_DEC = 4
_WBC_OBS_DIM = 86
_WBC_BODY = _JOINTS[:29]  # legs(12) + waist(3) + arms(14); gripper tendon-controlled.
_WBC_N = 29

_WBC_KP = np.array([
    150,150,150,200,40,40,  150,150,150,200,40,40,  250,250,250,
    0,0,0,0,0,0,0,  0,0,0,0,0,0,0], dtype=np.float32)
_WBC_KD = np.array([
    2,2,2,4,2,2,  2,2,2,4,2,2,  5,5,5,
    0,0,0,0,0,0,0,  0,0,0,0,0,0,0], dtype=np.float32)


def _gravity_orientation(quat):
    """Gravity direction in the base frame from the base quaternion (ONNX policy input)."""
    w, x, y, z = quat
    gv = np.array([0., 0., -1.])
    c = np.array([w, -x, -y, -z])
    return np.array([
        gv[0]*(c[0]**2+c[1]**2-c[2]**2-c[3]**2)+gv[1]*2*(c[1]*c[2]-c[0]*c[3])+gv[2]*2*(c[1]*c[3]+c[0]*c[2]),
        gv[0]*2*(c[1]*c[2]+c[0]*c[3])+gv[1]*(c[0]**2-c[1]**2+c[2]**2-c[3]**2)+gv[2]*2*(c[2]*c[3]-c[0]*c[1]),
        gv[0]*2*(c[1]*c[3]-c[0]*c[2])+gv[1]*2*(c[2]*c[3]+c[0]*c[1])+gv[2]*(c[0]**2-c[1]**2-c[2]**2+c[3]**2),
    ], dtype=np.float32)


_DEFAULT_MODELS_DIR = Path(__file__).resolve().parent.parent / "agents" / "models"


class G1Controller:
    def __init__(self, models_dir: Path | str | None = None):
        """Low-level G1 controller: load the ONNX walk/balance policies."""
        d = Path(models_dir) if models_dir is not None else _DEFAULT_MODELS_DIR
        self._stand_sess = ort.InferenceSession(str(d / "groot_balance.onnx"), providers=["CPUExecutionProvider"])
        self._walk_sess = ort.InferenceSession(str(d / "groot_walk.onnx"), providers=["CPUExecutionProvider"])

    def set_env(self, env):
        """Attach the env for sim-time access."""
        m = env.scene.model
        m.opt.timestep = 0.005
        m.opt.noslip_iterations = 5
        m.opt.impratio = 1.0
        m.opt.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
        m.opt.gravity[:] = [0, 0, -9.81]
        m.opt.enableflags = int(mujoco.mjtEnableBit.mjENBL_SLEEP)
        env.robot.n_substeps = 1
        for gid in range(m.ngeom):
            gname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
            if m.geom_type[gid] == mujoco.mjtGeom.mjGEOM_PLANE or "floor" in gname.lower():
                m.geom_friction[gid] = [1.0, 0.005, 0.0001]
                m.geom_solref[gid] = [0.02, 1.0]
        prefix = self._prefix if hasattr(self, '_prefix') else "robot_0/"
        for gid in range(m.ngeom):
            bid = m.geom_bodyid[gid]
            bname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
            if bname.startswith(prefix) and "ankle_roll" in bname and m.geom_type[gid] == mujoco.mjtGeom.mjGEOM_SPHERE:
                m.geom_friction[gid] = [1.0, 0.005, 0.0001]

    def setup(self, model, data, prefix="robot_0/"):
        """Bind joint/actuator/sensor addresses for the attached robot."""
        self._model, self._data, self._prefix = model, data, prefix

        self._jqpos = np.array([model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{prefix}{n}")]
                                for n in _WBC_BODY], dtype=np.int32)
        self._jdof = np.array([model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{prefix}{n}")]
                               for n in _WBC_BODY], dtype=np.int32)
        self._fj_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{prefix}floating_base_joint")
        self._fj_qa = model.jnt_qposadr[self._fj_id]
        self._fj_da = model.jnt_dofadr[self._fj_id]
        self._pelvis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{prefix}pelvis")

        self._act_ids = []
        for i, jname in enumerate(_WBC_BODY):
            aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{prefix}walk_{jname}")
            if aid < 0:
                aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{prefix}{jname}")
            self._act_ids.append(aid)
            if aid >= 0:
                if i < 15:
                    model.actuator_gainprm[aid, 0] = 1.0
                    model.actuator_biasprm[aid, :] = 0.0
                    model.actuator_biastype[aid] = mujoco.mjtBias.mjBIAS_NONE
                    model.actuator_gaintype[aid] = mujoco.mjtGain.mjGAIN_FIXED
                elif i < 22:
                    # Left arm: zero gain so it falls under gravity.
                    model.actuator_gainprm[aid, 0] = 0.0
                    model.actuator_biasprm[aid, :] = 0.0
                    model.actuator_forcerange[aid] = [-400, 400]
                else:
                    kp, kd = 2000.0, 60.0
                    model.actuator_gainprm[aid, 0] = kp
                    model.actuator_biasprm[aid, 0] = 0.0
                    model.actuator_biasprm[aid, 1] = -kp
                    model.actuator_biasprm[aid, 2] = -kd
                    model.actuator_forcerange[aid] = [-400, 400]
        self._act_ids = np.array(self._act_ids)

        for i in range(15, _WBC_N):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{prefix}{_WBC_BODY[i]}")
            if jid >= 0:
                model.jnt_actfrcrange[jid] = [-400, 400]
                model.dof_damping[model.jnt_dofadr[jid]] = 3.0


        self._rgripper = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{prefix}right_grip")

        self._obs_history = collections.deque(maxlen=_WBC_OBS_HISTORY)
        self._obs_buffer = np.zeros(_WBC_OBS_DIM * _WBC_OBS_HISTORY, dtype=np.float32)
        self._wbc_action = np.zeros(_WBC_NUM_ACTIONS, dtype=np.float32)
        self._target_lower = _WBC_DEFAULT.copy()
        self._cmd = np.zeros(3, dtype=np.float32)
        self._height_cmd = _WBC_HEIGHT_CMD
        self._step_counter = 0

        self._target_q = np.zeros(_WBC_N, dtype=np.float32)
        self._target_q[:15] = _WBC_DEFAULT
        self._waist_target = np.zeros(3, dtype=np.float32)

    def _set_groot_defaults(self):
        """Set the nominal pose the ONNX locomotion policy expects."""
        d = self._data
        for i in range(15):
            d.qpos[self._jqpos[i]] = _WBC_DEFAULT[i]
        # Left arm keeps DEFAULT_QPOS (unactuated, hangs); right arm zeroed for the policy to set.
        for i in range(15, 22):
            d.qpos[self._jqpos[i]] = _ROBOT_DEFAULT_QPOS[i]
        for i in range(22, _WBC_N):
            d.qpos[self._jqpos[i]] = 0.0
        d.qpos[self._fj_qa + 2] = 0.793

    def _reset_wbc_state(self):
        """Zero the locomotion policy's recurrent inputs and command state."""
        self._cmd[:] = 0
        self._wbc_action[:] = 0
        self._target_lower[:] = _WBC_DEFAULT
        self._height_cmd = _WBC_HEIGHT_CMD
        self._waist_target[:] = 0
        self._step_counter = 0
        self._obs_history.clear()
        for _ in range(_WBC_OBS_HISTORY):
            self._obs_history.append(np.zeros(_WBC_OBS_DIM, dtype=np.float32))

    def _write_default_pose(self):
        """Write the default pose into qpos and all actuators."""
        d = self._data
        for i in range(15):
            d.qpos[self._jqpos[i]] = _WBC_DEFAULT[i]
        for i in range(15, 22):
            d.qpos[self._jqpos[i]] = _ROBOT_DEFAULT_QPOS[i]
        for i in range(22, _WBC_N):
            d.qpos[self._jqpos[i]] = 0.0
        d.qpos[self._fj_qa + 2] = 0.793
        d.ctrl[:] = 0.0
        mujoco.mj_forward(self._model, d)

        self._target_q[:15] = _WBC_DEFAULT
        self._target_q[15:22] = _ROBOT_DEFAULT_QPOS[15:22]
        self._target_q[22:] = 0.0
        q = d.qpos[self._jqpos].astype(np.float32)
        dq = d.qvel[self._jdof].astype(np.float32)
        tau = (self._target_q[:15] - q[:15]) * _WBC_KP[:15] - dq[:15] * _WBC_KD[:15]
        for i in range(12, 15):
            tau[i] += d.qfrc_bias[self._jdof[i]]
        for i in range(15):
            if self._act_ids[i] >= 0:
                d.ctrl[self._act_ids[i]] = tau[i]
        for i in range(15, _WBC_N):
            if self._act_ids[i] >= 0:
                d.ctrl[self._act_ids[i]] = self._target_q[i]

    def execute_action(self, action):
        """Apply one 15-dim action: run the ONNX gait for the legs, PD torques for waist/arm, gripper ctrl."""
        d = self._data
        self._cmd[:] = action[0:3]
        self._height_cmd = float(action[3])
        waist = action[4:7]
        r_arm = action[7:14]
        r_grip = action[14]

        self._waist_target[:] = waist

        self._target_q[:15] = self._target_lower
        self._target_q[12:15] = waist
        self._target_q[15:22] = 0.0  # left arm: zero-gain, target unused.
        self._target_q[22:29] = r_arm

        q = d.qpos[self._jqpos].astype(np.float32)
        dq = d.qvel[self._jdof].astype(np.float32)
        tau = (self._target_q[:15] - q[:15]) * _WBC_KP[:15] - dq[:15] * _WBC_KD[:15]
        for i in range(12, 15):
            tau[i] += d.qfrc_bias[self._jdof[i]]
        for i in range(15):
            aid = self._act_ids[i]
            if aid >= 0:
                d.ctrl[aid] = tau[i]
        for i in range(15, _WBC_N):
            aid = self._act_ids[i]
            if aid >= 0:
                d.ctrl[aid] = self._target_q[i]

        if self._rgripper >= 0: d.ctrl[self._rgripper] = r_grip

        if self._step_counter % _WBC_CONTROL_DEC == 0:
            qj = d.qpos[self._jqpos[:29]].astype(np.float32)
            dqj = d.qvel[self._jdof[:29]].astype(np.float32)
            quat = d.qpos[self._fj_qa+3:self._fj_qa+7].astype(np.float32)
            omega = d.qvel[self._fj_da+3:self._fj_da+6].astype(np.float32)
            padded = np.zeros(29, dtype=np.float32)
            padded[:15] = _WBC_DEFAULT
            single_obs = np.zeros(_WBC_OBS_DIM, dtype=np.float32)
            single_obs[0:3] = self._cmd * _WBC_CMD_SCALE
            single_obs[3] = self._height_cmd
            single_obs[4:7] = [self._waist_target[1], self._waist_target[2], self._waist_target[0]]
            single_obs[7:10] = omega * _WBC_ANG_VEL_SCALE
            single_obs[10:13] = _gravity_orientation(quat)
            single_obs[13:42] = (qj - padded) * _WBC_DOF_POS_SCALE
            single_obs[42:71] = dqj * _WBC_DOF_VEL_SCALE
            single_obs[71:86] = self._wbc_action
            self._obs_history.append(single_obs)
            while len(self._obs_history) < _WBC_OBS_HISTORY:
                self._obs_history.appendleft(np.zeros_like(single_obs))
            for i, h in enumerate(self._obs_history):
                self._obs_buffer[i*_WBC_OBS_DIM:(i+1)*_WBC_OBS_DIM] = h
            obs_tensor = self._obs_buffer[None, :].astype(np.float32)
            if np.linalg.norm(self._cmd) < 0.05:
                self._wbc_action = self._stand_sess.run(None, {self._stand_sess.get_inputs()[0].name: obs_tensor})[0][0].astype(np.float32)
            else:
                self._wbc_action = self._walk_sess.run(None, {self._walk_sess.get_inputs()[0].name: obs_tensor})[0][0].astype(np.float32)
            self._target_lower = self._wbc_action * _WBC_ACTION_SCALE + _WBC_DEFAULT
