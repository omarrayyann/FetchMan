"""Delta-action transforms (Option B mask): dims 3..13 are delta, others absolute.

Layout of the 15-dim action (see molmospaces/agents/policy.py and env.py):
    [0:3]  base velocity command (vx, vy, omega)        — absolute (passthrough)
    [3]    base height target                            — DELTA
    [4:7]  waist joints (yaw, roll, pitch)               — DELTA
    [7:14] right arm joints (7)                          — DELTA
    [14]   gripper                                       — absolute (passthrough)

The "anchor state" for the 11 delta dims is concat(base_height, waist3, right_arm7) and
must be aligned with the chunk start (idx) — every action target in the chunk gets the
same anchor subtracted (training) / added (inference). This is the openpi convention.
"""
import numpy as np


DELTA_DIMS = slice(3, 14)            # dims 3..13 inclusive (11 dims)
DELTA_DIM_COUNT = 11


def anchor_from_cols(cols, idx):
    """Anchor (11,) float32 from the in-RAM 'cols' dict used by _WindowedDataset."""
    return np.concatenate([
        np.asarray(cols['observation.base_height'][idx], dtype=np.float32).reshape(-1),
        np.asarray(cols['observation.upper_joint_pos'][idx][0:3], dtype=np.float32).reshape(-1),
        np.asarray(cols['observation.upper_joint_pos'][idx][3:10], dtype=np.float32).reshape(-1),
    ])


def anchor_from_obs(obs):
    """Anchor (11,) float32 from a live env observation dict (eval/inference)."""
    return np.concatenate([
        np.asarray(obs['base_height'], dtype=np.float32).reshape(-1),
        np.asarray(obs['upper_joint_pos'][0:3], dtype=np.float32).reshape(-1),
        np.asarray(obs['upper_joint_pos'][3:10], dtype=np.float32).reshape(-1),
    ])


def target_point_xyv(p):
    """2D head-cam target point -> [x, y, visible] (point conditioning)."""
    p = np.asarray(p, dtype=np.float32)
    vis = ((p[..., 0] >= 0.0) & (p[..., 1] >= 0.0)).astype(np.float32)
    xy = p * vis[..., None]
    return np.concatenate([xy, vis[..., None]], axis=-1)
