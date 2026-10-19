"""Env for scoring checkpoints on the fixed benchmarks in assets/eval_sets/.

Each benchmark episode restores its own scene, poses and appearance, so all this
has to set is what a restore leaves to the config: the termination rules. Robot
contact with the world is logged rather than ending the episode, which is how
the runs behind the paper's numbers were scored. The env itself defaults both
flags to True, and that alone takes a 75% policy down to 19%.
"""
from molmospaces.env import get_config as _env_config


def get_config():
    """The default env with collision termination off."""
    config = _env_config()
    config.terminate_before_grasp_collision = False
    config.terminate_on_grasp_collision = False
    return config
