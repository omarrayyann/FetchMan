"""Bowl collection starting at the grasp, with no navigation.

Identical to `bowl_mixed.py` apart from the skill-profile weights, so it is
derived from that config rather than duplicating it.
"""
from molmospaces.configs.bowl_mixed import get_config as _bowl_mixed_config


def get_config():
    """bowl_mixed with the nav profile disabled."""
    config = _bowl_mixed_config()
    config.skill_profiles = [(name, 1.0 if name == "grasp" else 0.0, profile)
                             for name, weight, profile in config.skill_profiles]
    return config
