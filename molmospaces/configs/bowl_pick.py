"""Pick up a bowl in one fixed house, with no navigation.

This is the config the README quickstart collects with. It is `example.py`
with the nav skill profile switched off, so every episode starts with the
robot already standing in front of the bowl and the whole demonstration is
reach, grasp and lift. Pair it with `example.py` to add locomotion, or with
`bowl_mixed.py` to randomize across every house that contains a bowl.
"""
from molmospaces.configs.example import get_config as _example_config


def get_config():
    """example.py with the nav profile disabled: picking only."""
    config = _example_config()
    config.skill_profiles = [(name, 1.0 if name == "grasp" else 0.0, profile)
                             for name, _weight, profile in config.skill_profiles]
    return config
