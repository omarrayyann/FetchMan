import tempfile
from datetime import datetime

import ml_collections
import wandb
from absl import flags


def get_exp_name(seed):
    """Timestamped experiment name including the seed."""
    return f'sd{seed:03d}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'


def get_flag_dict():
    """All absl flags as a plain dict (for logging)."""
    flag_dict = {k: getattr(flags.FLAGS, k) for k in flags.FLAGS if '.' not in k}
    for k in flag_dict:
        if isinstance(flag_dict[k], ml_collections.ConfigDict):
            flag_dict[k] = flag_dict[k].to_dict()
    return flag_dict


def setup_wandb(entity=None, project='project', group=None, name=None,
                mode='online', resume='allow'):
    """Initialize wandb with the flag dict as config."""
    wandb_output_dir = tempfile.mkdtemp()
    tags = [group] if group is not None else None
    init_kwargs = dict(
        config=get_flag_dict(),
        project=project,
        entity=entity,
        tags=tags,
        group=group,
        dir=wandb_output_dir,
        name=name,
        settings=wandb.Settings(start_method='thread', _disable_stats=False),
        mode=mode,
        save_code=True,
        resume=resume,
    )
    return wandb.init(**init_kwargs)
