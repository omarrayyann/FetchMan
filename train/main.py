import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import wandb
from absl import app, flags
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from ml_collections import config_flags

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lightning.pytorch.callbacks.progress import TQDMProgressBar


class _StepFirstProgressBar(TQDMProgressBar):
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Show global step (not epoch progress) in the tqdm description."""
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)
        self.train_progress_bar.set_description(
            f"step {trainer.global_step} (epoch {trainer.current_epoch})"
        )

TRAIN_DIR = Path(__file__).resolve().parent
ROOT_DIR = TRAIN_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from algorithms import agents, EMAWeightAveraging
from checkpointing import _compile_agent, _resume_from_checkpoint
from data import (PRIVILEGED_OBS, _WindowedDataset, _action_stats, _build_video_cache,
                  _dl_worker_init, _filter_episodes, _load_state_columns, _resolve_stats,
                  _video_cum_counts_from_meta, split_val_episodes)
from eval_callback import EnvEvalCallback
from utils.log_utils import get_exp_name, setup_wandb, get_flag_dict


FLAGS = flags.FLAGS

flags.DEFINE_list('devices', ['1', '2', '3', '4', '5', '6', '7'], 'GPU device IDs, e.g. 0,1,2,3,4,5,6,7 for 8-GPU DDP. Default leaves GPU 0 free for eval render.')
flags.DEFINE_boolean('test', False, 'Test run without logging or checkpointing.')
flags.DEFINE_string('run_group', 'Debug', 'Run group.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_string('repo_id', 'allenai/fetchman-data', 'LeRobot dataset repo_id (Hugging Face Hub id, or a local name when --data_root is set).')
flags.DEFINE_string('data_root', '', 'Local LeRobot dataset root. Empty = pull --repo_id from the Hugging Face Hub (cached under ~/.cache/huggingface/lerobot).')
flags.DEFINE_string('revision', 'main', "Hub revision for --repo_id. lerobot defaults to a vX.Y tag; allenai/fetchman-data has none, so 'main' is used. Ignored when --data_root is set.")
flags.DEFINE_string('save_dir', 'exp/', 'Save directory.')
flags.DEFINE_integer('image_size', 224, 'Image height (H) for cached training frames and eval render cameras.')
flags.DEFINE_integer('image_size_w', 384, 'Image width (W). Defaults to 384 for ~16:9 aspect. Set equal to image_size for square.')

flags.DEFINE_integer('max_episodes', -1, 'Cap training data to the first N episodes (-1 = use all). Stats are still computed from the full dataset for scaling-law consistency.')
flags.DEFINE_string('bad_episodes_csv', None, 'CSV with an `episode_index` column; those episodes are dropped from training. Stats unaffected.')
flags.DEFINE_string('task_filter', '', 'Keep only episodes whose task prompt matches this regex (re.search). Empty = train on all tasks.')
flags.DEFINE_boolean('point_condition', False, 'Condition the policy on the 2D head-cam target point: un-privilege observation.target_point and feed it as [x, y, visible] (coords zeroed when the env reports the (-1,-1) not-visible sentinel).')
flags.DEFINE_integer('epochs', 10000, 'Maximum number of training epochs.')
flags.DEFINE_integer('val_episodes', 0, 'Hold out this many whole episodes as a validation split and log val/loss while training. 0 disables validation.')
flags.DEFINE_integer('val_split_seed', 0, 'Seed that picks which episodes land in the validation split. Separate from --seed so changing the training seed does not change the split.')
flags.DEFINE_integer('val_interval', 1, 'Run validation every N epochs (needs --val_episodes > 0).')
flags.DEFINE_integer('checkpoint_interval', 2000, 'Checkpoint saving interval in gradient steps.')
flags.DEFINE_float('gradient_clip_val', 1.0, 'Clip gradients to this max L2 norm. Caps outlier-batch gradients so a single bad batch cannot blow up the weights. Set <=0 to disable.')
flags.DEFINE_integer('num_workers', 4, 'Number of DataLoader workers per rank.')
flags.DEFINE_integer('prefetch_factor', 1, 'DataLoader prefetch_factor (samples preloaded per worker).')
flags.DEFINE_boolean('cudnn_benchmark', True, 'Enable cuDNN autotuning for fixed-size image training.')
flags.DEFINE_boolean('no_video_cache', True,
                     'Skip the .train_cache precompute and decode mp4 frames on-the-fly via lerobot/torchcodec. '
                     'Saves disk (~10 data size) at the cost of ~ms-per-frame decode latency (overlapped with GPU by dataloader workers).')
flags.DEFINE_integer('video_decode_debug', 0,
                     'When >0 and --no_video_cache is set, every N __getitem__ calls each worker '
                     'prints a timing summary (cold/warm decode ms, frames/call). 0 = silent.')
flags.DEFINE_boolean('use_ema', True, 'Use EMA/SWA weight averaging callback.')
flags.DEFINE_integer('ema_interval', 1, 'Update EMA every N optimizer steps.')
flags.DEFINE_integer('eval_interval', 2000, 'Run environment evaluation every N gradient steps. Disabled when 0.')
flags.DEFINE_integer('eval_episodes', 75, 'Number of episodes per environment evaluation run. Ignored when --eval_episodes_json points to an existing file (the whole JSON set is run).')
flags.DEFINE_integer('eval_num_envs', 10, 'Parallel env workers used during evaluation. 1 = serial.')
flags.DEFINE_list('eval_render_gpu', ['0'], 'Comma-separated nvidia-smi indices to round-robin eval workers across (via EGL). e.g. --eval_render_gpu=0,1,2,3 spreads 8 workers across 4 render GPUs (2/GPU). Single value (e.g. --eval_render_gpu=3) puts all workers on one render GPU. -1 = auto-pick the trainer GPU.')
flags.DEFINE_string('eval_episodes_json', '', 'JSONL of exported reset states (the fixed benchmark). When the file exists, training eval replays exactly these episodes instead of random resets. Empty string disables (random-reset eval).')
flags.DEFINE_integer('eval_seed', 777000, 'Seed for the flow sampler during environment evaluation. Fixed so each eval scores the policy rather than the policy plus a fresh noise draw; it is forked, so the training RNG stream is unaffected.')
flags.DEFINE_float('eval_time_limit', 50.0, 'Per-episode environment evaluation time limit in seconds.')
flags.DEFINE_boolean('eval_grasp_only', False, 'Evaluate in the grasp-only regime by teleporting the base to the grasp goal pose.')
flags.DEFINE_boolean('eval_at_resume_start', False, 'When resuming from a checkpoint, run one eval pass before training begins. Disable to skip the initial eval and jump straight into training.')
flags.DEFINE_string('resume_ckpt', None, 'Path to a checkpoint. Full checkpoints resume training state; weights-only checkpoints initialize model weights.')
flags.DEFINE_boolean('compile', False, 'torch.compile the encoder + flow_model. Adds ~1-2 min compile time at first step; ~20-40% speedup after.')
flags.DEFINE_string('compile_mode', 'default', 'torch.compile mode: default | reduce-overhead | max-autotune.')

config_flags.DEFINE_config_file('agent', str(TRAIN_DIR / 'algorithms/flow_policy.py'), lock_config=False)
config_flags.DEFINE_config_file('env', str(ROOT_DIR / 'molmospaces/configs/example.py'), lock_config=False)


def _write_run_artifacts(agent, flag_dict, dataset_stats, example_data, fps):
    """Dump flags.json and stats.json next to the checkpoints."""
    with open(os.path.join(FLAGS.save_dir, 'flags.json'), 'w') as f:
        flag_dict['agent'] = agent.config.to_dict()
        json.dump(flag_dict, f)
    with open(os.path.join(FLAGS.save_dir, 'stats.json'), 'w') as f:
        json.dump({
            'stats': {k: {sk: sv.tolist() for sk, sv in v.items()} for k, v in dataset_stats.items()},
            'example_shapes': {k: list(v.shape) for k, v in example_data.items() if hasattr(v, 'shape')},
            'fps': int(fps),
        }, f)


def _build_callbacks(fps):
    """Checkpointing, progress bar, and the optional EMA and environment-eval callbacks."""
    checkpoint_callback = ModelCheckpoint(
        dirpath=FLAGS.save_dir,
        save_top_k=-1,
        every_n_train_steps=FLAGS.checkpoint_interval,
        save_last=True,
        save_weights_only=False,
        filename='{step:08d}',
    )

    callbacks = [checkpoint_callback, _StepFirstProgressBar()]
    if FLAGS.use_ema:
        callbacks.append(EMAWeightAveraging(
            update_after_step=0, update_every=FLAGS.ema_interval,
            inv_gamma=1.0, power=2/3,
            min_value=0.0, max_value=0.9999, use_buffers=False,
        ))
    if FLAGS.eval_interval > 0:
        eval_env_cfg = FLAGS.env.copy_and_resolve_references()
        if FLAGS.eval_grasp_only:
            eval_env_cfg["spawn_at_grasp"] = True
        callbacks.append(
            EnvEvalCallback(
                env_config=eval_env_cfg,
                every_n_steps=FLAGS.eval_interval,
                episodes=FLAGS.eval_episodes,
                time_limit=FLAGS.eval_time_limit,
                num_envs=FLAGS.eval_num_envs,
                render_gpu=[int(g) for g in FLAGS.eval_render_gpu],
                dataset_fps=int(fps),
                eval_at_resume_start=FLAGS.eval_at_resume_start,
                episodes_json=FLAGS.eval_episodes_json,
                eval_seed=FLAGS.eval_seed,
            )
        )
    return callbacks


def main(_):
    """BC training entry point: load/pack the LeRobot data, build the policy, and run the Lightning loop."""
    torch.backends.cudnn.benchmark = FLAGS.cudnn_benchmark
    torch.set_float32_matmul_precision('high')

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    is_rank_zero = local_rank == 0

    exp_name = get_exp_name(FLAGS.seed)
    FLAGS.save_dir = os.path.join(FLAGS.save_dir, 'g1', FLAGS.run_group, exp_name)
    if not FLAGS.test and is_rank_zero:
        setup_wandb(project='g1', group=FLAGS.run_group, name=exp_name)
        logger = WandbLogger(project='g1', group=FLAGS.run_group,
                             name=exp_name, save_dir=wandb.run.dir)
        os.makedirs(FLAGS.save_dir, exist_ok=True)
    else:
        logger = None

    flag_dict = get_flag_dict()
    config = FLAGS.agent
    image_size = int(FLAGS.image_size)
    image_size_w = int(FLAGS.image_size_w)
    if image_size <= 0 or image_size_w <= 0:
        raise ValueError(f"--image_size and --image_size_w must be positive, got H={image_size}, W={image_size_w}")

    if FLAGS.eval_interval > 0:
        FLAGS.env['camera_size'] = (image_size, image_size_w)
        if not FLAGS.env.get('scene'):
            FLAGS.env['scene'] = 'sample_scenes/procthor-10k-val/val_1.xml'
        if FLAGS.env.get('objects', '.*') == '.*':
            FLAGS.env['objects'] = 'plu'
        if not FLAGS.env.get('randomize_object', False):
            FLAGS.env['randomize_object'] = True

    # Empty --data_root means "resolve --repo_id on the Hugging Face Hub";
    # lerobot caches it under ~/.cache/huggingface/lerobot.
    data_root = FLAGS.data_root or None
    # Pulling from the Hub downloads whatever episodes are requested, so cap the
    # fetch with --max_episodes rather than downloading all 548 GB to train on a few.
    hub_episodes = (list(range(FLAGS.max_episodes))
                    if (data_root is None and FLAGS.max_episodes > 0) else None)
    probe = LeRobotDataset(FLAGS.repo_id, root=data_root, episodes=hub_episodes,
                           revision=None if data_root else FLAGS.revision)
    fps = probe.fps
    priv = set(PRIVILEGED_OBS)
    if FLAGS.point_condition:
        priv.discard('observation.target_point')  # un-privilege -> fed as [x,y,visible]
    obs_keys = sorted(
        k for k in probe.meta.features
        if k.startswith('observation.') and k not in priv
    )
    video_keys = [k for k in obs_keys if probe.meta.features[k]['dtype'] == 'video']
    state_keys = [k for k in obs_keys if k not in video_keys]
    if is_rank_zero:
        dropped = sorted(priv & set(probe.meta.features))
        if dropped:
            print(f"  dropping privileged obs (not used for training): {dropped}")
        if FLAGS.point_condition:
            print("  point_condition ON: observation.target_point fed as [x, y, visible]")

    T_obs = config['observation_horizon']
    T_pred = config['prediction_horizon']
    dataset_stats = _resolve_stats(probe, obs_keys + ['action'])

    t0 = time.time()
    cols, obj_emb_unique, parquet_paths = _load_state_columns(
        probe, state_keys, config, data_root, dataset_stats, is_rank_zero)

    video_cache = _build_video_cache(
        probe, cols, video_keys, image_size, image_size_w, is_rank_zero)

    cols = _filter_episodes(cols, probe, parquet_paths, video_cache, is_rank_zero)

    cols, video_cache, val_cols, val_video_cache = split_val_episodes(
        cols, video_cache, FLAGS.val_episodes, is_rank_zero,
        split_seed=FLAGS.val_split_seed)

    # When --no_video_cache, hand the dataset the mp4 paths so it can decode
    # on-the-fly per worker via torchcodec.
    video_mp4_paths: dict[str, list[str]] = {}
    video_cum_counts: dict[str, np.ndarray] = {}
    if FLAGS.no_video_cache:
        for vk in video_keys:
            mp4s = sorted((Path(probe.root) / "videos" / vk).rglob("file-*.mp4"))
            if not mp4s:
                raise RuntimeError(f"No mp4s found for {vk}; cannot decode on-the-fly")
            video_mp4_paths[vk] = [str(p) for p in mp4s]
        if is_rank_zero:
            print(f"  building per-file frame counts from metadata "
                  f"(avoids per-worker mp4 scan)...")
        _t_cc = time.time()
        video_cum_counts = _video_cum_counts_from_meta(probe.root, video_keys, video_mp4_paths)
        if is_rank_zero:
            resolved = {vk: int(video_cum_counts[vk][-1]) for vk in video_cum_counts}
            print(f"    resolved frame counts for {list(video_cum_counts)} in "
                  f"{time.time()-_t_cc:.1f}s (totals={resolved})")
    train_dataset = _WindowedDataset(probe, cols, state_keys, video_keys, T_obs, T_pred,
                                     video_cache=video_cache,
                                     video_mp4_paths=video_mp4_paths,
                                     image_size_hw=(image_size, image_size_w),
                                     debug_every=FLAGS.video_decode_debug,
                                     use_action_delta=bool(config['use_action_delta']),
                                     obj_emb_unique=obj_emb_unique,
                                     video_cum_counts=video_cum_counts)
    example_data = train_dataset[0]
    if is_rank_zero:
        print(f"  total preload: {time.time()-t0:.1f}s, {len(train_dataset)} samples")

    val_dataset = None
    if val_cols is not None:
        val_dataset = _WindowedDataset(probe, val_cols, state_keys, video_keys, T_obs, T_pred,
                                       video_cache=val_video_cache,
                                       video_mp4_paths=video_mp4_paths,
                                       image_size_hw=(image_size, image_size_w),
                                       debug_every=0,
                                       use_action_delta=bool(config['use_action_delta']),
                                       obj_emb_unique=obj_emb_unique,
                                       video_cum_counts=video_cum_counts)

    dataset_stats = _action_stats(probe, cols, T_pred, dataset_stats, train_dataset, is_rank_zero)

    loader_kwargs = dict(
        batch_size=config['batch_size'],
        num_workers=FLAGS.num_workers,
        shuffle=True,
        pin_memory=True,
        persistent_workers=FLAGS.num_workers > 0,
    )
    if FLAGS.num_workers > 0:
        loader_kwargs['prefetch_factor'] = FLAGS.prefetch_factor
        loader_kwargs['worker_init_fn'] = _dl_worker_init
    train_loader = torch.utils.data.DataLoader(train_dataset, **loader_kwargs)
    val_loader = None
    if val_dataset is not None:
        # Shuffling is off so batch_idx maps to the same windows every pass, which
        # is what makes the seeded val loss in FlowPolicy.validation_step comparable
        # across epochs.
        val_loader = torch.utils.data.DataLoader(
            val_dataset, **{**loader_kwargs, 'shuffle': False})
    config['num_training_steps'] = FLAGS.epochs * len(train_loader)

    agent = agents[config['agent_name']](config, example_data=example_data, dataset_stats=dataset_stats)

    _compile_agent(agent, is_rank_zero)

    fit_ckpt_path = _resume_from_checkpoint(agent, is_rank_zero)

    if not FLAGS.test and is_rank_zero:
        _write_run_artifacts(agent, flag_dict, dataset_stats, example_data, fps)

    callbacks = _build_callbacks(fps)

    devices = [int(d) for d in FLAGS.devices]
    if len(devices) > 1:
        from datetime import timedelta
        from lightning.pytorch.strategies import DDPStrategy
        # NCCL timeout must outlast the eval window so non-rank-0 workers don't trip
        # while rank 0 is running the env eval. find_unused_parameters covers the
        # DINOv3 encoders, whose frozen submodules don't participate in the loss.
        needs_unused = 'dinov3' in str(config.get('encoder', ''))
        strategy = DDPStrategy(timeout=timedelta(hours=2), find_unused_parameters=needs_unused)
    else:
        strategy = 'auto'
    trainer = L.Trainer(
        max_epochs=FLAGS.epochs,
        accelerator='auto',
        devices=devices,
        strategy=strategy,
        callbacks=callbacks,
        logger=logger,
        check_val_every_n_epoch=FLAGS.val_interval,
        precision="bf16-mixed",
        gradient_clip_val=(FLAGS.gradient_clip_val if FLAGS.gradient_clip_val > 0 else None),
        gradient_clip_algorithm="norm",
    )

    trainer.fit(agent, train_loader, val_dataloaders=val_loader, ckpt_path=fit_ckpt_path)


if __name__ == '__main__':
    app.run(main)
