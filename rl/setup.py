"""Construction of everything the training loop needs before the first update.

Pure setup: each function takes what it needs and returns plain values, so the
loop in main.py reads as a loop instead of a thousand lines of wiring.
"""
import copy
import json
from pathlib import Path

import numpy as np
import torch
from absl import flags

import sys
ROOT_DIR = Path(__file__).resolve().parents[1]
TRAIN_DIR = ROOT_DIR / "train"
for _p in (ROOT_DIR, TRAIN_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from algorithms import agents

FLAGS = flags.FLAGS


def camera_shapes(payload):
    """Shapes of the checkpoint's image observations, used to size env cameras."""
    return [shape for k, shape in payload["example_shapes"].items() if k.endswith("_image")]


def load_policy(device):
    """Rebuild the BC policy from --checkpoint, plus its frozen KL reference.

    Returns (policy, ref_policy, config, payload, fps, ckpt_dir). ref_policy is
    None unless --kl_beta > 0, since nothing else reads it.
    """
    ckpt_dir = Path(FLAGS.checkpoint).resolve().parent
    payload = json.load(open(ckpt_dir / "stats.json"))
    dataset_stats = {k: {sk: np.asarray(sv, dtype=np.float32) for sk, sv in v.items()}
                     for k, v in payload["stats"].items()}
    example_data = {k: torch.zeros(shape, dtype=torch.float32)
                    for k, shape in payload["example_shapes"].items()}
    fps = payload.get("fps", 10)

    config = FLAGS.agent
    RUNTIME_DENYLIST = {"num_sampling_steps", "stabilization_level", "clip_noise",
                        "uncertainty_scale", "num_training_steps"}
    if (ckpt_dir / "flags.json").exists():
        saved = json.load(open(ckpt_dir / "flags.json"))
        for k, v in saved.get("agent", {}).items():
            if k in RUNTIME_DENYLIST: continue
            if k in config and config[k] != v: config[k] = v
    config["num_training_steps"] = 1
    # Finetune default: 10 flow steps (the agent config defaults to 20 for BC/eval_ckpt).
    # num_sampling_steps is in RUNTIME_DENYLIST so flags.json won't restore the BC value.
    config["num_sampling_steps"] = int(FLAGS.num_sampling_steps)

    model_cls = agents[config["agent_name"]]
    policy = model_cls.load_from_checkpoint(
        FLAGS.checkpoint, config=config, example_data=example_data,
        dataset_stats=dataset_stats, map_location=device,
    ).to(device).train()
    # load_from_checkpoint reads state_dict, the EMA weights. Raw training
    # weights live under current_model_state and are loaded over the top.
    if not FLAGS.use_ema_weights:
        _raw = torch.load(FLAGS.checkpoint, map_location="cpu", weights_only=False)
        if isinstance(_raw, dict) and "current_model_state" in _raw:
            missing, unexpected = policy.load_state_dict(_raw["current_model_state"], strict=False)
            print(f"[ppo] loaded RAW weights (current_model_state); EMA state_dict ignored "
                  f"(missing={len(missing)}, unexpected={len(unexpected)})")
        else:
            print("[ppo] --use_ema_weights=False but checkpoint has no current_model_state; "
                  "keeping the loaded state_dict")
        del _raw
    # Reference (frozen) — for KL only.
    ref_policy = None
    if FLAGS.kl_beta > 0:
        ref_policy = copy.deepcopy(policy).eval()
        for p in ref_policy.parameters(): p.requires_grad_(False)
    return policy, ref_policy, config, payload, fps, ckpt_dir


def build_replicas(policy, policy_devices):
    """Frozen policy copies for multi-GPU rollout, plus optional torch.compile.

    The primary on cuda:0 holds the trainable weights; the replicas are refreshed
    from it before each rollout and never take a gradient.
    """
    policy_replicas = [policy]
    if len(policy_devices) > 1:
        for d in policy_devices[1:]:
            rep = copy.deepcopy(policy).to(d).eval()
            for p in rep.parameters():
                p.requires_grad_(False)
            policy_replicas.append(rep)
        print(f"[grpo] policy replicas on devices: {[str(p.device) for p in policy_replicas]}")

    return policy_replicas


def build_optimizers(policy):
    """AdamW for the actor, plus the actor's trainable params.

    The param list comes back because gradient clipping in the update loop needs
    exactly the set the optimizer is stepping.
    """
    betas = (float(FLAGS.betas[0]), float(FLAGS.betas[1]))
    actor_params = [p for p in policy.parameters() if p.requires_grad]
    actor_optim = torch.optim.AdamW(actor_params, lr=FLAGS.actor_lr, betas=betas,
                                    weight_decay=FLAGS.weight_decay)
    return actor_optim, actor_params


def build_env_config(payload):
    """Env config from --env, plus per-worker scene overrides for a .txt scene list.

    Returns (env_cfg, per_worker_overrides); the overrides are None unless the
    scene field names a list file that pins one scene per worker.
    """
    # --- Env cfg comes from --env config file (default: molmospaces/configs/bowl_eval.py) ---
    image_shapes = camera_shapes(payload)
    try:
        env_cfg = FLAGS.env.copy_and_resolve_references().to_dict()
    except AttributeError:
        env_cfg = dict(FLAGS.env)
    env_cfg["seed"] = FLAGS.seed
    env_cfg["launch_viewer"] = False
    if image_shapes:
        env_cfg["camera_size"] = (int(image_shapes[0][-2]), int(image_shapes[0][-1]))
    # Group-mate workers must load identical Scene contents so within-group
    # rollouts differ only in SDE noise. (BC collection leaves this False, to
    # keep cross-worker visual diversity.)
    env_cfg["deterministic_scene_textures"] = True

    # randomize_scene=False pins one scene per worker round-robin (one MJCF
    # compile at startup, diversity capped at n_groups); True lets each worker
    # rotate per reset, covering every scene at the cost of a compile per swap.
    per_worker_overrides = None
    scene_field = env_cfg.get("scene", "")
    if isinstance(scene_field, str) and scene_field.endswith(".txt"):
        from molmospaces.env import _resolve_scene_paths as _resolve
        resolved_scenes = _resolve(scene_field)
        if resolved_scenes:
            cfg_randomize = bool(env_cfg.get("randomize_scene", False))
            if cfg_randomize:
                # Keep the .txt path so workers get the full list: pinning
                # scene to resolved_scenes[0] collapses the pool and _try_reset
                # only swaps when len(scene_paths) > 1.
                print(f"[ppo] {len(resolved_scenes)} scenes pool — workers rotate per reset "
                      f"(cfg.randomize_scene=True, freq={env_cfg.get('randomize_scene_freq', 1)})")
            else:
                gs_for_scene = FLAGS.parallel_envs // FLAGS.num_groups  # envs per group
                per_worker_overrides = [
                    {"scene": str(resolved_scenes[(i // gs_for_scene) % len(resolved_scenes)]),
                     "randomize_scene": False}
                    for i in range(FLAGS.parallel_envs)
                ]
                scene_assign_msg = "assigned round-robin per GRPO group"
                print(f"[ppo] {len(resolved_scenes)} scenes assigned round-robin to "
                      f"{FLAGS.parallel_envs} workers ({scene_assign_msg}, persistent)")
                env_cfg = dict(env_cfg)
                env_cfg["scene"] = str(resolved_scenes[0])
                env_cfg["randomize_scene"] = False
    return env_cfg, per_worker_overrides
