"""Construction of everything the training loop needs before the first update.

Pure setup: each function takes what it needs and returns plain values, so the
loop in main.py reads as a loop instead of a thousand lines of wiring.
"""
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch
from absl import flags

ROOT_DIR = Path(__file__).resolve().parents[1]
TRAIN_DIR = ROOT_DIR / "train"
for _p in (ROOT_DIR, TRAIN_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from algorithms import agents
from algorithms.flow_policy import get_config as get_agent_config
from molmospaces.agents.policy import G1Controller
from molmospaces.env import make_env

FLAGS = flags.FLAGS


def control_timing(env_cfg, policy, fps, seed):
    """(action_repeat, step_dt, image_keys) read off a throwaway probe env.

    The probe needs G1Controller applied: set_env() overrides the timestep to
    0.005 and n_substeps to 1, so without it the probe reads 0.02s, the workers
    run 0.005s, and action_repeat comes out 4x wrong.
    """
    env = make_env({**env_cfg, "seed": seed + 999})
    controller = G1Controller()
    controller.setup(env.scene.model, env.scene.data)
    controller.set_env(env)
    step_dt = env.robot.n_substeps * env.scene.model.opt.timestep
    action_repeat = max(1, int(round((1.0 / step_dt) / fps)))
    image_keys = [k for k in env.cameras if f"observation.{k}" in policy.obs_keys]
    if hasattr(env, "close"):
        env.close()
    return action_repeat, step_dt, image_keys


def camera_shapes(payload):
    """Shapes of the checkpoint's image observations, used to size env cameras."""
    return [shape for k, shape in payload["example_shapes"].items() if k.endswith("_image")]


def load_checkpoint(path, device, num_sampling_steps=10, use_ema=True, base_config=None):
    """Rebuild a FlowPolicy from a BC or RL checkpoint.

    BC checkpoints are Lightning's, holding EMA weights under `state_dict` and
    the raw training weights under `current_model_state`; RL checkpoints are a
    plain {"state_dict", "update", "sr"}. Constructing the policy and loading the
    weights directly handles both, so an RL checkpoint needs no Lightning
    version patched in before it can be evaluated or resumed.

    The architecture comes from the checkpoint directory's flags.json, so
    `base_config` only supplies the runtime-only keys (default: the agent
    config's own defaults). Returns (policy, config, payload, fps).
    """
    path = Path(path).resolve()
    payload = json.load(open(path.parent / "stats.json"))
    dataset_stats = {k: {sk: np.asarray(sv, dtype=np.float32) for sk, sv in v.items()}
                     for k, v in payload["stats"].items()}
    example_data = {k: torch.zeros(shape, dtype=torch.float32)
                    for k, shape in payload["example_shapes"].items()}

    config = base_config if base_config is not None else get_agent_config()
    runtime_only = {"num_sampling_steps", "stabilization_level", "clip_noise",
                    "uncertainty_scale", "num_training_steps"}
    flags_json = path.parent / "flags.json"
    if flags_json.exists():
        for k, v in json.load(open(flags_json)).get("agent", {}).items():
            if k not in runtime_only and k in config:
                config[k] = v
    config["num_training_steps"] = 1
    config["num_sampling_steps"] = int(num_sampling_steps)

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    weights = ckpt["state_dict"]
    if not use_ema:
        if "current_model_state" not in ckpt:
            raise ValueError(f"{path.name} has no raw weights (current_model_state); "
                             "only BC checkpoints carry both EMA and raw")
        weights = ckpt["current_model_state"]
    policy = agents[config["agent_name"]](config, example_data=example_data,
                                          dataset_stats=dataset_stats)
    policy.load_state_dict(weights, strict=True)
    return policy.to(device), config, payload, payload.get("fps", 10)


def load_policy(device):
    """The --checkpoint policy for training, plus its frozen KL reference.

    Returns (policy, ref_policy, config, payload, fps, ckpt_dir); ref_policy is
    None unless --kl_beta > 0, since nothing else reads it.
    """
    policy, config, payload, fps = load_checkpoint(
        FLAGS.checkpoint, device, num_sampling_steps=FLAGS.num_sampling_steps,
        use_ema=FLAGS.use_ema_weights, base_config=FLAGS.agent)
    policy.train()
    ref_policy = None
    if FLAGS.kl_beta > 0:
        ref_policy = copy.deepcopy(policy).eval()
        for p in ref_policy.parameters():
            p.requires_grad_(False)
    return policy, ref_policy, config, payload, fps, Path(FLAGS.checkpoint).resolve().parent


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
                print(f"[rl] {len(resolved_scenes)} scenes pool, workers rotate per reset "
                      f"(cfg.randomize_scene=True, freq={env_cfg.get('randomize_scene_freq', 1)})")
            else:
                gs_for_scene = FLAGS.parallel_envs // FLAGS.num_groups  # envs per group
                per_worker_overrides = [
                    {"scene": str(resolved_scenes[(i // gs_for_scene) % len(resolved_scenes)]),
                     "randomize_scene": False}
                    for i in range(FLAGS.parallel_envs)
                ]
                scene_assign_msg = "assigned round-robin per GRPO group"
                print(f"[rl] {len(resolved_scenes)} scenes assigned round-robin to "
                      f"{FLAGS.parallel_envs} workers ({scene_assign_msg}, persistent)")
                env_cfg = dict(env_cfg)
                env_cfg["scene"] = str(resolved_scenes[0])
                env_cfg["randomize_scene"] = False
    return env_cfg, per_worker_overrides
