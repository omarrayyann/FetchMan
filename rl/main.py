"""Flow-GRPO fine-tuning of a behavior-cloned flow policy.

Noise injected at one flow step (sigma_t = noise_level * sqrt((1-t)/t)) makes
each action chunk's log-prob tractable. Episodes run in groups that share a task,
each episode's advantage is its success relative to its group, and the policy
takes clipped-ratio steps on the chain log-prob. There is no value function.

Workers step their own envs and the master batches the policy over whichever
are ready. Every --eval_every_n_updates the policy is scored on --benchmark,
the same measurement benchmark/main.py makes.

    python rl/main.py --checkpoint=<bc step.ckpt> --env=molmospaces/configs/bowl_pick.py \
        --num_groups=32 --group_size=8 --parallel_envs=64 --benchmark=bowl_pick
"""
import os
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import tqdm
from absl import app, flags
from ml_collections import config_flags

TRAIN_DIR = Path(__file__).resolve().parents[1] / "train"
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(TRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(TRAIN_DIR))

from flow_sde import recompute_chain_kl_to_ref_cached, recompute_chain_logprob_cached, sde_sigma_schedule
from evaluate import load_benchmark, replay, sample
from rollouts import collect_rollouts
from setup import build_env_config, build_optimizers, build_replicas, control_timing, load_policy
from vec_env import VecEnv


FLAGS = flags.FLAGS

flags.DEFINE_string("checkpoint", None, "Pretrained BC checkpoint to fine-tune from.")
flags.DEFINE_bool("use_ema_weights", True, "Load the EMA-averaged weights (checkpoint['state_dict'], matches training-time eval). Set False to fine-tune from the RAW training weights (checkpoint['current_model_state']) instead.")
flags.DEFINE_string("objects", "bowl", "Object regex.")
flags.DEFINE_integer("gpu", 0, "GPU device for the policy. Should not also appear in --render_gpus or you get policy/render contention (small).")
flags.DEFINE_list("render_gpus", ["1", "2", "3", "4", "5", "6", "7"], "Comma-separated EGL render GPUs for parallel envs. Default excludes GPU 0 (the policy GPU) to avoid policy/render contention.")
flags.DEFINE_integer("seed", 0, "Random seed.")
flags.DEFINE_float("time_limit", 40.0, "Per-episode sim seconds.")
flags.DEFINE_float("eval_time_limit", 50.0, "Per-episode sim seconds during eval; training rollouts keep --time_limit. The paper benchmarks were measured at 50. Set <=0 to reuse --time_limit.")

flags.DEFINE_integer("parallel_envs", 64, "Number of parallel envs (subprocess workers). 64 is the measured sweet spot on 7 render GPUs (128 OOMs during EGL init).")
flags.DEFINE_integer("max_updates", 1000000000, "Total updates (default: run until stopped).")
flags.DEFINE_integer("max_steps_per_episode", 15000, "Hard cap on env substeps per episode (~75s at step_dt=0.005s).")
flags.DEFINE_float("dispatch_wait_ms", 30.0, "Async dispatch: ms to wait gathering more ready workers before dispatching a policy batch.")
flags.DEFINE_list("policy_gpus", [], "Comma-separated GPU indices for policy replication. Empty = single GPU = FLAGS.gpu.")

flags.DEFINE_float("noise_level", 0.7,
                   "Base sigma of the injected flow noise, in normalized action units.")
flags.DEFINE_integer("num_sampling_steps", 10,
                     "Flow denoising steps for RL collection + eval (overrides the "
                     "agent-config default of 20). 20 pushes the ratio out of the trust "
                     "region and diverges.")

flags.DEFINE_integer("num_groups", 16, "GRPO: number of distinct task-groups collected per update (baseline diversity). parallel_envs must be divisible by num_groups -> envs_per_group = parallel_envs / num_groups.")
flags.DEFINE_integer("group_size", 8, "GRPO group size G = EPISODES per group (the per-task baseline sample size). Total episodes/update = num_groups * group_size. A group's episodes are split across envs_per_group=(parallel_envs/num_groups) envs, each running group_size/envs_per_group of them; so group_size must be divisible by envs_per_group.")


flags.DEFINE_integer("epochs_per_update", 2, "Passes over each update's rollouts.")
flags.DEFINE_integer("minibatch_size", 128, "Minibatch size.")
flags.DEFINE_float("ppo_clip", 0.2, "Ratio clip: lower bound 1 - ppo_clip.")
flags.DEFINE_float("ppo_clip_high", 0.2, "Ratio clip: upper bound 1 + ppo_clip_high (DAPO clip-higher when larger).")
flags.DEFINE_bool("length_normalize", True, "Weight each transition's actor loss by mean_ep_len/ep_len so episodes contribute equally regardless of length. Down-weights long failed episodes.")
flags.DEFINE_float("grpo_filter_sr_low", 0.1, "GRPO group filter (RLinf filter_rewards): drop a group from the gradient if its success rate < this. 0 keeps all. 0.1 (default) drops all-fail groups.")
flags.DEFINE_float("grpo_filter_sr_high", 0.9, "GRPO group filter: drop a group if its success rate > this. 1 keeps all. 0.9 (default) drops all-success groups. Together with _low, keeps only mixed groups (the only ones with non-zero gradient).")
flags.DEFINE_float("ppo_dual_clip", 3.0, "Dual-clip coefficient: bounds the loss of negative-advantage samples with huge ratios.")
flags.DEFINE_float("kl_beta", 0.0, "KL-to-BC coefficient (0 = off).")
flags.DEFINE_float("actor_lr", 5e-6, "Actor learning rate.")
flags.DEFINE_list("betas", ["0.9", "0.95"], "Adam betas.")
flags.DEFINE_float("weight_decay", 0.01, "AdamW weight decay.")
flags.DEFINE_float("max_grad_norm", 1.0, "Grad clip.")

flags.DEFINE_string("save_dir", "exp/ppo_finetune", "Save dir.")
flags.DEFINE_integer("save_every_n_updates", 20, "Save policy state_dict every N updates. 0 disables periodic saving. A `highest.ckpt` is also saved whenever the rolling 5-update mean SR is a new high.")
flags.DEFINE_string("run_name", None, "Run name; auto if None.")
flags.DEFINE_bool("wandb", True, "Log to wandb.")
flags.DEFINE_string("wandb_project", "g1_ppo_finetune", "wandb project.")
flags.DEFINE_string("wandb_resume_id", "", "Resume this wandb run id instead of starting a new run. Pair with --start_update.")
flags.DEFINE_integer("start_update", 0, "Update number to start the loop counter at (for resuming).")
flags.DEFINE_string("video_dir", "", "If non-empty, save sample-rollout mp4s here each update.")
flags.DEFINE_integer("video_every_n_updates", 1, "Save videos every N updates (when video_dir set).")
flags.DEFINE_string("first_frame_grid_dir", "", "If non-empty, save a grid PNG of every episode's first head-cam frame here each update, to verify group-mates look identical.")

flags.DEFINE_integer("eval_every_n_updates", 10, "Run eval every N training updates. 0 disables.")
flags.DEFINE_integer("eval_episodes", 64, "Sampled eval tasks per pass when no --benchmark is given.")
flags.DEFINE_bool("eval_at_start", True, "Run one eval pass at update=0 before any training updates. Records the BC starting-point SR on the held-out eval scenes.")
flags.DEFINE_integer("eval_seed_base", 777000, "Seed for eval: the flow sampler, and the tasks when no --benchmark is given.")
flags.DEFINE_string("benchmark", "", "Eval set to replay every eval pass: a name under molmospaces/assets/eval_sets/ or a JSONL path. Empty samples --eval_episodes tasks instead.")

config_flags.DEFINE_config_file(
    "agent", str(TRAIN_DIR / "algorithms/flow_policy.py"), lock_config=False
)
config_flags.DEFINE_config_file(
    "env", str(ROOT_DIR / "molmospaces/configs/bowl_pick.py"), lock_config=False
)
flags.mark_flag_as_required("checkpoint")


# Main loop
def main(_):
    # Expose all GPUs we'll need (FLAGS.gpu plus any policy_gpus for replication).
    # Workers explicitly set CUDA_VISIBLE_DEVICES="" so they don't inherit this.
    """Spawn envs, then alternate group rollouts, GRPO updates, evals and checkpoints."""
    _gpus_to_expose = [str(FLAGS.gpu)]
    for g in FLAGS.policy_gpus:
        if g and str(g) not in _gpus_to_expose:
            _gpus_to_expose.append(str(g))
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(_gpus_to_expose)
    # Within torch, the exposed GPUs map to cuda:0, cuda:1, ... in the order
    # listed. FLAGS.gpu is always cuda:0 here.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    policy_devices = [torch.device(f"cuda:{i}") for i in range(len(_gpus_to_expose))]
    print(f"[rl] CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}  "
          f"policy_devices={[str(d) for d in policy_devices]}")

    policy, ref_policy, config, payload, fps, ckpt_dir = load_policy(device)
    policy_replicas = build_replicas(policy, policy_devices)
    actor_optim, actor_params = build_optimizers(policy)
    env_cfg, per_worker_overrides = build_env_config(payload)

    print(f"[rl] spawning {FLAGS.parallel_envs} parallel envs on GPUs {FLAGS.render_gpus}")
    _t_spawn = time.time()
    vec_env = VecEnv(env_cfg, FLAGS.parallel_envs, FLAGS.render_gpus, FLAGS.seed,
                     per_worker_overrides=per_worker_overrides)
    print(f"[rl] forked {FLAGS.parallel_envs} env processes in {time.time()-_t_spawn:.1f}s", flush=True)

    action_repeat, step_dt, image_keys = control_timing(env_cfg, policy, fps, FLAGS.seed)

    obs_horizon = int(policy.config["observation_horizon"])
    action_horizon = int(policy.config["action_horizon"])

    schedule = torch.linspace(0.0, 1.0, int(policy.config["num_sampling_steps"]), device=device)
    sigmas = sde_sigma_schedule(FLAGS.noise_level, int(policy.config["num_sampling_steps"]) - 1, device)
    print(f"[rl] schedule: {schedule.tolist()}")
    print(f"[rl] sigmas:   {[f'{s:.3f}' for s in sigmas.tolist()]}")
    print(f"[rl] action_repeat={action_repeat}  step_dt={step_dt:.4f}s  "
          f"time_limit={FLAGS.time_limit}s  max_steps={FLAGS.max_steps_per_episode} "
          f"(~{FLAGS.max_steps_per_episode * step_dt:.1f}s)")

    if FLAGS.wandb:
        import wandb
        if FLAGS.wandb_resume_id:
            wandb.init(project=FLAGS.wandb_project, id=FLAGS.wandb_resume_id,
                       resume="must", config=dict(FLAGS.flag_values_dict()))
            print(f"[wandb] resuming run id={FLAGS.wandb_resume_id} "
                  f"(start_update={FLAGS.start_update})", flush=True)
        else:
            wandb.init(project=FLAGS.wandb_project,
                       name=FLAGS.run_name or f"grpo_{int(time.time())}",
                       config=dict(FLAGS.flag_values_dict()))

    # --- GRPO setup: one-time init then per-update set_tasks. ---
    num_groups = int(FLAGS.num_groups)
    group_size = int(FLAGS.group_size)  # G = EPISODES per group (baseline sample size)
    assert FLAGS.parallel_envs % num_groups == 0, (
        f"parallel_envs ({FLAGS.parallel_envs}) must be divisible by "
        f"num_groups ({num_groups})")
    envs_per_group = FLAGS.parallel_envs // num_groups
    assert group_size % envs_per_group == 0, (
        f"group_size ({group_size} episodes/group) must be divisible by "
        f"envs_per_group ({envs_per_group} = parallel_envs/num_groups)")
    cap_per_env = group_size // envs_per_group
    episodes_per_update = num_groups * group_size  # derived total per update
    group_ids = [i // envs_per_group for i in range(FLAGS.parallel_envs)]

    def group_task_seeds():
        """One fresh task seed per group, repeated for each of its envs."""
        seeds = np.random.randint(0, 2**31 - 1, size=num_groups)
        return [int(seeds[g]) for g in group_ids]
    print(f"[grpo] {num_groups} groups x {group_size} episodes/group "
          f"= {episodes_per_update} episodes/update "
          f"({envs_per_group} envs/group x {cap_per_env} eps/env, {FLAGS.parallel_envs} envs)")
    # Init workers once; each update then locks groups onto tasks via set_tasks().
    _t_init = time.time()
    vec_env.init_workers(obs_horizon, action_horizon, action_repeat, FLAGS.time_limit,
                         FLAGS.max_steps_per_episode, policy.config.get("use_action_delta", True))
    print(f"[rl] initialized {FLAGS.parallel_envs} collection envs "
          f"(imports + first scene compile) in {time.time()-_t_init:.1f}s", flush=True)

    # Periodic eval replays a fixed benchmark when one is given, else samples
    # tasks from fixed seeds. Both run the flow noise-free.
    benchmark_name, benchmark_states = (load_benchmark(FLAGS.benchmark)
                                        if FLAGS.benchmark else (None, None))
    if benchmark_states:
        print(f"[eval] benchmark {benchmark_name}: {len(benchmark_states)} episodes", flush=True)
    eval_time_limit = FLAGS.eval_time_limit if FLAGS.eval_time_limit > 0 else FLAGS.time_limit
    eval_kwargs = dict(schedule=schedule, device=device, image_keys=image_keys,
                       obs_horizon=obs_horizon, action_horizon=action_horizon,
                       action_repeat=action_repeat, time_limit=eval_time_limit,
                       seed=FLAGS.eval_seed_base, policy_replicas=policy_replicas)

    # Checkpoint setup: copy BC's stats.json + flags.json into save_dir so
    # benchmark/main.py can load these RL checkpoints directly.
    run_save_dir = None
    sr_window = deque(maxlen=5)
    best_window_mean = -float("inf")
    if FLAGS.save_every_n_updates > 0:
        run_name = FLAGS.run_name or f"run_{int(time.time())}"
        run_save_dir = Path(FLAGS.save_dir) / run_name
        run_save_dir.mkdir(parents=True, exist_ok=True)
        for fname in ("stats.json", "flags.json"):
            src = ckpt_dir / fname
            dst = run_save_dir / fname
            if src.exists() and not dst.exists():
                import shutil; shutil.copy2(str(src), str(dst))
        print(f"[save] checkpoints -> {run_save_dir}")

    # Training loop
    try:
        for update in range(FLAGS.start_update, FLAGS.start_update + FLAGS.max_updates):
            t0 = time.time()
            policy.eval()

            if FLAGS.eval_every_n_updates > 0 and (
                    update % FLAGS.eval_every_n_updates == 0 if update > 0 else FLAGS.eval_at_start):
                t_eval = time.time()
                if benchmark_states:
                    eval_episodes = replay(policy, vec_env, benchmark_states, **eval_kwargs)
                    label = f"benchmark:{benchmark_name}"
                else:
                    eval_episodes = sample(policy, vec_env, FLAGS.eval_episodes, **eval_kwargs)
                    label = f"seeded:{FLAGS.eval_seed_base}"
                vec_env.set_time_limit(FLAGS.time_limit)
                eval_sr = float(np.mean([ep["success"] for ep in eval_episodes])) if eval_episodes else 0.0
                print(f"[eval] update={update} eval_sr_deterministic={eval_sr:.3f} ({label}) "
                      f"n={len(eval_episodes)} wall={time.time() - t_eval:.1f}s", flush=True)
                if FLAGS.wandb:
                    import wandb; wandb.log({"eval/sr_deterministic": eval_sr}, step=update)

            video_path = None
            if FLAGS.video_dir and (update % FLAGS.video_every_n_updates == 0):
                Path(FLAGS.video_dir).mkdir(parents=True, exist_ok=True)
                video_path = str(Path(FLAGS.video_dir) / f"update_{update:03d}.mp4")
            # Every env in a group locks onto the group's task and freezes
            # auto-resets, so all of the group's episodes share it.
            if getattr(vec_env, "_pending_set_tasks", False):
                set_payloads = vec_env.wait_set_tasks_ready()
            else:
                set_payloads = vec_env.set_tasks(group_task_seeds(), freeze_subsequent=True)
            vec_env._last_obs_histories = [p["obs_history"] for p in set_payloads]
            episodes = collect_rollouts(
                policy, vec_env, schedule, sigmas, device, image_keys,
                episodes_per_update, FLAGS.time_limit, FLAGS.max_steps_per_episode,
                obs_horizon, action_horizon, action_repeat, video_path=video_path,
                policy_replicas=policy_replicas, group_ids=group_ids, skip_init_workers=True,
                default_object=FLAGS.objects, dispatch_wait_ms=FLAGS.dispatch_wait_ms,
                capture_first_frames=bool(FLAGS.first_frame_grid_dir))
            # Prefetch the next update's tasks: workers reload scenes while the
            # master runs the update, logging and checkpointing below.
            if update + 1 < FLAGS.max_updates:
                vec_env.set_tasks_async(group_task_seeds(), freeze_subsequent=True)
            ep_successes = [float(ep["success"]) for ep in episodes]

            if FLAGS.first_frame_grid_dir:
                import imageio
                _grid_dir = Path(FLAGS.first_frame_grid_dir)
                _grid_dir.mkdir(parents=True, exist_ok=True)
                frames_with_ff = [ep for ep in episodes if ep.get("first_frame") is not None]
                if frames_with_ff:
                    eps_per_env = max(1, episodes_per_update // FLAGS.parallel_envs)
                    # rows = groups; cols = (env_within_group * cycles + cycle) = group_size
                    # episodes per group. Adjacent cells in a row are the same group/task
                    # (should look identical). envs_per_group = parallel_envs / num_groups.
                    n_rows = num_groups
                    n_cols = envs_per_group * eps_per_env
                    sample_img = frames_with_ff[0]["first_frame"]
                    H, W, C = sample_img.shape
                    grid = np.zeros((n_rows * H, n_cols * W, C), dtype=np.uint8)
                    for ep in frames_with_ff:
                        gid = ep["group_id"]; eidx = ep["env_idx"]; eie = ep["ep_idx_in_env"]
                        if eie >= eps_per_env: continue  # past the expected cycles
                        within = eidx - gid * envs_per_group
                        row = gid
                        col = within * eps_per_env + eie
                        if 0 <= row < n_rows and 0 <= col < n_cols:
                            grid[row*H:(row+1)*H, col*W:(col+1)*W] = ep["first_frame"]
                    grid_path = _grid_dir / f"update_{update:03d}.png"
                    imageio.imwrite(str(grid_path), grid)

            # Flow-GRPO: sparse episode returns, baselined against the mean
            # over envs sharing the task.
            group_returns = {g: [] for g in range(num_groups)}
            group_successes = {g: [] for g in range(num_groups)}
            for ep in episodes:
                if not ep["transitions"]: continue
                group_returns[int(ep["group_id"])].append(float(sum(t["reward"] for t in ep["transitions"])))
                group_successes[int(ep["group_id"])].append(float(ep["success"]))
            group_stats = {}
            skipped_groups = 0
            for g, Rs in group_returns.items():
                if not Rs:
                    group_stats[g] = (0.0, 1.0, True)
                    continue
                g_sr = float(np.mean(group_successes[g]))
                # Skip degenerate/saturated groups (RLinf filter_rewards): a group
                # contributes gradient only if its SR is in (low, high), which drops
                # all-fail and all-success groups whose advantage is ~0/pure noise.
                skip = (g_sr < FLAGS.grpo_filter_sr_low) or (g_sr > FLAGS.grpo_filter_sr_high)
                skipped_groups += skip
                group_stats[g] = (float(np.mean(Rs)), float(np.std(Rs)) + 1e-6, skip)
            if skipped_groups:
                print(f"[update {update}] dropped {skipped_groups}/{num_groups} degenerate/saturated groups")
            # Only mixed groups (0 < SR < 1) give a non-zero advantage, so this
            # is the diagnostic for whether GRPO has anything to learn from.
            group_srs = [float(np.mean(v)) for v in group_successes.values() if v]
            n_all_succ = sum(1 for s in group_srs if s >= 0.999)
            n_all_fail = sum(1 for s in group_srs if s <= 0.001)
            n_mixed = len(group_srs) - n_all_succ - n_all_fail
            grpo_mixed_frac = (n_mixed / len(group_srs)) if group_srs else 0.0
            print(f"[update {update}] group SR: " + " ".join(f"{s:.2f}" for s in group_srs))
            print(f"[update {update}] groups: {n_mixed} mixed / {n_all_succ} all-succ / "
                  f"{n_all_fail} all-fail  (only {n_mixed} give gradient)")
            flat = []
            for ep in episodes:
                trans = ep["transitions"]
                if not trans: continue
                gid = int(ep["group_id"])
                mu, sd, skip = group_stats[gid]
                if skip: continue
                R = float(sum(t["reward"] for t in trans))
                A = (R - mu) / sd
                ep_len = len(trans)
                for t in trans:
                    # One episode-level advantage broadcast to every chunk-step.
                    t["advantage"] = float(A)
                    t["return"] = float(R)  # for logging only; GRPO doesn't fit a value head
                    t["old_logprob"] = t["logprob"]
                    t["ep_len"] = ep_len
                    flat.append(t)
            if not flat:
                print(f"[update {update}] no transitions, skipping")
                continue

            # Policy stays in eval mode so the DiT's attention dropout is off.
            # With it on, new_logp drifts from old_logp through a fresh dropout
            # mask each forward and the update chases that noise.
            policy.eval()
            actor_losses, kl_logs, ratio_logs = [], [], []
            ratio_max_logs, ratio_min_logs = [], []
            clip_prop_logs, approx_kl_logs = [], []
            adv_mean_logs, adv_max_logs, adv_min_logs = [], [], []
            ret_mean_logs, ret_max_logs, ret_min_logs = [], [], []
            n_mb_per_epoch = max(1, (len(flat) + FLAGS.minibatch_size - 1) // FLAGS.minibatch_size)
            total_mb = FLAGS.epochs_per_update * n_mb_per_epoch
            train_pbar = tqdm.tqdm(total=total_mb, desc="update", unit="batch", leave=False)
            for _ in range(FLAGS.epochs_per_update):
                order = np.random.permutation(len(flat))
                for start in range(0, len(flat), FLAGS.minibatch_size):
                    idx = order[start:start + FLAGS.minibatch_size]
                    mb = [flat[i] for i in idx]

                    # Build minibatch from CACHED encoder outputs (rollout-time).
                    # Frozen encoder => safe to reuse. Skips the heavy DINOv3
                    # forward during every step (~5x faster per minibatch).
                    sample_enc = mb[0]["encoder_out"]
                    if isinstance(sample_enc, dict):
                        enc_batch = {}
                        for k in sample_enc:
                            vals = [t["encoder_out"][k] for t in mb]
                            if isinstance(vals[0], torch.Tensor):
                                enc_batch[k] = torch.cat(vals, dim=0).to(device)
                            else:
                                enc_batch[k] = sum([list(v) for v in vals], [])
                    else:
                        enc_batch = torch.cat([t["encoder_out"] for t in mb], dim=0).to(device)
                    chains = torch.stack([t["chain"] for t in mb]).to(device)
                    old_logp = torch.tensor([t["old_logprob"] for t in mb], device=device)
                    adv_b = torch.tensor([t["advantage"] for t in mb], device=device)
                    ret_b = torch.tensor([t["return"] for t in mb], device=device)
                    n_steps_b = torch.tensor(
                        [t.get("noise_step", 0) for t in mb], device=device, dtype=torch.long)

                    new_logp = recompute_chain_logprob_cached(
                        policy, enc_batch, chains, sigmas, schedule, batch_size=len(mb),
                        noise_step=n_steps_b)
                    # Clipped-ratio actor loss with dual clip.
                    log_ratio = (new_logp - old_logp).clamp(-10.0, 10.0)
                    ratio = log_ratio.exp()
                    surr1 = ratio * adv_b
                    # Asymmetric clip (DAPO clip-higher): lower bound 1-ppo_clip,
                    # upper bound 1+ppo_clip_high. Asymmetric prevents the upper
                    # clip from suppressing exploration while still bounding losses.
                    surr2 = torch.clamp(ratio, 1 - FLAGS.ppo_clip, 1 + FLAGS.ppo_clip_high) * adv_b
                    ppo_clip_loss = torch.min(surr1, surr2)
                    # Dual clip: cap loss at -dual_clip * |adv| for neg-adv samples.
                    # Prevents one outlier with huge ratio from dominating the batch.
                    if FLAGS.ppo_dual_clip > 0:
                        ppo_dual_floor = -torch.sign(adv_b) * FLAGS.ppo_dual_clip * adv_b
                        ppo_loss = torch.maximum(ppo_clip_loss, ppo_dual_floor)
                    else:
                        ppo_loss = ppo_clip_loss
                    if FLAGS.length_normalize and "ep_len" in mb[0]:
                        ep_len_b = torch.tensor([float(t["ep_len"]) for t in mb],
                                                 device=device, dtype=torch.float32)
                        length_weight = ep_len_b.mean() / ep_len_b.clamp(min=1.0)
                        actor_loss = -(ppo_loss * length_weight).mean()
                    else:
                        actor_loss = -ppo_loss.mean()

                    # Optional KL to BC reference (uses same cached encoder_out).
                    kl_term = torch.tensor(0.0, device=device)
                    if FLAGS.kl_beta > 0 and ref_policy is not None:
                        kl_term = recompute_chain_kl_to_ref_cached(
                            policy, ref_policy, enc_batch, chains, sigmas, schedule,
                            batch_size=len(mb), noise_step=n_steps_b).mean()
                        actor_loss = actor_loss + FLAGS.kl_beta * kl_term

                    actor_optim.zero_grad()
                    actor_loss.backward()
                    nn.utils.clip_grad_norm_(actor_params, FLAGS.max_grad_norm)
                    actor_optim.step()

                    # Per-batch diagnostics.
                    with torch.no_grad():
                        clip_prop = ((torch.clamp(ratio, 1 - FLAGS.ppo_clip, 1 + FLAGS.ppo_clip_high) * adv_b)
                                     < (ratio * adv_b)).float().mean().item()
                        ppo_approx_kl = (torch.exp(log_ratio) - log_ratio - 1.0).mean().item()
                    actor_losses.append(actor_loss.item())
                    kl_logs.append(kl_term.item())
                    ratio_logs.append(ratio.mean().item())
                    ratio_max_logs.append(ratio.max().item())
                    ratio_min_logs.append(ratio.min().item())
                    clip_prop_logs.append(clip_prop)
                    approx_kl_logs.append(ppo_approx_kl)
                    adv_mean_logs.append(adv_b.mean().item())
                    adv_max_logs.append(adv_b.max().item())
                    adv_min_logs.append(adv_b.min().item())
                    ret_mean_logs.append(ret_b.mean().item())
                    ret_max_logs.append(ret_b.max().item())
                    ret_min_logs.append(ret_b.min().item())
                    train_pbar.set_postfix(
                        actor=f"{actor_loss.item():.3f}",
                        ratio=f"{ratio.mean().item():.3f}",
                        clip=f"{clip_prop:.2f}",
                        kl=f"{ppo_approx_kl:.4f}",
                    )
                    train_pbar.update(1)
            train_pbar.close()

            dt = time.time() - t0
            sr = float(np.mean(ep_successes))
            ep_rewards = [sum(t["reward"] for t in ep["transitions"]) for ep in episodes
                          if ep["transitions"]]
            ep_collisions = [int(ep.get("n_collisions", 0)) for ep in episodes]
            collisions_mean = float(np.mean(ep_collisions)) if ep_collisions else 0.0
            collisions_rate = float(np.mean([1.0 if c > 0 else 0.0 for c in ep_collisions])) if ep_collisions else 0.0
            log = dict(
                update=update,
                success_rate=sr,
                ep_reward_mean=float(np.mean(ep_rewards)) if ep_rewards else 0.0,
                collisions_per_ep=collisions_mean,
                collision_rate=collisions_rate,
                grpo_mixed_groups=int(n_mixed),
                grpo_mixed_frac=float(grpo_mixed_frac),
                actor_loss=float(np.mean(actor_losses)),
                kl_to_bc=float(np.mean(kl_logs)),
                avg_ratio=float(np.mean(ratio_logs)),
                ratio_max=float(np.max(ratio_max_logs)) if ratio_max_logs else 0.0,
                ratio_min=float(np.min(ratio_min_logs)) if ratio_min_logs else 0.0,
                clip_proportion=float(np.mean(clip_prop_logs)),
                ppo_approx_kl=float(np.mean(approx_kl_logs)),
                adv_mean=float(np.mean(adv_mean_logs)),
                adv_max=float(np.max(adv_max_logs)) if adv_max_logs else 0.0,
                adv_min=float(np.min(adv_min_logs)) if adv_min_logs else 0.0,
                returns_mean=float(np.mean(ret_mean_logs)),
                returns_max=float(np.max(ret_max_logs)) if ret_max_logs else 0.0,
                returns_min=float(np.min(ret_min_logs)) if ret_min_logs else 0.0,
                transitions=len(flat),
                wall_s=dt,
            )
            print(f"[update {update}] SR={sr:.2f} reward={log['ep_reward_mean']:.2f} "
                  f"col/ep={collisions_mean:.1f} col_rate={collisions_rate:.2f} "
                  f"actor={log['actor_loss']:.3f} "
                  f"ratio={log['avg_ratio']:.3f} clip={log['clip_proportion']:.2f} "
                  f"kl={log['ppo_approx_kl']:.4f} "
                  f"({dt:.1f}s)")
            if FLAGS.wandb:
                import wandb; wandb.log(log, step=update)

            # Checkpointing: periodic step ckpt + rolling-window best.
            if run_save_dir is not None:
                sr_window.append(float(sr))
                step_idx = update + 1
                if step_idx % FLAGS.save_every_n_updates == 0:
                    step_path = run_save_dir / f"step={step_idx:06d}.ckpt"
                    torch.save({
                        "state_dict": policy.state_dict(),
                        "update": step_idx,
                        "sr": float(sr),
                    }, str(step_path))
                    print(f"[save] {step_path}  sr={sr:.3f}")
                if len(sr_window) == sr_window.maxlen:
                    win_mean = sum(sr_window) / len(sr_window)
                    if win_mean > best_window_mean:
                        best_window_mean = win_mean
                        best_path = run_save_dir / "highest.ckpt"
                        torch.save({
                            "state_dict": policy.state_dict(),
                            "update": step_idx,
                            "sr": float(sr),
                            "window_mean": win_mean,
                        }, str(best_path))
                        print(f"[save] highest.ckpt update={step_idx} window_mean={win_mean:.3f}")

    finally:
        vec_env.close()


if __name__ == "__main__":
    app.run(main)
