"""Rollout collection and advantage estimation."""
import sys
import time
from pathlib import Path

import numpy as np
import torch
import tqdm
from absl import flags

ROOT_DIR = Path(__file__).resolve().parents[1]
TRAIN_DIR = ROOT_DIR / "train"
for _p in (ROOT_DIR, TRAIN_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

FLAGS = flags.FLAGS

from utils.action_transforms import DELTA_DIMS, anchor_from_obs
from flow_sde import sample_actions_sde


class _NullCtx:
    def __enter__(self):
        """No-op context enter."""
        return self
    def __exit__(self, *a):
        """No-op context exit; never swallows exceptions."""
        return False


def _save_video(path: Path, frames, fps: int = 10):
    """Write frames to an mp4 (libx264, crf 20), swallowing encoder failures."""
    if not frames: return
    import imageio
    try:
        writer = imageio.get_writer(str(path), fps=fps, codec="libx264",
                                    pixelformat="yuv420p",
                                    output_params=["-crf", "20"])
        for f in frames:
            writer.append_data(f)
        writer.close()
    except Exception as e:
        print(f"[video] failed to save {path}: {e}")


def collect_rollouts(policy, vec_env, schedule, sigmas, device,
                     image_keys, n_episodes: int, time_limit: float,
                     max_steps: int, obs_horizon: int, action_horizon: int,
                     action_repeat: int, video_path: str = None,
                     policy_replicas=None, group_ids=None, skip_init_workers=False):
    """Async rollout: workers run their own chunk-batches and report when done.
    Master batches policy fwd across whichever workers are ready at any moment.
    Slow workers no longer block fast ones."""
    N = vec_env.n_envs
    use_delta = policy.config.get("use_action_delta", True)

    if policy_replicas is None: policy_replicas = [policy]
    n_replicas = len(policy_replicas)

    # Sync replica weights from the primary policy before this rollout.
    if n_replicas > 1:
        primary_sd = policy.state_dict()
        for rep in policy_replicas[1:]:
            rep.load_state_dict(primary_sd, strict=False)
            rep.eval()
    # text_adapter weights moved in the last update, so last rollout's cached
    # object-name tokens are stale. Within one rollout the cache is valid.
    for rep in policy_replicas:
        enc = getattr(rep, "encoder", None)
        if enc is not None and hasattr(enc, "reset_text_cache"):
            enc.reset_text_cache()
    # Pre-stage the schedule/sigmas tensors on each replica's device.
    schedules_per_replica = [schedule.to(p.device) for p in policy_replicas]
    sigmas_per_replica = [sigmas.to(p.device) for p in policy_replicas]

    # Init workers (they reset envs and send back initial obs_histories).
    # Skip when a higher-level coordinator has already initialized & set tasks
    # (GRPO path calls init once at the very start, then set_tasks each update).
    if not skip_init_workers:
        init_payloads = vec_env.init_workers(
            obs_horizon, action_horizon, action_repeat, time_limit, max_steps, use_delta,
        )
        worker_history = [p["obs_history"] for p in init_payloads]
    else:
        # Caller already (re)set tasks via set_tasks() and stashed the obs
        # histories on vec_env._last_obs_histories.
        worker_history = [list(h) for h in vec_env._last_obs_histories]

    # Each worker maintains a transition buffer until episode finishes.
    ep_transitions = [[] for _ in range(N)]
    # Pending chunk-step transitions that need to be attached to the LAST policy
    # output (chain/logprob/obs_batch). Per worker.
    pending_policy_output = [None] * N

    GRIP_OPEN, GRIP_CLOSED = -0.0222, 0.0245
    # 95%, matching the BC deployment paths. A 0.5 midpoint closes the gripper
    # early and fails the grasp.
    GRIP_MID = GRIP_OPEN + 0.95 * (GRIP_CLOSED - GRIP_OPEN)

    finished_eps = []
    # All workers start ready.
    ready_workers = set(range(N))
    executing_workers = set()
    env0_frames = [] if video_path else None
    env0_saved = False
    record_env0 = bool(video_path)

    # Optional: stash each episode's first head-cam frame (post-reset obs) so the
    # caller can write a per-update grid PNG verifying group-mates are identical.
    capture_first_frames = bool(FLAGS.first_frame_grid_dir)
    # Each group collects exactly its own quota.
    if group_ids is not None:
        n_groups = max(group_ids) + 1
        min_eps_per_group = n_episodes // n_groups
        group_ep_counter = [0] * n_groups
        # Cap per mate. Without it, fast-succeeding envs lap slow-failing ones
        # and fill the quota first, biasing the sample toward successes.
        _group_size = N // n_groups
        eps_per_env_cap = max(1, min_eps_per_group // max(1, _group_size))
        # Keep the group target equal to what the per-mate caps can actually
        # produce (cap * group_size), so the wait-loop can't hang when
        # min_eps_per_group isn't an exact multiple of group_size.
        min_eps_per_group = eps_per_env_cap * _group_size
    else:
        n_groups = 0
        min_eps_per_group = 0
        group_ep_counter = None
        # Same cap on the eval path, or fast envs crowd out slow ones and
        # inflate eval SR.
        eps_per_env_cap = max(1, n_episodes // N)
        n_episodes = min(n_episodes, eps_per_env_cap * N)  # achievable target; no hang
    def _head_cam_uint8(obs):
        """Extract the head-camera frame from an obs as HWC uint8."""
        key = next((k for k in image_keys if 'head' in k.lower()), image_keys[0] if image_keys else None)
        if key is None or key not in obs: return None
        img = obs[key]
        if hasattr(img, 'detach'): img = img.detach().cpu().numpy()
        img = np.asarray(img)
        if img.ndim == 3 and img.shape[0] in (1, 3):
            img = np.transpose(img, (1, 2, 0))
        return (img * 255.0).clip(0, 255).astype(np.uint8)
    pending_first_frames = (
        [_head_cam_uint8(worker_history[i][-1]) for i in range(N)]
        if capture_first_frames else None)
    ep_counter_per_env = [0] * N

    prof = {"policy": 0.0, "policy_calls": 0, "policy_batch_total": 0,
            "worker_step": 0.0, "worker_render": 0.0,
            "wait": 0.0, "process_done": 0.0, "dispatch": 0.0,
            "n_chunk_batches": 0}
    _t_start = time.perf_counter()

    pbar = tqdm.tqdm(total=n_episodes, desc="rollouts(async)", unit="ep", leave=False)

    # Persistent thread pool + per-replica CUDA streams for the multi-GPU path.
    # Pre-allocated stream lets each replica's PyTorch kernels run in parallel
    # with the others; the thread pool removes per-dispatch thread-spawn cost.
    from concurrent.futures import ThreadPoolExecutor
    _replica_streams = [torch.cuda.Stream(device=p.device) for p in policy_replicas] \
        if torch.cuda.is_available() and n_replicas > 1 else None
    _replica_pool = ThreadPoolExecutor(max_workers=max(1, n_replicas)) if n_replicas > 1 else None

    def _run_one_replica(rep_idx, sub_obs, sub_size, results, sub_noise_step=None):
        """Run the policy on one replica. Called from a thread.
        Uses a dedicated CUDA stream so multiple replicas can compute in
        parallel on different GPUs. Also returns the cached encoder output
        so we can skip the encoder during the update.

        bf16 autocast: the rollout-time forward (encoder + flow)
        runs in bf16 on tensor cores, ~1.5-2x faster on L40S with no accuracy
        loss in inference. We cast the returned tensors back to fp32 so the
        cached chain/logprob/encoder are stored in the same precision as the
        PPO/GRPO update expects."""
        rep = policy_replicas[rep_idx]
        stream = _replica_streams[rep_idx] if _replica_streams is not None else None
        ctx = torch.cuda.stream(stream) if stream is not None else _NullCtx()
        with torch.no_grad(), ctx:
            # Do NOT autocast to bf16: the update recomputes logp at fp32, and
            # the mismatch leaves a ~0.05 logp noise floor the policy chases
            # through the ratio. The encoder is bf16 internally; this path is not.
            actions, chain, logp, cond_out, n_step = sample_actions_sde(
                rep, sub_obs, schedules_per_replica[rep_idx],
                sigmas_per_replica[rep_idx], batch_size=sub_size,
                compute_logprob=True,
                noise_step=sub_noise_step,  # sampled once by master so all
                                            # replicas in this dispatch use
                                            # the same logged noise step.
                return_cond=True, return_noise_step=True,
            )
            # Cast back to fp32 so downstream PPO/GRPO update sees expected dtype.
            actions = actions.float()
            chain = chain.float()
            logp = logp.float()
            if isinstance(cond_out, dict):
                cond_out = {k: v.float() if isinstance(v, torch.Tensor) and v.dtype != torch.float32 else v
                            for k, v in cond_out.items()}
            elif isinstance(cond_out, torch.Tensor):
                cond_out = cond_out.float()
        if stream is not None:
            stream.synchronize()
        elif torch.cuda.is_available():
            torch.cuda.synchronize(rep.device)
        results[rep_idx] = (actions, chain, logp, cond_out, n_step)

    def dispatch_batch(ready_idxs):
        """Run policy on the obs_histories of the given ready workers and
        dispatch chunk-batches to them. Mark them executing. Splits the batch
        across policy_replicas using a thread per replica."""
        if not ready_idxs:
            return
        _td = time.perf_counter()
        # Build batched obs (numpy, then move per-replica below).
        batch_np = {}
        for key in policy.obs_keys:
            obs_key = key.removeprefix("observation.")
            stacked = np.stack(
                [np.stack([f[obs_key] for f in worker_history[idx]], axis=0)
                 for idx in ready_idxs], axis=0).astype(np.float32)
            batch_np[key] = stacked
        # Each env's actual object, tagged onto obs by the worker. Using one
        # global name feeds the wrong text condition on mixed-object sets.
        object_names = [(worker_history[idx][-1].get("_object_name") or FLAGS.objects)
                        for idx in ready_idxs]

        _t_pol = time.perf_counter()
        B = len(ready_idxs)
        K_noise = int(policy.config["num_sampling_steps"]) - 1
        dispatch_noise_step = int(torch.randint(0, K_noise, (1,)).item())
        if n_replicas == 1:
            sub_obs = {k: torch.from_numpy(v).to(policy_replicas[0].device)
                       for k, v in batch_np.items()}
            sub_obs["object_name"] = object_names
            results = [None]
            _run_one_replica(0, sub_obs, B, results,
                             sub_noise_step=dispatch_noise_step)
            actions, chain, logp, cond_out, n_step = results[0]
        else:
            edges = [(i * B) // n_replicas for i in range(n_replicas + 1)]
            results = [None] * n_replicas
            futures = []
            batch_cpu = {k: torch.from_numpy(v).pin_memory() if torch.cuda.is_available()
                            else torch.from_numpy(v)
                         for k, v in batch_np.items()}
            for i in range(n_replicas):
                s, e = edges[i], edges[i + 1]
                if e == s: continue
                rep_dev = policy_replicas[i].device
                sub_obs = {k: v[s:e].to(rep_dev, non_blocking=True)
                           for k, v in batch_cpu.items()}
                sub_obs["object_name"] = object_names[s:e]
                futures.append(_replica_pool.submit(
                    _run_one_replica, i, sub_obs, e - s, results,
                    dispatch_noise_step))
            for f in futures: f.result()
            parts_a, parts_c, parts_l = [], [], []
            cond_parts = None
            n_step = None
            for r in results:
                if r is None: continue
                a, c, l, co, ns = r
                if n_step is None: n_step = ns  # all replicas share same draw via the master
                parts_a.append(a.to(device))
                parts_c.append(c.to(device))
                parts_l.append(l.to(device))
                if isinstance(co, dict):
                    if cond_parts is None: cond_parts = {k: [] for k in co}
                    for k, t in co.items():
                        cond_parts[k].append(t.to(device) if isinstance(t, torch.Tensor) else t)
                else:
                    if cond_parts is None: cond_parts = []
                    cond_parts.append(co.to(device))
            actions = torch.cat(parts_a, dim=0)
            chain = torch.cat(parts_c, dim=0)
            logp = torch.cat(parts_l, dim=0)
            if isinstance(cond_parts, dict):
                cond_out = {k: torch.cat(v, dim=0) if isinstance(v[0], torch.Tensor) else v
                            for k, v in cond_parts.items()}
            else:
                cond_out = torch.cat(cond_parts, dim=0)
        prof["policy"] += time.perf_counter() - _t_pol
        prof["policy_calls"] += 1
        prof["policy_batch_total"] += B

        action_seqs = actions.cpu().numpy()
        for j, idx in enumerate(ready_idxs):
            if use_delta:
                action_seqs[j, :, DELTA_DIMS] += anchor_from_obs(worker_history[idx][-1])
            action_seqs[j, :, 14] = np.where(action_seqs[j, :, 14] > GRIP_MID,
                                             GRIP_CLOSED, GRIP_OPEN)
            # Slice the cached encoder output for env j and move to CPU for
            # storage. Used to skip the encoder forward during PPO update.
            if isinstance(cond_out, dict):
                cond_j = {k: (v[j:j+1].detach().cpu() if isinstance(v, torch.Tensor) else v[j:j+1])
                          for k, v in cond_out.items()}
            else:
                cond_j = cond_out[j:j+1].detach().cpu()
            pending_policy_output[idx] = {
                "encoder_out": cond_j,
                "chain": chain[j].detach().cpu(),
                "logprob": float(logp[j].item()),
                "noise_step": int(n_step),  # matches the step PPO must recompute at
            }
            vec_env.send_chunk(idx, action_seqs[j])
            executing_workers.add(idx)
        ready_workers.difference_update(ready_idxs)
        prof["dispatch"] += time.perf_counter() - _td

    def _process_one(idx, cmd, payload):
        """Fold one worker's finished episode into the rollout buffers (dropping errored workers)."""
        nonlocal env0_saved
        executing_workers.discard(idx)
        # A worker that hit an unrecoverable error (e.g. placement failure on a
        # frozen GRPO task) sends ("error", ...) and exits. Don't crash the whole
        # run on it: drop that worker for this rollout and continue with the rest.
        if cmd != "done" or "transitions" not in payload:
            print(f"[rollout] worker {idx} errored, dropping its episode this update: "
                  f"{payload.get('err', cmd)}", flush=True)
            return
        prof["worker_step"] += float(payload.get("_t_step", 0.0))
        prof["worker_render"] += float(payload.get("_t_render", 0.0))
        prof["n_chunk_batches"] += 1
        transitions = payload["transitions"]
        batch_reward = sum(float(t["reward"]) for t in transitions)
        batch_done = payload["episode_finished"]
        batch_success = payload["episode_success"] if batch_done else any(t["success"] for t in transitions)

        ep_transitions[idx].append({
            **pending_policy_output[idx],
            "reward": batch_reward,
            "done": bool(batch_done),
        })
        worker_history[idx] = payload["obs_history"]

        if record_env0 and idx == 0 and not env0_saved:
            final_obs = worker_history[idx][-1]
            panels = []
            for k in image_keys:
                if k not in final_obs: continue
                img = final_obs[k]
                if img.ndim == 3 and img.shape[0] in (1, 3):
                    img = np.transpose(img, (1, 2, 0))
                img = (np.asarray(img) * 255.0).clip(0, 255).astype(np.uint8)
                panels.append(img)
            if panels:
                env0_frames.append(np.concatenate(panels, axis=1))

        gid_now = int(group_ids[idx]) if group_ids is not None else None
        if gid_now is not None:
            accept = (batch_done
                      and ep_counter_per_env[idx] < eps_per_env_cap
                      and group_ep_counter[gid_now] < min_eps_per_group)
        else:
            accept = (batch_done
                      and ep_counter_per_env[idx] < eps_per_env_cap
                      and len(finished_eps) < n_episodes)
        if accept:
            ep_entry = {
                "transitions": ep_transitions[idx],
                "success": bool(batch_success),
                "n_collisions": int(payload.get("episode_collisions") or 0),
                "env_idx": int(idx),
                "group_id": gid_now,
                "ep_idx_in_env": int(ep_counter_per_env[idx]),
            }
            if capture_first_frames:
                ep_entry["first_frame"] = pending_first_frames[idx]
                # worker just auto-reset; obs_history[-1] is the next episode's first frame.
                pending_first_frames[idx] = _head_cam_uint8(worker_history[idx][-1])
            ep_counter_per_env[idx] += 1
            if gid_now is not None:
                group_ep_counter[gid_now] += 1
            finished_eps.append(ep_entry)
            running_sr = float(np.mean([float(ep["success"]) for ep in finished_eps]))
            pbar.set_postfix(sr=f"{running_sr:.2f}")
            pbar.update(1)
            if record_env0 and idx == 0 and not env0_saved:
                _save_video(Path(video_path), env0_frames, fps=10)
                print(f"[video] saved {video_path} ({len(env0_frames)} frames, "
                      f"success={batch_success})")
                env0_saved = True
            ep_transitions[idx] = []

        gid_for_ready = int(group_ids[idx]) if group_ids is not None else None
        if ((gid_for_ready is not None
                and ep_counter_per_env[idx] < eps_per_env_cap
                and group_ep_counter[gid_for_ready] < min_eps_per_group)
                or (gid_for_ready is None
                    and ep_counter_per_env[idx] < eps_per_env_cap
                    and len(finished_eps) < n_episodes)):
            ready_workers.add(idx)

    dispatch_wait_s = float(FLAGS.dispatch_wait_ms) / 1000.0
    while (len(finished_eps) < n_episodes
           or (min_eps_per_group and min(group_ep_counter) < min_eps_per_group)):
        # Briefly poll executing workers so stragglers can join this dispatch,
        # which grows the policy batch and amortizes the launch overhead.
        if ready_workers and executing_workers and dispatch_wait_s > 0:
            _tw = time.perf_counter()
            ready_done = vec_env.poll_ready(executing_workers, timeout=dispatch_wait_s)
            prof["wait"] += time.perf_counter() - _tw
            _tp = time.perf_counter()
            for idx, cmd, payload in ready_done:
                _process_one(idx, cmd, payload)
            prof["process_done"] += time.perf_counter() - _tp

        # Dispatch all ready workers as ONE policy batch.
        if ready_workers:
            dispatch_batch(sorted(ready_workers))

        if not executing_workers:
            break

        # Block until at least one worker finishes.
        _tw = time.perf_counter()
        ready_done = vec_env.poll_ready(executing_workers, timeout=None)
        prof["wait"] += time.perf_counter() - _tw
        _tp = time.perf_counter()
        for idx, cmd, payload in ready_done:
            _process_one(idx, cmd, payload)
        prof["process_done"] += time.perf_counter() - _tp

    pbar.close()
    _wall = time.perf_counter() - _t_start
    n_cb = max(1, prof["n_chunk_batches"])
    avg_batch = prof["policy_batch_total"] / max(1, prof["policy_calls"])
    print(f"\n[profile] rollout wall={_wall:.1f}s  N_envs={N}  episodes={len(finished_eps)}  chunk_batches={prof['n_chunk_batches']}")
    print(f"[profile]   policy (sde+val)  = {prof['policy']:7.2f}s  ({100*prof['policy']/_wall:5.1f}%)  "
          f"calls={prof['policy_calls']}  avg_batch={avg_batch:.1f}")
    print(f"[profile]   wait (worker)     = {prof['wait']:7.2f}s  ({100*prof['wait']/_wall:5.1f}%)")
    print(f"[profile]   dispatch          = {prof['dispatch']:7.2f}s  ({100*prof['dispatch']/_wall:5.1f}%)")
    print(f"[profile]   process_done      = {prof['process_done']:7.2f}s  ({100*prof['process_done']/_wall:5.1f}%)")
    print(f"[profile]   |- worker.step (sum)   = {prof['worker_step']:7.2f}s  avg={1000*prof['worker_step']/n_cb:.0f}ms/batch")
    print(f"[profile]   |- worker.render (sum) = {prof['worker_render']:7.2f}s  avg={1000*prof['worker_render']/n_cb:.0f}ms/batch")
    # GRPO stops each group once its quota is filled; non-GRPO stops at the
    # requested total episode count.
    return finished_eps


