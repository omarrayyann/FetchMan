"""Parallel environment workers and the master-side handle.

Each worker owns one env and runs whole chunk-batches locally, reporting once
per batch, so the master never coordinates per-substep timing. Workers hold no
flags: everything they need arrives in the env config dict, which keeps them
safe to spawn.
"""
import os
import sys
import time
from collections import deque
from pathlib import Path
from multiprocessing import get_context
from multiprocessing.connection import wait as mp_wait

import numpy as np
from absl import flags

ROOT_DIR = Path(__file__).resolve().parents[1]
TRAIN_DIR = ROOT_DIR / "train"
for _p in (ROOT_DIR, TRAIN_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

FLAGS = flags.FLAGS

from utils.action_transforms import anchor_from_obs
from molmospaces.egl import cuda_to_egl


# Parallel env workers


def _env_worker(rank: int, conn, env_cfg_dict: dict, egl_device_id: int, seed: int):
    """Async worker. Owns per-env rollout state (history, gripper_cmd, prev_smoothed,
    ep_succeeded, n_env_steps) and executes whole chunk-batches locally. Sends
    a single 'done' message per chunk-batch carrying per-chunk-step transitions
    and the new obs-history needed for the next policy call. Auto-resets after
    episodes finish so the master never has to coordinate per-substep timing."""
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["PYOPENGL_PLATFORM"] = "egl"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(egl_device_id)
    # One thread per worker. Uncapped, 64 workers x ~64 threads thrash the
    # cores and become the rollout bottleneck while the GPUs sit idle.
    for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
               "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[_v] = "1"
    try:
        import torch as _torch
        _torch.set_num_threads(1)
    except Exception:
        pass
    sys.path.insert(0, str(ROOT_DIR))
    from molmospaces.env import make_env as _make_env
    from molmospaces.agents.policy import G1Controller as _Controller

    cfg = dict(env_cfg_dict)
    cfg["seed"] = seed
    env = _make_env(cfg)
    agent = _Controller()
    agent.setup(env.scene.model, env.scene.data)
    agent.set_env(env)
    env.set_agent(agent)

    GRIP_OPEN, GRIP_CLOSED = -0.0222, 0.0245
    GRIP_RAMP_STEPS = 400
    GRIP_STEP = (GRIP_CLOSED - GRIP_OPEN) / GRIP_RAMP_STEPS

    history = None
    prev_smoothed = None
    gripper_cmd = GRIP_OPEN
    ep_succeeded = False
    ep_collisions = 0  # chunk-steps with robot-world contact in this episode
    n_env_steps = 0
    current_object_name = ""  # per-episode text condition; set on reset/restore,
                              # tagged onto every obs so the master conditions the
                              # text policy on the RIGHT object (not a global flag).
    obs_horizon = 1
    action_horizon = 8
    action_repeat = 20
    time_limit = 60.0
    max_steps = 15000
    success_bonus = 1.0  # sparse reward on first success; GRPO normalizes by the
                         # group std, so its scale cancels out of the advantage.

    # GRPO: when set by the master via "set_task", subsequent auto-resets at
    # episode end replay the same task (env.reset(freeze=True)). Envs in the
    # same group share a task via identical seed, then diverge via SDE noise.
    freeze_subsequent_resets = False
    require_exact_reset = False
    skip_autoreset_once = False  # set by restore_task: benchmark eval caps at 1 ep/env per cycle, so the end-of-episode auto-reset would be wasted work
    task_seed = 0  # captured at set_task time; re-applied to np.random on every auto-reset so the agent uses identical np.random sequences across group-mates AND across cycles within an env.

    def _do_reset(freeze=False):
        """Reset the env (retrying placement failures) and rebuild all per-episode bookkeeping."""
        nonlocal history, prev_smoothed, gripper_cmd, ep_succeeded, ep_collisions, n_env_steps
        nonlocal current_object_name
        # Robust to occasional "could not find valid placement" failures: keep
        # trying with fresh seeds until one works. Avoids killing the worker
        # (and therefore the master) when one scene/object combo is unlucky.
        _reset_tries = 20
        for attempt in range(_reset_tries):
            try:
                obs, info = env.reset(freeze=freeze)
                break
            except RuntimeError as e:
                if require_exact_reset:
                    raise
                print(f"[worker {rank}] env.reset failed (attempt {attempt+1}/{_reset_tries}): {e}")
                env.np_random = np.random.default_rng(int(np.random.SeedSequence().entropy & 0xFFFFFFFF))
                env._frozen_rng_state = None
        else:
            raise RuntimeError(f"[worker {rank}] env.reset failed {_reset_tries} times in a row, giving up")
        agent.reset(info)
        current_object_name = str(info.get("object_name") or "")
        obs = env._build_obs()
        cams = env.render_cameras()
        for k in cams:
            obs[k] = cams[k].astype(np.float32).transpose(2, 0, 1) / 255.0
        obs["_object_name"] = current_object_name
        history = deque([obs] * obs_horizon, maxlen=obs_horizon)
        prev_smoothed = None
        gripper_cmd = GRIP_OPEN
        ep_succeeded = False
        ep_collisions = 0
        n_env_steps = 0

    def _restore_state(st):
        # Fixed-benchmark eval: exact replay of an export_reset_state() dict.
        # Mirrors _do_reset's post-reset bookkeeping.
        """Exact-replay reset from an exported state dict, mirroring _do_reset bookkeeping."""
        nonlocal history, prev_smoothed, gripper_cmd, ep_succeeded, ep_collisions, n_env_steps
        nonlocal current_object_name
        obs, info = env.restore_reset_state(st)
        agent.reset(info)
        current_object_name = str(st.get("object_name") or info.get("object_name") or "")
        obs = env._build_obs()
        cams = env.render_cameras()
        for k in cams:
            obs[k] = cams[k].astype(np.float32).transpose(2, 0, 1) / 255.0
        obs["_object_name"] = current_object_name
        history = deque([obs] * obs_horizon, maxlen=obs_horizon)
        prev_smoothed = None
        gripper_cmd = GRIP_OPEN
        ep_succeeded = False
        ep_collisions = 0
        n_env_steps = 0


    try:
        while True:
            cmd, payload = conn.recv()
            if cmd == "init":
                obs_horizon = int(payload["obs_horizon"])
                action_horizon = int(payload["action_horizon"])
                action_repeat = int(payload["action_repeat"])
                time_limit = float(payload["time_limit"])
                max_steps = int(payload["max_steps"])
                if payload.get("reset", True):
                    _do_reset()
                conn.send(("ready", {"obs_history": list(history) if history is not None else []}))
            elif cmd == "set_time_limit":
                # Run the deterministic eval at a different per-episode limit than
                # training rollouts (workers are reused, so init's time_limit
                # sticks unless we update it here). Master restores it after eval.
                time_limit = float(payload["time_limit"])
                conn.send(("ok", {}))
            elif cmd == "set_task":
                # Group-shared seed, so this env draws the same task as its
                # mates. freeze_subsequent replays it on every auto-reset.
                seed = int(payload["seed"])
                freeze_subsequent_resets = bool(payload.get("freeze_subsequent", True))
                # Exact replay is only needed when mates share a frozen task.
                # Eval episodes are independent, so an unplaceable one resamples
                # rather than killing the worker.
                require_exact_reset = freeze_subsequent_resets
                # Unplaceable seeds advance deterministically so mates fail
                # identically and stay on the same task.
                def _seed_env(ts):
                    """Seed env + global numpy RNGs identically so GRPO group-mates share task and controller draws."""
                    env.np_random = np.random.default_rng(ts)
                    # G1Controller uses the GLOBAL np.random, not env.np_random,
                    # so seed both or mates diverge in controller behaviour even
                    # from identical env state.
                    np.random.seed(ts)
                    env._frozen_rng_state = None
                    env._frozen_obj_idx = None
                    env._reset_counter = 0
                    # From the group seed alone: the previous scene's object count
                    # differs across mates, and select_target takes this modulo the
                    # new scene's list.
                    if hasattr(env.task, "_obj_idx"):
                        env.task._obj_idx = int(np.random.default_rng(ts).integers(0, 2**31 - 1))

                _last_err = None
                # Generous retry budget: some scene/object pairs are unplaceable,
                # and exhausting it kills the worker and the run with it.
                _n_attempts = 40
                for _attempt in range(_n_attempts):
                    task_seed = seed if _attempt == 0 else seed + _attempt * 100003
                    _seed_env(task_seed)
                    try:
                        _do_reset(freeze=False)  # snapshots this seed's pre-reset state
                    except RuntimeError as e:
                        _last_err = e
                        print(f"[worker {rank}] set_task seed {task_seed} unplaceable "
                              f"(attempt {_attempt + 1}/{_n_attempts}); advancing deterministically", flush=True)
                        continue
                    _last_err = None
                    break
                if _last_err is not None:
                    raise _last_err
                conn.send(("ready", {"obs_history": list(history)}))
            elif cmd == "restore_task":
                # Fixed-benchmark eval: restore an exported reset state exactly.
                # Seed global np.random so agent.reset's draws (speed, yaw offset,
                # grasp speed scale) are identical across eval passes.
                np.random.seed(int(payload.get("seed", 0)))
                freeze_subsequent_resets = False
                require_exact_reset = False
                skip_autoreset_once = True
                _restore_state(payload["state"])
                conn.send(("ready", {"obs_history": list(history)}))
            elif cmd == "stop":
                try: env.close()
                except Exception: pass
                break
            elif cmd == "execute":
                # payload["action_seq"]: shape (action_horizon, 15), anchor-added,
                # gripper binarized. We do per-substep interp + gripper ramp here.
                action_seq = payload["action_seq"]
                n_chunk = min(action_horizon, action_seq.shape[0])

                # Build chunk_start
                if prev_smoothed is not None:
                    chunk_start = prev_smoothed.copy()
                else:
                    last_obs = history[-1]
                    chunk_start = np.zeros(15, dtype=np.float32)
                    chunk_start[3:14] = anchor_from_obs(last_obs)
                    chunk_start[14] = float(last_obs["right_gripper_pos"][0])

                transitions = []
                episode_finished = False
                episode_success_final = False
                _t_step_total = 0.0
                _t_render_total = 0.0

                for i in range(n_chunk):
                    a_curr = chunk_start if i == 0 else action_seq[i - 1]
                    a_next = action_seq[i]
                    total_r_chunk = 0.0  # accumulated env reward (we don't use it for sparse; tracked for diagnostic)
                    success_chunk = False
                    term = trunc = False
                    obs = None
                    n_exec = 0
                    for k in range(action_repeat):
                        alpha = k / max(action_repeat - 1, 1)
                        a = (1 - alpha) * a_curr + alpha * a_next
                        target = float(a[14])
                        if target > gripper_cmd:
                            gripper_cmd = min(gripper_cmd + GRIP_STEP, target)
                        else:
                            gripper_cmd = max(gripper_cmd - GRIP_STEP, target)
                        a = a.copy(); a[14] = gripper_cmd
                        prev_smoothed = a.copy()
                        _t0 = time.perf_counter()
                        obs, r, term, trunc, info = env.step(a)
                        _t_step_total += time.perf_counter() - _t0
                        agent.set_step_info(info)
                        total_r_chunk += float(r)
                        if bool(info.get("success", False)):
                            success_chunk = True
                        n_env_steps += 1
                        n_exec += 1
                        if term or trunc or success_chunk:
                            break
                    # Render once at the end of this chunk-step
                    _t1 = time.perf_counter()
                    cams = env.render_cameras()
                    for k_cam in cams:
                        obs[k_cam] = cams[k_cam].astype(np.float32).transpose(2, 0, 1) / 255.0
                    obs["_object_name"] = current_object_name
                    _t_render_total += time.perf_counter() - _t1
                    history.append(obs)

                    # Sparse success bonus: scaled by success_bonus on first hit.
                    sparse_r = 0.0
                    if success_chunk and not ep_succeeded:
                        sparse_r = success_bonus
                        ep_succeeded = True

                    # Collisions are tracked for logging only (cheap contact
                    # query); the reward itself is sparse.
                    if bool(env._robot_touches_world()):
                        ep_collisions += 1
                    total_r = sparse_r

                    sim_time = float(env.time)
                    chunk_done = bool(
                        term or trunc or success_chunk
                        or sim_time >= time_limit
                        or n_env_steps >= max_steps
                    )
                    transitions.append({
                        "reward": float(total_r),
                        "done": bool(chunk_done),
                        "success": bool(success_chunk),
                        "time": sim_time,
                        "n_executed": int(n_exec),
                        "_sparse": float(sparse_r),
                    })
                    if chunk_done:
                        episode_finished = True
                        episode_success_final = bool(ep_succeeded)
                        break

                payload_out = {
                    "transitions": transitions,
                    "episode_finished": episode_finished,
                    "episode_success": episode_success_final,
                    "episode_collisions": int(ep_collisions) if episode_finished else None,
                    "obs_history": None,  # populated below
                    "_t_step": _t_step_total,
                    "_t_render": _t_render_total,
                }

                if episode_finished:
                    if skip_autoreset_once:
                        # Benchmark eval runs exactly one episode per restore;
                        # the next restore_task rebuilds state, so a random
                        # auto-reset here would only waste a scene reload.
                        skip_autoreset_once = False
                    else:
                        # Re-seed so the next episode's agent.reset replays the
                        # same global-RNG sequence as its mates.
                        if freeze_subsequent_resets:
                            np.random.seed(task_seed)
                        _do_reset(freeze=freeze_subsequent_resets)
                payload_out["obs_history"] = list(history)
                conn.send(("done", payload_out))
    except Exception as e:
        import traceback; traceback.print_exc()
        try: conn.send(("error", {"err": repr(e)}))
        except Exception: pass


class VecEnv:
    """Asynchronous vec-env. Each worker runs its own chunk-batch loop and
    reports back when done. The master polls connections and batches policy
    fwds across whichever workers happen to be ready. Slow workers no longer
    block fast ones."""

    def __init__(self, env_cfg_dict: dict, n_envs: int, render_gpus, seed: int = 0,
                 per_worker_overrides=None):
        """`per_worker_overrides`: optional list of length n_envs, each entry a
        dict merged into the worker's env config. Used to assign a persistent
        scene to each worker so we get scene diversity without paying the
        scene-reload cost on every env reset."""
        self.n_envs = n_envs
        ctx = get_context("spawn")
        self._parents, self._procs = [], []
        egl_ids = cuda_to_egl([int(g) for g in render_gpus])
        for i in range(n_envs):
            parent, child = ctx.Pipe(duplex=True)
            gpu = egl_ids[i % len(egl_ids)]
            cfg_i = dict(env_cfg_dict)
            if per_worker_overrides is not None and i < len(per_worker_overrides):
                cfg_i.update(per_worker_overrides[i])
            p = ctx.Process(target=_env_worker,
                            args=(i, child, cfg_i, gpu, seed * 100 + i),
                            daemon=True)
            p.start()
            child.close()
            self._parents.append(parent)
            self._procs.append(p)

    def init_workers(self, obs_horizon, action_horizon, action_repeat,
                     time_limit, max_steps, use_delta, reset=True):
        """Broadcast rollout hyperparameters to all workers and collect their ready payloads.

        reset=False skips the sampled reset: a worker that will only replay
        restored states never needs one, and the env config it boots with may
        not even be able to place a robot.
        """
        for c in self._parents:
            c.send(("init", {
                "obs_horizon": obs_horizon,
                "action_horizon": action_horizon,
                "action_repeat": action_repeat,
                "time_limit": time_limit,
                "max_steps": max_steps,
                "use_delta": use_delta,
                "reset": reset,
            }))
        # Collect initial "ready" messages. The pipes MAY contain stale "done"
        # messages from a previous rollout's still-executing workers; discard
        # those and keep reading until we see "ready" from every worker.
        out = [None] * len(self._parents)
        remaining = list(range(len(self._parents)))
        while remaining:
            ready_conns = mp_wait([self._parents[i] for i in remaining])
            for c in ready_conns:
                i = self._parents.index(c)
                cmd, payload = c.recv()
                if cmd != "ready":
                    continue  # stale message from prior rollout; drain and wait for the real "ready"
                out[i] = payload
                remaining.remove(i)
        return out  # list of {"obs_history": [...]}

    def set_tasks(self, seeds_per_env, freeze_subsequent=True):
        """GRPO: broadcast a per-env seed and reset every worker's env to that
        task. Envs sharing a seed (= envs in the same group) land on the same
        task and diverge only via SDE noise. `freeze_subsequent`: future
        auto-resets at episode end replay the same task (so all episodes from
        env i within this collection come from env i's group's task)."""
        self.set_tasks_async(seeds_per_env, freeze_subsequent=freeze_subsequent)
        return self.wait_set_tasks_ready()

    def set_benchmark_tasks(self, states_per_env, seeds_per_env):
        """Fixed-benchmark eval: each worker restores an exported reset state
        (main.py --save_json format) via env.restore_reset_state. Synchronous."""
        assert len(states_per_env) == len(self._parents)
        if getattr(self, "_pending_set_tasks", False):
            self.wait_set_tasks_ready()
        for c, st, s in zip(self._parents, states_per_env, seeds_per_env):
            c.send(("restore_task", {"state": st, "seed": int(s)}))
        payloads = []
        for i, c in enumerate(self._parents):
            cmd, p = c.recv()
            if cmd != "ready":
                raise RuntimeError(f"worker {i} restore_task failed: {cmd} {p}")
            payloads.append(p)
        return payloads

    def set_time_limit(self, tl):
        """Update every worker's per-episode time_limit. Used to run the
        deterministic benchmark eval at --eval_time_limit while training
        rollouts keep --time_limit (workers are reused across both)."""
        # Drain in-flight async set_tasks: their "ready" replies are still in
        # the pipes and would be misread as our "ok".
        if getattr(self, "_pending_set_tasks", False):
            self.wait_set_tasks_ready()
        for c in self._parents:
            c.send(("set_time_limit", {"time_limit": float(tl)}))
        for i, c in enumerate(self._parents):
            # Tolerate any residual stale message in the pipe: read until our
            # own "ok" arrives (the worker handles commands in order, so after
            # any leftover "ready"/"done" it will emit the "ok").
            while True:
                cmd, _ = c.recv()
                if cmd == "ok":
                    break

    def set_tasks_async(self, seeds_per_env, freeze_subsequent=True):
        """Non-blocking variant: dispatch set_task to all workers and return.
        Call wait_set_tasks_ready() to drain the responses before next use.
        Lets the update overlap with worker reset (scene reload + snapshot)."""
        assert len(seeds_per_env) == len(self._parents)
        if getattr(self, "_pending_set_tasks", False):
            # Stale prefetch (e.g. interrupted by an eval that reused these workers).
            self.wait_set_tasks_ready()
        for c, s in zip(self._parents, seeds_per_env):
            c.send(("set_task", {"seed": int(s), "freeze_subsequent": bool(freeze_subsequent)}))
        self._pending_set_tasks = True

    def wait_set_tasks_ready(self):
        """Drain pending set_task 'ready' responses. Returns the per-env payloads
        (obs_history) in env order."""
        if not getattr(self, "_pending_set_tasks", False):
            return None
        out = [None] * len(self._parents)
        remaining = list(range(len(self._parents)))
        while remaining:
            ready_conns = mp_wait([self._parents[i] for i in remaining])
            for c in ready_conns:
                i = self._parents.index(c)
                cmd, payload = c.recv()
                if cmd != "ready":
                    continue
                out[i] = payload
                remaining.remove(i)
        self._pending_set_tasks = False
        return out

    def send_chunk(self, idx: int, action_seq):
        """Dispatch a chunk-batch to worker idx. Non-blocking on the worker."""
        self._parents[idx].send(("execute", {"action_seq": action_seq}))

    def poll_ready(self, executing_idx_set, timeout: float = None):
        """Block until at least one worker in `executing_idx_set` has a
        message ready; return list of (idx, payload) pairs for ALL workers
        that have messages waiting. Master can then process them and decide
        whether to wait for more or dispatch immediately."""
        if not executing_idx_set:
            return []
        conns = [self._parents[i] for i in executing_idx_set]
        ready_conns = mp_wait(conns, timeout=timeout)
        out = []
        for c in ready_conns:
            i = self._parents.index(c)
            cmd, payload = c.recv()
            out.append((i, cmd, payload))
        return out

    def close(self):
        # Parallel SIGTERM, not a sequential graceful join: workers are usually
        # mid-MJCF-compile and never return within a timeout, which made close
        # cost n_workers * 3s.
        """SIGTERM all workers in parallel and reap them (fast discard, no graceful join)."""
        for p in self._procs:
            try:
                if p.is_alive(): p.terminate()
            except Exception: pass
        for p in self._procs:
            p.join(timeout=5.0)
            if p.is_alive():
                try: p.kill()
                except Exception: pass
                p.join(timeout=2.0)
        for c in self._parents:
            try: c.close()
            except Exception: pass


# Rollout collection
def _obs_to_action(obs):
    """Recover the 15-dim action layout from an observation (height, waist, arm, gripper)."""
    jp = obs["joint_pos"]
    a = np.zeros(15, dtype=np.float32)
    a[3] = obs["base_height"][0]; a[4:7] = jp[12:15]; a[7:14] = jp[22:29]
    a[14] = obs["right_gripper_pos"][0]
    return a
