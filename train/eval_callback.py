from __future__ import annotations

import os
from collections import deque
from multiprocessing import get_context
from pathlib import Path
import sys

import numpy as np
import torch
from lightning.pytorch.callbacks import Callback
from tqdm import tqdm
from lightning.pytorch.callbacks import ModelCheckpoint

ROOT_DIR = Path(__file__).resolve().parent.parent
TRAIN_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(TRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(TRAIN_DIR))

from molmospaces.env import make_env
from molmospaces.agents.policy import G1Controller as RolloutController
from utils.action_transforms import DELTA_DIMS, anchor_from_obs, target_point_xyv

GRIP_OPEN, GRIP_CLOSED = -0.0222, 0.0245
# 95% of the way to closed (not the 50% midpoint): require a more committed
# close signal so the policy does not grasp too early.
GRIP_MID = GRIP_OPEN + 0.95 * (GRIP_CLOSED - GRIP_OPEN)
# Ramp the commanded gripper over this many env steps instead of snapping.
GRIP_STEP = (GRIP_CLOSED - GRIP_OPEN) / 400


def _eval_worker_main(rank, conn, env_cfg_dict, egl_device_id, time_limit, seed, dataset_fps):
    """Env-only eval worker process: owns one env, steps on command, streams observations back."""
    # EGL must be configured before mujoco is imported.
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(egl_device_id)
    os.environ["EGL_DEVICE_ID"] = str(egl_device_id)
    os.environ["PYOPENGL_PLATFORM"] = "egl"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    import sys as _sys
    from pathlib import Path as _Path
    _HERE = _Path(__file__).resolve().parent
    for _p in (str(_HERE.parent), str(_HERE)):
        if _p not in _sys.path:
            _sys.path.insert(0, _p)

    import numpy as _np
    from collections import deque as _deque
    from molmospaces.env import make_env as _make_env
    from molmospaces.agents.policy import G1Controller as _RolloutController
    from utils.action_transforms import DELTA_DIMS as _DELTA_DIMS, anchor_from_obs as _anchor_from_obs

    cfg = dict(env_cfg_dict)
    cfg["seed"] = int(seed)
    cfg["launch_viewer"] = False

    env = _make_env(cfg)
    agent = _RolloutController()
    agent.setup(env.scene.model, env.scene.data)
    agent.set_env(env)
    env.set_agent(agent)

    # env.step advances n_substeps mj_steps, so the relevant rate for chunk
    # execution is env.step rate, not mj_step rate.
    step_dt = env.robot.n_substeps * env.scene.model.opt.timestep
    step_rate = 1.0 / step_dt
    image_keys = list(env.cameras.keys())
    action_repeat = max(1, int(round(step_rate / float(dataset_fps))))

    obs_horizon = 2
    action_horizon = 8
    history = _deque(maxlen=obs_horizon)
    prev_smoothed = None
    gripper_cmd = GRIP_OPEN
    last_obs = None
    episode_steps = 0  # chunk-position count, mirrors single-worker semantics

    def augment(o):
        """Attach rendered camera images (CHW float [0,1]) to an observation."""
        cams = env.render_cameras()
        out = dict(o)
        for k in image_keys:
            out[k] = cams[k].astype(_np.float32).transpose(2, 0, 1) / 255.0
        return out

    current_task = ""

    def do_reset():
        """Reset with placement-failure retries and rebuild the observation history."""
        nonlocal prev_smoothed, history, gripper_cmd, last_obs, episode_steps, current_task
        # Eval episodes are independent, so a rare "could not find valid
        # placement" should resample with a fresh seed rather than kill the
        # worker (which otherwise raises and takes down the whole training run).
        obs = info = None
        for _attempt in range(20):
            try:
                obs, info = env.reset()
                break
            except RuntimeError as _e:
                env.np_random = _np.random.default_rng(
                    int(_np.random.SeedSequence().entropy & 0xFFFFFFFF))
                env._frozen_rng_state = None
        if obs is None:
            raise RuntimeError("env.reset failed 20x in a row (scene unplaceable)")
        current_task = str(info.get("object_name") or "")
        agent.reset(info)
        # agent.reset rewrites leg defaults, shifting pelvis height. The
        # anchor must come from a refreshed obs, not env.reset()'s return.
        last_obs = env._build_obs()
        aug = augment(last_obs)
        history = _deque([aug] * obs_horizon, maxlen=obs_horizon)
        prev_smoothed = None
        gripper_cmd = GRIP_OPEN
        episode_steps = 0

    try:
        while True:
            cmd, payload = conn.recv()
            if cmd == "init":
                obs_horizon = int(payload["obs_horizon"])
                action_horizon = int(payload["action_horizon"])
                use_action_delta = bool(payload.get("use_action_delta", True))
                history = _deque(maxlen=obs_horizon)
                if not payload.get("no_reset"):
                    do_reset()
                conn.send(("ready", {"history": list(history), "task": current_task}))

            elif cmd == "reset":
                do_reset()
                conn.send(("reset_done", {"history": list(history), "task": current_task}))

            elif cmd == "restore":
                # Exact replay of an export_reset_state() dict (fixed eval sets).
                obs, info = env.restore_reset_state(payload["state"])
                current_task = str(info.get("object_name") or payload["state"].get("object_name") or "")
                agent.reset(info)
                last_obs = env._build_obs()
                aug = augment(last_obs)
                history = _deque([aug] * obs_horizon, maxlen=obs_horizon)
                prev_smoothed = None
                gripper_cmd = GRIP_OPEN
                episode_steps = 0
                conn.send(("restore_done", {"history": list(history), "task": current_task}))

            elif cmd == "step":
                action_seq = payload["action_seq"].copy()
                if use_action_delta:
                    action_seq[:, _DELTA_DIMS] += _anchor_from_obs(last_obs)
                action_seq[:, 14] = _np.where(action_seq[:, 14] > GRIP_MID, GRIP_CLOSED, GRIP_OPEN)

                done = False
                success = False
                n_chunk = min(action_horizon, len(action_seq))
                if prev_smoothed is not None:
                    chunk_start = prev_smoothed.copy()
                else:
                    chunk_start = _np.zeros(15, dtype=_np.float32)
                    chunk_start[3:14] = _anchor_from_obs(last_obs)
                    chunk_start[14] = last_obs["right_gripper_pos"][0]
                for i in range(n_chunk):
                    if i == 0:
                        a_curr = chunk_start
                        a_next = action_seq[0]
                    else:
                        a_curr = action_seq[i - 1]
                        a_next = action_seq[i]
                    for k in range(action_repeat):
                        alpha = k / max(action_repeat - 1, 1)
                        action = (1 - alpha) * a_curr + alpha * a_next
                        prev_smoothed = action.copy()
                        target = action[14]
                        if target > gripper_cmd:
                            gripper_cmd = min(gripper_cmd + GRIP_STEP, target)
                        else:
                            gripper_cmd = max(gripper_cmd - GRIP_STEP, target)
                        prev_smoothed[14] = gripper_cmd
                        obs, _, terminated, truncated, info = env.step(prev_smoothed)
                        agent.set_step_info(info)
                        last_obs = obs
                        # Break on success so it's reported on the same chunk;
                        # env.step terminates at reward > 0.1 but info["success"]
                        # flips at 0.04, and a brief success can otherwise be dropped.
                        success = success or bool(info.get("success", False))
                        if terminated or truncated or success or env.time >= time_limit:
                            done = True
                            break
                    history.append(augment(obs))
                    episode_steps += 1
                    if done:
                        break

                conn.send(("step_done", {
                    "history": list(history),
                    "done": done,
                    "success": success,
                    "episode_steps": episode_steps,
                }))

            elif cmd == "close":
                env.close()
                conn.send(("closed", {}))
                break
    except Exception as e:
        import traceback
        traceback.print_exc()
        try:
            conn.send(("error", {"err": repr(e)}))
        except Exception:
            pass
    finally:
        try:
            env.close()
        except Exception:
            pass


def _obs_to_action(obs):
    """Recover the 15-dim action layout from an observation (height, waist, arm, gripper)."""
    jp = obs['joint_pos']
    action = np.zeros(15, dtype=np.float32)
    action[3] = obs['base_height'][0]
    action[4:7] = jp[12:15]
    action[7:14] = jp[22:29]
    action[14] = obs['right_gripper_pos'][0]
    return action


class EnvEvalCallback(Callback):
    def __init__(
        self,
        env_config,
        every_n_steps: int,
        episodes: int,
        time_limit: float = 60.0,
        num_envs: int = 16,
        render_gpu=-1,
        dataset_fps: int = 10,
        eval_at_resume_start: bool = True,
        episodes_json: str | None = None,
        eval_seed: int = 777000,
    ):
        """In-training eval: rolls the policy in env workers every N steps and logs success rate."""
        self._env_config = env_config
        self._every_n_steps = every_n_steps
        self._episodes = episodes
        self._eval_seed = int(eval_seed)
        self._time_limit = time_limit
        self._num_envs = int(num_envs)
        # Fixed benchmark: replay this JSONL of exported reset states every eval
        # (consistent numbers across runs) instead of random env.reset() episodes.
        self._episodes_json = None
        if episodes_json:
            p = Path(episodes_json)
            if p.is_file():
                self._episodes_json = str(p)
            else:
                print(f"[eval] episodes_json not found ({p}); using random-reset eval", flush=True)
        self._dataset_fps = int(dataset_fps)
        if isinstance(render_gpu, (list, tuple)):
            self._render_gpus = [int(g) for g in render_gpu] or [-1]
        else:
            self._render_gpus = [int(render_gpu)]
        self._last_eval_step = -1
        self._eval_at_resume_start = bool(eval_at_resume_start)

    def on_train_start(self, trainer, pl_module):
        """Log a zero baseline at step 0, or run a resume-time eval when configured."""
        if trainer.global_rank != 0:
            return
        if self._every_n_steps <= 0 or self._episodes <= 0:
            return
        step = trainer.global_step
        if step > 0:
            # Resuming from a checkpoint: optionally run one eval pass before training.
            if not self._eval_at_resume_start:
                return
            self._last_eval_step = step
            metrics = self._evaluate(pl_module, step)
            if trainer.logger is not None:
                trainer.logger.log_metrics(metrics, step=step)
            for key, value in metrics.items():
                trainer.callback_metrics[key] = torch.tensor(value, device=pl_module.device)
            return
        self._last_eval_step = 0
        metrics = {"eval/success_rate": 0.0, "eval/mean_steps": 0.0}
        if trainer.logger is not None:
            trainer.logger.log_metrics(metrics, step=0)
        for key, value in metrics.items():
            trainer.callback_metrics[key] = torch.tensor(value, device=pl_module.device)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Every N global steps: checkpoint, run eval, and log its metrics."""
        if trainer.global_rank != 0:
            return
        step = trainer.global_step
        if self._every_n_steps <= 0 or self._episodes <= 0:
            return
        if step <= 0 or step == self._last_eval_step or step % self._every_n_steps != 0:
            return
        self._last_eval_step = step

        for cb in trainer.callbacks:
            if isinstance(cb, ModelCheckpoint):
                try:
                    cb.on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)
                except Exception as e:
                    print(f"[eval] pre-eval checkpoint save failed: {type(e).__name__}: {e}")

        ema_cb = next(
            (cb for cb in trainer.callbacks
             if hasattr(cb, "_average_model") and hasattr(cb, "_swap_models")
             and getattr(cb, "_average_model", None) is not None),
            None,
        )
        swapped_ema = False
        if ema_cb is not None:
            ema_cb._swap_models(pl_module)
            swapped_ema = True
        # The flow sampler draws its starting noise from the global torch RNG
        # (FlowPolicy.sample_actions), which nothing seeds, so each eval scored the
        # policy plus a fresh random draw. Measured on the RL side of this codebase:
        # the same checkpoint returned 0.36-0.56 across identical runs and curves
        # swung +-11 points with no learning. Pinning it makes an eval measure the
        # policy. fork_rng keeps that fixed seed out of the training stream, which
        # would otherwise restart from the same point after every eval.
        rng_devices = [pl_module.device] if pl_module.device.type == "cuda" else []
        try:
            with torch.random.fork_rng(devices=rng_devices):
                torch.manual_seed(self._eval_seed)
                metrics = self._evaluate(pl_module, step)
        finally:
            if swapped_ema:
                ema_cb._swap_models(pl_module)
        if trainer.logger is not None:
            trainer.logger.log_metrics(metrics, step=step)
        for key, value in metrics.items():
            trainer.callback_metrics[key] = torch.tensor(value, device=pl_module.device)

    @torch.no_grad()
    def _evaluate(self, pl_module, step: int) -> dict[str, float]:
        """Dispatch to fixed-benchmark, parallel, or single-env evaluation."""
        if self._episodes_json and self._num_envs > 1:
            return self._evaluate_fixed(pl_module, step)
        if self._num_envs > 1:
            return self._evaluate_parallel(pl_module, step)
        return self._evaluate_single(pl_module, step)

    @torch.no_grad()
    def _egl_device_ids(self, pl_module, num_envs: int) -> list[int]:
        """One EGL device id per worker, resolving render_gpu=-1 to the trainer's GPU."""
        from molmospaces.egl import cuda_to_egl
        visible_idx = pl_module.device.index if pl_module.device.type == "cuda" else 0
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        ids = [int(x) for x in cvd.split(",") if x.strip().isdigit()] if cvd else []
        auto = ids[visible_idx] if 0 <= visible_idx < len(ids) else visible_idx
        pool = cuda_to_egl([(auto if g < 0 else int(g)) for g in self._render_gpus])
        return [pool[rank % len(pool)] for rank in range(num_envs)]

    def _base_env_cfg(self) -> dict:
        """Env config as a plain dict with the viewer off, ready to send to a worker."""
        env_cfg = self._env_config.copy_and_resolve_references()
        env_cfg["launch_viewer"] = False
        try:
            return env_cfg.to_dict()
        except AttributeError:
            return dict(env_cfg)

    def _spawn_workers(self, per_worker_cfgs, egl_device_ids):
        """Start one env worker per config; returns (parent connections, processes)."""
        ctx = get_context("spawn")
        parents, workers = [], []
        for rank, cfg in enumerate(per_worker_cfgs):
            parent_conn, child_conn = ctx.Pipe(duplex=True)
            p = ctx.Process(
                target=_eval_worker_main,
                args=(rank, child_conn, cfg, egl_device_ids[rank],
                      self._time_limit, rank, self._dataset_fps),
                daemon=True,
            )
            p.start()
            workers.append(p)
            parents.append(parent_conn)
            child_conn.close()
        return parents, workers

    def _horizons(self, pl_module):
        """(observation horizon, action horizon, flow-sampling schedule) for this policy."""
        return (
            int(pl_module.config["observation_horizon"]),
            int(pl_module.config["action_horizon"]),
            torch.linspace(0.0, 1.0, int(pl_module.config["num_sampling_steps"]),
                           device=pl_module.device),
        )

    def _evaluate_fixed(self, pl_module, step: int) -> dict[str, float]:
        """Replay the fixed JSON benchmark across self._num_envs env workers with one shared model doing batched inference."""
        import json
        from molmospaces import ASSETS_DIR

        with open(self._episodes_json) as f:
            eps = [json.loads(line) for line in f if line.strip()]
        n = len(eps)
        if n == 0:
            return {"eval/success_rate": 0.0, "eval/mean_steps": 0.0}
        num_envs = max(1, min(self._num_envs, n))

        egl_device_ids = self._egl_device_ids(pl_module, num_envs)
        env_cfg_dict = self._base_env_cfg()

        # Contiguous blocks; each worker boots into its block's first scene.
        blocks = [[int(i) for i in b] for b in np.array_split(np.arange(n), num_envs)]
        per_worker_cfgs = []
        for block in blocks:
            cfg = dict(env_cfg_dict)
            scene = eps[block[0]]["scene"]
            cfg["scene"] = scene if Path(scene).is_absolute() else str(ASSETS_DIR / scene)
            cfg["randomize_scene"] = False
            per_worker_cfgs.append(cfg)

        obs_horizon, action_horizon, schedule = self._horizons(pl_module)

        parents, workers = self._spawn_workers(per_worker_cfgs, egl_device_ids)

        was_training = pl_module.training
        pl_module.eval()
        results: dict[int, bool] = {}
        step_counts: list[float] = []

        queues = [list(b) for b in blocks]
        current = [None] * num_envs
        histories: list = [None] * num_envs
        tasks = [""] * num_envs

        def send_restore(i):
            """Send the next benchmark episode to worker i for exact replay."""
            idx = queues[i].pop(0)
            current[i] = idx
            parents[i].send(("restore", {"state": eps[idx]}))

        def recv_restore(i):
            """Receive worker i's restore ack and seed its observation history."""
            cmd, payload = parents[i].recv()
            if cmd != "restore_done":
                raise RuntimeError(f"eval worker {i} restore of episode {current[i]} failed: {cmd} {payload}")
            histories[i] = payload["history"]
            tasks[i] = payload.get("task", "")

        try:
            for c in parents:
                c.send(("init", {
                    "obs_horizon": obs_horizon,
                    "action_horizon": action_horizon,
                    "use_action_delta": bool(pl_module.config.get("use_action_delta", True)),
                    "no_reset": True,
                }))
            for i, c in enumerate(parents):
                cmd, payload = c.recv()
                if cmd != "ready":
                    raise RuntimeError(f"eval worker {i} init failed: {cmd} {payload}")

            for i in range(num_envs):
                send_restore(i)
            for i in range(num_envs):
                recv_restore(i)
            active = list(range(num_envs))

            iter_bar = tqdm(total=n, desc=f"eval step {step}", unit="ep", leave=False)
            while active:
                batch = self._build_obs_batch_parallel([histories[i] for i in active], pl_module,
                                                       tasks=[tasks[i] for i in active])
                action_seq = pl_module.sample_actions(batch, schedule, batch_size=len(active))
                action_seq_np = action_seq.detach().cpu().numpy()
                for k, i in enumerate(active):
                    parents[i].send(("step", {"action_seq": action_seq_np[k]}))
                done_workers = []
                for k, i in enumerate(active):
                    cmd, payload = parents[i].recv()
                    if cmd != "step_done":
                        raise RuntimeError(f"eval worker {i} step failed: {cmd} {payload}")
                    histories[i] = payload["history"]
                    if payload["done"]:
                        results[current[i]] = bool(payload["success"])
                        step_counts.append(float(payload["episode_steps"]))
                        iter_bar.update(1)
                        iter_bar.set_postfix(
                            succ=f"{int(sum(results.values()))}/{len(results)}",
                            rate=f"{np.mean(list(results.values())):.0%}",
                        )
                        done_workers.append(i)
                for i in done_workers:
                    if queues[i]:
                        send_restore(i)
                    else:
                        current[i] = None
                        active.remove(i)
                for i in done_workers:
                    if current[i] is not None:
                        recv_restore(i)
            iter_bar.close()
        finally:
            if was_training:
                pl_module.train()
            for c in parents:
                try:
                    c.send(("close", {}))
                except Exception:
                    pass
            for p in workers:
                p.join(timeout=5.0)
                if p.is_alive():
                    p.terminate()

        succ = list(results.values())
        return {
            "eval/success_rate": float(np.mean(succ)) if succ else 0.0,
            "eval/mean_steps": float(np.mean(step_counts)) if step_counts else 0.0,
        }

    @torch.no_grad()
    def _evaluate_parallel(self, pl_module, step: int) -> dict[str, float]:
        """Random-reset eval across workers with one shared model doing batched inference."""

        egl_device_ids = self._egl_device_ids(pl_module, self._num_envs)
        env_cfg_dict = self._base_env_cfg()

        # A .txt scene list is dealt out round-robin, so --eval_num_envs equal to
        # the number of scenes covers each one exactly once.
        per_worker_cfgs = [dict(env_cfg_dict) for _ in range(self._num_envs)]
        scene_field = env_cfg_dict.get("scene", "")
        if isinstance(scene_field, str) and scene_field.endswith(".txt"):
            from molmospaces.env import _resolve_scene_paths
            resolved = _resolve_scene_paths(scene_field)
            if resolved:
                for rank in range(self._num_envs):
                    per_worker_cfgs[rank]["scene"] = str(resolved[rank % len(resolved)])
                    per_worker_cfgs[rank]["randomize_scene"] = False

        obs_horizon, action_horizon, schedule = self._horizons(pl_module)

        parents, workers = self._spawn_workers(per_worker_cfgs, egl_device_ids)

        was_training = pl_module.training
        pl_module.eval()
        successes: list[float] = []
        episode_step_counts: list[float] = []

        try:
            for c in parents:
                c.send(("init", {
                    "obs_horizon": obs_horizon,
                    "action_horizon": action_horizon,
                    "use_action_delta": bool(pl_module.config.get("use_action_delta", True)),
                }))
            histories: list = [None] * self._num_envs
            tasks = [""] * self._num_envs
            for i, c in enumerate(parents):
                cmd, payload = c.recv()
                if cmd != "ready":
                    raise RuntimeError(f"eval worker {i} init failed: {cmd} {payload}")
                histories[i] = payload["history"]
                tasks[i] = payload.get("task", "")

            per_env_target = (self._episodes + self._num_envs - 1) // self._num_envs
            per_env_count = [0] * self._num_envs
            active = list(range(self._num_envs))
            total_target = per_env_target * self._num_envs

            iter_bar = tqdm(total=total_target, desc=f"eval step {step}", unit="ep", leave=False)
            while active:
                active_histories = [histories[i] for i in active]
                batch = self._build_obs_batch_parallel(active_histories, pl_module,
                                                       tasks=[tasks[i] for i in active])
                action_seq = pl_module.sample_actions(batch, schedule, batch_size=len(active))
                action_seq_np = action_seq.detach().cpu().numpy()

                for k, i in enumerate(active):
                    parents[i].send(("step", {"action_seq": action_seq_np[k]}))

                done_set: set[int] = set()
                for k, i in enumerate(active):
                    cmd, payload = parents[i].recv()
                    if cmd != "step_done":
                        raise RuntimeError(f"eval worker {i} step failed: {cmd} {payload}")
                    histories[i] = payload["history"]
                    if payload["done"]:
                        done_set.add(i)
                        per_env_count[i] += 1
                        successes.append(1.0 if payload["success"] else 0.0)
                        episode_step_counts.append(float(payload["episode_steps"]))
                        iter_bar.update(1)
                        iter_bar.set_postfix(
                            succ=f"{int(sum(successes))}/{len(successes)}",
                            rate=f"{np.mean(successes):.0%}",
                        )

                reset_idx: list[int] = []
                new_active: list[int] = []
                for i in active:
                    if i not in done_set:
                        new_active.append(i)
                        continue
                    if per_env_count[i] >= per_env_target:
                        continue
                    reset_idx.append(i)
                    new_active.append(i)
                for i in reset_idx:
                    parents[i].send(("reset", {}))
                failed = set()
                for i in reset_idx:
                    cmd, payload = parents[i].recv()
                    if cmd != "reset_done":
                        # Worker couldn't reset (e.g. unplaceable scene). Drop it
                        # from this eval pass instead of crashing the whole run.
                        print(f"[eval] worker {i} reset failed, dropping it: {cmd} {payload}", flush=True)
                        failed.add(i)
                        continue
                    histories[i] = payload["history"]
                    tasks[i] = payload.get("task", "")
                active = [i for i in new_active if i not in failed]
            iter_bar.close()
        finally:
            if was_training:
                pl_module.train()
            for c in parents:
                try:
                    c.send(("close", {}))
                except Exception:
                    pass
            for p in workers:
                p.join(timeout=5.0)
                if p.is_alive():
                    p.terminate()

        return {
            "eval/success_rate": float(np.mean(successes)) if successes else 0.0,
            "eval/mean_steps": float(np.mean(episode_step_counts)) if episode_step_counts else 0.0,
        }

    def _build_obs_batch_parallel(self, histories, pl_module, tasks=None) -> dict[str, torch.Tensor]:
        """Stack per-worker observation histories into one batched policy input (with per-episode object names)."""
        B = len(histories)
        T = len(histories[0])
        batch = {}
        for key in pl_module.obs_keys:
            obs_key = key.removeprefix("observation.")
            sample0 = np.asarray(histories[0][0][obs_key])
            stacked = np.empty((B, T) + sample0.shape, dtype=np.float32)
            for b in range(B):
                for t in range(T):
                    stacked[b, t] = histories[b][t][obs_key]
            # Point conditioning: match training's [x, y, visible] transform so the
            # policy sees the same 3D layout at eval (only present when --point_condition).
            if key == "observation.target_point" and stacked.shape[-1] == 2:
                stacked = target_point_xyv(stacked)
            batch[key] = torch.from_numpy(stacked).to(pl_module.device, non_blocking=True)
        # Per-episode object names from the workers (matches training, which
        # conditions on each frame's object_name). Fallback: env objects regex.
        if tasks is not None and any(t for t in tasks):
            batch["object_name"] = [str(t) for t in tasks]
        else:
            task_str = self._env_config.get("objects", "")
            if isinstance(task_str, str) and task_str:
                batch["object_name"] = [task_str] * B
        return batch

    @torch.no_grad()
    def _evaluate_single(self, pl_module, step: int) -> dict[str, float]:
        """Sequential single-env eval, used when --eval_num_envs=1."""
        env = make_env(self._base_env_cfg())

        rollout_agent = RolloutController()
        rollout_agent.setup(env.scene.model, env.scene.data)
        rollout_agent.set_env(env)
        env.set_agent(rollout_agent)

        try:
            obs_horizon, action_horizon, schedule = self._horizons(pl_module)

            step_dt = env.robot.n_substeps * env.scene.model.opt.timestep
            step_rate = int(round(1.0 / step_dt))
            action_repeat = max(1, int(round(step_rate / float(self._dataset_fps))))

            successes = []
            steps = []

            was_training = pl_module.training
            pl_module.eval()

            image_keys = [k for k in env.cameras if f"observation.{k}" in pl_module.obs_keys]

            def _augment(o):
                """Attach rendered camera images to an observation."""
                if image_keys:
                    cams = env.render_cameras()
                    for k in image_keys:
                        img = cams[k].astype(np.float32).transpose(2, 0, 1) / 255.0
                        o = {**o, k: img}
                return o

            episode_iter = tqdm(
                range(self._episodes),
                desc=f"eval step {step}",
                unit="ep",
                leave=False,
            )
            for _ in episode_iter:
                obs, info = env.reset()
                rollout_agent.reset(info)
                episode_task = str(info.get("object_name") or "")
                obs = _augment(env._build_obs())
                history = deque([obs] * obs_horizon, maxlen=obs_horizon)
                episode_steps = 0
                success = False
                prev_smoothed = None
                gripper_cmd = GRIP_OPEN

                while env.time < self._time_limit:
                    batch = self._build_obs_batch(history, pl_module, task=episode_task)
                    action_seq = pl_module.sample_actions(batch, schedule, batch_size=1)[0]
                    action_seq = action_seq.detach().cpu().numpy()
                    if pl_module.config.get("use_action_delta", True):
                        action_seq[:, DELTA_DIMS] += anchor_from_obs(obs)
                    action_seq[:, 14] = np.where(action_seq[:, 14] > GRIP_MID, GRIP_CLOSED, GRIP_OPEN)

                    done = False
                    n_chunk = min(action_horizon, len(action_seq))
                    chunk_start = prev_smoothed.copy() if prev_smoothed is not None else _obs_to_action(obs)
                    for i in range(n_chunk):
                        if i == 0:
                            a_curr = chunk_start
                            a_next = action_seq[0]
                        else:
                            a_curr = action_seq[i - 1]
                            a_next = action_seq[i]
                        for k in range(action_repeat):
                            alpha = k / max(action_repeat - 1, 1)
                            action = (1 - alpha) * a_curr + alpha * a_next
                            prev_smoothed = action.copy()
                            target = action[14]
                            if target > gripper_cmd:
                                gripper_cmd = min(gripper_cmd + GRIP_STEP, target)
                            else:
                                gripper_cmd = max(gripper_cmd - GRIP_STEP, target)
                            prev_smoothed[14] = gripper_cmd
                            obs, _, terminated, truncated, info = env.step(prev_smoothed)
                            rollout_agent.set_step_info(info)
                            success = success or bool(info.get("success", False))
                            if terminated or truncated:
                                done = True
                                break
                        history.append(_augment(obs))
                        episode_steps += 1
                        if done:
                            break
                    if done:
                        break

                successes.append(float(success))
                steps.append(float(episode_steps))
                episode_iter.set_postfix(
                    success=f"{int(sum(successes))}/{len(successes)}",
                    rate=f"{np.mean(successes):.0%}",
                )

            if was_training:
                pl_module.train()

            return {
                "eval/success_rate": float(np.mean(successes)) if successes else 0.0,
                "eval/mean_steps": float(np.mean(steps)) if steps else 0.0,
            }
        finally:
            env.close()

    def _build_obs_batch(self, history, pl_module, task: str = "") -> dict[str, torch.Tensor]:
        """Stack one observation history into a batch-of-1 policy input."""
        batch = {}
        for key in pl_module.obs_keys:
            obs_key = key.removeprefix("observation.")
            stacked = np.stack([frame[obs_key] for frame in history], axis=0).astype(np.float32)
            batch[key] = torch.from_numpy(stacked).unsqueeze(0).to(pl_module.device)
        task_str = task or self._env_config.get("objects", "")
        if isinstance(task_str, str) and task_str:
            batch["object_name"] = [task_str]
        return batch
