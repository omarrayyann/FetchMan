"""Fixed-benchmark evaluation: replay exported reset states with a noise-free flow.

rl/main.py runs this every few updates and benchmark/main.py runs it once, so a
success rate logged during training and one printed by the benchmark tool are
the same measurement.
"""
import json
from pathlib import Path

import torch

from rollouts import collect_rollouts

EVAL_SETS = Path(__file__).resolve().parents[1] / "molmospaces" / "assets" / "eval_sets"


def load_benchmark(name):
    """(name, states) for an eval set named by stem under eval_sets/, or a JSONL path."""
    path = Path(name)
    if not (path.suffix == ".jsonl" and path.exists()):
        path = EVAL_SETS / f"{name}.jsonl"
    if not path.exists():
        known = ", ".join(sorted(p.stem for p in EVAL_SETS.glob("*.jsonl")))
        raise SystemExit(f"unknown benchmark {name!r}; known: {known}")
    return path.stem, [json.loads(line) for line in open(path) if line.strip()]


def replay(policy, vec_env, states, *, schedule, device, image_keys, obs_horizon,
           action_horizon, action_repeat, time_limit, seed, policy_replicas=None,
           progress=None):
    """Run every state once and return one episode result per state, in order.

    Episodes go out one per worker per cycle so a fast env cannot crowd out slow
    ones; the last cycle pads by wrapping and the duplicates are dropped. The
    flow runs without SDE noise (sigmas = 0). If given, progress(done, successes)
    is called after every cycle.
    """
    sigmas = torch.zeros(len(schedule) - 1, device=device)
    vec_env.set_time_limit(time_limit)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    n, n_envs = len(states), vec_env.n_envs
    results = {}
    for cycle in range((n + n_envs - 1) // n_envs):
        idxs = [(cycle * n_envs + j) % n for j in range(n_envs)]
        seeds = [max(0, seed) + i for i in idxs]
        payloads = vec_env.set_benchmark_tasks([states[i] for i in idxs], seeds)
        vec_env._last_obs_histories = [p["obs_history"] for p in payloads]
        # time_limit / max_steps are only read when collect_rollouts initializes
        # workers itself; here they already hold the limit set above.
        episodes = collect_rollouts(
            policy, vec_env, schedule, sigmas, device, image_keys,
            n_envs, time_limit, 0, obs_horizon, action_horizon, action_repeat,
            policy_replicas=policy_replicas, group_ids=None, skip_init_workers=True)
        for ep in episodes:
            i = cycle * n_envs + int(ep["env_idx"])
            if i < n and i not in results:
                results[i] = ep
        if progress is not None:
            progress(len(results), sum(bool(ep["success"]) for ep in results.values()))
    return [results[i] for i in sorted(results)]


def sample(policy, vec_env, n_episodes, *, schedule, device, image_keys, obs_horizon,
           action_horizon, action_repeat, time_limit, seed, policy_replicas=None):
    """Score the policy on freshly sampled tasks when no benchmark is given.

    Worker i resets with seed + i, so every pass draws the same tasks and the
    score stays comparable across updates, though not across env configs.
    """
    sigmas = torch.zeros(len(schedule) - 1, device=device)
    vec_env.set_time_limit(time_limit)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    payloads = vec_env.set_tasks([seed + i for i in range(vec_env.n_envs)], freeze_subsequent=False)
    vec_env._last_obs_histories = [p["obs_history"] for p in payloads]
    return collect_rollouts(
        policy, vec_env, schedule, sigmas, device, image_keys,
        n_episodes, time_limit, 0, obs_horizon, action_horizon, action_repeat,
        policy_replicas=policy_replicas, group_ids=None, skip_init_workers=True)
