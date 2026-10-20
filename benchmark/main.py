"""Score a policy checkpoint on a fixed benchmark.

    python benchmark/main.py --checkpoint=path/to/step.ckpt --benchmark=bowls

A benchmark is a JSONL of exported reset states in molmospaces/assets/eval_sets/,
named by its stem (bowls, all, bowl_pick) or given as a path. Every episode
restores its saved state, through the same evaluation rl/main.py logs during
training. Takes BC and RL checkpoints alike.

Physics is deterministic but GPU rendering is not: shadows flicker by a few
pixels and policy batches vary with worker timing, and the closed loop amplifies
both. One pass over 100 episodes moves by about 4 points, so the score is the
mean over --repeats passes.
"""
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
from absl import app, flags

ROOT_DIR = Path(__file__).resolve().parents[1]
for _p in (ROOT_DIR, ROOT_DIR / "train", ROOT_DIR / "rl"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from evaluate import load_benchmark, replay
from molmospaces.configs.benchmark import get_config as get_env_config
from setup import camera_shapes, control_timing, load_checkpoint
from vec_env import VecEnv

flags.DEFINE_string("checkpoint", None, "BC or RL checkpoint to score.")
flags.DEFINE_string("benchmark", "bowls",
                    "Eval set name (bowls, all, bowl_pick) or a path to a JSONL of reset states.")
flags.DEFINE_integer("gpu", 0, "GPU for the policy.")
flags.DEFINE_list("render_gpus", ["0"], "GPUs the env workers render on.")
flags.DEFINE_integer("num_envs", 64, "Parallel env workers, capped at the number of episodes.")
flags.DEFINE_integer("num_sampling_steps", 10, "Flow denoising steps.")
flags.DEFINE_float("time_limit", 50.0, "Per-episode limit in sim seconds.")
flags.DEFINE_bool("use_ema", True, "Score the EMA weights. BC checkpoints also hold the raw ones.")
flags.DEFINE_integer("repeats", 3, "Passes over the benchmark; the score is their mean.")
flags.DEFINE_integer("seed", 777000, "Seed for the flow sampler and the per-episode env RNG.")
flags.DEFINE_string("output", "", "Optional JSON file for per-episode results.")
flags.mark_flag_as_required("checkpoint")
FLAGS = flags.FLAGS


def main(_):
    """Load the checkpoint, replay every benchmark episode, report the success rate."""
    name, states = load_benchmark(FLAGS.benchmark)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.gpu)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    policy, config, payload, fps = load_checkpoint(
        FLAGS.checkpoint, device, num_sampling_steps=FLAGS.num_sampling_steps,
        use_ema=FLAGS.use_ema)
    policy.eval()

    # Episodes restore their own scene, poses and appearance; the config sets
    # the termination rules. Boot workers on a benchmark scene rather than a
    # training pool they would never use.
    env_cfg = get_env_config().to_dict()
    height, width = camera_shapes(payload)[0][-2:]
    env_cfg.update(scene=states[0]["scene"], camera_size=(int(height), int(width)),
                   launch_viewer=False, seed=FLAGS.seed)
    action_repeat, step_dt, image_keys = control_timing(env_cfg, policy, fps, FLAGS.seed)
    obs_horizon = int(config["observation_horizon"])
    action_horizon = int(config["action_horizon"])

    n_envs = min(FLAGS.num_envs, len(states))
    print(f"[benchmark] {name}: {len(states)} episodes, {n_envs} envs, "
          f"checkpoint {Path(FLAGS.checkpoint).name}", flush=True)
    t0 = time.time()
    vec_env = VecEnv(env_cfg, n_envs, FLAGS.render_gpus, FLAGS.seed)
    try:
        # Room for the time limit to bind first: max_steps is a backstop only.
        vec_env.init_workers(obs_horizon, action_horizon, action_repeat, FLAGS.time_limit,
                             int(1.5 * FLAGS.time_limit / step_dt),
                             config.get("use_action_delta", True), reset=False)
        schedule = torch.linspace(0.0, 1.0, int(config["num_sampling_steps"]), device=device)
        passes = []
        for p in range(FLAGS.repeats):
            def progress(done, hits, p=p):
                if done < len(states):
                    print(f"[benchmark] pass {p + 1}/{FLAGS.repeats}: {done}/{len(states)} "
                          f"episodes, {hits / done:.1%} so far  "
                          f"({(time.time() - t0) / 60:.1f} min)", flush=True)
            episodes = replay(policy, vec_env, states, schedule=schedule, device=device,
                              image_keys=image_keys, obs_horizon=obs_horizon,
                              action_horizon=action_horizon, action_repeat=action_repeat,
                              time_limit=FLAGS.time_limit, seed=FLAGS.seed + p,
                              progress=progress)
            passes.append([bool(ep["success"]) for ep in episodes])
            print(f"[benchmark] pass {p + 1}/{FLAGS.repeats}: "
                  f"{sum(passes[-1]) / len(states):.1%}", flush=True)
    finally:
        vec_env.close()

    results = [{"index": i, "scene": st["scene"], "object": st.get("object_name", ""),
                "success": sum(hits) / len(hits)}
               for i, (st, hits) in enumerate(zip(states, zip(*passes)))]
    sr = report(name, results, [sum(h) / len(h) for h in passes], time.time() - t0)
    if FLAGS.output:
        Path(FLAGS.output).write_text(json.dumps(
            {"benchmark": name, "checkpoint": str(FLAGS.checkpoint), "success_rate": sr,
             "passes": [sum(h) / len(h) for h in passes], "episodes": results}, indent=1))


def report(name, results, pass_srs, wall_s):
    """Print the mean success rate and its standard error.

    The error is over episodes, with each episode's success averaged across
    passes, so it covers both which episodes the set holds and render noise.
    """
    n = len(results)
    rates = [r["success"] for r in results]
    sr = sum(rates) / n
    se = math.sqrt(sum((x - sr) ** 2 for x in rates) / (n - 1) / n) if n > 1 else 0.0
    print(f"\n{name}: {sr:.1%} success +/- {se:.1%} over {len(pass_srs)} passes "
          f"({', '.join(f'{s:.0%}' for s in pass_srs)})  in {wall_s / 60:.1f} min")
    return sr


if __name__ == "__main__":
    app.run(main)
