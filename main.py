import os
import signal
import time
from pathlib import Path

import numpy as np
from absl import app, flags
from ml_collections import config_flags

from molmospaces.egl import set_egl_device

flags.DEFINE_integer("gpu", 0, "GPU device ID")
flags.DEFINE_integer("episodes", 1000, "max episodes")
flags.DEFINE_integer("seed", 55, "random seed")
flags.DEFINE_bool("render", False, "render to viewer")
flags.DEFINE_bool("cv2", False, "with --render, show face/head/wrist cameras in a cv2 window instead of the MuJoCo viewer")
flags.DEFINE_bool("record", False, "record successful episodes to a LeRobot dataset")
flags.DEFINE_bool("save_failures", False, "also save failed episodes (when --record)")
flags.DEFINE_bool("realtime", False, "Throttle rollout so wall-clock time matches sim time. Useful with --render.")
flags.DEFINE_bool("debug", False, "Overlay debug markers in the MuJoCo viewer (nav waypoints, etc.). No-op when off.")
flags.DEFINE_string("repo_id", "local/g1_pick", "LeRobot dataset repo_id")
flags.DEFINE_string("data_dir", "data", "local root for LeRobot dataset")
flags.DEFINE_integer("fps", 10, "dataset logging fps (physics runs at 200)")
flags.DEFINE_string("save_json", "", "Append each SUCCESSFUL episode's exact reset state (scene, poses, lights, textures, cameras) to this JSONL file for fixed-eval replay.")
config_flags.DEFINE_config_file("env", "molmospaces/env.py", lock_config=False)
config_flags.DEFINE_config_file("agent", "molmospaces/agents/policy.py", lock_config=False)
FLAGS = flags.FLAGS

CV2_WINDOW = "cameras (face | head | wrist)"
SETTLE_STEPS = 70          # let the WBC reach its commanded pose before recording frame 0
TIME_LIMIT_S = 60.0
GRIP_OPEN, GRIP_CLOSED = -0.0222, 0.0245


def _build_env_and_agent():
    """Create the env and the scripted controller, wired to each other."""
    from molmospaces.env import make_env
    from molmospaces.agents import agents

    FLAGS.env["seed"] = FLAGS.seed
    FLAGS.env["launch_viewer"] = FLAGS.render and not FLAGS.cv2
    env = make_env(FLAGS.env)
    env.debug = bool(FLAGS.debug)

    agent = agents[FLAGS.agent["agent_name"]].create(config=FLAGS.agent)
    agent.setup(env.scene.model, env.scene.data)
    agent.set_env(env)
    env.set_agent(agent)
    return env, agent


def _build_recorder(env):
    """LeRobot recorder when --record is set, else None."""
    if not FLAGS.record:
        return None
    from molmospaces.dataset import LeRobotRecorder
    return LeRobotRecorder(
        repo_id=FLAGS.repo_id,
        root=Path(FLAGS.data_dir) / FLAGS.repo_id,
        env=env,
        action_dim=env.action_space.shape[0],
        fps=FLAGS.fps,
        physics_fps=int(round(1.0 / env.scene.model.opt.timestep)),
    )


class _Display:
    """Optional cv2 camera panel; also carries the stop flag for Ctrl+C and window close."""

    def __init__(self, env):
        self.env = env
        self.stop = False
        self.cv2 = None
        if FLAGS.render and FLAGS.cv2:
            import cv2
            self.cv2 = cv2
            cv2.namedWindow(CV2_WINDOW, cv2.WINDOW_NORMAL)

    def refresh(self):
        """Draw the camera panel and poll for space (skip) and q/esc (quit)."""
        if self.cv2 is None:
            return
        panel = self.env.render_debug_panel()
        if panel is None:
            return
        self.cv2.imshow(CV2_WINDOW, self.cv2.cvtColor(panel, self.cv2.COLOR_RGB2BGR))
        # pollKey is non-blocking; waitKey(1) costs 10-30ms per call on macOS.
        key = self.cv2.pollKey() & 0xFF
        if key == ord(' '):
            self.env._skip_episode = True
        elif key in (ord('q'), 27):
            self.stop = True
        if self.cv2.getWindowProperty(CV2_WINDOW, self.cv2.WND_PROP_VISIBLE) < 1:
            self.stop = True

    def close(self):
        """Tear down the cv2 window if one was opened."""
        if self.cv2 is None:
            return
        try:
            self.cv2.destroyAllWindows()
        except Exception:
            pass


def _reset_until_navigable(env, agent, ui):
    """Reset until the agent has a path to the target; returns (obs, reset_info) or None if stopped."""
    while not ui.stop:
        try:
            obs, info = env.reset()
        except RuntimeError as e:
            # Rejection sampling: a cramped scene can exhaust reset's own retries.
            print(f"[main] resampling placement ({e})", flush=True)
            continue
        reset_info = dict(info)
        agent.reset(info)
        obs = env._build_obs()   # agent.reset may have teleported the base
        if FLAGS.render:
            env.sync_viewer()
            ui.refresh()
        if agent.has_path:
            return obs, reset_info
    return None


def _settle(env, obs, agent, ui):
    """Hold the reset pose for a few frames so frame 0 is not captured mid-drop."""
    hold = np.zeros(15, dtype=np.float32)
    hold[3] = float(getattr(agent, "_height_cmd", obs["base_height"][0]))
    hold[4:7] = obs["joint_pos"][12:15]
    hold[7:14] = obs["joint_pos"][22:29]
    hold[14] = float(obs["joint_pos"][29])
    info = {}
    for _ in range(SETTLE_STEPS):
        if ui.stop:
            break
        obs, _, terminated, truncated, info = env.step(hold)
        if FLAGS.render:
            ui.refresh()
        if terminated or truncated or not env.occ.is_free(env.robot.get_xy()):
            return obs, info, False
    return obs, info, True


def _run_episode(env, agent, recorder, obs, ui):
    """Roll the controller to termination, recording each step; returns the final info dict."""
    step_dt = env.robot.n_substeps * env.scene.model.opt.timestep
    info = {}
    while env.time < TIME_LIMIT_S and not ui.stop:
        action = agent.sample_actions(obs)
        if recorder is not None:
            # The saved gripper label is the controller's phase-level intent, so it
            # flips cleanly at CLOSE; the executed action keeps its gradual ramp.
            save_action = action.copy()
            intent = float(getattr(agent, "_gripper_intent", 0.0))
            save_action[14] = GRIP_CLOSED if intent > 0.5 else GRIP_OPEN
            recorder.add_step(obs, save_action)
        exec_action = (agent.perturb_action_for_rollout(action)
                       if hasattr(agent, "perturb_action_for_rollout") else action)
        step_t0 = time.perf_counter()
        obs, _, terminated, truncated, info = env.step(exec_action)
        agent.set_step_info(info)
        if FLAGS.render:
            ui.refresh()
            if env.consume_skip_episode():
                break
        if terminated or truncated or agent.done:
            break
        if FLAGS.render and not FLAGS.cv2 and not env.viewer_running:
            ui.stop = True
            break
        if FLAGS.realtime:
            remaining = step_dt - (time.perf_counter() - step_t0)
            if remaining > 0:
                time.sleep(remaining)
    return info


def _finalize(env, recorder, ui):
    """Flush the dataset and close the env, ignoring further signals while writing."""
    if recorder is not None:
        # Ignore INT/TERM here or the parquet footer never gets written.
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        print("Finalizing dataset... (do NOT Ctrl+C)")
        try:
            recorder.close()
            print("Dataset finalized.")
        except Exception as e:
            print(f"[warn] recorder.close() failed: {type(e).__name__}: {e}")
    try:
        env.close()
    except Exception:
        pass
    ui.close()


def main(_):
    """Data-collection entry point: run the scripted controller and optionally record episodes."""
    if FLAGS.gpu != -1:
        set_egl_device(FLAGS.gpu)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.gpu)

    env, agent = _build_env_and_agent()
    recorder = _build_recorder(env)
    ui = _Display(env)

    def _shutdown(*_args):
        """Ask the loop to stop at the next safe point."""
        ui.stop = True
        print("\nStopping...")
    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    results = []
    try:
        episode = 0
        while episode < FLAGS.episodes and not ui.stop:
            started = _reset_until_navigable(env, agent, ui)
            if started is None:
                break
            obs, reset_info = started

            ep_state = env.export_reset_state(info=reset_info) if FLAGS.save_json else None
            obs, info, settled = _settle(env, obs, agent, ui)
            if not settled:
                print(f"[main] rejecting reset after settle: xy={env.robot.get_xy()} "
                      f"sim_error={info.get('sim_error')}", flush=True)
                continue
            agent.set_step_info(info)

            if recorder is not None:
                # Reset metadata (scene, object, skill_profile, prompt) must be used
                # here: the settle steps overwrite `info` with per-step metadata.
                recorder.begin_episode(reset_info)

            info = _run_episode(env, agent, recorder, obs, ui)
            success = info.get("success", False)
            results.append(success)

            if recorder is not None:
                if success or FLAGS.save_failures:
                    recorder.save_episode()
                else:
                    recorder.discard_episode()
            if FLAGS.save_json and success and ep_state is not None:
                import json
                with open(FLAGS.save_json, "a") as f:
                    f.write(json.dumps(ep_state) + "\n")

            label = reset_info.get("prompt", reset_info.get("target_name", info.get("target_name")))
            print(f"Ep {episode+1}: {label!r} {'OK' if success else ''} "
                  f"({sum(results)}/{len(results)}={np.mean(results):.0%})")
            episode += 1
    finally:
        _finalize(env, recorder, ui)
    if results:
        print(f"\n{sum(results)}/{len(results)} ({np.mean(results):.0%})")


if __name__ == "__main__":
    app.run(main)
