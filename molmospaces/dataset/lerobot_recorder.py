from __future__ import annotations

from pathlib import Path

import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets import video_utils as _lr_video_utils

# LeRobot hardcodes crf=30, which was tuned for libsvtav1 and looks visibly
# worse on h264; 23 matches libsvtav1@30 perceptually at the same speed.
_H264_CRF = 23
_orig_sve_init = _lr_video_utils.StreamingVideoEncoder.__init__

def _patched_sve_init(self, *args, **kwargs):
    """Patch LeRobot's StreamingVideoEncoder to use our h264 CRF default."""
    vcodec = kwargs.get("vcodec", args[1] if len(args) > 1 else "libsvtav1")
    if "h264" in vcodec and kwargs.get("crf", 30) == 30:
        kwargs["crf"] = _H264_CRF
    return _orig_sve_init(self, *args, **kwargs)

_lr_video_utils.StreamingVideoEncoder.__init__ = _patched_sve_init


# LeRobot's concat demuxer does not propagate DTS offsets to remuxed packets, so
# with h264 the new episode's DTS lands before the shard's last one and the mp4
# muxer rejects it with `[Errno 22] Invalid argument`.
def _concat_video_files_fixed(input_video_paths, output_video_path, overwrite=True):
    """Replacement for LeRobot's video concat: stream-copy with monotonically rewritten DTS/PTS."""
    import shutil
    import tempfile as _tf
    import av as _av
    from pathlib import Path as _Path

    output_video_path = _Path(output_video_path)
    if output_video_path.exists() and not overwrite:
        return
    output_video_path.parent.mkdir(parents=True, exist_ok=True)

    if len(input_video_paths) == 0:
        raise FileNotFoundError("No input video paths provided.")

    with _tf.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        tmp_out = f.name

    out = _av.open(tmp_out, mode="w", options={"movflags": "faststart"})
    smap = None  # input_stream_index -> output_stream
    last_dts: dict[int, int] = {}  # by output stream index

    try:
        for path in input_video_paths:
            inp = _av.open(str(path), "r")
            if smap is None:
                smap = {}
                for s_in in inp.streams:
                    if s_in.type in ("video", "audio", "subtitle"):
                        s_out = out.add_stream_from_template(template=s_in, opaque=True)
                        s_out.time_base = s_in.time_base
                        smap[s_in.index] = s_out
                        last_dts[s_out.index] = -1
            dts_offsets: dict[int, int] = {}
            seen_first: dict[int, bool] = {}
            for pkt in inp.demux():
                if pkt.stream.index not in smap or pkt.dts is None:
                    continue
                out_stream = smap[pkt.stream.index]
                if not seen_first.get(out_stream.index, False):
                    dts_offsets[out_stream.index] = (last_dts[out_stream.index] + 1) - pkt.dts
                    seen_first[out_stream.index] = True
                off = dts_offsets[out_stream.index]
                pkt.dts = pkt.dts + off
                if pkt.pts is not None:
                    pkt.pts = pkt.pts + off
                pkt.stream = out_stream
                out.mux(pkt)
                last_dts[out_stream.index] = pkt.dts
            inp.close()
        out.close()
        shutil.move(tmp_out, str(output_video_path))
    except Exception:
        try:
            out.close()
        except Exception:
            pass
        if _Path(tmp_out).exists():
            _Path(tmp_out).unlink()
        raise

# Patch both module-level and the lerobot_dataset.py import.
_lr_video_utils.concatenate_video_files = _concat_video_files_fixed
import lerobot.datasets.lerobot_dataset as _lr_dataset_mod
_lr_dataset_mod.concatenate_video_files = _concat_video_files_fixed


class LeRobotRecorder:
    """Records episodes as a LeRobot v3 dataset, subsampling physics steps down to fps."""

    def __init__(self, repo_id, env, action_dim, root=None, fps=10, physics_fps=200, robot_type="g1",
                 data_files_size_in_mb=0, video_files_size_in_mb=0):
        """Create or append to a LeRobot v3 dataset with our feature schema."""
        self._env = env
        # `physics_fps` is the rate `add_step` is called at (env.step rate, not the
        # mj_step rate): callers pass 1 / (mj_timestep * n_substeps).
        self._stride = max(1, int(round(physics_fps / fps)))
        self._step_in_episode = 0
        self._task = ""
        self._object_name = ""
        self._object_id = ""
        self._skill_profile = "default"
        self._scene = ""
        self._cameras = list(env.cameras)

        root = Path(root) if root is not None else None
        features = self._build_features(env, action_dim)
        if root is not None and root.exists():
            self._dataset = LeRobotDataset(repo_id, root=root)
            self._validate_existing_dataset(features, fps)
        else:
            self._dataset = LeRobotDataset.create(
                repo_id=repo_id,
                root=root,
                fps=fps,
                features=features,
                robot_type=robot_type,
                use_videos=bool(self._cameras),
                vcodec="h264",
                streaming_encoding=bool(self._cameras),
            )
            self._configure_flush_policy(data_files_size_in_mb, video_files_size_in_mb)

    def _build_features(self, env, action_dim):
        """LeRobot feature schema: action, episode metadata strings, observations, camera videos."""
        features = {
            "action": {
                "dtype": "float32",
                "shape": (action_dim,),
                "names": [f"a{i}" for i in range(action_dim)],
            },
            "object_name": {"dtype": "string", "shape": (1,)},
            "object_id": {"dtype": "string", "shape": (1,)},
            "skill_profile": {"dtype": "string", "shape": (1,)},
            "scene": {"dtype": "string", "shape": (1,)},
        }
        for key, space in env.observation_space.spaces.items():
            shape = tuple(int(s) for s in space.shape)
            n = int(np.prod(shape))
            features[f"observation.{key}"] = {
                "dtype": "float32",
                "shape": shape,
                "names": [f"{key}_{i}" for i in range(n)],
            }
        for name in self._cameras:
            features[f"observation.{name}"] = {
                "dtype": "video",
                "shape": env.camera_shape,
                "names": ["height", "width", "channels"],
            }
        return features

    def _configure_flush_policy(self, data_files_size_in_mb, video_files_size_in_mb):
        """Write one file and one metadata row per episode so a crash loses at most one."""
        info = self._dataset.meta.info
        info["data_files_size_in_mb"] = int(data_files_size_in_mb)
        info["video_files_size_in_mb"] = int(video_files_size_in_mb)
        info["chunks_size"] = int(info.get("chunks_size", 1000) or 1000)
        self._dataset.meta.metadata_buffer_size = 1   # default 10 loses metadata on a hard kill
        from lerobot.datasets.utils import write_info
        write_info(info, self._dataset.meta.root)

    def _validate_existing_dataset(self, features, fps):
        """Assert an existing dataset's features/fps match before appending."""
        if int(self._dataset.fps) != int(fps):
            raise ValueError(
                f"Existing dataset fps={self._dataset.fps} does not match requested fps={fps}."
            )
        existing = self._dataset.meta.features
        missing = sorted(set(features) - set(existing))
        extra = sorted(set(existing) - set(features) - {"index", "episode_index", "task_index", "timestamp", "frame_index"})
        if missing or extra:
            raise ValueError(
                f"Existing dataset schema mismatch. missing={missing}, extra={extra}"
            )
        for key, spec in features.items():
            es = existing[key]
            if es.get("dtype") != spec["dtype"] or tuple(es.get("shape", ())) != tuple(spec["shape"]):
                raise ValueError(
                    f"Existing dataset feature {key!r} mismatch: existing={es}, requested={spec}"
                )

    def begin_episode(self, info):
        """Start buffering a new episode (task = the episode's object name)."""
        self._step_in_episode = 0
        self._task = info.get("prompt") or info.get("task") or ""
        scene = info.get("scene") or str(getattr(self._env, "_current_scene_path", "") or "")
        tgt = getattr(getattr(self._env, "task", None), "target", None)
        obj = info.get("object_name") or getattr(tgt, "category", None) or getattr(tgt, "asset_id", "") or ""
        self._object_name = str(obj)
        # Unique asset identity (objaverse UID / THOR asset id) so two different
        # meshes of the same category are distinguishable; "" if unavailable.
        self._object_id = str(info.get("object_id") or getattr(tgt, "asset_id", "") or "")
        self._skill_profile = str(info.get("skill_profile", "default"))
        self._scene = str(scene)

    def add_step(self, obs, action):
        """Append one (obs, action) frame to the episode buffer."""
        if self._step_in_episode % self._stride != 0:
            self._step_in_episode += 1
            return
        self._step_in_episode += 1
        frame = {
            "action": np.asarray(action, dtype=np.float32).reshape(-1),
            "task": self._task,
            "object_name": self._object_name,
            "object_id": self._object_id,
            "skill_profile": self._skill_profile,
            "scene": self._scene,
        }
        for key, val in obs.items():
            frame[f"observation.{key}"] = np.asarray(val, dtype=np.float32).reshape(-1)
        if self._cameras:
            images = self._env.render_cameras()
            for name in self._cameras:
                frame[f"observation.{name}"] = np.asarray(images[name], dtype=np.uint8)
        self._dataset.add_frame(frame)

    def save_episode(self):
        """Flush the buffered episode into the dataset."""
        self._dataset.save_episode()

    def discard_episode(self):
        """Drop the buffered episode."""
        self._dataset.clear_episode_buffer()

    def close(self):
        """Finalize the dataset writer."""
        self._dataset.finalize()
