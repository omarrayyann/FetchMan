"""Merge the per-worker LeRobot datasets that collect_parallel.sh writes into one.

Each collection worker writes a standalone dataset, so a parallel run leaves N
shards that training cannot read together. This merges them:

    python molmospaces/scripts/merge_datasets.py \
        --input_glob='data/data_20260101_120000/*' \
        --output_root=data/bowl_pick --output_repo_id=local/bowl_pick

Then train on the result with
`--data_root=data/bowl_pick --repo_id=local/bowl_pick`.

Shards killed mid-write (Ctrl+C during collection) are repaired first; pass
--norecover to skip that and drop them instead.
"""
import glob
import json
import os
import re
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from absl import app, flags

import pyarrow.parquet as pq
import lerobot.datasets.aggregate as _agg
from lerobot.datasets.aggregate import aggregate_datasets
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

FLAGS = flags.FLAGS

flags.DEFINE_list("input_roots", None, "Comma-separated shard dataset roots to merge.")
flags.DEFINE_string("input_glob", None, "Glob matching the shard roots to merge, e.g. 'data/data_*/*'.")
flags.DEFINE_string("output_root", None, "Directory to write the merged dataset to. Must not exist.")
flags.DEFINE_string("output_repo_id", "local/combined", "repo_id to record in the merged dataset.")
flags.DEFINE_bool("dry_run", False, "Print what would be merged, then stop.")
flags.DEFINE_bool("recover", True, "Repair shards killed mid-write before merging: drop the truncated "
                                   "in-flight episode and fix info.json totals. Only ever deletes files "
                                   "that are already incomplete, and is idempotent.")
flags.DEFINE_integer("jobs", min(32, os.cpu_count() or 8), "Worker processes for the per-shard scan.")

# Captured before the monkeypatch below, since shutil is a shared module.
_ORIG_COPY = shutil.copy


def _video_keys(root: Path) -> list[str]:
    """Camera feature keys of a shard, e.g. ['observation.head_image', ...]."""
    info = json.load(open(root / "meta" / "info.json"))
    return [k for k, v in info.get("features", {}).items() if v.get("dtype") == "video"]


def _chunk_file(path) -> tuple[int, int]:
    """Parse (chunk_index, file_index) out of a chunk-CCC/file-NNN.* path."""
    return (int(re.search(r"chunk-(\d+)", str(path)).group(1)),
            int(re.search(r"file-(\d+)\.", str(path)).group(1)))


def _parquet_ok(path: Path) -> bool:
    """Cheap truncation check: a complete parquet ends with a PAR1 footer."""
    try:
        with open(path, "rb") as f:
            f.seek(-4, os.SEEK_END)
            return f.read(4) == b"PAR1"
    except OSError:
        return False


def _hardlink_or_copy(src, dst, *args, **kwargs):
    """shutil.copy replacement used inside lerobot's aggregate: hardlink when both
    paths are on one filesystem, else copy. The aggregate never mutates video
    bytes, so sharing the inode is safe."""
    try:
        os.link(str(src), str(dst))
        return str(dst)
    except OSError:
        return _ORIG_COPY(str(src), str(dst), *args, **kwargs)


def _resolve_input_roots() -> list[Path]:
    """Expand --input_roots and --input_glob into a deduped list of shard roots."""
    roots = [Path(p).resolve() for p in (FLAGS.input_roots or []) if p]
    if FLAGS.input_glob:
        roots += [Path(p).resolve() for p in sorted(glob.glob(FLAGS.input_glob)) if Path(p).is_dir()]
    output_root = Path(FLAGS.output_root).resolve()
    deduped, seen = [], set()
    for root in roots:
        if root in seen or root == output_root:
            continue
        seen.add(root)
        deduped.append(root)
    return deduped


def _recover_shard(root: Path) -> str:
    """Make a shard loadable again after a kill mid-write.

    Every episode is self-contained (one data parquet, one meta row, one mp4 per
    camera), so recovery is: keep the longest contiguous prefix of episodes whose
    files are all present and untruncated, delete the rest along with any tmp
    streaming dirs, and correct info.json. Returns a status, '' when clean.
    """
    info_path = root / "meta" / "info.json"
    if not info_path.exists():
        return ""
    cams = _video_keys(root)

    cols = (["episode_index", "length", "dataset_to_index", "data/chunk_index", "data/file_index"]
            + [f"videos/{c}/{f}" for c in cams for f in ("chunk_index", "file_index")])
    meta_files = sorted(root.glob("meta/episodes/chunk-*/file-*.parquet"))
    episodes = []
    for meta_file in meta_files:
        if not _parquet_ok(meta_file):
            continue
        try:
            episodes.extend(pq.read_table(meta_file, columns=cols).to_pylist())
        except Exception:
            pass
    episodes.sort(key=lambda r: r["episode_index"])

    def data_path(row):
        return (root / "data" / f"chunk-{row['data/chunk_index']:03d}"
                / f"file-{row['data/file_index']:03d}.parquet")

    def video_path(row, cam):
        return (root / "videos" / cam / f"chunk-{row[f'videos/{cam}/chunk_index']:03d}"
                / f"file-{row[f'videos/{cam}/file_index']:03d}.mp4")

    keep = keep_frames = 0
    keep_data: set = set()
    keep_videos: dict[str, set] = {cam: set() for cam in cams}
    for i, row in enumerate(episodes):
        if row["episode_index"] != i:
            break
        data = data_path(row)
        videos = {cam: video_path(row, cam) for cam in cams}
        if not (data.exists() and _parquet_ok(data)
                and all(v.exists() and v.stat().st_size > 0 for v in videos.values())):
            break
        keep, keep_frames = i + 1, row["dataset_to_index"]
        keep_data.add(_chunk_file(data))
        for cam, path in videos.items():
            keep_videos[cam].add(_chunk_file(path))

    tmp_dirs = list(root.glob("tmp*"))
    if keep == len(episodes) and not tmp_dirs and all(_parquet_ok(p) for p in root.rglob("*.parquet")):
        return ""

    deleted = 0
    for path in root.glob("data/chunk-*/file-*.parquet"):
        if _chunk_file(path) not in keep_data:
            path.unlink()
            deleted += 1
    for cam, keepset in keep_videos.items():
        for path in root.glob(f"videos/{cam}/chunk-*/file-*.mp4"):
            if _chunk_file(path) not in keepset:
                path.unlink()
                deleted += 1
    for meta_file in meta_files:
        drop = not _parquet_ok(meta_file)
        if not drop:
            indices = pq.read_table(meta_file, columns=["episode_index"]).column(0).to_pylist()
            drop = any(i >= keep for i in indices)
        if drop:
            meta_file.unlink()
            deleted += 1
    for tmp in tmp_dirs:
        shutil.rmtree(tmp, ignore_errors=True)

    info = json.load(open(info_path))
    info["total_episodes"] = int(keep)
    info["total_frames"] = int(keep_frames)
    info["splits"] = {"train": f"0:{int(keep)}"}
    json.dump(info, open(info_path, "w"), indent=4)
    return f"kept {keep} episodes, dropped {deleted} files and {len(tmp_dirs)} tmp dirs"


def _is_loadable(root: Path) -> tuple[bool, str]:
    """Check a shard has an info.json, data parquets, and no truncated parquet."""
    if not (root / "meta" / "info.json").exists():
        return False, "no meta/info.json"
    parquets = list(root.rglob("*.parquet"))
    if not parquets:
        return False, "no parquet files"
    if not list((root / "data").rglob("*.parquet")):
        return False, "no data parquet files"
    truncated = [p for p in parquets if not _parquet_ok(p)]
    if truncated:
        return False, f"truncated parquet ({len(truncated)}x, e.g. {truncated[0].relative_to(root)})"
    try:
        LeRobotDatasetMetadata(root.name, root=root)
    except Exception as e:
        return False, f"{type(e).__name__}: {str(e)[:80]}"
    return True, ""


def _scan_shard(args) -> dict:
    """Recover, validate and count one shard. Returns plain data so it pickles back."""
    root_str, do_recover = args
    root = Path(root_str)
    out = {"root": root_str, "name": root.name, "ok": False, "why": "",
           "recovered": "", "episodes": 0, "frames": 0}
    if do_recover:
        try:
            out["recovered"] = _recover_shard(root)
        except Exception as e:
            out["why"] = f"recovery failed: {type(e).__name__}: {e}"
            return out
    out["ok"], out["why"] = _is_loadable(root)
    if out["ok"]:
        try:
            meta = LeRobotDatasetMetadata(root.name, root=root)
            out["episodes"], out["frames"] = meta.total_episodes, meta.total_frames
        except Exception as e:
            out["ok"], out["why"] = False, f"{type(e).__name__}: {str(e)[:80]}"
    return out


def _normalize_one_meta(path_str: str) -> int:
    """Re-stamp one meta/episodes parquet's chunk/file index to match its filename."""
    import pandas as pd
    path = Path(path_str)
    chunk, file_idx = _chunk_file(path)
    chunk_col, file_col = "meta/episodes/chunk_index", "meta/episodes/file_index"
    df = pd.read_parquet(path)
    if file_col not in df.columns:
        return 0
    if sorted(set(df[file_col])) == [file_idx] and sorted(set(df[chunk_col])) == [chunk]:
        return 0
    df[chunk_col] = chunk
    df[file_col] = file_idx
    df.to_parquet(path, index=False)
    return 1


def _normalize_meta_episode_index(root: Path, jobs: int = 1) -> int:
    """Point every meta/episodes row at the physical parquet it actually lives in.

    lerobot's aggregate advances file_index as a logical counter while rotating
    the physical files by size, so the column stops matching the filenames. That
    is harmless for training, which reads the episodes table whole, but it makes
    the merged dataset un-re-mergeable: a second aggregate looks for file-NNN
    parquets that were never written. Only the location pointers are touched.
    """
    files = [str(p) for p in sorted(root.glob("meta/episodes/chunk-*/file-*.parquet"))]
    if not files:
        return 0
    jobs = max(1, min(jobs, len(files)))
    if jobs == 1:
        return sum(_normalize_one_meta(f) for f in files)
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        return sum(ex.map(_normalize_one_meta, files, chunksize=64))


def main(_):
    """Merge the given shards into one dataset, repairing broken shards first."""
    if not FLAGS.output_root:
        raise ValueError("--output_root is required.")
    if not (FLAGS.input_roots or FLAGS.input_glob):
        raise ValueError("Pass --input_roots or --input_glob to say which shards to merge.")
    input_roots = _resolve_input_roots()
    if not input_roots:
        raise ValueError("No shard directories matched.")

    # The scan reads every per-episode meta parquet in every shard, which is the
    # bulk of the wall clock, and shards are independent.
    jobs = max(1, min(FLAGS.jobs, len(input_roots)))
    print(f"Scanning {len(input_roots)} shards with {jobs} workers (recover={FLAGS.recover})...")
    work = [(str(r), FLAGS.recover) for r in input_roots]
    if jobs == 1:
        results = [_scan_shard(w) for w in work]
    else:
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            results = list(ex.map(_scan_shard, work))
    results.sort(key=lambda d: d["name"])

    kept, total_episodes, total_frames = [], 0, 0
    for r in results:
        if r["recovered"]:
            print(f"  ~ repaired {r['name']}: {r['recovered']}")
        if r["ok"]:
            kept.append(Path(r["root"]))
            total_episodes += r["episodes"]
            total_frames += r["frames"]
            print(f"  - {r['name']}: {r['episodes']} episodes, {r['frames']} frames")
        else:
            print(f"  ! skipping {r['name']}: {r['why']}")
    if not kept:
        raise RuntimeError("No loadable shards found.")

    print(f"\nMerging {len(kept)} shards into {FLAGS.output_repo_id} "
          f"({total_episodes} episodes, {total_frames} frames)")
    if FLAGS.dry_run:
        return

    output_root = Path(FLAGS.output_root).resolve()
    if output_root.exists():
        raise RuntimeError(f"{output_root} already exists; delete it or pick another --output_root.")

    # Keep one file per episode instead of size-packing. Packing makes the
    # aggregate re-concatenate videos through ffmpeg and re-read and rewrite the
    # growing episodes table once per episode, both O(n^2); at a thousand
    # episodes that is nearly all of the runtime. Hardlinking moves no video
    # bytes at all. _normalize_meta_episode_index below fixes the index columns
    # this leaves inconsistent.
    _agg.shutil.copy = _hardlink_or_copy
    _agg.DEFAULT_DATA_FILE_SIZE_IN_MB = 1e-6
    aggregate_datasets(
        repo_ids=[root.name for root in kept],
        aggr_repo_id=FLAGS.output_repo_id,
        roots=list(kept),
        aggr_root=output_root,
        video_files_size_in_mb=1e-6,
        data_files_size_in_mb=1e-6,
    )
    normalized = _normalize_meta_episode_index(output_root, jobs=FLAGS.jobs)
    print(f"Done. Merged dataset at {output_root}"
          f"{f' (normalized {normalized} meta files)' if normalized else ''}")


if __name__ == "__main__":
    app.run(main)
