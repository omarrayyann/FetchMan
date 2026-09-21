"""Summarize the LeRobot datasets under a data root: episode counts and coverage.

Reads each shard's meta/info.json for totals, then the data parquets for the
per-object and per-scene breakdown. Safe to run while collection is in flight;
shards whose parquets are still being written are skipped.

Usage:
    python molmospaces/scripts/dataset_stats.py [DATA_ROOT]
"""
import argparse
import glob
import json
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

BREAKDOWN_COLUMNS = ("object_name", "object_id", "scene")


def shard_totals(root):
    """Per-shard (episodes, frames, fps) from every meta/info.json under the root."""
    shards = []
    for info_path in sorted(root.glob("*/meta/info.json")):
        try:
            info = json.loads(info_path.read_text())
        except Exception:
            continue
        shards.append((info_path.parent.parent.name,
                       int(info.get("total_episodes", 0)),
                       int(info.get("total_frames", 0)),
                       float(info.get("fps", 10))))
    return shards


def episode_breakdown(root):
    """Episode and frame counts per object_name, object_id and scene."""
    eps = {c: Counter() for c in BREAKDOWN_COLUMNS}
    frames = {c: Counter() for c in BREAKDOWN_COLUMNS}
    n_eps = n_frames = 0
    for path in glob.glob(f"{root}/**/data/**/file-*.parquet", recursive=True):
        try:
            names = pq.read_schema(path).names
        except Exception:
            continue          # still being written
        if "episode_index" not in names:
            continue
        present = [c for c in BREAKDOWN_COLUMNS if c in names]
        table = pq.read_table(path, columns=["episode_index"] + present)
        rows = zip(table.column("episode_index").to_pylist(),
                   *[table.column(c).to_pylist() for c in present])
        last_ep = object()
        for ep, *vals in rows:
            n_frames += 1
            new_ep = ep != last_ep
            n_eps += new_ep
            last_ep = ep
            for col, val in zip(present, vals):
                frames[col][val or "<empty>"] += 1
                if new_ep:
                    eps[col][val or "<empty>"] += 1
    return eps, frames, n_eps, n_frames


def print_totals(shards):
    """One line per shard plus the aggregate."""
    for name, n_eps, n_frames, fps in shards:
        print(f"  {name:<40} episodes={n_eps:<7} frames={n_frames:<10} "
              f"hours={n_frames / fps / 3600:.2f}")
    total_eps = sum(s[1] for s in shards)
    total_frames = sum(s[2] for s in shards)
    fps = shards[0][3]
    print(f"\nDatasets:       {len(shards)}\nTotal episodes: {total_eps}\n"
          f"Total frames:   {total_frames}\nTotal hours:    "
          f"{total_frames / fps / 3600:.2f}  (at fps={fps:g})")


def print_distribution(title, ep_counts, frame_counts, n_eps, n_frames, top=None):
    """Episode/frame share per key, most common first."""
    print(f"\n{title}:")
    if not ep_counts:
        print("  (no data)")
        return
    items = ep_counts.most_common(top)
    width = max(len(str(k)) for k, _ in items)
    for key, n in items:
        fr = frame_counts.get(key, 0)
        print(f"  {str(key):<{width}}  episodes={n:<7} ({n * 100 / max(n_eps, 1):5.1f}%)  "
              f"frames={fr:<10} ({fr * 100 / max(n_frames, 1):5.1f}%)")
    if top and len(ep_counts) > top:
        print(f"  ... and {len(ep_counts) - top} more")


def print_spread(title, ep_counts):
    """Min/max/mean/median episodes per key, for coverage at a glance."""
    if not ep_counts:
        return
    vals = sorted(ep_counts.values())
    n = len(vals)
    median = vals[n // 2] if n % 2 else (vals[n // 2 - 1] + vals[n // 2]) / 2
    print(f"\n{title}: {n} unique")
    print(f"  episodes each  min={vals[0]}  max={vals[-1]}  "
          f"mean={sum(vals) / n:.1f}  median={median}")


def main():
    """Print dataset totals and the object/scene coverage breakdown."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data_root", nargs="?", default="data", type=Path)
    ap.add_argument("--top", type=int, default=10, help="Rows to show in each ranking.")
    args = ap.parse_args()
    if not args.data_root.is_dir():
        raise SystemExit(f"data root not found: {args.data_root}")

    shards = shard_totals(args.data_root)
    if not shards:
        raise SystemExit(f"no LeRobot datasets found under {args.data_root}")
    print_totals(shards)

    eps, frames, n_eps, n_frames = episode_breakdown(args.data_root)
    if not n_eps:
        print("\n(no readable parquets yet; collection may still be running)")
        return
    print_distribution("Target object distribution", eps["object_name"], frames["object_name"],
                       n_eps, n_frames, args.top)
    print_spread("Object asset ids (unique meshes)", eps["object_id"])
    print_distribution("Most-used assets", eps["object_id"], frames["object_id"],
                       n_eps, n_frames, args.top)
    print_spread("Scenes used", eps["scene"])
    print_distribution("Most-used scenes", eps["scene"], frames["scene"], n_eps, n_frames, args.top)


if __name__ == "__main__":
    main()
