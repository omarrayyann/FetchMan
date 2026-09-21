#!/usr/bin/env python3
"""Fast asset sanity check: counts scenes / objects / grasps under
molmospaces/assets and flags common setup problems (broken symlinks,
missing scene-list entries, stale grasp UID cache).

Run after `run_download.sh` on a new machine:

    python molmospaces/scripts/check_assets.py
"""
import argparse
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from molmospaces.components.constants import has_valid_grasp   # noqa: E402
from _paths import ASSETS, scene_roots                         # noqa: E402
GRASPS = Path(os.environ["MOLMOSPACES_GRASPS_DIR"]).expanduser() \
    if os.environ.get("MOLMOSPACES_GRASPS_DIR") else ASSETS / "grasps"
SCENE_VARIANT_SUFFIXES = ("_ceiling.xml", "_orig.xml", "_non_settled.xml", "_noskybox.xml")

problems = []


def warn(msg):
    """Print a warning line and count it."""
    problems.append(msg)
    print(f"  !! {msg}")


def scan_dir(path):
    """Returns (entries, broken_symlinks) for one directory level."""
    entries, broken = [], 0
    try:
        with os.scandir(path) as it:
            for e in it:
                if e.is_symlink() and not os.path.exists(e.path):
                    broken += 1
                    continue
                entries.append(e)
    except FileNotFoundError:
        pass
    return entries, broken


def check_scenes():
    """Count scene XMLs per source, across the sample set and any downloaded set."""
    print("== scenes ==")
    roots = scene_roots()
    if not roots:
        warn("no scenes dir — run molmospaces/scripts/run_download.sh")
        return
    total = 0
    for root in roots:
        for src in sorted(p for p in root.iterdir() if p.is_dir()):
            entries, broken = scan_dir(src)
            xmls = [e.name for e in entries if e.name.endswith(".xml")]
            scenes = [x for x in xmls if not x.endswith(SCENE_VARIANT_SUFFIXES)]
            total += len(scenes)
            label = f"{root.name}/{src.name}"
            print(f"  {label:36s} {len(scenes):6d} scenes  ({len(xmls)} xml incl. variants)")
            if broken:
                warn(f"{label}: {broken} broken symlinks (re-run run_download.sh)")
    print(f"  {'TOTAL':36s} {total:6d} scenes")
    if total == 0:
        warn("no scenes found — run molmospaces/scripts/run_download.sh")


def check_scene_lists():
    """Verify scene-list files point at existing XMLs."""
    print("== scene_lists ==")
    root = ASSETS / "scene_lists"
    if not root.is_dir():
        warn(f"missing {root}")
        return
    for txt in sorted(root.glob("*.txt")):
        lines = [l.strip() for l in txt.read_text().splitlines()
                 if l.strip() and not l.strip().startswith("#")]
        missing = [l for l in lines if not (ASSETS / l).exists() and not Path(l).exists()]
        if missing and not (ASSETS / "scenes").is_dir() and not txt.name.startswith("sample"):
            # Lists index the downloadable corpus; before download they are all missing.
            print(f"  {txt.name:28s} {len(lines):6d} entries  (not downloaded)")
            continue
        status = "ok" if not missing else f"{len(missing)} MISSING"
        print(f"  {txt.name:28s} {len(lines):6d} entries  {status}")
        if missing:
            warn(f"{txt.name}: first missing entry: {missing[0]}")


def check_objects():
    """Verify object asset dirs and metadata files exist."""
    print("== objects ==")
    root = ASSETS / "objects"
    thor = root / "thor"
    if thor.is_dir():
        n = sum(1 for p in thor.iterdir() if p.is_dir())
        print(f"  {'thor':28s} {n:6d} category dirs")
        if n == 0:
            warn("objects/thor is empty")
    else:
        warn("missing objects/thor")
    meta = root / "objathor_metadata" / "objects_metadata.json.gz"
    if meta.exists():
        print(f"  {'objathor_metadata':28s}     ok (objects_metadata.json.gz)")
    else:
        print(f"  {'objathor_metadata':28s}     (not downloaded; only needed for objaverse names)")
    objaverse = root / "objaverse"
    if objaverse.is_dir():
        entries, broken = scan_dir(objaverse)
        dirs = [e for e in entries if e.is_dir()]
        print(f"  {'objaverse':28s} {len(dirs):6d} object dirs")
        if broken:
            warn(f"objects/objaverse: {broken} broken symlinks")
    else:
        print(f"  {'objaverse':28s}     (not downloaded; run_download.sh fetches it)")


def grasp_source(root, name):
    """grasps/<source>, descending into a version dir (raw resources-cache layout)."""
    d = root / name if name else root
    if d.is_dir():
        kids = [k for k in d.iterdir() if k.is_dir()]
        if kids and all(re.fullmatch(r"\d{8}", k.name) for k in kids):
            return max(kids, key=lambda k: k.name)
    return d


def grasp_uids(sub):
    """Asset dirs under `sub` holding a whole-object <ID>_grasps_filtered.npz."""
    uids, joint_only, broken = set(), 0, 0
    entries, broken = scan_dir(sub)
    for e in entries:
        if not e.is_dir():
            continue
        if os.path.exists(os.path.join(e.path, f"{e.name}_grasps_filtered.npz")):
            uids.add(e.name)
        else:
            joint_only += 1
    return uids, joint_only, broken


def check_grasps():
    """Count grasp files per source."""
    print("== grasps ==")
    root = GRASPS
    if not root.is_dir():
        warn(f"missing {root}")
        return
    all_uids = set()
    for name in ("droid", "droid_objaverse"):
        sub = grasp_source(root, name)
        if not sub.is_dir():
            # Not an error: download_grasps.py creates it on demand.
            print(f"  {name:28s}     (not downloaded)")
            continue
        uids, no_grasp_file, broken = grasp_uids(sub)
        all_uids |= uids
        extra = f"  (+{no_grasp_file} dirs with no grasp file)" if no_grasp_file else ""
        if uids:
            print(f"  {name:28s} {len(uids):6d} object grasps{extra}")
        else:
            print(f"  {name:28s}     (not downloaded)")
        if broken:
            warn(f"grasps/{name}: {broken} broken symlinks")
    # legacy flat layout (real dirs directly under grasps/) still counts
    flat_uids, _, _ = grasp_uids(grasp_source(root, ""))
    all_uids |= flat_uids
    print(f"  {'TOTAL unique':28s} {len(all_uids):6d} objects with grasps")

    cache = root / ".valid_uids.txt"
    if cache.exists():
        cached = {l.strip() for l in cache.read_text().splitlines() if l.strip()}
        if cached != all_uids:
            warn(f".valid_uids.txt is STALE ({len(cached)} cached vs {len(all_uids)} on disk) "
                 f"— delete it: rm {cache}")
        else:
            print(f"  .valid_uids.txt cache up to date ({len(cached)})")


def check_scene_compile(n_per_source):
    """Compile a random sample of scenes under MuJoCo and report grasp coverage."""
    import json
    import random
    import mujoco

    print(f"== scene compile (sampling {n_per_source} per source) ==")
    rng = random.Random(0)
    for root in scene_roots():
        for src in sorted(p for p in root.iterdir() if p.is_dir()):
            xmls = [p for p in src.rglob("*.xml") if not p.name.endswith(SCENE_VARIANT_SUFFIXES)]
            if not xmls:
                continue
            n_ok = n_pick = n_grasped = 0
            for xml in rng.sample(xmls, min(n_per_source, len(xmls))):
                try:
                    mujoco.MjModel.from_xml_path(str(xml))
                    n_ok += 1
                except Exception as e:
                    warn(f"{xml.name}: {type(e).__name__}: {str(e)[:120]}")
                    continue
                meta = xml.with_name(xml.stem + "_metadata.json")
                try:
                    objects = json.loads(meta.read_text())["objects"]
                except Exception:
                    warn(f"{xml.name}: unreadable metadata sidecar")
                    continue
                for obj in objects.values():
                    if obj.get("is_static", False) or not obj.get("asset_id"):
                        continue
                    n_pick += 1
                    n_grasped += bool(has_valid_grasp(obj["asset_id"]))
            print(f"  {root.name}/{src.name:26s} {n_ok:4d} compiled  "
                  f"{n_grasped}/{n_pick} objects with grasps")


def main():
    """Run all asset checks and exit nonzero on missing pieces."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--compile", type=int, default=0, metavar="N",
                    help="Also compile N scenes per source under MuJoCo (slow).")
    args = ap.parse_args()

    t0 = time.time()
    print(f"assets root: {ASSETS}")
    if not ASSETS.is_dir():
        print(f"FATAL: {ASSETS} does not exist")
        sys.exit(2)
    check_scenes()
    check_scene_lists()
    check_objects()
    check_grasps()
    if args.compile:
        check_scene_compile(args.compile)
    dt = time.time() - t0
    if problems:
        print(f"\nFAIL: {len(problems)} problem(s) found in {dt:.1f}s")
        sys.exit(1)
    print(f"\nOK: all checks passed in {dt:.1f}s")


if __name__ == "__main__":
    main()
