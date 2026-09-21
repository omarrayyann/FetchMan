"""List scenes containing a pickable instance of an object, for use as a `scene=` glob.

Pickable means what PickTask means by it: a body carrying a freejoint whose
asset id has a downloaded grasp file. Both facts are needed and they live in
different places, so this reads both: the freejoint from the scene XML, and the
asset id from the sibling `_metadata.json`. Neither alone is reliable
(`is_static` disagrees with the real freejoint on ~8% of bodies, and the asset
id is not recoverable from the body name). Scenes are parsed, never compiled,
which keeps a full-corpus scan cheap.

Usage:
    python molmospaces/scripts/find_scenes.py bowl
    python molmospaces/scripts/find_scenes.py 'bowl|plunger' --out bowl_scenes.txt
    python molmospaces/scripts/find_scenes.py bowl --scene_list scene_lists/all_scenes.txt
"""
import argparse
import json
import re
import sys
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from molmospaces import ASSETS_DIR                                # noqa: E402
from molmospaces.components.constants import has_valid_grasp      # noqa: E402
from _paths import scene_roots                                    # noqa: E402

EXCLUDE_SUFFIXES = ("_ceiling.xml", "_orig.xml", "_non_settled.xml", "_noskybox.xml")
PAT = None


def _init(pattern):
    """Pool initializer: compile the category regex once per worker."""
    global PAT
    PAT = re.compile(pattern, re.IGNORECASE)


def freejoint_bodies(xml_path):
    """Names of bodies in the scene XML that carry a freejoint."""
    names = set()
    for _, body in ET.iterparse(str(xml_path), events=("end",)):
        if body.tag != "body":
            continue
        name = body.get("name") or ""
        if body.find("freejoint") is not None or any(
                j.get("type") == "free" for j in body.findall("joint")):
            names.add(name)
        # iterparse over 'end' events keeps peak memory to one body subtree
        body.clear()
    return names


def scan(rel):
    """Return 'rel<TAB>asset_id,asset_id' for this scene's pickable matches, or None."""
    xml_path = Path(rel)
    if not xml_path.is_absolute():
        xml_path = ASSETS_DIR / xml_path
    meta_path = xml_path.with_name(xml_path.stem + "_metadata.json")
    try:
        objects = json.loads(meta_path.read_text())["objects"]
        free = freejoint_bodies(xml_path)
    except Exception:
        return None

    hits = []
    for body_name, meta in objects.items():
        if body_name not in free:
            continue
        if not PAT.search(meta.get("category") or "") and not PAT.search(body_name):
            continue
        asset_id = meta.get("asset_id") or ""
        if has_valid_grasp(asset_id) and asset_id not in hits:
            hits.append(asset_id)
    return f"{rel}\t{','.join(hits)}" if hits else None


def collect_scenes(scene_list):
    """Scene paths to scan: an explicit list file, else every installed scene."""
    if scene_list:
        src = Path(scene_list)
        if not src.exists():
            src = ASSETS_DIR / "scene_lists" / src.name
        return [ln.strip().split("\t")[0] for ln in src.read_text().splitlines() if ln.strip()]
    return sorted(str(p.relative_to(ASSETS_DIR))
                  for root in scene_roots()
                  for p in root.rglob("*.xml")
                  if not p.name.endswith(EXCLUDE_SUFFIXES))


def main():
    """Scan scenes in parallel and report the ones where the object is truly pickable."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pattern", help="Regex matched against object category, case-insensitive.")
    ap.add_argument("--scene_list", default=None,
                    help="Scan this list file instead of every installed scene.")
    ap.add_argument("--out", default=None, help="Write the matching scenes here, one per line.")
    ap.add_argument("--jobs", type=int, default=16)
    args = ap.parse_args()

    scenes = collect_scenes(args.scene_list)
    if not scenes:
        raise SystemExit("no scenes found; ship sample_scenes or run download_scenes.py")
    print(f"scanning {len(scenes)} scenes with {args.jobs} workers", flush=True)

    keep = []
    with ProcessPoolExecutor(args.jobs, initializer=_init, initargs=(args.pattern,)) as ex:
        for done, r in enumerate(ex.map(scan, scenes, chunksize=8), 1):
            if r:
                keep.append(r)
            if done % 250 == 0:
                print(f"  {done}/{len(scenes)}  kept={len(keep)}", flush=True)

    if args.out:
        Path(args.out).write_text("".join(f"{line}\n" for line in keep))
        print(f"kept {len(keep)}/{len(scenes)} -> {args.out}", flush=True)
    else:
        for line in keep:
            print(line)
        print(f"\nkept {len(keep)}/{len(scenes)}", flush=True)


if __name__ == "__main__":
    main()
