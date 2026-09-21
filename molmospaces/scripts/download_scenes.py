"""Download ProcTHOR / Holodeck / Objaverse scenes + required objects + grasps.

Streams .tar.zst archives from Cloudflare R2, decompresses, extracts into
~/.cache/molmo-spaces-resources/, then symlinks into
molmospaces/assets/scenes/<source>/.

Script version of download_scenes.ipynb so you can run it in a tmux session.

Usage:
    # Default: download a moderate set (~5k scenes, ~37GB extracted total)
    python molmospaces/scripts/download_scenes.py

    # Custom counts per source (`-1` = all, `0` = skip):
    python molmospaces/scripts/download_scenes.py \\
        --procthor-10k-train 10000 \\
        --procthor-10k-val 1000 \\
        --procthor-objaverse-train 5000 \\
        --holodeck-objaverse-train 5000 \\
        --ithor -1

    # Skip object / grasp install (just scene XMLs):
    python molmospaces/scripts/download_scenes.py --no-objects --no-grasps

In tmux:
    tmux new -s scenes
    conda activate g1   # (or whatever env has molmospaces_resources)
    python molmospaces/scripts/download_scenes.py --procthor-10k-train -1
    # Ctrl+B then D to detach
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

VERSIONS = dict(
    scenes={
        "ithor":                    "20251217",
        "procthor-10k-train":       "20251122",
        "procthor-10k-val":         "20251217",
        "procthor-10k-test":        "20251121",
        "procthor-objaverse-train": "20251205",
        "procthor-objaverse-val":   "20251205",
        "holodeck-objaverse-train": "20251217",
        "holodeck-objaverse-val":   "20251217",
    },
    objects={
        "thor":              "20251117",   # ~1.5 GB, used by procthor-10k-* and ithor scenes
        "objaverse":         "20260131",   # ~138 GB total -- fetched lazily per scene
        "objathor_metadata": "20251117",   # ~880 MB, objaverse object names (prompt_sampler) + clip features
    },
    grasps={
        "droid":           "20251116",   # ~366 MB, THOR-named grasp transforms
        "droid_objaverse": "20251218",   # ~951 MB, objaverse UID grasp transforms
    },
)

# Sources we install fully upfront — skip per-scene resolution entirely.
FULLY_INSTALLED_OBJECT_SOURCES = {"thor", "objathor_metadata"}


def build_argparser() -> argparse.ArgumentParser:
    """CLI for choosing scene sources, counts, and cache locations."""
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--project-root", type=Path,
                   default=Path(__file__).resolve().parents[2],
                   help="Repo root containing molmospaces/assets")
    p.add_argument("--cache-dir", type=Path,
                   default=Path("~/.cache/molmo-spaces-resources").expanduser(),
                   help="Where extracted archives live")
    p.add_argument("--start", type=int, default=0,
                   help="Skip the first N archives per source before applying the "
                        "count. Lets a second machine grab a disjoint slice: "
                        "machine A `--<src> 50000`, machine B `--<src> -1 --start 50000`.")

    # Per-source counts. -1 = all available archives. 0 = skip.
    # Defaults mirror the notebook's TO_DOWNLOAD example.
    p.add_argument("--ithor", type=int, default=-1,
                   help="iTHOR scenes count (-1=all 122). Default: -1 (~0.5GB)")
    p.add_argument("--procthor-10k-train", type=int, default=1000,
                   help="ProcTHOR-10K train scenes count (-1=all 10000). Default: 1000")
    p.add_argument("--procthor-10k-val", type=int, default=1000,
                   help="ProcTHOR-10K val scenes count (-1=all 1001). Default: 1000")
    p.add_argument("--procthor-10k-test", type=int, default=0,
                   help="ProcTHOR-10K test scenes count (-1=all 1002). Default: 0")
    p.add_argument("--procthor-objaverse-train", type=int, default=1000,
                   help="ProcTHOR-Objaverse train scenes (-1=all 99925). Default: 1000")
    p.add_argument("--procthor-objaverse-val", type=int, default=0,
                   help="ProcTHOR-Objaverse val scenes (-1=all 10000). Default: 0")
    p.add_argument("--holodeck-objaverse-train", type=int, default=1000,
                   help="Holodeck-Objaverse train scenes (-1=all 99997). Default: 1000")
    p.add_argument("--holodeck-objaverse-val", type=int, default=0,
                   help="Holodeck-Objaverse val scenes (-1=all 10001). Default: 0")

    p.add_argument("--no-objects", action="store_true",
                   help="Skip downloading object meshes (scenes won't render properly without these)")
    p.add_argument("--no-grasps", action="store_true",
                   help="Skip downloading grasp transforms")
    p.add_argument("--no-verify", action="store_true",
                   help="Skip the final disk-size verification step")
    p.add_argument("--dry-run", action="store_true",
                   help="Plan & print download size, do NOT actually download")
    return p


def setup_env(args):
    """Prepare cache/assets dirs and construct the resource manager."""
    assets_dir = args.project_root / "molmospaces" / "assets"
    assert assets_dir.is_dir(), f"ASSETS_DIR {assets_dir} does not exist — set --project-root manually"

    os.environ["MLSPACES_ASSETS_DIR"] = str(assets_dir)
    os.environ["MLSPACES_CACHE_DIR"] = str(args.cache_dir)

    # Drop a stale local manifest if the cache is missing — forces fresh install.
    local_manifest = assets_dir / "mjthor_data_type_to_source_to_versions.json"
    if local_manifest.is_file() and not args.cache_dir.is_dir():
        print(f"[setup] stale local manifest detected (cache missing). removing → {local_manifest.name}")
        local_manifest.unlink()

    from molmospaces_resources import R2RemoteStorage, setup_resource_manager  # noqa: E402

    rm = setup_resource_manager(
        R2RemoteStorage("mujoco-thor-resources"),
        symlink_dir=assets_dir,
        versions=VERSIONS,
        cache_dir=args.cache_dir,
        env_prefix="MLSPACES",
    )
    print(f"[setup] PROJECT_ROOT = {args.project_root}")
    print(f"[setup] ASSETS_DIR   = {assets_dir}")
    print(f"[setup] CACHE_DIR    = {args.cache_dir}")
    return assets_dir, args.cache_dir, rm


def collect_plan(args, rm) -> dict[str, list[str]]:
    """Resolve per-source archive lists from CLI counts."""
    counts = {
        "ithor":                    args.ithor,
        "procthor-10k-train":       args.procthor_10k_train,
        "procthor-10k-val":         args.procthor_10k_val,
        "procthor-10k-test":        args.procthor_10k_test,
        "procthor-objaverse-train": args.procthor_objaverse_train,
        "procthor-objaverse-val":   args.procthor_objaverse_val,
        "holodeck-objaverse-train": args.holodeck_objaverse_train,
        "holodeck-objaverse-val":   args.holodeck_objaverse_val,
    }
    start = max(0, int(getattr(args, "start", 0)))
    plan: dict[str, list[str]] = {}
    for source, count in counts.items():
        if count == 0:
            continue
        all_archives = sorted(rm.find_all_packages_for_source("scenes", source))
        # `--start N` skips the first N archives so a second machine can grab a
        # disjoint slice (e.g. --start 50000 + count -1 = the back half).
        sliced = all_archives[start:]
        plan[source] = sliced if count < 0 else sliced[:count]
    return plan


def print_plan_summary(plan: dict[str, list[str]], rm) -> None:
    """Print which archives will be downloaded per source."""
    from molmospaces_resources.constants import REMOTE_MANIFEST_NAME
    print(f"\n{'source':30s} {'archives':>10s} {'download MB':>14s} {'~extracted MB':>16s}")
    print("-" * 73)
    total_mb = 0.0
    total_count = 0
    for source, archives in plan.items():
        with open(rm.symlink_path("scenes", source) / REMOTE_MANIFEST_NAME) as f:
            manifest = json.load(f)
        mb = sum(manifest.get(a, 0) for a in archives)
        total_mb += mb
        total_count += len(archives)
        print(f"{source:30s} {len(archives):>10d} {mb:>14.1f} {mb * 40:>16.0f}")
    print("-" * 73)
    print(f"{'TOTAL':30s} {total_count:>10d} {total_mb:>14.1f} {total_mb * 40 / 1024:>14.1f} GB")


def install_scenes(plan: dict[str, list[str]], rm) -> None:
    """Download, extract, and symlink the planned scene archives."""
    from molmospaces_resources.constants import REMOTE_MANIFEST_NAME

    for source, archives in plan.items():
        if not archives:
            continue

        # PER_FILE sources store their manifest in assets/ rather than cache/.
        # install_packages() reads from cache/, so copy if missing.
        cache_manifest = rm.cache_path("scenes", source) / REMOTE_MANIFEST_NAME
        if not cache_manifest.exists():
            assets_manifest = rm.symlink_path("scenes", source) / REMOTE_MANIFEST_NAME
            cache_manifest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(assets_manifest, cache_manifest)

        print(f"\n=== Installing {len(archives)} scene archive(s) from {source} ===")
        rm.install_packages("scenes", {source: archives})


def _ensure_object_manifest_in_cache(source: str, rm) -> bool:
    """Copy the manifest from the assets dir into the cache; False if it isn't bootstrapped."""
    from molmospaces_resources.constants import REMOTE_MANIFEST_NAME
    cache_manifest = rm.cache_path("objects", source) / REMOTE_MANIFEST_NAME
    if cache_manifest.exists():
        return True
    assets_manifest = rm.symlink_path("objects", source) / REMOTE_MANIFEST_NAME
    if not assets_manifest.exists():
        return False
    cache_manifest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(assets_manifest, cache_manifest)
    return True


def install_eager_objects(rm) -> None:
    """Fully install upfront sources: thor objects + objathor metadata."""
    for source in sorted(FULLY_INSTALLED_OBJECT_SOURCES):
        if not _ensure_object_manifest_in_cache(source, rm):
            print(f"[objects] {source} manifest missing — skipping")
            continue
        archives = rm.find_all_packages_for_source("objects", source)
        print(f"\n=== Installing {len(archives)} object archive(s) from {source} ===")
        rm.install_packages("objects", {source: archives})


def find_object_refs(xml_path: Path, assets_dir: Path):
    """Yield (object_source, relative_path_within_source) for each external mesh/texture."""
    try:
        tree = ET.parse(xml_path)
    except ET.ParseError:
        return
    root = tree.getroot()
    scene_dir = xml_path.parent
    objects_root = assets_dir / "objects"
    for asset_type in ("mesh", "texture", "material", "hfield", "skin"):
        for elem in root.findall(f".//asset/{asset_type}"):
            fp = elem.attrib.get("file")
            if not fp or not fp.startswith("../"):
                continue
            normalized = Path(os.path.normpath(scene_dir / fp))
            try:
                rel_to_objects = normalized.relative_to(objects_root)
            except ValueError:
                continue
            parts = rel_to_objects.parts
            if len(parts) < 2:
                continue
            yield parts[0], Path(*parts[1:])


def install_objects_for_scenes(scene_paths: list[Path], assets_dir: Path, rm) -> None:
    """Install only the object archives referenced by the actually-downloaded scenes."""
    from tqdm import tqdm

    # Collect (source, folder_key, representative_path) — one rep per folder.
    source_to_key_to_rep = defaultdict(dict)
    print(f"\n[objects] scanning {len(scene_paths)} scene XMLs for object refs...")
    for scene_xml in tqdm(scene_paths, unit="scene"):
        for source, rel in find_object_refs(scene_xml, assets_dir):
            if source in FULLY_INSTALLED_OBJECT_SOURCES:
                continue
            key = rel.parts[0]  # UID / category folder
            if key not in source_to_key_to_rep[source]:
                source_to_key_to_rep[source][key] = rel

    for source, key_to_rep in source_to_key_to_rep.items():
        reps = list(key_to_rep.values())
        print(f"\n[objects] {source}: {len(reps)} unique folders across {len(scene_paths)} scenes")
        archives: set[str] = set()
        for chunk_start in tqdm(range(0, len(reps), 1000), desc=f"  resolving {source}", unit="chunk"):
            chunk = reps[chunk_start:chunk_start + 1000]
            archives.update(rm.find_archives("objects", source, chunk))
        archives_sorted = sorted(archives)
        if not archives_sorted:
            print("  No archives needed.")
            continue
        if not _ensure_object_manifest_in_cache(source, rm):
            print(f"  [skip] {source} manifest missing")
            continue
        print(f"  installing {len(archives_sorted)} {source} archives...")
        rm.install_packages("objects", {source: archives_sorted})


def install_grasps(rm) -> None:
    """Download and install the grasp archives."""
    from molmospaces_resources.constants import REMOTE_MANIFEST_NAME

    for source in ("droid", "droid_objaverse"):
        cache_manifest = rm.cache_path("grasps", source) / REMOTE_MANIFEST_NAME
        if not cache_manifest.exists():
            assets_manifest = rm.symlink_path("grasps", source) / REMOTE_MANIFEST_NAME
            if not assets_manifest.exists():
                print(f"[grasps] {source} manifest missing on disk; skipping")
                continue
            cache_manifest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(assets_manifest, cache_manifest)
        archives = rm.find_all_packages_for_source("grasps", source)
        print(f"\n=== Installing {len(archives)} {source} grasp archive(s) ===")
        rm.install_packages("grasps", {source: archives})


def post_install_fixup(cache_dir: Path, assets_dir: Path) -> None:
    """Re-symlink sources that were materialised as real dirs; safe to re-run."""
    GLOBAL_PAIRS = [
        ("objects", "thor"),
        ("objects", "objaverse"),
        ("grasps", "droid"),
        ("grasps", "droid_objaverse"),
    ]
    print("\n[fixup] reconciling cache → assets symlinks...")
    for data_type, source in GLOBAL_PAIRS:
        cache_base = cache_dir / data_type / source
        if not cache_base.is_dir():
            continue
        versions = sorted((d for d in cache_base.iterdir() if d.is_dir()), reverse=True)
        if not versions:
            continue
        cache_versioned = versions[0]
        target_dir = assets_dir / data_type / source
        if target_dir.is_symlink():
            print(f"  {data_type}/{source}: managed symlink (skipped)")
            continue
        target_dir.mkdir(parents=True, exist_ok=True)
        linked = already = 0
        for entry in cache_versioned.iterdir():
            if not entry.is_dir():
                continue
            link_target = target_dir / entry.name
            if link_target.exists() or link_target.is_symlink():
                already += 1
            else:
                os.symlink(entry, link_target, target_is_directory=True)
                linked += 1
        print(f"  {data_type}/{source}: linked {linked} new (already had {already})")


def verify(assets_dir: Path) -> None:
    """Count installed scenes/objects/grasps and report."""
    scenes_root = assets_dir / "scenes"
    print(f"\n{'source':30s} {'xml_files':>10s} {'disk size':>12s}")
    print("-" * 56)
    for d in sorted(scenes_root.iterdir()):
        if not d.is_dir():
            continue
        xmls = [p for p in d.rglob("*.xml")
                if not any(p.name.endswith(s) for s in ("_ceiling.xml", "_orig.xml", "_non_settled.xml"))]
        try:
            size = subprocess.check_output(["du", "-sh", "-L", str(d)]).decode().split()[0]
        except Exception:
            size = "?"
        print(f"{d.name:30s} {len(xmls):>10d} {size:>12s}")


def main():
    """Scene/object/grasp downloader entry point."""
    args = build_argparser().parse_args()
    assets_dir, cache_dir, rm = setup_env(args)

    plan = collect_plan(args, rm)
    if plan:
        print_plan_summary(plan, rm)
    else:
        print("No scene archives requested (all counts 0); still installing objects/grasps.")
    if args.dry_run:
        print("\n[dry-run] exiting before download")
        return

    if plan:
        install_scenes(plan, rm)

    if not args.no_objects:
        install_eager_objects(rm)
        # Collect all scene XMLs currently on disk (not just from this run's plan)
        # so newly-added scenes can pull their referenced objaverse objects.
        scenes_root = assets_dir / "scenes"
        scene_xmls = [
            p for p in scenes_root.rglob("*.xml")
            if not any(p.name.endswith(s) for s in ("_ceiling.xml", "_orig.xml", "_non_settled.xml"))
        ]
        print(f"\n[objects] found {len(scene_xmls)} scene XMLs across all sources")
        install_objects_for_scenes(scene_xmls, assets_dir, rm)

    if not args.no_grasps:
        install_grasps(rm)

    post_install_fixup(cache_dir, assets_dir)

    if not args.no_verify:
        verify(assets_dir)

    print("\nDone.")


if __name__ == "__main__":
    sys.exit(main())
