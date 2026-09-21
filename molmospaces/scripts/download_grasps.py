"""Download the droid and droid_objaverse grasp transforms.

download_scenes.py skips grasps when the local manifest is missing. This helper
bootstraps the grasp manifests from R2 and installs the archives, so run it
after download_scenes.py.

Usage:
    python molmospaces/scripts/download_grasps.py [--cache-dir=DIR]
"""
import argparse
import os
from pathlib import Path

from download_scenes import VERSIONS

R2_BASE = "https://pub-3555e9bb2d304fab9c6c79819e48aa40.r2.dev"
SOURCES = ("droid", "droid_objaverse")


def parse_args():
    """Command-line interface."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    ap.add_argument("--cache-dir", type=Path,
                    default=Path("~/.cache/molmo-spaces-resources").expanduser())
    return ap.parse_args()


def bootstrap_manifest(source, version, cache_dir):
    """Pre-fetch a source's R2 manifests so the resource manager sees its archives."""
    import requests
    from molmospaces_resources.constants import REMOTE_MANIFEST_NAME, COMBINED_TRIES_NAME

    version_dir = cache_dir / "grasps" / source / version
    version_dir.mkdir(parents=True, exist_ok=True)
    for fname in (REMOTE_MANIFEST_NAME, COMBINED_TRIES_NAME):
        target = version_dir / fname
        if target.exists() and target.stat().st_size > 0:
            continue
        print(f"  bootstrapping {fname} from R2")
        resp = requests.get(f"{R2_BASE}/grasps/{source}/{version}/{fname}", timeout=60)
        resp.raise_for_status()
        target.write_bytes(resp.content)
        print(f"    wrote {target} ({len(resp.content):,} bytes)")


def make_resource_manager(assets_dir, cache_dir):
    """Resource manager pointed at this repo's assets dir and the given cache."""
    from molmospaces_resources import R2RemoteStorage, setup_resource_manager
    rm = setup_resource_manager(
        R2RemoteStorage("mujoco-thor-resources"),
        symlink_dir=assets_dir,
        versions=VERSIONS,
        cache_dir=cache_dir,
        env_prefix="MLSPACES",
    )
    print(f"[setup] ASSETS_DIR = {assets_dir}")
    print(f"[setup] CACHE_DIR  = {cache_dir}")
    return rm


def install_grasps(rm):
    """Resolve and install every available grasp archive per source."""
    for source in SOURCES:
        print(f"\n=== Resolving {source} grasp archives from R2 ===")
        archives = rm.find_all_packages_for_source("grasps", source)
        print(f"  {len(archives)} archive(s) available remotely")
        if archives:
            print(f"=== Installing {len(archives)} {source} grasp archive(s) ===")
            rm.install_packages("grasps", {source: archives})


def reconcile_symlinks(cache_dir, assets_dir):
    """Link each cached grasp dir into assets/grasps, mirroring download_scenes.py's fix-up."""
    print("\n[fixup] reconciling cache -> assets symlinks for grasps...")
    for source in SOURCES:
        cache_base = cache_dir / "grasps" / source
        versions = sorted((d for d in cache_base.iterdir() if d.is_dir()), reverse=True) \
            if cache_base.is_dir() else []
        if not versions:
            continue
        target_dir = assets_dir / "grasps" / source
        if target_dir.is_symlink():
            print(f"  grasps/{source}: managed symlink (skipped)")
            continue
        target_dir.mkdir(parents=True, exist_ok=True)
        linked = already = 0
        for entry in versions[0].iterdir():
            if not entry.is_dir():
                continue
            link = target_dir / entry.name
            if link.exists() or link.is_symlink():
                already += 1
            else:
                os.symlink(entry, link, target_is_directory=True)
                linked += 1
        print(f"  grasps/{source}: linked {linked} new (already had {already})")


def main():
    """Download and install the grasp archives into assets/grasps."""
    args = parse_args()
    assets_dir = args.project_root / "molmospaces" / "assets"
    os.environ["MLSPACES_ASSETS_DIR"] = str(assets_dir)
    os.environ["MLSPACES_CACHE_DIR"] = str(args.cache_dir)

    print("=== Bootstrapping grasp manifests from R2 ===")
    for source, version in VERSIONS["grasps"].items():
        bootstrap_manifest(source, version, args.cache_dir)

    rm = make_resource_manager(assets_dir, args.cache_dir)
    install_grasps(rm)
    reconcile_symlinks(args.cache_dir, assets_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()
