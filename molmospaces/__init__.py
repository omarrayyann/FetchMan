import functools
import os
import re
from pathlib import Path

# Assets bundled inside the molmospaces package. MOLMOSPACES_ASSETS_DIR points
# at a full tree elsewhere, e.g. one kept on a data disk or shared across clones.
_assets_env = os.environ.get("MOLMOSPACES_ASSETS_DIR", "")
ASSETS_DIR: Path = Path(_assets_env).expanduser() if _assets_env else Path(__file__).parent / "assets"

# Grasp files default to assets/grasps but can live anywhere via env var
_grasps_env = os.environ.get("MOLMOSPACES_GRASPS_DIR", "")
GRASPS_DIR: Path = Path(_grasps_env).expanduser() if _grasps_env else ASSETS_DIR / "grasps"


@functools.lru_cache(maxsize=None)
def grasp_source_dir(source: str) -> Path:
    """Path to grasps/<source>, accepting the cache layout that nests a version dir."""
    d = GRASPS_DIR / source if source else GRASPS_DIR
    if d.is_dir():
        versions = []
        for k in d.iterdir():
            if not (k.is_dir() and re.fullmatch(r"\d{8}", k.name)):
                versions = []
                break
            versions.append(k)
        if versions:
            return max(versions, key=lambda k: k.name)
    return d
