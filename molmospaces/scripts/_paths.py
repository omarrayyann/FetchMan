"""Asset directory lookup shared by the scripts in this directory."""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ASSETS = REPO_ROOT / "molmospaces" / "assets"


def scene_roots(subdir=None):
    """Scene directories that exist: the downloaded set, the sample set, or both."""
    roots = [ASSETS / name for name in ("scenes", "sample_scenes")]
    if subdir:
        roots = [r / subdir for r in roots]
    return [r for r in roots if r.is_dir()]
