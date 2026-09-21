import functools

from molmospaces import GRASPS_DIR, grasp_source_dir

THOR_PICKUP_TYPES = [
    "alarmclock", "aluminumfoil", "apple", "bathtowel", "boot", "bottle",
    "bowl", "box", "bread", "butterknife", "candle", "cd", "cellphone",
    "cloth", "creditcard", "cup", "dishsponge", "dogstatue", "dumbbell",
    "egg", "eggcracked", "fork", "handtowel", "kettle", "keychain",
    "knife", "ladle", "laptop", "lettuce", "mug", "newspaper", "pan",
    "papertowel", "pen", "pencil", "peppershaker", "pillow", "plate",
    "plunger", "pot", "potato", "remote", "saltshaker", "scrubbrush",
    "soapbar", "soapbottle", "spatula", "spoon", "spraybottle",
    "stonestatue", "tabletopdecor", "teddybear", "tennisracquet",
    "tissuebox", "toilet", "toiletpaper", "toiletpaperused up", "tomato",
    "towelstatue", "vasedecorative", "vaseflat", "vasemedium", "vaseopen",
    "vasetall", "walldecorpainting", "watch", "wateringcan", "wavingstatue",
    "winebottle",
]

# Categories excluded from THOR_PICKUP_TYPES — too soft / large / deformable to grasp well.
THOR_PICKUP_BLACKLIST = [
    "pillow",
]


# Maps a substring of a body/geom name to a THOR material category. Used by the
# scene texture randomizer to pick category-appropriate textures (walls get wall
# textures, floors get floor textures, etc.). The right-hand sides must match
# keys in objects/thor/material-database.json.
SCENE_TEXTURE_CATEGORIES = {
    "wall":       "Wall",
    "backsplash": "Wall",
    "ceiling":    "Wall",
    "floor":      "Floor",
    "room":       "Floor",       # ProcTHOR floor body name
    "counter":    "CounterTop",
    "countertop": "CounterTop",
    "island":     "Table",       # upstream convention
    "table":      "Table",
    "desk":       "Table",
    "shelf":      "Table",
    "shelving":   "Table",
    "doorway":    "Doorway",
    "door":       "Doorway",
    "drawer":     "Doorway",
    "cabinet":    "Doorway",
    "handle":     "Doorway",
}


def classify_scene_geom(body_name: str, geom_name: str) -> str | None:
    """THOR category for a scene geom from its body/geom name, or None if architectural keywords don't match."""
    b = (body_name or "").lower()
    g = (geom_name or "").lower()
    for kw, cat in SCENE_TEXTURE_CATEGORIES.items():
        if kw in b or kw in g:
            return cat
    return None


def is_pickup_type(category):
    """True if the THOR category is a pickupable type."""
    cat = category.lower().replace("_", "")
    if any(cat.startswith(t) or t.startswith(cat) for t in THOR_PICKUP_BLACKLIST):
        return False
    return any(cat.startswith(t) or t.startswith(cat) for t in THOR_PICKUP_TYPES)


_VALID_GRASP_UIDS_CACHE = GRASPS_DIR / ".valid_uids.txt"


@functools.lru_cache(maxsize=1)
def _valid_grasp_uids() -> frozenset:
    """Asset IDs that have a grasp file on disk, cached in assets/grasps/.valid_uids.txt."""
    base = GRASPS_DIR
    if not base.is_dir():
        return frozenset()
    if _VALID_GRASP_UIDS_CACHE.exists():
        with open(_VALID_GRASP_UIDS_CACHE) as f:
            return frozenset(line.strip() for line in f if line.strip())
    uids = set()
    for sub in (grasp_source_dir(""), grasp_source_dir("droid"), grasp_source_dir("droid_objaverse")):
        if not sub.is_dir():
            continue
        for d in sub.iterdir():
            if not d.is_dir():
                continue
            if (d / f"{d.name}_grasps_filtered.npz").exists():
                uids.add(d.name)
    try:
        with open(_VALID_GRASP_UIDS_CACHE, "w") as f:
            for u in sorted(uids):
                f.write(u + "\n")
    except OSError:
        pass
    return frozenset(uids)


def has_valid_grasp(asset_id: str) -> bool:
    """True iff a grasp transform file exists for this asset_id."""
    if not asset_id:
        return False
    return asset_id in _valid_grasp_uids()

