import gzip
import json

import ml_collections

from molmospaces import ASSETS_DIR

TEMPLATES = [
    "pick up the {object}.",
    "grab the {object}.",
    "grasp the {object}.",
    "lift the {object}.",
    "take the {object}.",
    "get the {object}.",
]

_WORD_KEYS = ["one_word", "two_words", "three_words", "four_words", "five_words"]
_METADATA = None


def _load_metadata():
    """Load (cached) the objaverse/THOR annotation metadata table; empty if not downloaded."""
    global _METADATA
    if _METADATA is None:
        path = ASSETS_DIR / "objects" / "objathor_metadata" / "objects_metadata.json.gz"
        if not path.is_file():
            # Optional asset. THOR ids still yield a usable name from the id itself;
            # objaverse ids fall back to their uid prefix until it is downloaded.
            print(f"[prompt_sampler] {path.name} not found; naming objects from asset ids")
            _METADATA = {}
            return _METADATA
        with gzip.open(path) as f:
            raw = json.load(f)
        # older versions: dict keyed by assetId; 20251117+: list of entries
        _METADATA = raw if isinstance(raw, dict) else {e["assetId"]: e for e in raw}
    return _METADATA


def get_object_name(asset_id, num_words=1):
    """Short human name for an asset id from its annotations (num_words cap)."""
    entry = _load_metadata().get(asset_id)
    if entry is None:
        return asset_id.split("_")[0].lower()
    key = _WORD_KEYS[min(num_words, len(_WORD_KEYS)) - 1]
    return entry.get("description_short", {}).get(key, asset_id).lower()


class PromptSampler:
    def __init__(self, config=None):
        """Sampler of natural-language pick prompts from templates + object names."""
        config = config or get_config()
        self.templates = TEMPLATES if config.randomize else TEMPLATES[:1]
        self.num_words = config.num_words

    def sample(self, asset_id, rng):
        """Sample one prompt string for the asset."""
        name = get_object_name(asset_id, self.num_words)
        idx = int(rng.integers(len(self.templates)))
        return self.templates[idx].format(object=name)


def get_config():
    """Default prompt-sampler ConfigDict."""
    return ml_collections.ConfigDict(dict(
        randomize=False,
        num_words=1,
    ))