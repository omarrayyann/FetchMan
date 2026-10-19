"""Write a BC or RL checkpoint as a self-contained release folder.

    python train/export_checkpoint.py --checkpoint=exp/.../step=00010000.ckpt --output=release/bowls_bc

The folder holds model.ckpt (the policy weights alone; EMA for a BC checkpoint),
stats.json and a flags.json reduced to the agent config. That config records the
DINOv3 backbone's architecture, so the folder loads without downloading Meta's
gated pretrained weights: every backbone weight is already in model.ckpt.
"""
import json
import shutil
from pathlib import Path

import torch
from absl import app, flags
from transformers import AutoConfig

flags.DEFINE_string("checkpoint", None, "BC or RL checkpoint to export.")
flags.DEFINE_string("output", None, "Release folder to create.")
flags.DEFINE_bool("use_ema", True, "Ship a BC checkpoint's EMA weights rather than its raw ones.")
flags.mark_flags_as_required(["checkpoint", "output"])
FLAGS = flags.FLAGS

DEFAULT_DINO = "facebook/dinov3-vitb16-pretrain-lvd1689m"


def main(_):
    """Copy the weights, normalization stats and agent config into --output."""
    src = Path(FLAGS.checkpoint).resolve()
    out = Path(FLAGS.output)
    out.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(src, map_location="cpu", weights_only=False)
    key = "state_dict" if FLAGS.use_ema or "current_model_state" not in ckpt else "current_model_state"
    torch.save({"state_dict": ckpt[key]}, out / "model.ckpt")

    agent = json.load(open(src.parent / "flags.json"))["agent"]
    if str(agent.get("encoder", "")).startswith("dinov3_patches"):
        agent["dino_config"] = AutoConfig.from_pretrained(agent.get("dino_model") or DEFAULT_DINO).to_diff_dict()
    json.dump({"agent": agent}, open(out / "flags.json", "w"), indent=1)
    shutil.copy2(src.parent / "stats.json", out / "stats.json")
    print(f"{src.name} ({key}) -> {out}/", flush=True)


if __name__ == "__main__":
    app.run(main)
