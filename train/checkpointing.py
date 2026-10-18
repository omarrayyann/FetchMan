"""Checkpoint resume and torch.compile wiring."""
import os

import torch
from absl import flags

FLAGS = flags.FLAGS


def _checkpoint_has_training_state(ckpt):
    """True if the checkpoint carries optimizer + scheduler state (resumable)."""
    return (
        isinstance(ckpt, dict)
        and bool(ckpt.get('optimizer_states'))
        and 'lr_schedulers' in ckpt
    )


def _load_weights_only(agent, ckpt_path):
    """Load just the model weights from a checkpoint (non-strict); returns (missing, unexpected)."""
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state_dict = ckpt.get('state_dict', ckpt) if isinstance(ckpt, dict) else ckpt
    missing, unexpected = agent.load_state_dict(state_dict, strict=False)
    return missing, unexpected


def _compile_agent(agent, is_rank_zero):
    """torch.compile the encoder and velocity model, rewriting checkpoint keys to match."""
    if FLAGS.compile:
        compiled_prefixes = []
        try:
            if getattr(agent, 'encoder', None) is not None:
                if hasattr(agent.encoder, 'backbone'):
                    agent.encoder.backbone = torch.compile(agent.encoder.backbone, mode=FLAGS.compile_mode)
                    compiled_prefixes.append('encoder.backbone.')
                else:
                    agent.encoder = torch.compile(agent.encoder, mode=FLAGS.compile_mode)
                    compiled_prefixes.append('encoder.')
            if getattr(agent, 'flow_model', None) is not None:
                agent.flow_model = torch.compile(agent.flow_model, mode=FLAGS.compile_mode)
                compiled_prefixes.append('flow_model.')
            if is_rank_zero:
                print(f"[compile] enabled with mode={FLAGS.compile_mode}; first step will be slow.")

            # torch.compile prefixes compiled-module params with `_orig_mod.`,
            # so pre-compile checkpoints don't load directly. Strip/insert in-place
            # on the incoming state_dict so the EMA callback sees the rewritten keys too.
            _orig_load = agent.load_state_dict

            def _patched_load(state_dict, strict=True, assign=False):
                """Translate state_dict keys between compiled (_orig_mod) and plain module layouts on resume."""
                for k in list(state_dict.keys()):
                    if '._orig_mod.' in k:
                        state_dict[k.replace('._orig_mod.', '.')] = state_dict.pop(k)
                for k in list(state_dict.keys()):
                    for p in compiled_prefixes:
                        if k.startswith(p) and not k.startswith(p + '_orig_mod.'):
                            state_dict[k.replace(p, p + '_orig_mod.', 1)] = state_dict.pop(k)
                            break
                return _orig_load(state_dict, strict=strict, assign=assign)

            agent.load_state_dict = _patched_load
        except Exception as e:
            if is_rank_zero:
                print(f"[compile] failed: {type(e).__name__}: {e}; running uncompiled.")


def _resume_from_checkpoint(agent, is_rank_zero):
    """Return a Trainer ckpt_path for a full checkpoint, or load weights-only in place."""
    fit_ckpt_path = None
    if FLAGS.resume_ckpt:
        resume_path = os.path.expanduser(FLAGS.resume_ckpt)
        if not os.path.exists(resume_path):
            raise FileNotFoundError(f"--resume_ckpt does not exist: {resume_path}")
        ckpt_meta = torch.load(resume_path, map_location='cpu')
        if _checkpoint_has_training_state(ckpt_meta):
            fit_ckpt_path = resume_path
            if is_rank_zero:
                print(f"  resuming full training state from {resume_path}")
        else:
            del ckpt_meta
            missing, unexpected = _load_weights_only(agent, resume_path)
            if is_rank_zero:
                print(f"  loaded model weights from weights-only checkpoint {resume_path}")
                if missing:
                    print(f"    missing keys: {len(missing)}")
                if unexpected:
                    print(f"    unexpected keys: {len(unexpected)}")
    return fit_ckpt_path


