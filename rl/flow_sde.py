"""Flow-SDE math: stochastic sampling and chain log-probabilities.

Euler-Maruyama over the rectified flow with sigma_t = noise_level * sqrt((1-t)/t)
in the t=0->1 (noise->clean) convention, which is what makes the flow's
log-density tractable and therefore makes a PPO ratio well defined.
"""
import sys
from pathlib import Path

import numpy as np
import torch
from absl import flags

ROOT_DIR = Path(__file__).resolve().parents[1]
TRAIN_DIR = ROOT_DIR / "train"
for _p in (ROOT_DIR, TRAIN_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

FLAGS = flags.FLAGS


# Flow-SDE math
def sde_sigma_schedule(noise_level: float, num_steps: int, device) -> torch.Tensor:
    """sigma_t = noise_level * sqrt((1-t)/t) for our t=0→1 (noise→clean) convention."""
    ts = torch.linspace(0.0, 1.0, num_steps + 1, device=device)[:-1]
    safe_t = torch.where(ts == 0.0, ts.new_tensor([ts[1].item()]), ts)
    return noise_level * torch.sqrt((1.0 - safe_t) / safe_t)


def _gaussian_logprob(sample: torch.Tensor, mean: torch.Tensor, sigma: torch.Tensor):
    """Elementwise diagonal-gaussian log-density with the configured sigma floor."""
    sigma = sigma.clamp_min(1e-6)   # numerical guard only
    return -torch.log(sigma) - 0.5 * np.log(2.0 * np.pi) - 0.5 * ((sample - mean) / sigma) ** 2


def sample_actions_sde(policy, obs_batch, schedule, sigmas, batch_size: int,
                       compute_logprob: bool = False, noise_step: int = None,
                       return_cond: bool = False, return_noise_step: bool = False):
    """Stochastic flow rollout. Returns (actions, chain, [logprob], [cond]).

    Noise is injected at EXACTLY ONE flow step (`noise_step`); every other step
    runs the deterministic flow update. Scoring a single step keeps the PPO ratio
    in a trust region: summing log-probs over all K steps compounds the 1/sigma^2
    sensitivity of the smallest-sigma step into ratio spikes.

    If return_cond=True, also returns the encoder output (`global_cond`). For
    token-memory encoders (dinov3_patches) this is a dict; for pooled encoders
    it's a tensor. Caching this output lets the PPO update skip the heavy
    encoder forward pass during recompute when the encoder is frozen.
    """
    cfg = policy.config
    K = int(cfg["num_sampling_steps"]) - 1
    H, A = int(cfg["prediction_horizon"]), int(cfg["action_dim"])
    device = policy.device

    global_cond = policy._build_cond(obs_batch)
    cond_for_cache = global_cond  # before view-reshape; preserved for caching
    if policy._uses_token_memory:
        memory = policy._build_memory(global_cond)
    elif cfg.architecture in ("unet", "dit"):
        global_cond = global_cond.view(batch_size, -1)

    naction = torch.randn(batch_size, H, A, device=device)
    chain = torch.empty(batch_size, K + 1, H, A, device=device)
    chain[:, 0] = naction
    logprob = chain.new_zeros(batch_size) if compute_logprob else None
    # Default to a RANDOM noise_step (uniform across flow steps) matching
    # main_ppopi's `per_sample_logprob=False` behavior. Hardcoding noise_step=0
    # only ever supervises the very first flow step, biasing the policy.
    if noise_step is None:
        noise_step_int = int(torch.randint(0, K, (1,)).item())
    else:
        noise_step_int = int(noise_step)

    for m in range(K):
        curr_t, next_t = schedule[m], schedule[m + 1]
        if policy._uses_token_memory:
            v_t = policy.flow_model(naction, curr_t, memory)
        elif cfg.architecture in ("unet", "dit"):
            v_t = policy.flow_model(naction, curr_t, global_cond)
        else:
            v_t = policy.flow_model(naction, curr_t, global_cond, is_causal=cfg.is_causal)
        dt = (next_t - curr_t).unsqueeze(-1)
        mean = naction + v_t * dt
        inject_noise = (m == noise_step_int)
        if inject_noise:
            sigma_eff = sigmas[m] * dt.abs().sqrt()
            eps = torch.randn_like(naction)
            naction = mean + sigma_eff * eps
            if compute_logprob:
                lp = _gaussian_logprob(naction, mean, sigma_eff)
                logprob = logprob + lp.sum(dim=(1, 2))
        else:
            naction = mean
        chain[:, m + 1] = naction

    actions = policy._unnormalize(naction, "action")
    out = (actions, chain)
    if compute_logprob: out = out + (logprob,)
    if return_cond:     out = out + (cond_for_cache,)
    if return_noise_step: out = out + (noise_step_int,)
    return out


def recompute_chain_logprob_cached(policy, encoder_out, chain, sigmas, schedule, batch_size: int,
                                    noise_step=None):
    """Recompute chain log-prob skipping the encoder. `encoder_out` is the
    cached output of `policy._build_cond` from rollout time. Safe because
    the encoder is frozen. `noise_step` may be a scalar or a tensor of shape
    (B,) — when a tensor, we do a SINGLE flow_model forward with per-sample
    t (matching main_ppopi). This is critical for memory: looping over
    unique noise_steps and re-running the flow model would keep K forwards
    of activations for backward."""
    cfg = policy.config
    if policy._uses_token_memory:
        memory = policy._build_memory(encoder_out)
        global_cond = None
    elif cfg.architecture in ("unet", "dit"):
        global_cond = encoder_out.view(batch_size, -1)
        memory = None
    else:
        global_cond = encoder_out
        memory = None

    # Per-sample noise step (or scalar) — single flow forward, per-sample t.
    if isinstance(noise_step, torch.Tensor):
        n = noise_step.long()
    else:
        n = torch.full((batch_size,), int(noise_step), dtype=torch.long,
                       device=chain.device)
    arange_b = torch.arange(batch_size, device=chain.device)
    x_t = chain[arange_b, n]              # (B, H, A)
    x_t_next = chain[arange_b, n + 1]     # (B, H, A)
    curr_t = schedule[n]                  # (B,)
    next_t = schedule[n + 1]              # (B,)
    dt = (next_t - curr_t).view(batch_size, 1, 1)  # (B, 1, 1)
    sigma_step = (sigmas[n].view(batch_size, 1, 1)) * dt.abs().sqrt()

    if policy._uses_token_memory:
        v_t = policy.flow_model(x_t, curr_t, memory)
    elif cfg.architecture in ("unet", "dit"):
        v_t = policy.flow_model(x_t, curr_t, global_cond)
    else:
        v_t = policy.flow_model(x_t, curr_t, global_cond, is_causal=cfg.is_causal)
    mean = x_t + v_t * dt
    lp = _gaussian_logprob(x_t_next, mean, sigma_step).sum(dim=(1, 2))
    return lp


def recompute_chain_kl_to_ref_cached(policy, ref_policy, encoder_out, chain, sigmas, schedule,
                                     batch_size: int,
                                     noise_step=None):
    """Exact Gaussian KL for the SDE transition(s), using cached encoder output.

    Both policies use the same sigma schedule, so each step's KL is
    0.5 * ||mean_theta - mean_ref||^2 / sigma^2, summed over action dims. This is
    a real positive anchor to the reference policy. It replaces the sampled
    log-prob difference heuristic, whose gradient is not an exact KL gradient.
    """
    cfg = policy.config

    def _cond_for(p):
        """Conditioning tensors (global_cond, memory) for a policy per its architecture."""
        if p._uses_token_memory:
            return None, p._build_memory(encoder_out)
        if cfg.architecture in ("unet", "dit"):
            return encoder_out.view(batch_size, -1), None
        return encoder_out, None

    global_cond, memory = _cond_for(policy)
    ref_global_cond, ref_memory = _cond_for(ref_policy)

    def _flow(p, x_t, t, gc, mem):
        """Dispatch one velocity-model forward for the policy's architecture."""
        if p._uses_token_memory:
            return p.flow_model(x_t, t, mem)
        if cfg.architecture in ("unet", "dit"):
            return p.flow_model(x_t, t, gc)
        return p.flow_model(x_t, t, gc, is_causal=cfg.is_causal)

    def _step_kl(m, batch_index=None):
        """Per-step gaussian KL between policy and reference at flow step m."""
        curr_t, next_t = schedule[m], schedule[m + 1]
        if batch_index is None:
            x_t = chain[:, m]
            dt = (next_t - curr_t).unsqueeze(-1)
            sigma_step = sigmas[m] * dt.abs().sqrt()
        else:
            x_t = chain[batch_index, m]
            dt = (next_t - curr_t).view(batch_size, 1, 1)
            sigma_step = sigmas[m].view(batch_size, 1, 1) * dt.abs().sqrt()
        v_t = _flow(policy, x_t, curr_t, global_cond, memory)
        with torch.no_grad():
            v_ref = _flow(ref_policy, x_t, curr_t, ref_global_cond, ref_memory)
        mean = x_t + v_t * dt
        ref_mean = x_t + v_ref * dt
        return 0.5 * (((mean - ref_mean) / sigma_step.clamp_min(1e-6)) ** 2).sum(dim=(1, 2))

    if isinstance(noise_step, torch.Tensor):
        n = noise_step.long()
    else:
        n = torch.full((batch_size,), int(noise_step), dtype=torch.long,
                       device=chain.device)
    arange_b = torch.arange(batch_size, device=chain.device)
    curr_t = schedule[n]
    next_t = schedule[n + 1]
    x_t = chain[arange_b, n]
    dt = (next_t - curr_t).view(batch_size, 1, 1)
    sigma_step = sigmas[n].view(batch_size, 1, 1) * dt.abs().sqrt()
    v_t = _flow(policy, x_t, curr_t, global_cond, memory)
    with torch.no_grad():
        v_ref = _flow(ref_policy, x_t, curr_t, ref_global_cond, ref_memory)
    mean = x_t + v_t * dt
    ref_mean = x_t + v_ref * dt
    return 0.5 * (((mean - ref_mean) / sigma_step.clamp_min(1e-6)) ** 2).sum(dim=(1, 2))


def recompute_chain_logprob(policy, obs_batch, chain, sigmas, schedule, batch_size: int,
                            noise_step: int = None):
    """Re-roll the deterministic flow on a stored chain. Returns log-prob (B,).

    Scores the single flow step `noise_step` across the (H, A) dims.
    If False, only one step (noise_step) is evaluated — must pass noise_step.
    """
    cfg = policy.config

    global_cond = policy._build_cond(obs_batch)
    if policy._uses_token_memory:
        memory = policy._build_memory(global_cond)
    elif cfg.architecture in ("unet", "dit"):
        global_cond = global_cond.view(batch_size, -1)

    steps_to_eval = [int(noise_step)]
    total = chain.new_zeros(batch_size)
    for m in steps_to_eval:
        curr_t, next_t = schedule[m], schedule[m + 1]
        x_t = chain[:, m]
        x_t_next = chain[:, m + 1]
        if policy._uses_token_memory:
            v_t = policy.flow_model(x_t, curr_t, memory)
        elif cfg.architecture in ("unet", "dit"):
            v_t = policy.flow_model(x_t, curr_t, global_cond)
        else:
            v_t = policy.flow_model(x_t, curr_t, global_cond, is_causal=cfg.is_causal)
        dt = (next_t - curr_t).unsqueeze(-1)
        mean = x_t + v_t * dt
        sigma_step = sigmas[m] * dt.abs().sqrt()
        lp = _gaussian_logprob(x_t_next, mean, sigma_step)
        total = total + lp.sum(dim=(1, 2))
    return total
