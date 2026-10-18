import math
import torch
import torch.nn as nn


class SinusoidalPosEmb(nn.Module):
    # For [0,1] flow-matching timesteps pass scale=1000 so the encoding uses the
    # full frequency band (default scale=1 leaves half the dims near-constant).
    def __init__(self, dim: int, theta: int = 10000, scale: float = 1.0):
        """Sinusoidal time/position embedding module."""
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.scale = float(scale)

    def forward(self, x: torch.Tensor):
        """Embed scalar positions into sin/cos features."""
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = (x * self.scale)[..., None] * emb[None, ...]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb
