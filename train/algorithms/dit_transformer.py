import torch
import torch.nn as nn

from algorithms.common import SinusoidalPosEmb


def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """AdaLN modulation: scale-and-shift x by conditioning-derived parameters."""
    return x * (1 + scale) + shift


class DiTBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, cond_dim: int, dropout: float = 0.0):
        """DiT block: self-attention + MLP with adaLN-Zero conditioning."""
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_size, num_heads=num_heads,
                                          batch_first=True, dropout=dropout)
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.GELU(approximate="tanh"),
            nn.Linear(4 * hidden_size, hidden_size),
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 6 * hidden_size, bias=True),
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Apply gated, conditioned self-attention and MLP residuals."""
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(cond).chunk(6, dim=1)
        )
        attn_input = _modulate(self.norm1(x), shift_msa.unsqueeze(1), scale_msa.unsqueeze(1))
        attn_out, _ = self.attn(attn_input, attn_input, attn_input, need_weights=False)
        x = x + gate_msa.unsqueeze(1) * attn_out
        mlp_input = _modulate(self.norm2(x), shift_mlp.unsqueeze(1), scale_mlp.unsqueeze(1))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(mlp_input)
        return x


class DiTPolicy(nn.Module):
    def __init__(self, in_dim: int, cond_dim: int, horizon: int,
                 hidden_dim: int = 384, num_heads: int = 8, num_layers: int = 8,
                 dropout: float = 0.0, time_embed_dim: int = 128):
        """DiT velocity model conditioned on a flat global vector via adaLN."""
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, in_dim)
        self.pos_embedding = nn.Parameter(torch.empty(1, horizon, hidden_dim).normal_(std=0.02))

        self.time_mlp = nn.Sequential(
            # scale=1000: flow timesteps live in [0,1]; without scaling, high-period
            # sinusoidal dims carry no signal.
            SinusoidalPosEmb(time_embed_dim, scale=1000.0),
            nn.Linear(time_embed_dim, 2 * time_embed_dim),
            nn.GELU(),
            nn.Linear(2 * time_embed_dim, time_embed_dim),
            nn.GELU(),
        )

        cond_proj_dim = time_embed_dim + cond_dim
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_dim, num_heads, cond_proj_dim, dropout=dropout)
            for _ in range(num_layers)
        ])

        # AdaLN-Zero: zero-init the modulation Linear so each block is identity at init.
        for block in self.blocks:
            nn.init.zeros_(block.adaLN_modulation[-1].weight)
            nn.init.zeros_(block.adaLN_modulation[-1].bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor,
                is_causal: bool = False) -> torch.Tensor:
        """Predict velocity for the action sequence x at time t under cond."""
        if t.dim() == 0:
            t = t.expand(x.shape[0])
        elif t.dim() == 1 and t.shape[0] != x.shape[0]:
            t = t.expand(x.shape[0])
        time_feats = self.time_mlp(t)
        cond_feats = torch.cat([time_feats, cond], dim=-1)
        h = self.input_proj(x) + self.pos_embedding[:, :x.shape[1]]
        for block in self.blocks:
            h = block(h, cond_feats)
        return self.output_proj(h)


class DiTCrossAttnBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, time_dim: int,
                 kv_dim: int, dropout: float = 0.0):
        """DiT block with cross-attention into an external token memory."""
        super().__init__()
        self.norm_sa = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.self_attn = nn.MultiheadAttention(hidden_size, num_heads=num_heads,
                                               batch_first=True, dropout=dropout)
        self.norm_ca = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.cross_attn = nn.MultiheadAttention(
            hidden_size, num_heads=num_heads, batch_first=True, dropout=dropout,
            kdim=kv_dim, vdim=kv_dim,
        )
        self.norm_mlp = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.GELU(approximate="tanh"),
            nn.Linear(4 * hidden_size, hidden_size),
        )
        # 9 = (shift, scale, gate) x (self-attn, cross-attn, mlp)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, 9 * hidden_size, bias=True),
        )

    def forward(self, x: torch.Tensor, memory: torch.Tensor, t_feat: torch.Tensor) -> torch.Tensor:
        """Self-attend, cross-attend into memory, then MLP (all adaLN-gated)."""
        (shift_sa, scale_sa, gate_sa,
         shift_ca, scale_ca, gate_ca,
         shift_mlp, scale_mlp, gate_mlp) = self.adaLN_modulation(t_feat).chunk(9, dim=1)

        sa_in = _modulate(self.norm_sa(x), shift_sa.unsqueeze(1), scale_sa.unsqueeze(1))
        sa_out, _ = self.self_attn(sa_in, sa_in, sa_in, need_weights=False)
        x = x + gate_sa.unsqueeze(1) * sa_out

        ca_in = _modulate(self.norm_ca(x), shift_ca.unsqueeze(1), scale_ca.unsqueeze(1))
        ca_out, _ = self.cross_attn(ca_in, memory, memory, need_weights=False)
        x = x + gate_ca.unsqueeze(1) * ca_out

        mlp_in = _modulate(self.norm_mlp(x), shift_mlp.unsqueeze(1), scale_mlp.unsqueeze(1))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(mlp_in)
        return x


class DiTPolicyCrossAttn(nn.Module):
    def __init__(self, in_dim: int, kv_dim: int, horizon: int,
                 hidden_dim: int = 512, num_heads: int = 8, num_layers: int = 8,
                 dropout: float = 0.0, time_embed_dim: int = 128):
        """DiT velocity model that cross-attends into encoder tokens (memory)."""
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, in_dim)
        self.pos_embedding = nn.Parameter(torch.empty(1, horizon, hidden_dim).normal_(std=0.02))

        self.time_mlp = nn.Sequential(
            # scale=1000: flow timesteps live in [0,1]; without scaling, high-period
            # sinusoidal dims carry no signal.
            SinusoidalPosEmb(time_embed_dim, scale=1000.0),
            nn.Linear(time_embed_dim, 2 * time_embed_dim),
            nn.GELU(),
            nn.Linear(2 * time_embed_dim, time_embed_dim),
            nn.GELU(),
        )

        self.blocks = nn.ModuleList([
            DiTCrossAttnBlock(hidden_dim, num_heads, time_dim=time_embed_dim,
                              kv_dim=kv_dim, dropout=dropout)
            for _ in range(num_layers)
        ])

        # AdaLN-Zero: zero-init the modulation Linear so each block is identity at init.
        for block in self.blocks:
            nn.init.zeros_(block.adaLN_modulation[-1].weight)
            nn.init.zeros_(block.adaLN_modulation[-1].bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor, memory: torch.Tensor,
                is_causal: bool = False) -> torch.Tensor:
        """Predict velocity for x at time t attending over the token memory."""
        if t.dim() == 0:
            t = t.expand(x.shape[0])
        elif t.dim() == 1 and t.shape[0] != x.shape[0]:
            t = t.expand(x.shape[0])
        time_feats = self.time_mlp(t)
        h = self.input_proj(x) + self.pos_embedding[:, :x.shape[1]]
        for block in self.blocks:
            h = block(h, memory, time_feats)
        return self.output_proj(h)
