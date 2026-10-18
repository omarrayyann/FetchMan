import torch
import torch.nn as nn

from algorithms.common import SinusoidalPosEmb


class DiffusionPolicyTransformer(nn.Module):
    def __init__(self,
                 in_dim: int,
                 external_cond_dim: int,
                 horizon: int,
                 num_obs: int = 0,
                 hidden_dim: int = 256,
                 num_heads: int = 4,
                 num_layers: int = 8,
                 attn_dropout: float = 0.01,
                 embed_dropout: float = 0.0,
                 ):
        """Decoder-style transformer denoiser conditioned on obs tokens (diffusion-policy layout)."""
        super(DiffusionPolicyTransformer, self).__init__()
        self.input_embedding = nn.Linear(in_dim, hidden_dim)
        self.pos_embedding = nn.Parameter(torch.zeros(1, horizon, hidden_dim))

        self.cond_obs_embedding = nn.Linear(external_cond_dim, hidden_dim)
        self.cond_pos_embedding = nn.Parameter(torch.zeros(1, 1 + num_obs, hidden_dim))

        # scale=1000 for [0,1] flow timesteps; see common.py.
        self.diffusion_time_embedding = SinusoidalPosEmb(hidden_dim, scale=1000.0)
        self.dropout = nn.Dropout(embed_dropout)
        self.encoder = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.Mish(),
            nn.Linear(4 * hidden_dim, hidden_dim),
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=4 * hidden_dim,
            dropout=attn_dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=num_layers,
        )
        self.decoder_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, in_dim),
        )

        memory_mask = torch.triu(
            torch.full((horizon, 1 + num_obs), float("-inf")),
            diagonal=2,
        )
        tgt_mask = torch.triu(
            torch.full((horizon, horizon), float("-inf")),
            diagonal=1,
        )
        self.register_buffer("_memory_mask", memory_mask, persistent=False)
        self.register_buffer("_tgt_mask", tgt_mask, persistent=False)

    def forward(self, x: torch.Tensor, k: torch.Tensor, global_cond: torch.Tensor = None, is_causal: bool = False):
        """Predict the denoised action sequence for x at diffusion step k."""
        diffusion_time_embedding = self.diffusion_time_embedding(k)[:, None, :]
        input_embedding = self.input_embedding(x)

        cond_embedding = self.cond_obs_embedding(global_cond)
        cond_embedding = torch.cat([diffusion_time_embedding, cond_embedding], dim=1)
        cond_seq_len = cond_embedding.shape[1]
        cond_positional_embeddings = self.cond_pos_embedding[:, :cond_seq_len, :]

        cond_embedding = self.dropout(cond_embedding + cond_positional_embeddings)
        cond_embedding = self.encoder(cond_embedding)

        action_pred_horizon = input_embedding.shape[1]
        positional_embeddings = self.pos_embedding[:, :action_pred_horizon, :]
        x = self.dropout(input_embedding + positional_embeddings)

        if is_causal:
            tgt_mask = self._tgt_mask[:action_pred_horizon, :action_pred_horizon]
            memory_mask = self._memory_mask[:action_pred_horizon, :cond_seq_len]
        else:
            tgt_mask = None
            memory_mask = None
        x = self.decoder(
            tgt=x, memory=cond_embedding,
            tgt_mask=tgt_mask, tgt_is_causal=is_causal,
            memory_mask=memory_mask, memory_is_causal=False,
        )

        return self.decoder_head(x)
