from functools import partial
from transformers import AutoModel
import torch
import torch.nn as nn

# DINOv3 uses ImageNet normalization.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


def _register_imagenet_norm_buffers(module: nn.Module):
    """Attach ImageNet mean/std buffers so normalization moves with the module's device."""
    module.register_buffer('_im_mean', torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
    module.register_buffer('_im_std', torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))


def _imagenet_normalize(module: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Normalize an image batch with the module's ImageNet mean/std buffers."""
    return (x - module._im_mean.to(x.dtype)) / module._im_std.to(x.dtype)


class MultimodalEncoder(nn.Module):
    def __init__(self, obs_shape_dict: dict):
        """State-only encoder: concatenates the proprioceptive observation vectors."""
        super().__init__()
        self.state_key_order = []
        self.out_dim = 0
        for obs_name, obs_shape in obs_shape_dict.items():
            if 'image' in obs_name or obs_name.endswith('depth'):
                raise ValueError(
                    f"encoder='multimodal_state' takes state observations only, got {obs_name!r}. "
                    "Use 'dinov3_patches' to train on images.")
            self.state_key_order.append(obs_name)
            self.out_dim += obs_shape[-1]

    def forward(self, batch: dict):
        """Concatenate the state observations along the feature axis."""
        return torch.cat([batch[k] for k in self.state_key_order], dim=-1)


class DINOv3PatchEncoder(nn.Module):
    # Forward returns {visual_tokens: (B,T,V*n_patches,D), state: (B,T,state_dim)}.
    # Pair with architecture='dit_xattn'.
    def __init__(
        self,
        obs_shape_dict: dict,
        model_name: str = 'facebook/dinov3-vitb16-pretrain-lvd1689m',
        freeze: bool = True,
        fusion_dim: int = 512,
        n_special_tokens: int = 5,  # DINOv3: 1 CLS + 4 register tokens
    ):
        """DINOv3 patch-token encoder: per-view learned tokens + adapter, emits visual tokens for a DiT."""
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        self.freeze = freeze
        self.fusion_dim = int(fusion_dim)
        self.n_special = int(n_special_tokens)
        self._image_dim = int(self.backbone.config.hidden_size)

        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad_(False)
            self.backbone.eval()
            # Keep fp32 weights; bf16-mixed autocast handles matmuls. Manually
            # casting the backbone to bf16 here degrades frozen-feature quality.
        _register_imagenet_norm_buffers(self)

        self.image_keys = []
        self.state_key_order = []
        for k in obs_shape_dict:
            if 'image' in k:
                self.image_keys.append(k)
            elif k.endswith('depth'):
                raise NotImplementedError('Depth support not yet implemented!')
            else:
                self.state_key_order.append(k)

        self.view_tokens = nn.ParameterDict({
            k.replace('.', '_'): nn.Parameter(0.02 * torch.randn(self.fusion_dim))
            for k in self.image_keys
        })
        self.adapter = nn.Sequential(
            nn.LayerNorm(self._image_dim),
            nn.Linear(self._image_dim, self.fusion_dim),
            nn.GELU(),
            nn.Linear(self.fusion_dim, self.fusion_dim),
            nn.LayerNorm(self.fusion_dim),
        )

        self.state_dim = 0
        for k in self.state_key_order:
            shape = obs_shape_dict[k]
            self.state_dim += int(shape[-1])
        self.out_dim = self.fusion_dim
        self.returns_tokens = True

    def train(self, mode: bool = True):
        """Keep the frozen backbone in eval mode when the policy switches to train."""
        super().train(mode)
        if self.freeze:
            self.backbone.eval()
        return self

    def forward(self, batch: dict):
        """Adapted DINOv3 patch tokens for all views, tagged with view embeddings; state returned separately."""
        any_key = self.image_keys[0]
        bsz, seq_len = batch[any_key].shape[:2]
        V = len(self.image_keys)

        x = torch.stack([batch[k] for k in self.image_keys], dim=0)
        x = x.flatten(0, 2)
        x = _imagenet_normalize(self, x)
        with torch.set_grad_enabled(not self.freeze):
            vis_out = self.backbone(pixel_values=x).last_hidden_state
        patches = vis_out[:, self.n_special:, :]
        patches = patches.to(x.dtype)
        patches = self.adapter(patches)
        n_patches = patches.shape[1]

        # Out-of-place view-token add: adapter ends with LN; an in-place write
        # here would otherwise trip autograd's saved-output version check.
        patches = patches.view(V, bsz, seq_len, n_patches, self.fusion_dim)
        view_toks = torch.stack(
            [self.view_tokens[k.replace('.', '_')] for k in self.image_keys],
            dim=0,
        ).view(V, 1, 1, 1, self.fusion_dim)
        patches = patches + view_toks
        patches = patches.permute(1, 2, 0, 3, 4).contiguous()
        patches = patches.view(bsz, seq_len, V * n_patches, self.fusion_dim)

        state_obs = [batch[k] for k in self.state_key_order]
        state = torch.cat(state_obs, dim=-1) if state_obs else None

        return {"visual_tokens": patches, "state": state}


class _CrossAttentionBlock(nn.Module):
    """Pre-norm cross-attention + FFN block. Query attends to key/value."""
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0, ffn_mult: int = 4):
        """Pre-norm cross-attention + MLP block used to inject text into patch tokens."""
        super().__init__()
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.xattn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm_ffn = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * ffn_mult),
            nn.GELU(),
            nn.Linear(d_model * ffn_mult, d_model),
        )

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        """Cross-attend queries into kv and apply the MLP residual."""
        q_n = self.norm_q(q)
        kv_n = self.norm_kv(kv)
        attn_out, _ = self.xattn(q_n, kv_n, kv_n, need_weights=False)
        q = q + attn_out
        q = q + self.ffn(self.norm_ffn(q))
        return q


class DINOv3DinoTxtEncoder(nn.Module):
    """Frozen Meta dino.txt (DINOv3 ViT-L/16 backbone + text head, pre-aligned vision<->text)."""
    def __init__(
        self,
        obs_shape_dict: dict,
        model_name: str = 'dinov3_vitl16_dinotxt_tet1280d20h24l',
        freeze: bool = True,
        fusion_dim: int = 512,
        fusion_layers: int = 2,
        fusion_heads: int = 8,
        use_text: bool = True,
        pretrained: bool = True,
    ):
        """Load the frozen dino.txt model + tokenizer and build adapters/fusion blocks."""
        super().__init__()
        # Meta gated the dinov3 weight URLs (dl.fbaipublicfiles.com now 403s), so a
        # pretrained build can fail even though nothing is wrong locally. Every
        # dino.txt parameter is saved inside our policy checkpoints, so when the
        # download is unavailable we can build the architecture untrained and let
        # load_from_checkpoint supply the weights. Only training FROM SCRATCH needs
        # the download; evaluating a checkpoint does not.
        try:
            self.model, self.tokenizer = torch.hub.load(
                'facebookresearch/dinov3', model_name, trust_repo=True,
                pretrained=bool(pretrained))
        except Exception as e:
            if not pretrained:
                raise
            print(f"[dinotxt] pretrained weights unavailable ({e}); building "
                  f"untrained architecture. This is only valid when the weights "
                  f"come from a policy checkpoint.", flush=True)
            self.model, self.tokenizer = torch.hub.load(
                'facebookresearch/dinov3', model_name, trust_repo=True,
                pretrained=False)
        self.freeze = bool(freeze)
        self.use_text = bool(use_text)
        self.fusion_dim = int(fusion_dim)
        if self.freeze:
            for p in self.model.parameters():
                p.requires_grad_(False)
            self.model.eval()
        _register_imagenet_norm_buffers(self)
        # object_name -> adapted text token, for the live-tokenize eval path.
        # Only valid while the text_adapter is frozen, which it is during eval.
        self._text_cache = {}

        # Infer the aligned patch dim and the patch-half text dim once (CPU).
        with torch.no_grad():
            _g, al, _raw = self.model.encode_image_with_patch_tokens(torch.zeros(1, 3, 224, 224))
            self._patch_dim = int(al.shape[-1])
            self._text_dim = int(self.model.encode_text(self.tokenizer.tokenize(["object"])).shape[-1]) // 2

        self.image_keys, self.state_key_order = [], []
        for k in obs_shape_dict:
            if 'image' in k:
                self.image_keys.append(k)
            elif k.endswith('depth'):
                raise NotImplementedError('Depth support not yet implemented!')
            else:
                self.state_key_order.append(k)

        self.view_tokens = nn.ParameterDict({
            k.replace('.', '_'): nn.Parameter(0.02 * torch.randn(self.fusion_dim))
            for k in self.image_keys
        })
        self.vision_adapter = nn.Sequential(
            nn.LayerNorm(self._patch_dim),
            nn.Linear(self._patch_dim, self.fusion_dim), nn.GELU(),
            nn.Linear(self.fusion_dim, self.fusion_dim), nn.LayerNorm(self.fusion_dim),
        )
        if self.use_text:
            self.text_adapter = nn.Sequential(
                nn.LayerNorm(self._text_dim),
                nn.Linear(self._text_dim, self.fusion_dim), nn.GELU(),
                nn.Linear(self.fusion_dim, self.fusion_dim), nn.LayerNorm(self.fusion_dim),
            )
            self.fusion_blocks = nn.ModuleList([
                _CrossAttentionBlock(self.fusion_dim, fusion_heads) for _ in range(fusion_layers)
            ])

        self.state_dim = sum(int(obs_shape_dict[k][-1]) for k in self.state_key_order)
        self.out_dim = self.fusion_dim
        self.returns_tokens = True

    def train(self, mode: bool = True):
        """Keep the frozen dino.txt model in eval mode when the policy switches to train."""
        super().train(mode)
        if self.freeze:
            self.model.eval()
        return self

    def _encode_text(self, batch, bsz, ref_dtype, device):
        """Embed object names with the dino.txt text tower (patch-aligned half), with a per-batch cache."""
        # Train: data.py pre-embeds object names with the same text tower into
        # object_name_embedding, so here it only needs adapting.
        emb = batch.get("object_name_embedding")
        if emb is not None:
            return self.text_adapter(emb.to(device).to(ref_dtype))
        # Eval/rollout: live tokenize the raw name with dino.txt's own tokenizer.
        task = batch.get("object_name") or batch.get("task")
        if task is None:
            return torch.zeros(bsz, self.fusion_dim, dtype=ref_dtype, device=device)
        task_list = list(task) if isinstance(task, (list, tuple)) else [task] * bsz
        # The adapted token for a name is constant while the adapter is frozen,
        # so encode only the unique uncached names, then gather per-sample.
        cache = self._text_cache
        missing = list(dict.fromkeys(t for t in task_list if t not in cache))
        if missing:
            toks = self.tokenizer.tokenize(missing).to(device)
            with torch.no_grad():
                feats = self.model.encode_text(toks)        # [U, 2D]
                feats = feats[:, feats.shape[1] // 2:]      # patch-aligned half
            adapted = self.text_adapter(feats.to(ref_dtype))  # [U, fusion_dim]
            for i, name in enumerate(missing):
                cache[name] = adapted[i].detach()
        return torch.stack([cache[t] for t in task_list], dim=0).to(device=device, dtype=ref_dtype)

    def forward(self, batch: dict):
        """Aligned dino.txt patch tokens fused with the text token via cross-attention; returns visual tokens + state."""
        any_key = self.image_keys[0]
        bsz, seq_len = batch[any_key].shape[:2]
        V = len(self.image_keys)
        device = next(self.parameters()).device

        x = torch.stack([batch[k] for k in self.image_keys], dim=0).flatten(0, 2)
        x = _imagenet_normalize(self, x)
        with torch.set_grad_enabled(not self.freeze):
            _g, aligned_patches, _raw = self.model.encode_image_with_patch_tokens(x)
        patches = self.vision_adapter(aligned_patches.to(x.dtype))
        n_patches = patches.shape[1]

        patches = patches.view(V, bsz, seq_len, n_patches, self.fusion_dim)
        view_toks = torch.stack(
            [self.view_tokens[k.replace('.', '_')] for k in self.image_keys], dim=0
        ).view(V, 1, 1, 1, self.fusion_dim)
        patches = patches + view_toks
        patches = patches.permute(1, 2, 0, 3, 4).contiguous()
        patches = patches.view(bsz, seq_len, V * n_patches, self.fusion_dim)

        if self.use_text:
            text_feat = self._encode_text(batch, bsz, patches.dtype, device)
            text_token = text_feat.view(bsz, 1, 1, self.fusion_dim).expand(bsz, seq_len, 1, self.fusion_dim)
            BT = bsz * seq_len
            q = patches.reshape(BT, V * n_patches, self.fusion_dim)
            kv = text_token.reshape(BT, 1, self.fusion_dim)
            for block in self.fusion_blocks:
                q = block(q, kv)
            patches = q.view(bsz, seq_len, V * n_patches, self.fusion_dim)

        state_obs = [batch[k] for k in self.state_key_order]
        state = torch.cat(state_obs, dim=-1) if state_obs else None
        return {"visual_tokens": patches, "state": state}


# Text tower used by each text-conditioned encoder. main.py's object_name
# pre-embedding MUST use the same model as the encoder's live tokenize path
# (eval/rollout), or train and eval see different text distributions.
encoders = {
    'multimodal_state': MultimodalEncoder,
    'dinov3_patches': partial(DINOv3PatchEncoder, freeze=True),
    'dinov3_patches_trainable': DINOv3PatchEncoder,
    'dinov3_dinotxt': partial(DINOv3DinoTxtEncoder, freeze=True),
}
