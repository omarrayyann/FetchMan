import numpy as np
from diffusers import get_scheduler
from ml_collections import ConfigDict
from ml_collections.config_dict import placeholder
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L

from algorithms.dit_transformer import DiTPolicy, DiTPolicyCrossAttn
from algorithms.dp_transformer import DiffusionPolicyTransformer
from algorithms.flow_matching import FlowMatchingScheduler
from algorithms.encoders import encoders
from algorithms.unet import ConditionalUnet1D


_OBS_PREFIX = "observation."


def _safe(key: str) -> str:
    """Sanitize an obs key into a valid buffer-name suffix."""
    return key.replace(".", "_")


class FlowPolicy(L.LightningModule):
    def __init__(self, config: ConfigDict, example_data: dict, dataset_stats: dict):
        """Flow-matching action policy: encoder + velocity model + min-max normalization buffers from dataset stats."""
        super().__init__()
        self.config = ConfigDict(config)

        self.obs_keys = sorted(
            k for k in example_data.keys()
            if k.startswith(_OBS_PREFIX) and not k.endswith("_is_pad")
        )
        self.image_keys = {k for k in self.obs_keys if "image" in k}
        self.config['obs_keys'] = list(self.obs_keys)
        self.config['action_dim'] = int(example_data['action'].shape[-1])

        self._register_norm_buffers(dataset_stats)

        self._build_encoder(config, example_data)

        self._uses_token_memory = bool(self.encoder is not None and getattr(self.encoder, 'returns_tokens', False))
        if self._uses_token_memory and config['architecture'] != 'dit_xattn':
            raise ValueError(
                f"encoder {config['encoder']!r} returns token sequence — "
                f"architecture must be 'dit_xattn' (got {config['architecture']!r})."
            )
        if config['architecture'] == 'dit_xattn' and self.encoder is not None and not self._uses_token_memory:
            raise ValueError(
                f"encoder {config['encoder']!r} returns flat features (no token sequence) — "
                f"pair it with architecture='dit' or 'unet', not 'dit_xattn'."
            )

        self._build_velocity_model(config)

        self.flow_scheduler = FlowMatchingScheduler(config)

    def _register_norm_buffers(self, dataset_stats):
        """Register the per-key (loc, scale) buffers that normalize observations and actions."""
        for k in self.obs_keys + ['action']:
            if k in self.image_keys:
                continue
            s = dataset_stats[k]
            if 'mean' in s and 'std' in s:
                loc = torch.as_tensor(np.asarray(s['mean']), dtype=torch.float32).reshape(-1)
                scale = torch.as_tensor(np.asarray(s['std']), dtype=torch.float32).reshape(-1)
            else:
                mn = torch.as_tensor(np.asarray(s['min']), dtype=torch.float32).reshape(-1)
                mx = torch.as_tensor(np.asarray(s['max']), dtype=torch.float32).reshape(-1)
                loc = (mn + mx) / 2
                scale = (mx - mn) / 2
            scale = scale.clamp(min=1e-6)
            self.register_buffer(f"_loc_{_safe(k)}", loc)
            self.register_buffer(f"_scale_{_safe(k)}", scale)

    def _build_encoder(self, config, example_data):
        """Instantiate the observation encoder and record its output shape."""
        if config['encoder'] is not None:
            obs_shapes = {k.removeprefix(_OBS_PREFIX): tuple(example_data[k].shape) for k in self.obs_keys}
            enc_kwargs = {}
            # Optionally override the DINOv3 backbone size (e.g. vitl16 for the
            # 300M model). The adapter auto-adapts to the backbone hidden size.
            dino_model = config.get('dino_model', None)
            if dino_model and 'dino' in str(config['encoder']):
                enc_kwargs['model_name'] = str(dino_model)
            if config.get('dino_config') and config['encoder'].startswith('dinov3_patches'):
                enc_kwargs['backbone_config'] = config['dino_config'].to_dict() \
                    if hasattr(config['dino_config'], 'to_dict') else dict(config['dino_config'])
            self.encoder = encoders[config['encoder']](obs_shapes, **enc_kwargs)
            self.config['observation_dim'] = self.encoder.out_dim
        else:
            self.encoder = None
            self.config['observation_dim'] = sum(
                int(example_data[k].shape[-1]) if example_data[k].dim() > 1 else 1
                for k in self.obs_keys
            )

    def _build_velocity_model(self, config):
        """Instantiate the velocity network for the configured architecture."""
        if config['architecture'] == 'unet':
            self.flow_model = ConditionalUnet1D(
                input_dim=self.config['action_dim'],
                global_cond_dim=self.config['observation_dim'] * config['observation_horizon'],
                down_dims=config.get('unet_dims', [64, 128, 256]),
                diffusion_step_embed_dim=config.get('unet_step_dim', 128),
            )
        elif config['architecture'] == 'dit':
            self.flow_model = DiTPolicy(
                in_dim=self.config['action_dim'],
                cond_dim=self.config['observation_dim'] * config['observation_horizon'],
                horizon=config['prediction_horizon'],
                hidden_dim=config['hidden_dim'],
                num_heads=config['num_heads'],
                num_layers=config['num_layers'],
                dropout=config['attention_dropout'],
                time_embed_dim=config.get('unet_step_dim', 128),
            )
        elif config['architecture'] == 'dit_xattn':
            kv_dim = int(self.encoder.fusion_dim)
            if self.encoder.state_dim > 0:
                # Tiny init so the proprio token doesn't dominate softmax in
                # cross-attention before the model learns to scale it.
                self.state_proj = nn.Linear(int(self.encoder.state_dim), kv_dim)
                nn.init.uniform_(self.state_proj.weight, -0.01, 0.01)
                nn.init.zeros_(self.state_proj.bias)
            else:
                self.state_proj = None
            self.flow_model = DiTPolicyCrossAttn(
                in_dim=self.config['action_dim'],
                kv_dim=kv_dim,
                horizon=config['prediction_horizon'],
                hidden_dim=config['hidden_dim'],
                num_heads=config['num_heads'],
                num_layers=config['num_layers'],
                dropout=config['attention_dropout'],
                time_embed_dim=config.get('unet_step_dim', 128),
            )
        else:
            self.flow_model = DiffusionPolicyTransformer(
                self.config['action_dim'],
                external_cond_dim=self.config['observation_dim'],
                horizon=config['prediction_horizon'],
                num_obs=config['observation_horizon'],
                hidden_dim=config['hidden_dim'],
                num_heads=config['num_heads'],
                num_layers=config['num_layers'],
                attn_dropout=config['attention_dropout'],
                embed_dropout=config['embedding_dropout'],
            )

    def _stats(self, key):
        """(loc, scale) normalization buffers for an obs/action key."""
        return getattr(self, f"_loc_{_safe(key)}"), getattr(self, f"_scale_{_safe(key)}")

    def _normalize(self, x, key):
        """Map raw values to the normalized training range for `key`."""
        loc, scale = self._stats(key)
        return (x - loc) / scale

    def _unnormalize(self, x, key):
        """Map normalized values back to raw units for `key`."""
        loc, scale = self._stats(key)
        return x * scale + loc

    def _build_cond(self, batch):
        """Run the encoder over prepped observations (plus text-conditioning keys) to get the conditioning."""
        def _prep(k):
            """Scale images to [0,1] and normalize state keys, adding a feature dim to 2-D state."""
            if k in self.image_keys:
                v = batch[k]
                if v.dtype == torch.uint8:
                    v = v.float() / 255.0
                return v
            v = self._normalize(batch[k], k)
            return v.unsqueeze(-1) if v.dim() == 2 else v
        if self.encoder is not None:
            enc_in = {k.removeprefix(_OBS_PREFIX): _prep(k) for k in self.obs_keys}
            # Text conditioning inputs are not observation.* keys, so pass them
            # through untouched or the encoder's _encode_text only ever sees None.
            for tk in ("object_name_embedding", "object_name", "task"):
                if tk in batch:
                    enc_in[tk] = batch[tk]
            return self.encoder(enc_in)
        return torch.cat([_prep(k) for k in self.obs_keys], dim=-1)

    def configure_optimizers(self):
        """AdamW over head (+ encoder at a lower LR) with a warmup+cosine schedule."""
        head_params = [p for p in self.flow_model.parameters() if p.requires_grad]
        if getattr(self, 'state_proj', None) is not None:
            head_params += [p for p in self.state_proj.parameters() if p.requires_grad]
        optimizer_list = [{'params': head_params,
                           'lr': self.config['lr'],
                           'weight_decay': self.config['weight_decay']}]
        if self.encoder is not None:
            encoder_params = [p for p in self.encoder.parameters() if p.requires_grad]
            if encoder_params:
                if self.config['encoder_lr'] is not None:
                    enc_lr = self.config['encoder_lr']
                elif self._uses_token_memory and getattr(self.encoder, 'freeze', False):
                    enc_lr = self.config['lr']
                else:
                    enc_lr = self.config['lr'] * float(self.config.get('vision_lr_multiplier', 1.0))
                optimizer_list.append({'params': encoder_params,
                                       'lr': enc_lr,
                                       'weight_decay': self.config['encoder_weight_decay']
                                       if self.config['encoder_weight_decay'] is not None else self.config['weight_decay']})
        optimizer_kwargs = dict(betas=self.config['betas'])
        # foreach (not fused): a fused AdamW unscales grads internally, which is
        # incompatible with Lightning AMP (bf16-mixed) gradient clipping. foreach
        # is clip-compatible and nearly as fast.
        if self.config.get('fused_adamw', True) and torch.cuda.is_available():
            optimizer_kwargs['foreach'] = True
        try:
            optimizer = torch.optim.AdamW(optimizer_list, **optimizer_kwargs)
        except TypeError:
            optimizer_kwargs.pop('foreach', None)
            optimizer = torch.optim.AdamW(optimizer_list, **optimizer_kwargs)
        scheduler = get_scheduler(name=self.config['scheduler'],
                                  optimizer=optimizer,
                                  num_warmup_steps=self.config['num_warmup_steps'],
                                  num_training_steps=self.config['num_training_steps'])
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }

    def _color_aug(self, image):
        """Per-sample brightness/contrast/saturation/white-balance jitter on (N,3,H,W) images."""
        N = image.shape[0]
        device, dtype = image.device, image.dtype
        def u(lo, hi, c=1):
            """Uniform per-sample scalar (or per-channel) sampler shaped for broadcasting."""
            return torch.empty(N, c, 1, 1, device=device, dtype=dtype).uniform_(lo, hi)
        br = self.config.get('aug_brightness', (0.6, 1.4))
        ct = self.config.get('aug_contrast', (0.5, 1.5))
        sat = self.config.get('aug_saturation', (0.4, 1.6))
        wb = self.config.get('aug_white_balance', (0.85, 1.15))
        image = image * u(br[0], br[1])
        mean = image.mean(dim=[1, 2, 3], keepdim=True)
        image = (image - mean) * u(ct[0], ct[1]) + mean
        gray = image.mean(dim=1, keepdim=True)
        image = gray + (image - gray) * u(sat[0], sat[1])
        image = image * u(wb[0], wb[1], c=3)  # per-channel white-balance gain
        return image.clamp(0.0, 1.0)

    def _gaussian_blur(self, image, p_apply, max_sigma, ksize=5):
        """Randomly blur a subset of images with per-sample sigma via grouped conv."""
        N, C, H, W = image.shape
        dev, dt = image.device, image.dtype
        apply = torch.rand(N, device=dev) < p_apply
        if not bool(apply.any()):
            return image
        sigma = torch.empty(N, device=dev, dtype=dt).uniform_(0.3, max_sigma)
        half = ksize // 2
        coords = torch.arange(ksize, device=dev, dtype=dt) - half
        g1 = torch.exp(-coords[None, :] ** 2 / (2.0 * sigma[:, None] ** 2))
        g1 = g1 / g1.sum(dim=1, keepdim=True)
        k2 = (g1[:, :, None] * g1[:, None, :])            # (N, ksize, ksize)
        k2 = k2[:, None].expand(N, C, ksize, ksize).reshape(N * C, 1, ksize, ksize)
        blurred = F.conv2d(image.reshape(1, N * C, H, W), k2,
                           padding=half, groups=N * C).reshape(N, C, H, W)
        return torch.where(apply[:, None, None, None], blurred, image)

    def _add_noise(self, image, max_std):
        """Add per-sample-strength gaussian pixel noise, clamped to [0,1]."""
        N = image.shape[0]
        std = torch.empty(N, 1, 1, 1, device=image.device, dtype=image.dtype).uniform_(0.0, max_std)
        return (image + torch.randn_like(image) * std).clamp(0.0, 1.0)

    def _augment_images(self, batch):
        """Apply color, blur, and noise augmentation to every image key in the batch."""
        out = dict(batch)
        for k in self.image_keys:
            v = batch[k]
            if v.dtype == torch.uint8:
                v = v.float() / 255.0
            B, T = v.shape[:2]
            x = v.flatten(0, 1)
            x = self._color_aug(x)
            x = self._gaussian_blur(x, p_apply=self.config.get('aug_blur_p', 0.5),
                                    max_sigma=self.config.get('aug_blur_sigma', 1.5))
            x = self._add_noise(x, max_std=self.config.get('aug_noise_std', 0.04))
            out[k] = x.view(B, T, *x.shape[1:])
        return out

    def _build_memory(self, encoder_out: dict) -> torch.Tensor:
        """Assemble the DiT cross-attention memory: [text tokens?, last-frame visual tokens, state token?]."""
        memory_parts = []
        text = encoder_out.get("text_features")
        if text is not None:
            memory_parts.append(text)
        vis = encoder_out["visual_tokens"]
        memory_parts.append(vis[:, -1, :, :])
        state = encoder_out.get("state")
        if state is not None and self.state_proj is not None:
            state_tok = self.state_proj(state[:, -1, :]).unsqueeze(1)
            memory_parts.append(state_tok)
        return torch.cat(memory_parts, dim=1)

    def _flow_loss(self, batch, augment: bool):
        """Flow-matching MSE: predict the noise->action velocity at a sampled time."""
        actions = self._normalize(batch['action'], 'action')
        if augment:
            batch = self._augment_images(batch)
        global_cond = self._build_cond(batch)
        bsz = actions.shape[0]

        if self._uses_token_memory:
            memory = self._build_memory(global_cond)
        elif self.config.architecture in ('unet', 'dit'):
            global_cond = global_cond.view(bsz, -1)

        t = self.flow_scheduler.sample_time(bsz, 1, self.device).squeeze()
        noise = torch.randn_like(actions, device=self.device)
        if self.config.clip_noise < float('inf'):
            noise = torch.clamp(noise, min=-self.config.clip_noise, max=self.config.clip_noise)

        x_t = self.flow_scheduler.get_flow_trajectory(noise, actions, t)
        target_velocity = self.flow_scheduler.compute_target_velocity(noise, actions)

        if self._uses_token_memory:
            pred_velocity = self.flow_model(x_t, t, memory)
        elif self.config['architecture'] in ('unet', 'dit'):
            pred_velocity = self.flow_model(x_t, t, global_cond)
        else:
            pred_velocity = self.flow_model(x_t, t, global_cond, is_causal=self.config.is_causal)

        return F.mse_loss(pred_velocity, target_velocity)

    def training_step(self, batch, batch_idx):
        """One flow-matching step on the training split."""
        loss = self._flow_loss(batch, augment=self.config.get('augment', True))
        self.log('train/loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Same loss on held-out episodes, with augmentation off.

        The flow time and the noise are resampled every call, so an unseeded val
        loss would wander by more than the signal we are watching for. Forking
        the RNG and seeding it from batch_idx makes every validation pass draw
        the same (t, noise) for the same batch, which leaves the model as the
        only thing that changed between epochs. The fork keeps that seed out of
        the training stream.
        """
        devices = [self.device] if self.device.type == 'cuda' else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(0x5EED + batch_idx)
            loss = self._flow_loss(batch, augment=False)
        self.log('val/loss', loss, prog_bar=True, sync_dist=True)
        return loss

    @torch.no_grad()
    def sample_actions(self, obs_batch: dict, schedule: torch.Tensor, batch_size: int = 1):
        """Integrate the learned velocity field from noise along `schedule` and unnormalize to actions."""
        global_cond = self._build_cond(obs_batch)
        if self._uses_token_memory:
            memory = self._build_memory(global_cond)
        elif self.config.architecture in ('unet', 'dit'):
            global_cond = global_cond.view(batch_size, -1)

        naction = torch.randn(
            (batch_size, self.config['prediction_horizon'], self.config['action_dim']),
            device=self.device,
        )

        for m in range(self.config['num_sampling_steps'] - 1):
            curr_t = schedule[m]
            next_t = schedule[m + 1]
            if self._uses_token_memory:
                pred_velocity = self.flow_model(naction, curr_t, memory)
            elif self.config['architecture'] in ('unet', 'dit'):
                pred_velocity = self.flow_model(naction, curr_t, global_cond)
            else:
                pred_velocity = self.flow_model(naction, curr_t, global_cond, is_causal=self.config.is_causal)
            dt = (next_t - curr_t).unsqueeze(-1)
            naction = naction + pred_velocity * dt

        return self._unnormalize(naction, 'action')


def get_config():
    """Default FlowPolicy ConfigDict (architecture, horizons, augmentation, optimizer)."""
    config = ConfigDict(dict(
        agent_name='flow_policy',
        batch_size=512,
        observation_dim=placeholder(int),
        action_dim=placeholder(int),
        prediction_horizon=16,
        observation_horizon=1,
        action_horizon=8,
        architecture='dit_xattn',
        hidden_dim=768,
        num_heads=12,
        num_layers=12,
        unet_dims=[64, 128, 256],
        unet_step_dim=128,
        attention_dropout=0.0,
        embedding_dropout=0.0,
        is_causal=True,
        encoder='dinov3_patches',
        # DINOv3 backbone override. None -> encoder's default (vitb16, ~86M).
        # e.g. 'facebook/dinov3-vitl16-pretrain-lvd1689m' for the ~300M model.
        dino_model=placeholder(str),
        # The backbone's architecture, set in released checkpoints so loading one
        # builds it without downloading the gated pretrained weights.
        dino_config=placeholder(dict),
        augment=True,
        timestep_sampling='beta',
        timestep_beta_alpha=1.5,
        timestep_beta_beta=1.0,
        timestep_beta_s=0.999,
        lr=1e-4,
        betas=(0.9, 0.999),
        weight_decay=1e-6,
        scheduler='cosine',
        num_warmup_steps=1500,
        num_training_steps=placeholder(int),
        encoder_lr=placeholder(float),
        encoder_weight_decay=1e-6,
        vision_lr_multiplier=0.1,
        fused_adamw=True,
        num_sampling_steps=20,
        clip_noise=float('inf'),
        use_action_delta=True,
    ))
    return config
