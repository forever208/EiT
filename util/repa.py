# --------------------------------------------------------
# References:
# PixelU: https://github.com/gzp6688/PixelU
# iREPA: https://github.com/End2End-Diffusion/iREPA
# --------------------------------------------------------
import os

import torch
import torch.nn as nn
import torch.nn.functional as F


DINOV2_REPO = 'facebookresearch_dinov2_main'
DINOV2_CKPT = 'dinov2_vitb14_reg4_pretrain.pth'


class DINOv2Encoder(nn.Module):
    """Frozen DINOv2-B/14-reg4 targets [B, 256, 768], without spatial normalization."""

    def __init__(self, repo_dir=None, ckpt_path=None, input_size=224):
        super().__init__()
        hub_dir = torch.hub.get_dir()
        repo_dir = repo_dir or os.path.join(hub_dir, DINOV2_REPO)
        ckpt_path = ckpt_path or os.path.join(hub_dir, 'checkpoints', DINOV2_CKPT)
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f'DINOv2 checkpoint not found: {ckpt_path}')

        self.model = torch.hub.load(repo_dir, 'dinov2_vitb14_reg', source='local', pretrained=False)
        self.model.load_state_dict(torch.load(ckpt_path, map_location='cpu'), strict=True)
        self.model.eval()
        self.model.requires_grad_(False)
        print(f'Loaded REPA encoder from {ckpt_path}')

        self.input_size = input_size
        self.num_prefix_tokens = 1 + self.model.num_register_tokens
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)

    @torch.no_grad()
    def forward(self, x):
        """x: images in [0, 1], (B, 3, H, W)."""
        if x.shape[-2:] != (self.input_size, self.input_size):
            x = F.interpolate(x, size=(self.input_size, self.input_size), mode='bicubic', align_corners=False)
        x = (x - self.mean) / self.std

        # bf16 backbone: with cuBLAS builds that run fp32 GEMMs on CUDA cores (e.g. CUDA 12.8),
        # an fp32 forward costs more than the whole diffusion training step.
        with torch.autocast(device_type=x.device.type, dtype=torch.bfloat16):
            feat = self.model.forward_features(x)['x_prenorm'][:, self.num_prefix_tokens:]

        # Keep the existing per-token LayerNorm, without affine parameters.
        # Do not center or scale features across spatial tokens.
        return F.layer_norm(feat.float(), feat.shape[-1:], eps=self.model.norm.eps)
