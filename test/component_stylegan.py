"""
models/component_stylegan.py
Modified StyleGAN2 for Khitan character/component generation.
Key changes vs original:
  - Replace single learned constant with discrete codebook entries
  - Remove per-layer noise (for structure clarity)
  - Add recognition loss support
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ── Building blocks ────────────────────────────────────────────────────────

class EqualLinear(nn.Module):
    """Equalized learning rate linear layer."""
    def __init__(self, in_dim, out_dim, bias=True, lr_mul=1.0):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))
        self.bias = nn.Parameter(torch.zeros(out_dim)) if bias else None
        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, x):
        return F.linear(x, self.weight * self.scale,
                        self.bias * self.lr_mul if self.bias is not None else None)


class MappingNetwork(nn.Module):
    """Maps latent z to disentangled W space."""
    def __init__(self, z_dim=512, w_dim=512, num_layers=8):
        super().__init__()
        layers = []
        for i in range(num_layers):
            in_d = z_dim if i == 0 else w_dim
            layers.append(EqualLinear(in_d, w_dim, lr_mul=0.01))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        z = F.normalize(z, dim=1)  # pixel-wise normalisation
        return self.net(z)


class ModulatedConv2d(nn.Module):
    """Modulated convolution (weight demodulation)."""
    def __init__(self, in_ch, out_ch, kernel_size, style_dim, demodulate=True,
                 upsample=False):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.kernel_size = kernel_size
        self.demodulate = demodulate
        self.upsample = upsample
        self.padding = kernel_size // 2
        self.scale = 1 / math.sqrt(in_ch * kernel_size ** 2)
        self.weight = nn.Parameter(
            torch.randn(1, out_ch, in_ch, kernel_size, kernel_size)
        )
        self.modulation = EqualLinear(style_dim, in_ch, bias=True)

    def forward(self, x, style):
        B, C, H, W = x.shape
        # Modulate
        s = self.modulation(style).view(B, 1, C, 1, 1)  # (B,1,in,1,1)
        weight = self.scale * self.weight * s             # (B,out,in,k,k)
        # Demodulate
        if self.demodulate:
            denom = torch.rsqrt((weight ** 2).sum([2, 3, 4]) + 1e-8)
            weight = weight * denom.view(B, self.out_ch, 1, 1, 1)
        # Reshape for grouped conv
        weight = weight.view(B * self.out_ch, C, self.kernel_size, self.kernel_size)
        x = x.view(1, B * C, H, W)
        if self.upsample:
            x = F.interpolate(x.view(B, C, H, W), scale_factor=2,
                              mode='nearest')
            x = x.view(1, B * C, H * 2, W * 2)
        x = F.conv2d(x, weight, padding=self.padding, groups=B)
        x = x.view(B, self.out_ch, x.shape[2], x.shape[3])
        return x


class NoiseInjection(nn.Module):
    """Compatibility noise layer for older Stage1 checkpoints."""

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, x, noise=None):
        if noise is None:
            noise = x.new_zeros(x.shape[0], 1, x.shape[2], x.shape[3])
        return x + self.weight.view(1, 1, 1, 1) * noise


class StyleBlock(nn.Module):
    """One upsampling StyleGAN block."""
    def __init__(self, in_ch, out_ch, style_dim, upsample=True, use_noise=False):
        super().__init__()
        self.use_noise = use_noise
        self.conv1 = ModulatedConv2d(in_ch, out_ch, 3, style_dim, upsample=upsample)
        self.conv2 = ModulatedConv2d(out_ch, out_ch, 3, style_dim)
        if self.use_noise:
            self.noise1 = NoiseInjection()
            self.noise2 = NoiseInjection()
        self.activate = nn.LeakyReLU(0.2, inplace=True)
        self.to_rgb = ModulatedConv2d(out_ch, 1, 1, style_dim, demodulate=False)

    def forward(self, x, w):
        x = self.conv1(x, w)
        if self.use_noise:
            x = self.noise1(x)
        x = self.activate(x)
        x = self.conv2(x, w)
        if self.use_noise:
            x = self.noise2(x)
        x = self.activate(x)
        rgb = self.to_rgb(x, w)
        return x, rgb


class ComponentCodebook(nn.Module):
    """
    Discrete codebook storing per-character/component features.
    Each entry is a learnable 1x1x512 constant vector.
    """
    def __init__(self, num_entries: int, dim: int = 512):
        super().__init__()
        self.num_entries = num_entries
        self.dim = dim
        # Each code: (dim,) vector, will be used as (1, dim, 1, 1) constant
        self.codes = nn.Embedding(num_entries, dim)
        nn.init.normal_(self.codes.weight, 0, 1)

    def forward(self, indices: Tensor) -> Tensor:
        """
        Args:
            indices: (B,) LongTensor of codebook indices
        Returns:
            codes: (B, dim, 1, 1) feature maps
        """
        emb = self.codes(indices)          # (B, dim)
        return emb.unsqueeze(-1).unsqueeze(-1)  # (B, dim, 1, 1)

    def get_all_codes(self) -> Tensor:
        """Return all codebook entries for visualisation."""
        idx = torch.arange(self.num_entries,
                           device=self.codes.weight.device)
        return self.forward(idx)


class ComponentStyleGAN(nn.Module):
    """
    StyleGAN2 for character/component generation with codebook input.

    Architecture:
      - Input: codebook entry c (B, 512, 1, 1) + style w (B, 512)
      - 7 upsampling blocks: 1x1 -> 4 -> 8 -> 16 -> 32 -> 64 -> 128
      - Output: binary structure image (B, 1, 128, 128)
    """

    def __init__(
        self,
        codebook_dim=512,
        w_dim=512,
        base_ch=512,
        use_noise_in_blocks=False,
        legacy_output_head=False,
    ):
        super().__init__()
        self.w_dim = w_dim
        self.use_noise_in_blocks = use_noise_in_blocks
        self.legacy_output_head = legacy_output_head

        # Initial 1x1 conv from codebook to 4x4
        self.init_conv = nn.ConvTranspose2d(codebook_dim, base_ch, 4, 1, 0)

        # Progressive upsampling blocks
        # 4->8->16->32->64->128
        ch_schedule = [base_ch, base_ch, 256, 256, 128, 64]
        self.blocks = nn.ModuleList()
        in_ch = base_ch
        for i, out_ch in enumerate(ch_schedule):
            upsample = (i > 0)
            self.blocks.append(
                StyleBlock(
                    in_ch,
                    out_ch,
                    w_dim,
                    upsample=upsample,
                    use_noise=use_noise_in_blocks,
                )
            )
            in_ch = out_ch

        # Final output conv
        if legacy_output_head:
            self.final_conv = nn.Sequential(
                nn.Conv2d(1, 32, 3, 1, 1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(32, 1, 3, 1, 1),
                nn.Sigmoid(),
            )
        else:
            self.final_conv = nn.Sequential(
                nn.Conv2d(64, 32, 3, 1, 1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(32, 1, 1),
                nn.Sigmoid(),
            )

    def forward(self, code: Tensor, w: Tensor,
                return_features: bool = False) -> Tensor:
        """
        Args:
            code: (B, codebook_dim, 1, 1) from ComponentCodebook
            w: (B, w_dim) style vector from MappingNetwork
            return_features: if True, return intermediate feature maps
        Returns:
            output: (B, 1, 128, 128) structure image
            features (optional): list of intermediate feature tensors
        """
        x = self.init_conv(code)   # (B, 512, 4, 4)

        features = []

        last_rgb = None
        for block in self.blocks:
            x, rgb = block(x, w)
            features.append(x)
            last_rgb = rgb

        final_input = last_rgb if self.legacy_output_head else x
        out = self.final_conv(final_input)

        if return_features:
            return out, features
        return out

    def get_structure_prior(self, code: Tensor, w: Tensor,
                            layer_indices: list = None) -> list:
        """
        Extract intermediate features as structure priors.
        Returns list of feature maps at specified layer indices.
        Default: last two blocks (32x32 and 64x64 features).
        """
        if layer_indices is None:
            layer_indices = [-2, -1]  # 64x64 and 128x128

        _, features = self.forward(code, w, return_features=True)
        return [features[i] for i in layer_indices]


class StyleGANDiscriminator(nn.Module):
    """
    Progressive discriminator for StyleGAN training.
    Uses spectral normalization for training stability.
    """
    def __init__(self, in_ch=1, base_ch=64):
        super().__init__()
        ch_list = [base_ch, base_ch*2, base_ch*4, base_ch*8, base_ch*8]
        layers = [nn.utils.spectral_norm(nn.Conv2d(in_ch, base_ch, 3, 1, 1)),
                  nn.LeakyReLU(0.2, inplace=True)]
        in_c = base_ch
        for out_c in ch_list:
            layers += [
                nn.utils.spectral_norm(nn.Conv2d(in_c, out_c, 4, 2, 1)),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            in_c = out_c
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),
            nn.utils.spectral_norm(nn.Linear(in_c * 16, 512)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1)
        )

    def forward(self, img, struct=None):
        if struct is not None:
            x = torch.cat([img, struct], dim=1)
        else:
            x = img
        x = self.features(x)
        return self.head(x)
