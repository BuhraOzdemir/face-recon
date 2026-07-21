"""
PatchGAN Discriminator — SADECE egitimde kullanilir, export/deploy edilen
FaceDecoder modeline hic dahil edilmez. QR/boyut kisitini etkilemez.

128x128 RGB girdi -> patch-level "gercek/sahte" logit haritasi.
Spectral normalization ile stabilize edilir.
Ara katman aktivasyonlari feature matching loss icin expose edilir.
"""

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
from typing import List, Tuple


def _sn_conv(in_ch, out_ch, k=4, s=2, p=1):
    return spectral_norm(nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p))


class PatchDiscriminator(nn.Module):
    """
    128 -> 64 -> 32 -> 16 -> 8 -> patch logits (~7x7 alan).
    Varsayilan base_channels=64 (daha guclu adversarial sinyal).
    """

    def __init__(self, in_channels: int = 3, base_channels: int = 64):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Sequential(
                _sn_conv(in_channels, base_channels),
                nn.LeakyReLU(0.2, inplace=True),
            ),
            nn.Sequential(
                _sn_conv(base_channels, base_channels * 2),
                nn.LeakyReLU(0.2, inplace=True),
            ),
            nn.Sequential(
                _sn_conv(base_channels * 2, base_channels * 4),
                nn.LeakyReLU(0.2, inplace=True),
            ),
            nn.Sequential(
                _sn_conv(base_channels * 4, base_channels * 8),
                nn.LeakyReLU(0.2, inplace=True),
            ),
            spectral_norm(
                nn.Conv2d(base_channels * 8, 1, kernel_size=4, stride=1, padding=1)
            ),
        ])

    def forward_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Returns:
            logits: (B, 1, ~7, ~7)
            features: ara katman aktivasyonlari (feature matching icin)
        """
        features: List[torch.Tensor] = []
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i < len(self.layers) - 1:
                features.append(h)
        return h, features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, 128, 128), [-1, 1]
        Returns:
            (B, 1, ~7, ~7) patch logits (ham, sigmoid uygulanmamis)
        """
        logits, _ = self.forward_features(x)
        return logits

    @torch.no_grad()
    def count_parameters(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        return {"total_params": total, "float32_mb": round(total * 4 / 1e6, 2)}


if __name__ == "__main__":
    disc = PatchDiscriminator()
    info = disc.count_parameters()
    print(f"Discriminator parametre: {info['total_params']:,} ({info['float32_mb']} MB)")
    dummy = torch.randn(4, 3, 128, 128)
    out, feats = disc.forward_features(dummy)
    print(f"Girdi: {dummy.shape} -> Cikti: {out.shape}, feat layers: {len(feats)}")
