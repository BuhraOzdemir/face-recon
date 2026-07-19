"""
PatchGAN Discriminator — SADECE egitimde kullanilir, export/deploy edilen
FaceDecoder modeline hic dahil edilmez. QR/boyut kisitini etkilemez.

128x128 RGB girdi -> patch-level "gercek/sahte" logit haritasi.
Spectral normalization ile stabilize edilir (R1 penalty gibi ekstra
backward gerektirmez, kucuk cozunurlukte yeterince etkili ve ucuz).
"""

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


def _sn_conv(in_ch, out_ch, k=4, s=2, p=1):
    return spectral_norm(nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p))


class PatchDiscriminator(nn.Module):
    """
    128 -> 64 -> 32 -> 16 -> 8 -> patch logits (~7x7 alan).
    """

    def __init__(self, in_channels: int = 3, base_channels: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            _sn_conv(in_channels, base_channels),                 # 128 -> 64
            nn.LeakyReLU(0.2, inplace=True),

            _sn_conv(base_channels, base_channels * 2),           # 64 -> 32
            nn.LeakyReLU(0.2, inplace=True),

            _sn_conv(base_channels * 2, base_channels * 4),       # 32 -> 16
            nn.LeakyReLU(0.2, inplace=True),

            _sn_conv(base_channels * 4, base_channels * 8),       # 16 -> 8
            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm(
                nn.Conv2d(base_channels * 8, 1, kernel_size=4, stride=1, padding=1)
            ),  # 8 -> ~7x7 patch logits, sigmoid YOK (BCEWithLogits kullanilacak)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, 128, 128), [-1, 1]
        Returns:
            (B, 1, ~7, ~7) patch logits (ham, sigmoid uygulanmamis)
        """
        return self.net(x)

    @torch.no_grad()
    def count_parameters(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        return {"total_params": total, "float32_mb": round(total * 4 / 1e6, 2)}


if __name__ == "__main__":
    disc = PatchDiscriminator()
    info = disc.count_parameters()
    print(f"Discriminator parametre: {info['total_params']:,} ({info['float32_mb']} MB)")
    dummy = torch.randn(4, 3, 128, 128)
    out = disc(dummy)
    print(f"Girdi: {dummy.shape} -> Cikti: {out.shape}")