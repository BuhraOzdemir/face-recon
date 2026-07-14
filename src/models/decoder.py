"""
FaceDecoder: Embedding (512-dim) → 128×128 RGB yüz görüntüsü.

Mimari:
    MLP Mapping Head   : 512 → 512 → 4×4×128  (~790K param)
    UpsampleBlock ×5   : 4→8→16→32→64→128 px  (~355K param)
    Output Conv        : 16ch → 3ch RGB         (~0.4K param)
    ─────────────────────────────────────────
    Toplam             : ~1.15M parametre
    float32 boyut      : ~4.4 MB
    float16 boyut      : ~2.2 MB
    INT8 (TFLite)      : ~1.1 MB
"""

import torch
import torch.nn as nn
from typing import Tuple


# ─── Temel bloklar ─────────────────────────────────────────────────────────────

class DWResBlock(nn.Module):
    """
    Depthwise Separable Residual Block.

    DW-Conv 3×3 (groups=ch) → PW-Conv 1×1 → BN → LeakyReLU
    Artı: düşük FLOPs, düşük parametre, stabil gradyan akışı.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.dw  = nn.Conv2d(channels, channels, kernel_size=3,
                             padding=1, groups=channels, bias=False)
        self.pw  = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.bn  = nn.BatchNorm2d(channels)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.act(self.bn(self.pw(self.dw(x))))


class UpsampleBlock(nn.Module):
    """
    Bilinear Upsample (×2) + Conv2d + DWResBlock.

    Bilinear upsample checkerboard artifact üretmez (ConvTranspose2d'e göre avantajlı).
    Her blok spatial boyutu iki katına çıkarır ve kanal sayısını ayarlar.
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.res  = DWResBlock(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.res(self.conv(self.up(x)))


# ─── Ana model ─────────────────────────────────────────────────────────────────

class FaceDecoder(nn.Module):
    """
    Embedding-to-Face Reconstruction Decoder.

    Args:
        embedding_dim (int): Giriş embedding boyutu. Varsayılan: 512.
        initial_spatial (int): MLP çıktısı reshape boyutu. Varsayılan: 4 (4×4).
        initial_channels (int): İlk feature map kanal sayısı. Varsayılan: 128.
        decoder_channels (tuple): Her UpsampleBlock'un çıktı kanalları.
            Varsayılan: (128, 128, 64, 32, 16) → 5 blok → 4→128 px.

    Forward:
        z: (B, embedding_dim) → output: (B, 3, 128, 128), aralık [-1, 1]
    """

    def __init__(
        self,
        embedding_dim:   int   = 512,
        initial_spatial: int   = 4,
        initial_channels: int  = 128,
        decoder_channels: Tuple[int, ...] = (128, 128, 64, 32, 16),
    ):
        super().__init__()

        self.initial_spatial   = initial_spatial
        self.initial_channels  = initial_channels
        spatial_flat = initial_spatial * initial_spatial * initial_channels  # 4*4*128 = 2048

        # MLP Mapping Head
        # 512 → 512 → 4×4×128
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, spatial_flat),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # 5× UpsampleBlock:  4→8→16→32→64→128
        channels = [initial_channels] + list(decoder_channels)
        self.up_blocks = nn.ModuleList([
            UpsampleBlock(channels[i], channels[i + 1])
            for i in range(len(decoder_channels))
        ])

        # Çıktı katmanı: son kanal sayısı → RGB, Tanh [-1, 1]
        self.output_conv = nn.Sequential(
            nn.Conv2d(decoder_channels[-1], 3, kernel_size=3, padding=1),
            nn.Tanh(),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (B, 512) L2-normalized embedding
        Returns:
            image: (B, 3, 128, 128), değerler [-1, 1]
        """
        B = z.size(0)

        # MLP → spatial feature map
        x = self.mlp(z)                                            # (B, 2048)
        x = x.view(B, self.initial_channels,
                   self.initial_spatial, self.initial_spatial)     # (B, 128, 4, 4)

        # Upsample ×5: 4→8→16→32→64→128
        for block in self.up_blocks:
            x = block(x)

        return self.output_conv(x)                                 # (B, 3, 128, 128)

    @torch.no_grad()
    def count_parameters(self) -> dict:
        """Model parametre sayısını ve boyutunu raporla."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total_params":      total,
            "trainable_params":  trainable,
            "float32_mb":        round(total * 4 / 1e6, 2),
            "float16_mb":        round(total * 2 / 1e6, 2),
            "int8_mb":           round(total * 1 / 1e6, 2),
        }


# ─── Hızlı test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    model = FaceDecoder()
    info  = model.count_parameters()

    print("FaceDecoder Parametre Raporu:")
    print(f"  Toplam parametre : {info['total_params']:,}")
    print(f"  float32 boyutu   : {info['float32_mb']} MB")
    print(f"  float16 boyutu   : {info['float16_mb']} MB")
    print(f"  INT8 boyutu      : {info['int8_mb']} MB")

    dummy_z = torch.randn(4, 512)
    out     = model(dummy_z)
    print(f"\nGiriş: {dummy_z.shape}  →  Çıktı: {out.shape}")
    print(f"Değer aralığı: [{out.min():.3f}, {out.max():.3f}]")
