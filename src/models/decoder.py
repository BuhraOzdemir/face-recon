"""
FaceDecoder: Embedding (512-dim) → 128×128 RGB yüz görüntüsü.

Mimari (kalite odaklı, mobil üst sınır içinde):
    MLP Mapping Head   : 512 → 512 → 4×4×192
    UpsampleBlock ×5   : 4→8→16→32→64→128 px  (bilinear + Conv + 2× DWRes)
    Output Conv        : 32ch → 3ch RGB
    ─────────────────────────────────────────
    INT8 TFLite hedefi: < 5 MB
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


# ─── Noise Modulation (AdaIN-tarzı, opsiyonel) ─────────────────────────────────

class NoiseModulation(nn.Module):
    """
    Stil vektörü w'den üretilen kanal-başı (scale, shift) ile x'i modüle eder:
        x' = x * scale(w) + shift(w)

    Kimlik-taşıyıcı embedding z, poz/ifade/aydınlatmadan bağımsız (invaryant)
    tasarlandığı için tek başına decoder'ı deterministik "ortalama yüz"e
    yönlendirir. w (ayrı bir gürültü z'den mapping ile üretilir) her
    UpsampleBlock çıktısına bu modülasyonla enjekte edilerek modele z'nin
    açıklamadığı yüksek-frekans/doku detayı üretme serbestliği tanır.

    Init: scale ağırlığı=0/bias=1, shift ağırlığı=0/bias=0 → ilk anda TAM
    KİMLİK dönüşümü (w'den bağımsız), eğitim ilerledikçe modülasyon öğrenilir.
    """

    def __init__(self, style_dim: int, channels: int):
        super().__init__()
        self.to_scale = nn.Linear(style_dim, channels)
        self.to_shift = nn.Linear(style_dim, channels)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.to_scale.weight)
        nn.init.ones_(self.to_scale.bias)
        nn.init.zeros_(self.to_shift.weight)
        nn.init.zeros_(self.to_shift.bias)

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        B, C = x.size(0), x.size(1)
        scale = self.to_scale(w).view(B, C, 1, 1)
        shift = self.to_shift(w).view(B, C, 1, 1)
        return x * scale + shift


# ─── Norm fabrikası ─────────────────────────────────────────────────────────────

def _make_norm(norm_type: str, channels: int) -> nn.Module:
    """
    "batch" (varsayılan) veya "instance".
    state_dict anahtarları farklı olduğu için norm_type değiştirilince
    eski checkpoint'lerle UYUMSUZDUR.
    """
    if norm_type == "batch":
        return nn.BatchNorm2d(channels)
    elif norm_type == "instance":
        return nn.InstanceNorm2d(channels, affine=True)
    raise ValueError(f"Bilinmeyen norm_type: {norm_type} (batch|instance)")


# ─── Temel bloklar ─────────────────────────────────────────────────────────────

class DWResBlock(nn.Module):
    """
    Depthwise Separable Residual Block.

    DW-Conv 3×3 (groups=ch) → PW-Conv 1×1 → Norm → LeakyReLU
    """

    def __init__(self, channels: int, norm_type: str = "batch"):
        super().__init__()
        self.dw  = nn.Conv2d(channels, channels, kernel_size=3,
                             padding=1, groups=channels, bias=False)
        self.pw  = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.bn  = _make_norm(norm_type, channels)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.act(self.bn(self.pw(self.dw(x))))


class UpsampleBlock(nn.Module):
    """
    Bilinear Upsample (×2) + Conv2d + 2× DWResBlock.

    PixelShuffle checkerboard artifact üretir (özellikle erken eğitimde
    yoğun ızgara deseni). Teknik rapordaki öneri: Upsample + Conv —
    checkerboard yok, eğitim stabil.

    ── Cascade skip (opsiyonel) ──
    Bloğun girdisi 1×1 conv + bilinear-upsample ile çıktıya eklenir.
    Kapalıyken mimari birebir aynı kalır.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        norm_type: str = "batch",
        use_noise_injection: bool = False,
        noise_style_dim: int = 64,
        use_cascade_skip: bool = False,
    ):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            _make_norm(norm_type, out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.use_noise_injection = use_noise_injection
        if use_noise_injection:
            self.noise_mod = NoiseModulation(noise_style_dim, out_ch)
        self.res1 = DWResBlock(out_ch, norm_type=norm_type)
        self.res2 = DWResBlock(out_ch, norm_type=norm_type)

        self.use_cascade_skip = use_cascade_skip
        if use_cascade_skip:
            self.skip_proj = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            )
            # Sıfır-init: ilk forward'da skip'siz halle matematiksel özdeş
            nn.init.zeros_(self.skip_proj[0].weight)

    def forward(self, x: torch.Tensor, w: Optional[torch.Tensor] = None) -> torch.Tensor:
        skip_input = x
        x = self.up(x)
        if self.use_noise_injection and w is not None:
            x = self.noise_mod(x, w)
        out = self.res2(self.res1(x))
        if self.use_cascade_skip:
            out = out + self.skip_proj(skip_input)
        return out


# ─── Ana model ─────────────────────────────────────────────────────────────────

class FaceDecoder(nn.Module):
    """
    Embedding-to-Face Reconstruction Decoder.

    Args:
        embedding_dim (int): Giriş embedding boyutu. Varsayılan: 512.
        initial_spatial (int): MLP çıktısı reshape boyutu. Varsayılan: 4 (4×4).
        initial_channels (int): İlk feature map kanal sayısı. Varsayılan: 192.
        decoder_channels (tuple): Her UpsampleBlock'un çıktı kanalları.
            Varsayılan: (192, 128, 96, 64, 32) → 5 blok → 4→128 px.
        use_cascade_skip (bool): Opsiyonel cascade skip.
        cascade_skip_last_n_blocks (int): Skip alan son N upsample bloğu.

    Forward:
        z: (B, embedding_dim) → output: (B, 3, 128, 128), aralık [-1, 1]
    """

    def __init__(
        self,
        embedding_dim:   int   = 512,
        initial_spatial: int   = 4,
        initial_channels: int  = 192,
        decoder_channels: Tuple[int, ...] = (192, 128, 96, 64, 32),
        norm_type: str = "batch",
        use_noise_injection: bool = False,
        noise_dim: int = 64,
        use_cascade_skip: bool = False,
        cascade_skip_last_n_blocks: int = 2,
    ):
        super().__init__()

        self.initial_spatial   = initial_spatial
        self.initial_channels  = initial_channels
        spatial_flat = initial_spatial * initial_spatial * initial_channels

        # MLP Mapping Head: 512 → 512 → 4×4×C
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, spatial_flat),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # ── Noise injection (opsiyonel) ─────────────────────────────
        # noise_dim: dogrudan embedding_dim (512, kimlik bilgisi) DEGIL —
        # ayri, kucuk bir stokastik latent. Girdi sozlesmesini (sadece z)
        # bozmaz: forward(z) tek basina calisir, gurultu otomatik ornekelenir.
        self.use_noise_injection = use_noise_injection
        self.noise_dim = noise_dim
        if use_noise_injection:
            self.noise_mlp = nn.Sequential(
                nn.Linear(noise_dim, noise_dim),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(noise_dim, noise_dim),
                nn.LeakyReLU(0.2, inplace=True),
            )

        # 5× UpsampleBlock:  4→8→16→32→64→128
        channels = [initial_channels] + list(decoder_channels)
        n_blocks = len(decoder_channels)
        # Cascade skip: sadece son N bloğa uygulanır (<5MB INT8 bütçesi)
        # (mobil INT8 <5MB bütçesi). Kapalıyken mimari birebir aynı.
        self.use_cascade_skip = use_cascade_skip
        self.cascade_skip_last_n_blocks = cascade_skip_last_n_blocks
        skip_flags = [
            use_cascade_skip and (i >= n_blocks - cascade_skip_last_n_blocks)
            for i in range(n_blocks)
        ]
        self.up_blocks = nn.ModuleList([
            UpsampleBlock(
                channels[i], channels[i + 1], norm_type=norm_type,
                use_noise_injection=use_noise_injection, noise_style_dim=noise_dim,
                use_cascade_skip=skip_flags[i],
            )
            for i in range(n_blocks)
        ])

        # Çıktı katmanı: son kanal sayısı → RGB, Tanh [-1, 1]
        self.output_conv = nn.Sequential(
            nn.Conv2d(decoder_channels[-1], 3, kernel_size=3, padding=1),
            nn.Tanh(),
        )

        self._init_weights()
        if use_noise_injection:
            # _init_weights genel Linear init'i NoiseModulation'in ozenle
            # secilmis "kimlik donusumu" init'ini (scale=1,shift=0) ezer —
            # bu yuzden ozel init'i genel pass'tan SONRA tekrar uyguluyoruz.
            for m in self.modules():
                if isinstance(m, NoiseModulation):
                    m.reset_parameters()
        if use_cascade_skip:
            # _init_weights Kaiming'i skip_proj'u ezer — sıfır-init'i yenile
            for block in self.up_blocks:
                if block.use_cascade_skip:
                    nn.init.zeros_(block.skip_proj[0].weight)

    def _init_weights(self):
        out_conv = self.output_conv[0]
        for m in self.modules():
            if m is out_conv:
                # Tanh'a besleyen son katman - asagida AYRI ve UYGUN init alir.
                continue
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # ── output_conv: Tanh-uyumlu init ───────────────────────────
        # Kaiming(nonlinearity="leaky_relu") ReLU-ailesi aktivasyonlar icin
        # ileri-yon varyansini korumaya gore kalibre edilmistir; Tanh gibi
        # DOYAN (saturating) bir aktivasyona bu gain'i uygulamak, ozellikle
        # bu katmanin fan_out'u kucuk oldugundan (3 kanal -> std~0.27),
        # pre-aktivasyonlari erken doygunluga (|x|>2-3) itebilir. Doygun
        # Tanh -> vanishing gradient -> R/G/B kanallarindaki rastgele
        # init dengesizligi egitimle duzelemez (periyodik/kanal-baskin
        # gurultu semptomuyla ortusuyor). Xavier/Glorot + tanh gain,
        # PyTorch'un bu aktivasyon icin onerdigi standart eslesmedir.
        nn.init.xavier_normal_(out_conv.weight, gain=nn.init.calculate_gain("tanh"))
        if out_conv.bias is not None:
            nn.init.zeros_(out_conv.bias)

    def forward(
        self,
        z: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
        noise_seed: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Args:
            z: (B, 512) L2-normalized embedding
            noise: (B, noise_dim) opsiyonel; verilmezse ve use_noise_injection
                aktifse otomatik torch.randn ile örneklenir. Girdi
                sözleşmesini bozmaz — normal çağrı hâlâ forward(z)'dir.
            noise_seed: verilirse (ve noise=None ise) gürültü bu seed'le
                DETERMİNİSTİK üretilir (ör. inference'ta tekrarlanabilirlik
                için). use_noise_injection kapalıyken tamamen yoksayılır.
        Returns:
            image: (B, 3, 128, 128), değerler [-1, 1]
        """
        B = z.size(0)

        x = self.mlp(z)
        x = x.view(B, self.initial_channels,
                   self.initial_spatial, self.initial_spatial)

        w = None
        if self.use_noise_injection:
            if noise is not None:
                noise_z = noise
            elif noise_seed is not None:
                gen = torch.Generator(device="cpu").manual_seed(int(noise_seed))
                noise_z = torch.randn(B, self.noise_dim, generator=gen).to(z.device)
            else:
                noise_z = torch.randn(B, self.noise_dim, device=z.device)
            w = self.noise_mlp(noise_z.to(dtype=x.dtype))

        for block in self.up_blocks:
            x = block(x, w)

        return self.output_conv(x)

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
    print(f"\nGiris: {dummy_z.shape}  ->  Cikti: {out.shape}")
    print(f"Değer aralığı: [{out.min():.3f}, {out.max():.3f}]")
