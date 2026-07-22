"""
Decoder Init Karşılaştırması (ESKİ Kaiming-her-yer init vs YENİ Tanh-uyumlu
output_conv init) — GERÇEK VERİ GEREKTİRMEZ, sadece rastgele embedding ile
forward pass alır.

AMAÇ: src/models/decoder.py FaceDecoder._init_weights()'teki düzeltmenin
(output_conv artık Kaiming(leaky_relu) yerine Xavier(tanh) ile init
ediliyor) rastgele-init çıktıdaki kanal dengesizliğini ve yüksek-frekans
("periyodik") enerjiyi GERÇEKTEN azalttığını sayısal olarak göstermek.

Ölçülen istatistikler (B rastgele embedding, EĞİTİMSİZ model, tek forward):
    - Kanal başı ortalama/std (R,G,B)  -> "kanal-baskınlık" testi
    - |x|>0.95 oranı (Tanh doygunluğu) -> "doygun/vanishing-gradient" testi
    - Yüksek/düşük frekans enerji oranı (2D FFT) -> "periyodiklik" testi

Kullanım (Kaggle/Colab hücresinde veya yerelde):
    from src.inspect_decoder_init import run_init_comparison
    run_init_comparison()
"""

import logging

import torch
import torch.nn as nn

from ..models.decoder import FaceDecoder

log = logging.getLogger(__name__)


# ─── ESKİ (düzeltme öncesi) init'in birebir kopyası ────────────────────────────
# NOT: Bu, decoder.py'deki GERÇEK kodun bir parçası DEĞİL — sadece "önce/sonra"
# karşılaştırması yapabilmek için düzeltmeden ÖNCEKİ davranışın burada
# yeniden üretilmiş bir kopyasıdır. decoder.py'nin kendisi artık bunu
# kullanmıyor (bkz. FaceDecoder._init_weights, output_conv artık ayrı).

def _apply_legacy_all_kaiming_init(model: FaceDecoder):
    """output_conv dahil TÜM Conv2d/Linear katmanlarına Kaiming(leaky_relu)
    uygular — düzeltmeden ÖNCEKİ _init_weights() davranışının kopyası."""
    for m in model.modules():
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


# ─── İstatistikler ──────────────────────────────────────────────────────────

def _channel_stats(x: torch.Tensor) -> dict:
    """x: (B,3,H,W) [-1,1]. Kanal başı mean/std + aralarındaki spread."""
    means = x.mean(dim=[0, 2, 3])
    stds = x.std(dim=[0, 2, 3])
    return {
        "mean_R": means[0].item(), "mean_G": means[1].item(), "mean_B": means[2].item(),
        "std_R": stds[0].item(), "std_G": stds[1].item(), "std_B": stds[2].item(),
        "mean_spread": (means.max() - means.min()).item(),  # 0'a yakın = kanallar dengeli
    }


def _saturation_fraction(x: torch.Tensor, threshold: float = 0.95) -> float:
    """|x|>threshold oranı — Tanh doygunluğu (yüksek = kötü, gradyan sönümü riski)."""
    return (x.abs() > threshold).float().mean().item()


def _high_freq_energy_ratio(x: torch.Tensor) -> float:
    """
    2D FFT ile yüksek/toplam frekans enerjisi oranı ("periyodiklik" proxy'si).
    Görüntü merkezi (DC + düşük frekans) hariç enerjinin toplam enerjiye oranı.
    Yüksek deger = daha fazla yuksek-frekans/gurultu-benzeri icerik.
    x: (B,3,H,W)
    """
    gray = x.mean(dim=1)  # (B,H,W)
    fft = torch.fft.fft2(gray)
    mag = torch.fft.fftshift(fft.abs(), dim=(-2, -1))
    power = (mag ** 2)

    H, W = gray.shape[-2:]
    cy, cx = H // 2, W // 2
    radius = min(H, W) // 8  # merkez = "dusuk frekans" bolgesi
    yy, xx = torch.meshgrid(
        torch.arange(H, device=x.device), torch.arange(W, device=x.device), indexing="ij"
    )
    low_mask = ((yy - cy) ** 2 + (xx - cx) ** 2) <= radius ** 2

    total = power.sum(dim=[-2, -1])
    low = power[:, low_mask].sum(dim=-1)
    high_ratio = 1.0 - (low / total.clamp(min=1e-12))
    return high_ratio.mean().item()


def _report_row(label: str, x: torch.Tensor):
    ch = _channel_stats(x)
    sat = _saturation_fraction(x)
    hf = _high_freq_energy_ratio(x)
    log.info(
        f"[{label:22s}] "
        f"mean(R,G,B)=({ch['mean_R']:+.3f},{ch['mean_G']:+.3f},{ch['mean_B']:+.3f}) "
        f"spread={ch['mean_spread']:.3f}  "
        f"std(R,G,B)=({ch['std_R']:.3f},{ch['std_G']:.3f},{ch['std_B']:.3f})  "
        f"|x|>0.95 oranı={sat:.3f}  "
        f"yüksek-frekans enerji oranı={hf:.3f}"
    )
    return {"channel": ch, "saturation": sat, "high_freq_ratio": hf}


# ─── Ana karşılaştırma ──────────────────────────────────────────────────────

def run_init_comparison(
    batch_size: int = 64,
    embedding_dim: int = 512,
    seed: int = 0,
) -> dict:
    """
    EĞİTİMSİZ (rastgele-init) FaceDecoder'ı iki init modunda karşılaştırır:
        "legacy_kaiming" : output_conv dahil HER YER Kaiming(leaky_relu) (ESKİ/hatalı)
        "fixed_xavier"   : output_conv Xavier(tanh), geri kalan aynı (YENİ/decoder.py)

    Gerçek veri KULLANILMAZ — sadece torch.randn embedding ile forward pass.

    Returns:
        dict: {"legacy": {...}, "fixed": {...}} istatistik raporları
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    torch.manual_seed(seed)
    z = torch.randn(batch_size, embedding_dim)

    log.info("=" * 100)
    log.info(f"Decoder Init Karşılaştırması — B={batch_size}, EĞİTİMSİZ (rastgele ağırlık), aynı z")
    log.info("=" * 100)

    # ── legacy (düzeltme öncesi davranışın kopyası) ─────────────────
    torch.manual_seed(seed)
    model_legacy = FaceDecoder()
    _apply_legacy_all_kaiming_init(model_legacy)  # output_conv'u ESKİ haline geri döndür
    model_legacy.eval()
    with torch.no_grad():
        out_legacy = model_legacy(z)
    report_legacy = _report_row("legacy_kaiming (ESKİ)", out_legacy)

    # ── fixed (decoder.py'nin GERÇEK, güncel _init_weights'i) ───────
    torch.manual_seed(seed)
    model_fixed = FaceDecoder()  # varsayılan __init__ zaten YENİ init'i kullanır
    model_fixed.eval()
    with torch.no_grad():
        out_fixed = model_fixed(z)
    report_fixed = _report_row("fixed_xavier (YENİ)", out_fixed)

    log.info("-" * 100)
    d_spread = report_legacy["channel"]["mean_spread"] - report_fixed["channel"]["mean_spread"]
    d_sat = report_legacy["saturation"] - report_fixed["saturation"]
    d_hf = report_legacy["high_freq_ratio"] - report_fixed["high_freq_ratio"]
    log.info(
        f"[FARK, legacy-fixed]  kanal-spread azalması={d_spread:+.4f}  "
        f"doygunluk azalması={d_sat:+.4f}  yüksek-frekans-enerji azalması={d_hf:+.4f}"
    )
    if d_sat > 0.05 or d_hf > 0.02:
        log.info(
            "SONUÇ: Düzeltme, eğitimsiz çıktıdaki Tanh doygunluğunu ve/veya yüksek-frekans "
            "'gürültü' enerjisini ÖLÇÜLEBİLİR şekilde azaltıyor — hipotez destekleniyor."
        )
    else:
        log.info(
            "SONUÇ: Fark küçük — bu tek başına ana sebep olmayabilir, başka bir "
            "kaynağı (eğitim dinamiği, veri) da araştırmak gerekebilir."
        )
    log.info("=" * 100)

    return {"legacy": report_legacy, "fixed": report_fixed}


# ─── Regresyon kontrolü: mevcut mimari toggle kombinasyonları ─────────────────

def run_architecture_regression_check(batch_size: int = 4, embedding_dim: int = 512) -> bool:
    """
    NOT: GAN altyapısı (discriminator, r1_gamma, adv_weight vb.) kod tabanından
    TAMAMEN kaldırılmış durumda (config.py'de artık use_gan yok) — bu yüzden
    "GAN'lı config" regresyon testi yapılamıyor/uygulanamıyor. Bunun yerine
    decoder.py'de GÜNCEL olarak var olan tüm mimari toggle kombinasyonlarını
    (norm_type, use_noise_injection, use_cascade_skip) init-fix SONRASI
    hatasız forward+backward alabildiklerini doğrulayarak kontrol ediyor.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    configs = [
        dict(norm_type="batch", use_noise_injection=False, use_cascade_skip=False),
        dict(norm_type="instance", use_noise_injection=False, use_cascade_skip=False),
        dict(norm_type="batch", use_noise_injection=True, noise_dim=32, use_cascade_skip=False),
        dict(norm_type="batch", use_noise_injection=False, use_cascade_skip=True,
             cascade_skip_last_n_blocks=2),
        dict(norm_type="batch", use_noise_injection=True, noise_dim=32, use_cascade_skip=True,
             cascade_skip_last_n_blocks=3),
    ]

    all_ok = True
    for i, kwargs in enumerate(configs):
        try:
            model = FaceDecoder(embedding_dim=embedding_dim, **kwargs)
            z = torch.randn(batch_size, embedding_dim, requires_grad=True)
            out = model(z)
            assert out.shape == (batch_size, 3, 128, 128), f"beklenmeyen sekil: {out.shape}"
            loss = out.mean()
            loss.backward()
            has_grad = all(p.grad is not None for p in model.parameters() if p.requires_grad)
            status = "OK" if has_grad else "UYARI: bazi parametrelerde gradyan yok"
            log.info(f"[Regresyon {i+1}/{len(configs)}] {kwargs} -> {status}")
        except Exception as exc:
            all_ok = False
            log.error(f"[Regresyon {i+1}/{len(configs)}] {kwargs} -> HATA: {exc}")

    log.info(f"Regresyon sonucu: {'TÜMÜ BAŞARILI' if all_ok else 'HATA VAR'}")
    return all_ok


if __name__ == "__main__":
    run_init_comparison()
    run_architecture_regression_check()
