"""
"Renk öğreniliyor ama periyodik/noktalı doku hiç değişmiyor" bulgusunu test
eden diagnostik — GERÇEK VERİ/EĞİTİM GEREKTİRMEZ.

Hipotez: 10 residual blok (5×UpsampleBlock × 2×DWResBlock), rastgele
Kaiming-init depthwise kernellerin (3×3, groups=channels) BİR KISMI şans
eseri yüksek-frekans/checkerboard-benzeri desenler kodluyor. Bu desenler
`x + branch(x)` (kimlik-koruyan residual toplama) yoluyla girdiden
BAĞIMSIZ olarak tüm derinlik boyunca taşınıyor. L1/perceptual loss
gradyanı büyüklük olarak DÜŞÜK-FREKANS (renk/parlaklık) hatasına
hakimdir — bu yüzden eğitim rengi hızla düzeltir ama küçük-büyüklükte,
girdiden bağımsız yüksek-frekans "donmuş gürültü"yü değiştirmekte çok
daha yavaştır/isteksizdir.

Üç test:
  1) Embedding darboğazı: SABİT ağırlıklarla, FARKLI rastgele z'lerin
     (kimlik simülasyonu) çıktıları birbirinden yeterince ayrışıyor mu?
  2) Kimlikler-arası yüksek-frekans korelasyonu (ANA TEST): SABİT
     ağırlıklarla, FARKLI z'lerin yüksek-frekans REZİDÜELLERİ (Laplacian-
     benzeri high-pass) birbirine ne kadar benziyor? Yüksek korelasyon =
     desen girdiden bağımsız/ağırlıklara "gömülü".
  3) Sentetik mini-eğitim: aynı model, saf L1 loss ile birkaç yüz rastgele
     "hedef" görüntüye karşı optimize edilir (gerçek renk/doku öğrenmeyi
     simüle eder) — HF-korelasyon/enerji eğitim ÖNCESİ/SONRASI ne kadar
     değişiyor?

Kullanım:
    from src.inspect_residual_texture import run_all_tests
    run_all_tests()
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..models.decoder import FaceDecoder

log = logging.getLogger(__name__)


# ─── Yardımcılar ────────────────────────────────────────────────────────────

def _high_freq_residual(x: torch.Tensor) -> torch.Tensor:
    """x: (B,3,H,W). Basit high-pass: x - avg_pool(x) (düşük-frekansı çıkar)."""
    gray = x.mean(dim=1, keepdim=True)
    low = F.avg_pool2d(gray, kernel_size=5, stride=1, padding=2, count_include_pad=False)
    return (gray - low).squeeze(1)  # (B,H,W)


def _cross_sample_hf_correlation(x: torch.Tensor) -> float:
    """
    SABİT ağırlık, FARKLI z varsayımıyla: yüksek-frekans rezidüellerin
    örnekler-arası ortalama kosinüs benzerliği. ~1.0 = desen z'den
    BAĞIMSIZ (ağırlıklara gömülü); ~0.0 = desen z'ye göre değişiyor (normal).
    """
    hf = _high_freq_residual(x)
    flat = hf.reshape(hf.size(0), -1)
    flat = flat - flat.mean(dim=1, keepdim=True)
    norm = flat.norm(dim=1, keepdim=True).clamp(min=1e-8)
    flat_n = flat / norm
    corr = flat_n @ flat_n.T
    B = corr.size(0)
    mask = ~torch.eye(B, dtype=torch.bool, device=x.device)
    return corr[mask].mean().item()


def _hf_energy(x: torch.Tensor) -> float:
    hf = _high_freq_residual(x)
    return hf.pow(2).mean().item()


def _pairwise_output_distance(x: torch.Tensor) -> float:
    """Farklı z'lerin çıktıları piksel uzayında ne kadar ayrışıyor (embedding darboğazı testi)."""
    B = x.size(0)
    flat = x.reshape(B, -1)
    d = torch.cdist(flat, flat, p=1) / flat.size(1)
    mask = ~torch.eye(B, dtype=torch.bool, device=x.device)
    return d[mask].mean().item()


# ─── Test 1+2: Embedding darboğazı + kimlikler-arası HF korelasyonu ───────────

def test_identity_independence(batch_size: int = 8, embedding_dim: int = 512, seed: int = 0) -> dict:
    torch.manual_seed(seed)
    model = FaceDecoder().eval()

    torch.manual_seed(123)  # z'ler icin AYRI seed (agirliklardan bagimsiz)
    z = torch.randn(batch_size, embedding_dim)
    z = F.normalize(z, dim=1)  # gerçek ArcFace embedding'leri L2-normalized

    with torch.no_grad():
        out = model(z)

    pix_dist = _pairwise_output_distance(out)
    hf_corr = _cross_sample_hf_correlation(out)
    hf_energy = _hf_energy(out)

    log.info("=" * 90)
    log.info("TEST 1+2: Embedding darboğazı + kimlikler-arası yüksek-frekans korelasyonu")
    log.info(f"  {batch_size} FARKLI rastgele z (L2-normalized), AYNI (eğitimsiz) ağırlıklar")
    log.info(f"  Ortalama piksel-uzayı çift-farkı     : {pix_dist:.5f}  (0'a yakınsa = darboğaz riski)")
    log.info(f"  Kimlikler-arası HF-korelasyon        : {hf_corr:.4f}  (1'e yakınsa = desen z'den BAĞIMSIZ)")
    log.info(f"  Ortalama HF enerjisi                 : {hf_energy:.5f}")
    log.info("=" * 90)

    return {"pixel_distance": pix_dist, "hf_correlation": hf_corr, "hf_energy": hf_energy}


# ─── Test 3: Sentetik mini-eğitim (renk mi düzeliyor, doku mu kalıyor?) ────────

def test_synthetic_training(
    batch_size: int = 8,
    embedding_dim: int = 512,
    steps: int = 300,
    lr: float = 3e-4,
    seed: int = 0,
) -> dict:
    """
    Gerçek veri YOK — rastgele "renkli blob" hedeflere karşı saf L1 ile
    kısa bir optimize turu. Amaç: gerçek eğitimin ilk aşamasını (baskın
    olarak düşük-frekans/renk hatasını azaltma) ucuza simüle etmek.
    """
    torch.manual_seed(seed)
    model = FaceDecoder().train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    torch.manual_seed(123)
    z = torch.randn(batch_size, embedding_dim)
    z = F.normalize(z, dim=1)

    # Sentetik "hedef": duz renkli + hafif gradyanli blob (dusuk-frekans agirlikli,
    # gercek yuz fotograflarindaki gibi "renk/ton" baskin, ince doku YOK)
    torch.manual_seed(456)
    base_color = torch.rand(batch_size, 3, 1, 1) * 2 - 1
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, 128), torch.linspace(-1, 1, 128), indexing="ij")
    grad = (0.15 * (xx + yy)).unsqueeze(0).unsqueeze(0)
    target = (base_color + grad).clamp(-1, 1)

    with torch.no_grad():
        out_before = model(z)
    stats_before = {
        "hf_correlation": _cross_sample_hf_correlation(out_before),
        "hf_energy": _hf_energy(out_before),
        "l1_to_target": F.l1_loss(out_before, target).item(),
    }

    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        out = model(z)
        loss = F.l1_loss(out, target)
        loss.backward()
        opt.step()
        if step % 100 == 0:
            log.info(f"  [sentetik] step={step:4d}  L1={loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        out_after = model(z)
    stats_after = {
        "hf_correlation": _cross_sample_hf_correlation(out_after),
        "hf_energy": _hf_energy(out_after),
        "l1_to_target": F.l1_loss(out_after, target).item(),
    }

    log.info("=" * 90)
    log.info(f"TEST 3: Sentetik mini-eğitim ({steps} adım, saf L1, renk-baskın hedef)")
    log.info(f"  L1(hedef)         : {stats_before['l1_to_target']:.4f} -> {stats_after['l1_to_target']:.4f}")
    log.info(f"  HF-korelasyon     : {stats_before['hf_correlation']:.4f} -> {stats_after['hf_correlation']:.4f}")
    log.info(f"  HF enerjisi       : {stats_before['hf_energy']:.5f} -> {stats_after['hf_energy']:.5f}")
    l1_drop = 1 - stats_after["l1_to_target"] / max(stats_before["l1_to_target"], 1e-8)
    hf_corr_drop = stats_before["hf_correlation"] - stats_after["hf_correlation"]
    log.info(f"  L1 azalma oranı={l1_drop:.2%}  HF-korelasyon azalma={hf_corr_drop:+.4f}")
    if l1_drop > 0.5 and abs(hf_corr_drop) < 0.1:
        log.info(
            "SONUÇ: Renk/ton (L1) hızla düzeliyor AMA kimlikler-arası HF-korelasyon "
            "NEREDEYSE SABİT kalıyor — 'desen ağırlıklara gömülü, eğitimle silinmiyor' "
            "hipotezi DESTEKLENİYOR."
        )
    else:
        log.info("SONUÇ: HF-korelasyon da eğitimle belirgin değişti — hipotez bu haliyle desteklenmiyor.")
    log.info("=" * 90)

    return {"before": stats_before, "after": stats_after}


def run_all_tests():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    r1 = test_identity_independence()
    r2 = test_synthetic_training()
    return {"identity_independence": r1, "synthetic_training": r2}


if __name__ == "__main__":
    run_all_tests()
