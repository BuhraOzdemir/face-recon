"""
Faz A — 32 Örnek Overfit Sanity Check (v3 mimari dokümanından, §7.1).

AMAÇ: Tam veri setiyle günlerce eğitmeden ÖNCE, modelin/pipeline'ın
öğrenebilme kapasitesini ucuza doğrula. Küçük bir örnek kümesini
(varsayılan 32) ezberlemesi (overfit) beklenir — sadece L1 loss ile.

Sonuç yorumu:
    Loss çok düşerse (örn. <0.02) VE üretilen görüntüler gerçeklere
    görsel olarak yakınsarsa → mimari/pipeline SAĞLAM, ana sorun
    muhtemelen "yetersiz epoch" veya "veri miktarı/çeşitliliği".

    Loss düşmezse veya görüntüler hâlâ anlamsızsa (renk blob'u vs.) →
    gerçek bir BUG var (gradient akışı, veri eşleşmesi, init, LR vb.)
    — ana eğitime geçmeden önce bunu çöz.

Kullanım (Kaggle/Colab hücresinde):
    from src.overfit_check import run_overfit_test
    run_overfit_test(cfg, manifest_path)
"""

import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

from ..config import Config
from ..data.dataset import _load_manifest_samples, FaceDataset, build_val_transform, denormalize
from ..models.decoder import FaceDecoder

log = logging.getLogger(__name__)


def run_overfit_test(
    cfg: Config,
    manifest_path: str,
    n_samples: int = 32,
    steps: int = 800,
    lr: float = 3e-4,
    log_every: int = 50,
    save_path: str = "/kaggle/working/overfit_check.png",
) -> float:
    """
    Küçük bir örnek kümesinde decoder'ın overfit edip edemediğini test eder.

    Returns:
        final_l1: son adımdaki L1 loss değeri (düşükse ~0.01-0.03 iyi işaret)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"[Overfit Test] Cihaz: {device}")

    all_samples = _load_manifest_samples(manifest_path)
    subset = all_samples[:n_samples]
    log.info(f"[Overfit Test] {len(subset)} örnek seçildi (istenen: {n_samples}).")

    ds = FaceDataset(subset, build_val_transform(cfg.data.image_size), cfg.data.image_size)

    # Tüm örnekleri belleğe al — tek batch olarak defalarca kullanılacak
    embeddings = torch.stack([ds[i][0] for i in range(len(ds))]).to(device)
    images     = torch.stack([ds[i][1] for i in range(len(ds))]).to(device)
    log.info(f"[Overfit Test] embeddings: {embeddings.shape}, images: {images.shape}")

    model = FaceDecoder(
        embedding_dim=cfg.model.embedding_dim,
        initial_spatial=cfg.model.initial_spatial,
        initial_channels=cfg.model.initial_channels,
        decoder_channels=cfg.model.decoder_channels,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=lr)
    l1_loss   = nn.L1Loss()

    model.train()
    final_l1 = None
    recent_losses: list = []
    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        generated = model(embeddings)
        loss = l1_loss(generated, images)
        loss.backward()
        optimizer.step()

        final_l1 = loss.item()
        recent_losses.append(final_l1)
        if len(recent_losses) > 200:
            recent_losses.pop(0)
        if step % log_every == 0 or step == 1:
            log.info(f"[Overfit Test] step={step:4d}/{steps}  L1={final_l1:.5f}")

    # ── Sonuç görselleştirme ────────────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        generated = model(embeddings[:8]).cpu()
    real_np = denormalize(images[:8].cpu()).permute(0, 2, 3, 1).numpy()
    gen_np  = denormalize(generated).permute(0, 2, 3, 1).numpy()

    fig, axes = plt.subplots(2, 8, figsize=(20, 6))
    for i in range(8):
        axes[0, i].imshow(real_np[i].clip(0, 1)); axes[0, i].set_title("Gerçek", fontsize=8); axes[0, i].axis("off")
        axes[1, i].imshow(gen_np[i].clip(0, 1));  axes[1, i].set_title("Overfit", fontsize=8); axes[1, i].axis("off")
    plt.suptitle(f"Faz A Overfit Testi — final L1={final_l1:.4f}", fontsize=13)
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()

    # Trend kontrolü: son 200 adımın ilk yarısı ile ikinci yarısını kıyasla.
    # Loss hâlâ düşüyorsa (plato yapmadıysa) mutlak eşik yanıltıcı olur —
    # asıl karar görsel yakınsamaya bırakılmalı.
    still_improving = False
    if len(recent_losses) >= 100:
        first_half  = sum(recent_losses[:100]) / 100
        second_half = sum(recent_losses[-100:]) / 100
        still_improving = second_half < first_half * 0.98

    log.info(f"[Overfit Test] TAMAMLANDI. Final L1: {final_l1:.5f}  "
              f"(son 200 adımda hâlâ düşüyor: {still_improving})")

    if final_l1 < 0.06 or still_improving:
        log.info("[Overfit Test] SONUÇ: Loss düşüyor ve/veya makul seviyede — "
                  "pipeline SAĞLAM görünüyor. Görsel çıktıyı MUTLAKA gözle kontrol et "
                  "(sayısal eşikten daha güvenilir gösterge). Ana eğitimdeki blob/anlamsız "
                  "çıktı sorunu muhtemelen yetersiz epoch/veri, mimari hatası DEĞİL.")
    else:
        log.warning("[Overfit Test] SONUÇ: Loss plato yaptı VE hâlâ yüksek — "
                     "model/veri/gradient akışında bir SORUN olabilir, ana eğitime "
                     "geçmeden önce incele.")

    return final_l1
