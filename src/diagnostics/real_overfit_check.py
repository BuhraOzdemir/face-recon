"""
5/5 — real_overfit_check.py: Küçük GERÇEK veri ezberleme testi.

overfit_check.py'nin sentetik-veri sürümünün aksine, burada 32 GERÇEK
(JPG, embedding) eşleşmiş çifti kullanılır. EMA KAPALI, scheduler KAPALI,
augmentation KAPALI (build_val_transform — flip/color-jitter yok), SADECE
L1 loss, SABİT öğrenme oranı. Model bu 32 örneği ezberleyemiyorsa
(loss yeterince düşmüyorsa/görsel olarak yakınsamıyorsa), sorun eğitim
fazlarından (identity ağırlığı vb.) değil — temel veri eşleşmesi, gradyan
akışı veya normalizasyon zincirinden kaynaklanıyor demektir.

Varsayılan: SIFIRDAN (rastgele init) model — checkpoint'ten değil, çünkü
amaç mimarinin/pipeline'ın TEMEL kapasitesini test etmek (checkpoint
kullanmak istenirse checkpoint_path verilebilir).

Kullanım:
    from src.real_overfit_check import run_real_overfit_test
    run_real_overfit_test(manifest_path="/path/manifest.txt")
"""

import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

from ..config import Config, DEFAULT_CONFIG
from ..data.dataset import _load_manifest_samples, FaceDataset, build_val_transform, denormalize
from ..models.decoder import FaceDecoder
from .common import build_model_from_checkpoint

log = logging.getLogger(__name__)


def run_real_overfit_test(
    manifest_path: str,
    cfg: Config = DEFAULT_CONFIG,
    checkpoint_path: str = None,
    n_samples: int = 32,
    steps: int = 1500,
    lr: float = 3e-4,
    log_every: int = 100,
    save_path: str = "/kaggle/working/real_overfit_check.png",
) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"[Real Overfit] Cihaz: {device}")

    all_samples = _load_manifest_samples(manifest_path)
    subset = all_samples[:n_samples]
    log.info(f"[Real Overfit] {len(subset)} GERÇEK örnek seçildi (istenen: {n_samples}).")

    # build_val_transform: augmentation YOK (RandomHorizontalFlip/ColorJitter içermez)
    ds = FaceDataset(subset, build_val_transform(cfg.data.image_size), cfg.data.image_size)

    embeddings = torch.stack([ds[i][0] for i in range(len(ds))]).to(device)
    images = torch.stack([ds[i][1] for i in range(len(ds))]).to(device)
    log.info(f"[Real Overfit] embeddings: {embeddings.shape}, images: {images.shape}")
    log.info(
        f"[Real Overfit] embedding std (kimlikler-arası çeşitlilik kontrolü): "
        f"{embeddings.std(dim=0).mean().item():.4f} (0'a çok yakınsa embedding'ler şüpheli derecede benzer)"
    )

    if checkpoint_path:
        model, _ = build_model_from_checkpoint(checkpoint_path, cfg, device)
        log.info("[Real Overfit] Checkpoint'ten başlatıldı (varsayılan: sıfırdan önerilir).")
    else:
        model = FaceDecoder(
            embedding_dim=cfg.model.embedding_dim,
            initial_spatial=cfg.model.initial_spatial,
            initial_channels=cfg.model.initial_channels,
            decoder_channels=cfg.model.decoder_channels,
            norm_type=cfg.model.norm_type,
        ).to(device)
        log.info("[Real Overfit] SIFIRDAN (rastgele init) model — mimari/pipeline temel testi.")

    # KASITLI OLARAK YOK: scheduler, EMA, AMP, augmentation, GAN — sadece
    # sabit-LR AdamW + saf L1. Amaç mumkun olan en az degiskenle test etmek.
    optimizer = AdamW(model.parameters(), lr=lr)
    l1_loss = nn.L1Loss()

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
            log.info(f"[Real Overfit] step={step:4d}/{steps}  L1={final_l1:.5f}")

    model.eval()
    with torch.no_grad():
        generated = model(embeddings[:8]).cpu()
    real_np = denormalize(images[:8].cpu()).permute(0, 2, 3, 1).numpy()
    gen_np = denormalize(generated).permute(0, 2, 3, 1).numpy()

    fig, axes = plt.subplots(2, 8, figsize=(20, 6))
    for i in range(8):
        axes[0, i].imshow(real_np[i].clip(0, 1)); axes[0, i].set_title("Gerçek", fontsize=8); axes[0, i].axis("off")
        axes[1, i].imshow(gen_np[i].clip(0, 1)); axes[1, i].set_title("Ezber", fontsize=8); axes[1, i].axis("off")
    plt.suptitle(f"GERÇEK Veri Ezberleme Testi — final L1={final_l1:.4f}", fontsize=13)
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()

    still_improving = False
    if len(recent_losses) >= 100:
        first_half = sum(recent_losses[:100]) / 100
        second_half = sum(recent_losses[-100:]) / 100
        still_improving = second_half < first_half * 0.98

    log.info("=" * 90)
    log.info(
        f"[Real Overfit] TAMAMLANDI. Final L1: {final_l1:.5f}  "
        f"(son 200 adımda hâlâ düşüyor: {still_improving})"
    )
    if final_l1 < 0.06 or still_improving:
        verdict = (
            "SAĞLIKLI — temel veri/gradyan/normalizasyon akışı ÇALIŞIYOR. Ana eğitimdeki "
            "sorun muhtemelen 'yetersiz eğitim süresi/faz zamanlaması' türünden, "
            "mimari/veri-hattı bir bug DEĞİL."
        )
    else:
        verdict = (
            "ŞÜPHELİ — 32 örneği bile ezberleyemiyor. Sorun identity fazından ÖNCE, "
            "temel veri eşleşmesi/gradyan akışı/normalizasyon zincirinde aranmalı "
            "(audit_pairing.py ve audit_identity_gradient.py sonuçlarına bakın)."
        )
    log.info(f"[Yorum] {verdict}")
    log.info("=" * 90)

    return {"final_l1": final_l1, "still_improving": still_improving, "verdict": verdict}


if __name__ == "__main__":
    import sys
    run_real_overfit_test(sys.argv[1])
