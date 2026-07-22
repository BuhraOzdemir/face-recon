"""
2/5 — inspect_conditioning.py: Decoder'ın embedding girdisine duyarlılığı.

"Conditioning collapse" testi: decoder çıktısı gerçekten z'ye mi bağlı,
yoksa z'den bağımsız (öğrenilmiş) bir "ortalama çıktı"ya mı yakınsamış?
Gerçek eğitim YAPILMAZ — mevcut bir checkpoint + gerçek val batch ile
sadece forward pass'ler.

Testler:
    a) normal        : recon = D(z),            loss_correct = L1(recon, target)
    b) shuffled       : recon = D(z[perm]),      loss_shuffled = L1(recon, target)
                        SAĞLIKLI modelde loss_shuffled >> loss_correct olmalı.
    c) zero embedding : D(0) normal çıktıya çok benziyorsa conditioning zayıf.
    d) mean embedding : D(mean(z)) farklı z'lerin çıktılarına yakınsa
                        model ortalamaya düşmüş.
    e) latent duyarlılık: ||D(z+eps)-D(z)|| / ||eps||, birkaç eps büyüklüğünde.
                        ~0 ise decoder'ın girdiye duyarlılığı çökmüş.

Kullanım:
    from src.inspect_conditioning import run_conditioning_check
    run_conditioning_check(cfg, manifest_path, "/path/checkpoint.pt")
"""

import logging
from pathlib import Path

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from ..config import Config
from ..data.dataset import denormalize
from .common import build_model_from_checkpoint, sample_val_batch

log = logging.getLogger(__name__)


def run_conditioning_check(
    cfg: Config,
    manifest_path: str,
    checkpoint_path: str,
    n_samples: int = 8,
    eps_list=(0.01, 0.1, 0.5, 1.0),
    save_path: str = "/kaggle/working/conditioning_check.png",
) -> dict:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, state = build_model_from_checkpoint(checkpoint_path, cfg, device)
    model.eval()
    log.info(f"[Conditioning] Checkpoint yüklendi: epoch={state.get('epoch')}")

    z, target = sample_val_batch(manifest_path, cfg, n_samples=n_samples, device=device)
    B = z.size(0)

    with torch.no_grad():
        recon = model(z)
        loss_correct = F.l1_loss(recon, target).item()

        perm = torch.randperm(B, device=device)
        recon_shuffled = model(z[perm])
        loss_shuffled = F.l1_loss(recon_shuffled, target).item()

        recon_zero = model(torch.zeros_like(z))
        z_mean = z.mean(dim=0, keepdim=True).repeat(B, 1)
        recon_mean = model(z_mean)

        # zero/mean ciktilarinin NORMAL ciktiya (ayni ornekler) ortalama mesafesi
        dist_zero_to_normal = (recon_zero - recon).abs().mean().item()
        dist_mean_to_normal = (recon_mean - recon).abs().mean().item()
        # normal ciktilarin KENDI ARALARINDAKI ortalama mesafesi (referans olcek)
        pairwise_normal = torch.cdist(
            recon.reshape(B, -1), recon.reshape(B, -1), p=1
        ) / recon[0].numel()
        mask = ~torch.eye(B, dtype=torch.bool, device=device)
        ref_pairwise_dist = pairwise_normal[mask].mean().item()

        sensitivity = {}
        for eps in eps_list:
            delta = torch.randn_like(z) * eps
            recon_pert = model(z + delta)
            num = (recon_pert - recon).reshape(B, -1).norm(dim=1).mean().item()
            den = delta.reshape(B, -1).norm(dim=1).mean().item()
            sensitivity[eps] = num / max(den, 1e-8)

    log.info("=" * 90)
    log.info("INSPECT_CONDITIONING — Decoder'ın embedding girdisine duyarlılığı")
    log.info("=" * 90)
    log.info(f"  loss_correct (normal z)          : {loss_correct:.5f}")
    log.info(f"  loss_shuffled (karıştırılmış z)  : {loss_shuffled:.5f}  "
              f"(oran shuffled/correct = {loss_shuffled / max(loss_correct,1e-8):.2f}x)")
    log.info(f"  Referans: normal çıktıların kendi-aralarında ort. mesafesi = {ref_pairwise_dist:.5f}")
    log.info(f"  D(0) - D(z) ort. mesafe          : {dist_zero_to_normal:.5f}  "
              f"(referansın {dist_zero_to_normal/max(ref_pairwise_dist,1e-8):.2f}x'i)")
    log.info(f"  D(mean(z)) - D(z) ort. mesafe    : {dist_mean_to_normal:.5f}  "
              f"(referansın {dist_mean_to_normal/max(ref_pairwise_dist,1e-8):.2f}x'i)")
    log.info("  Latent duyarlılık ||D(z+eps)-D(z)||/||eps||:")
    for eps, val in sensitivity.items():
        log.info(f"    eps={eps:5.2f} -> {val:.5f}")
    log.info("-" * 90)

    shuffle_ratio = loss_shuffled / max(loss_correct, 1e-8)
    verdict_shuffle = "SAĞLIKLI" if shuffle_ratio > 1.3 else "ŞÜPHELİ (embedding'e duyarsız olabilir)"
    verdict_zero = (
        "SAĞLIKLI" if dist_zero_to_normal > 0.5 * ref_pairwise_dist
        else "ŞÜPHELİ (D(0) normale çok yakın)"
    )
    verdict_mean = (
        "SAĞLIKLI" if dist_mean_to_normal > 0.5 * ref_pairwise_dist
        else "ŞÜPHELİ (ortalamaya çökme belirtisi)"
    )
    sens_min = min(sensitivity.values())
    verdict_sens = "SAĞLIKLI" if sens_min > 0.01 else "ŞÜPHELİ (duyarlılık çökmüş)"

    log.info(f"  [Yorum] shuffle testi   : {verdict_shuffle}  (oran={shuffle_ratio:.2f}x, eşik>1.3x)")
    log.info(f"  [Yorum] zero-embedding  : {verdict_zero}")
    log.info(f"  [Yorum] mean-embedding  : {verdict_mean}")
    log.info(f"  [Yorum] latent duyarlılık: {verdict_sens}  (min={sens_min:.5f}, eşik>0.01)")
    log.info("=" * 90)

    # ── Görselleştirme ───────────────────────────────────────────────
    real_np = denormalize(target.cpu()).permute(0, 2, 3, 1).numpy()
    normal_np = denormalize(recon.cpu()).permute(0, 2, 3, 1).numpy()
    shuffled_np = denormalize(recon_shuffled.cpu()).permute(0, 2, 3, 1).numpy()
    zero_np = denormalize(recon_zero.cpu()).permute(0, 2, 3, 1).numpy()
    mean_np = denormalize(recon_mean.cpu()).permute(0, 2, 3, 1).numpy()

    rows = [
        ("Gerçek (target)", real_np), ("Normal D(z)", normal_np),
        ("Karıştırılmış D(z[perm])", shuffled_np),
        ("Sıfır D(0)", zero_np), ("Ortalama D(mean(z))", mean_np),
    ]
    fig, axes = plt.subplots(len(rows), B, figsize=(2.3 * B, 2.5 * len(rows)), squeeze=False)
    for r, (title, imgs) in enumerate(rows):
        for c in range(B):
            axes[r, c].imshow(imgs[c].clip(0, 1))
            axes[r, c].axis("off")
        axes[r, 0].text(-0.3, 0.5, title, fontsize=9, ha="right", va="center",
                         transform=axes[r, 0].transAxes)
    plt.suptitle(
        f"Conditioning testi — epoch={state.get('epoch')}  "
        f"shuffle_oran={shuffle_ratio:.2f}x  D0_mesafe={dist_zero_to_normal:.3f}",
        fontsize=10,
    )
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    log.info(f"[Conditioning] Görsel kaydedildi: {save_path}")

    return {
        "loss_correct": loss_correct, "loss_shuffled": loss_shuffled,
        "shuffle_ratio": shuffle_ratio, "ref_pairwise_dist": ref_pairwise_dist,
        "dist_zero_to_normal": dist_zero_to_normal, "dist_mean_to_normal": dist_mean_to_normal,
        "sensitivity": sensitivity,
        "verdicts": {
            "shuffle": verdict_shuffle, "zero": verdict_zero,
            "mean": verdict_mean, "sensitivity": verdict_sens,
        },
    }


if __name__ == "__main__":
    import sys
    from ..config import DEFAULT_CONFIG
    run_conditioning_check(DEFAULT_CONFIG, sys.argv[1], sys.argv[2])
