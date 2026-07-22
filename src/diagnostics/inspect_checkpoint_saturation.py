"""
Checkpoint(ler) arası Tanh-doygunluk karşılaştırması.

AMAÇ: "epoch 2'den itibaren decoder çıktısı Tanh'ın ±1 uçlarına kilitleniyor
mu" sorusunu, GERÇEK checkpoint'ler + gerçek val batch ile ölçmek. 1-2
checkpoint verilebilir (2 verilirse yan yana karşılaştırır).

NOT: save_every_epochs=5 varsayılanıyla kısa (ör. 5 epoch) bir run'da SADECE
en iyi (is_best) ve son epoch checkpoint'i kaydedilir — ara epoch'lar
(2,3,4) için ayrı checkpoint YOKTUR. Bu script elinizdeki checkpoint'lerle
çalışır; ara epoch'ları da görmek isterseniz bir sonraki run'da
save_every_epochs=1 yapın.

Kullanım:
    from src.diagnostics.inspect_checkpoint_saturation import run_saturation_check
    run_saturation_check(cfg, manifest_path, ["/path/checkpoint_epoch0001.pt",
                                               "/path/checkpoint_epoch0005.pt"])
"""

import logging
from pathlib import Path
from typing import List

import torch
import matplotlib.pyplot as plt

from ..config import Config
from ..data.dataset import denormalize
from .common import build_model_from_checkpoint, sample_val_batch

log = logging.getLogger(__name__)

SAT_THRESHOLD = 0.95


def _saturation_stats(x: torch.Tensor) -> dict:
    """x: (B,3,H,W) [-1,1]. |x|>0.95 oranı + kanal başı istatistik."""
    sat_frac = (x.abs() > SAT_THRESHOLD).float().mean().item()
    sat_pos = (x > SAT_THRESHOLD).float().mean().item()
    sat_neg = (x < -SAT_THRESHOLD).float().mean().item()
    means = x.mean(dim=[0, 2, 3])
    stds = x.std(dim=[0, 2, 3])
    return {
        "sat_frac": sat_frac, "sat_pos_frac": sat_pos, "sat_neg_frac": sat_neg,
        "mean_R": means[0].item(), "mean_G": means[1].item(), "mean_B": means[2].item(),
        "std_R": stds[0].item(), "std_G": stds[1].item(), "std_B": stds[2].item(),
    }


def run_saturation_check(
    cfg: Config,
    manifest_path: str,
    checkpoint_paths: List[str],
    n_samples: int = 8,
    save_path: str = "/kaggle/working/saturation_check.png",
) -> dict:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    embeddings, real_imgs = None, None
    results = []
    outputs = []

    for ckpt_path in checkpoint_paths:
        model, state = build_model_from_checkpoint(ckpt_path, cfg, device)
        model.eval()
        if embeddings is None:
            embeddings, real_imgs = sample_val_batch(manifest_path, cfg, n_samples=n_samples, device=device)
        with torch.no_grad():
            out = model(embeddings)
        stats = _saturation_stats(out)
        stats["epoch"] = state.get("epoch")
        stats["checkpoint"] = ckpt_path
        results.append(stats)
        outputs.append(out)

    log.info("=" * 100)
    log.info("INSPECT_CHECKPOINT_SATURATION")
    log.info(f"{'Checkpoint (epoch)':<28}{'|x|>0.95':>12}{'+1 ucu':>10}{'-1 ucu':>10}"
              f"{'mean(R,G,B)':>26}{'std(R,G,B)':>26}")
    for r in results:
        log.info(
            f"epoch={r['epoch']!s:<21}{r['sat_frac']:>12.4f}{r['sat_pos_frac']:>10.4f}"
            f"{r['sat_neg_frac']:>10.4f}"
            f"  ({r['mean_R']:+.2f},{r['mean_G']:+.2f},{r['mean_B']:+.2f})      "
            f"  ({r['std_R']:.2f},{r['std_G']:.2f},{r['std_B']:.2f})"
        )
    log.info("-" * 100)

    if len(results) >= 2:
        d_sat = results[-1]["sat_frac"] - results[0]["sat_frac"]
        log.info(f"[Fark] son - ilk doygunluk oranı = {d_sat:+.4f}")
        if d_sat > 0.15:
            log.info(
                "SONUÇ: Doygunluk ANLAMLI şekilde ARTMIŞ — çıktı Tanh'ın uçlarına "
                "kilitlenmiş olabilir (raporda tartışılan LR/phase1 çakışması hipoteziyle tutarlı)."
            )
        else:
            log.info("SONUÇ: Doygunlukta büyük bir artış görülmüyor.")
    log.info("=" * 100)

    # ── Görselleştirme ───────────────────────────────────────────────
    rows = [("Gerçek", denormalize(real_imgs.cpu()).permute(0, 2, 3, 1).numpy())]
    for r, out in zip(results, outputs):
        rows.append((f"epoch={r['epoch']}", denormalize(out.cpu()).permute(0, 2, 3, 1).numpy()))

    B = embeddings.size(0)
    fig, axes = plt.subplots(len(rows), B, figsize=(2.3 * B, 2.5 * len(rows)), squeeze=False)
    for r, (title, imgs) in enumerate(rows):
        for c in range(B):
            axes[r, c].imshow(imgs[c].clip(0, 1))
            axes[r, c].axis("off")
        axes[r, 0].text(-0.3, 0.5, title, fontsize=9, ha="right", va="center",
                         transform=axes[r, 0].transAxes)
    plt.suptitle("Checkpoint Doygunluk Karşılaştırması", fontsize=11)
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    log.info(f"Görsel kaydedildi: {save_path}")

    return {"results": results}


if __name__ == "__main__":
    import sys
    from ..config import DEFAULT_CONFIG
    run_saturation_check(DEFAULT_CONFIG, sys.argv[1], sys.argv[2:])
