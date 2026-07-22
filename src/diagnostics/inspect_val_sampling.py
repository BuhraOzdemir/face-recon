"""
TensorBoard val/generated grid'inin SABİT-batch yanılsaması olup olmadığını
ve renk kaymasının gerçek/sistematik mi yoksa o sabit batch'e mi özgü
olduğunu ayırt eder.

validate()'in TB'ye yazdığı grid, val_loader'ın SON batch'i (shuffle=False
olduğu için her epoch'ta AYNI 8 örnek) — bu script bunun yerine val
dataset'inden GERÇEKTEN RASTGELE örnekler çeker (DataLoader'ın sabit
sırasını bypass eder), böylece "8 sabit örnek atipik miydi" sorusuna
cevap verir.

Kullanım:
    from src.diagnostics.inspect_val_sampling import run_random_val_check
    run_random_val_check(cfg, manifest_path, "/path/best_model.pt", n_samples=16)
"""

import logging
import random
from pathlib import Path

import torch
import matplotlib.pyplot as plt

from ..config import Config
from ..data.dataset import denormalize
from .common import build_model_from_checkpoint

log = logging.getLogger(__name__)


def _sample_random_val_items(val_loader, n: int, seed: int):
    """
    val_loader.dataset'ten DataLoader'ın sabit (shuffle=False) sırasını
    BYPASS ederek gerçekten rastgele n örnek çeker.
    """
    ds = val_loader.dataset
    rng = random.Random(seed)
    idx = rng.sample(range(len(ds)), min(n, len(ds)))
    pairs = [ds[i] for i in idx]
    embeddings = torch.stack([p[0] for p in pairs])
    images = torch.stack([p[1] for p in pairs])
    return embeddings, images, idx


def _channel_report(x: torch.Tensor, label: str):
    """x: [-1,1] tensor. Denormalize SONRASI [0,1] kanal ortalaması/std'si."""
    x01 = denormalize(x)
    means = x01.mean(dim=[0, 2, 3])
    stds = x01.std(dim=[0, 2, 3])
    log.info(
        f"  [{label:10s}] [0,1] mean(R,G,B)=({means[0]:.3f},{means[1]:.3f},{means[2]:.3f})  "
        f"std(R,G,B)=({stds[0]:.3f},{stds[1]:.3f},{stds[2]:.3f})"
    )
    return {"mean": means.tolist(), "std": stds.tolist()}


def run_random_val_check(
    cfg: Config,
    manifest_path: str,
    checkpoint_path: str,
    n_samples: int = 16,
    seed: int = 123,
    save_path: str = "/kaggle/working/random_val_check.png",
) -> dict:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, state = build_model_from_checkpoint(checkpoint_path, cfg, device)
    model.eval()
    log.info(f"[Random Val Check] Checkpoint: epoch={state.get('epoch')}")

    from ..data.dataset import build_dataloaders

    _, val_loader, _ = build_dataloaders(
        manifest_path=manifest_path, image_size=cfg.data.image_size,
        batch_size=cfg.train.batch_size, val_split=cfg.data.val_split,
        test_split=cfg.data.test_split, num_workers=0, max_samples=cfg.data.max_samples,
    )
    log.info(f"[Random Val Check] val dataset boyutu: {len(val_loader.dataset)}")

    embeddings, real_imgs, idx = _sample_random_val_items(val_loader, n_samples, seed)
    embeddings, real_imgs = embeddings.to(device), real_imgs.to(device)
    log.info(f"[Random Val Check] Seçilen rastgele indeksler: {idx}")

    with torch.no_grad():
        generated = model(embeddings)

    log.info("=" * 100)
    log.info("RENK KAYMASI KARŞILAŞTIRMASI ([0,1] denormalize sonrası)")
    real_stats = _channel_report(real_imgs.cpu(), "Gerçek")
    gen_stats = _channel_report(generated.cpu(), "Üretilen")
    diff = [g - r for g, r in zip(gen_stats["mean"], real_stats["mean"])]
    log.info(f"  [FARK] (üretilen-gerçek) mean(R,G,B) = ({diff[0]:+.3f},{diff[1]:+.3f},{diff[2]:+.3f})")
    channel_names = ["R", "G", "B"]
    biggest = max(range(3), key=lambda i: abs(diff[i]))
    log.info("-" * 100)
    if max(abs(d) for d in diff) > 0.08:
        log.info(
            f"SONUÇ: {channel_names[biggest]} kanalında GERÇEK/sistematik bir sapma var "
            f"(fark={diff[biggest]:+.3f}) — bu RASTGELE {n_samples} örnekte de sürüyorsa "
            f"sabit-batch yanılsaması DEĞİL, model gerçekten renk öğrenmede sorunlu."
        )
    else:
        log.info(
            "SONUÇ: Kanal farkları küçük (<0.08) — TensorBoard'daki 'camgöbeği/pembe' izlenimi "
            "büyük ihtimalle o SABİT 8 örneğe özgüydü, genel bir renk-öğrenme sorunu değil."
        )
    log.info("=" * 100)

    # ── Görselleştirme ───────────────────────────────────────────────
    real_np = denormalize(real_imgs.cpu()).permute(0, 2, 3, 1).numpy()
    gen_np = denormalize(generated.cpu()).permute(0, 2, 3, 1).numpy()
    ncols = 8
    nrows_per = (n_samples + ncols - 1) // ncols
    fig, axes = plt.subplots(2 * nrows_per, ncols, figsize=(2.2 * ncols, 2.2 * 2 * nrows_per), squeeze=False)
    for i in range(n_samples):
        r_block, c = divmod(i, ncols)
        axes[2 * r_block, c].imshow(real_np[i].clip(0, 1)); axes[2 * r_block, c].axis("off")
        axes[2 * r_block + 1, c].imshow(gen_np[i].clip(0, 1)); axes[2 * r_block + 1, c].axis("off")
    for r_block in range(nrows_per):
        axes[2 * r_block, 0].text(-0.3, 0.5, "Gerçek", fontsize=9, ha="right", va="center",
                                   transform=axes[2 * r_block, 0].transAxes)
        axes[2 * r_block + 1, 0].text(-0.3, 0.5, "Üretilen", fontsize=9, ha="right", va="center",
                                       transform=axes[2 * r_block + 1, 0].transAxes)
    plt.suptitle(
        f"RASTGELE {n_samples} val örneği — epoch={state.get('epoch')}  seed={seed}", fontsize=11,
    )
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    log.info(f"[Random Val Check] Görsel kaydedildi: {save_path}")

    return {"real_stats": real_stats, "gen_stats": gen_stats, "diff": diff, "indices": idx}


if __name__ == "__main__":
    import sys
    from ..config import DEFAULT_CONFIG
    run_random_val_check(DEFAULT_CONFIG, sys.argv[1], sys.argv[2])
