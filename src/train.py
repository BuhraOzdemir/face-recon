"""
Ana eğitim döngüsü.

Colab'da doğrudan import edilebilir veya bağımsız çalıştırılabilir.
"""

import os
import logging
import shutil
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

from .config import Config, DEFAULT_CONFIG
from .data.dataset import build_dataloaders, denormalize
from .models.decoder import FaceDecoder
from .models.losses import ReconstructionLoss

log = logging.getLogger(__name__)


# ─── Checkpoint yönetimi ──────────────────────────────────────────────────────

def save_checkpoint(
    state: dict,
    save_dir: str,
    epoch: int,
    is_best: bool,
    keep_last_n: int = 3,
):
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    ckpt_file = save_path / f"checkpoint_epoch{epoch:04d}.pt"
    torch.save(state, ckpt_file)

    if is_best:
        best_file = save_path / "best_model.pt"
        shutil.copyfile(ckpt_file, best_file)
        log.info(f"[Best] {best_file}")

    # Eski checkpoint'leri temizle
    ckpts = sorted(save_path.glob("checkpoint_epoch*.pt"))
    for old in ckpts[:-keep_last_n]:
        old.unlink()


def load_checkpoint(model: nn.Module, optimizer, scheduler, ckpt_path: str, device: torch.device):
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    if scheduler and "scheduler" in state:
        scheduler.load_state_dict(state["scheduler"])
    start_epoch     = state["epoch"] + 1
    best_val_score  = state.get("best_val_score", 0.0)
    log.info(f"Checkpoint yüklendi: epoch={state['epoch']}, best_score={best_val_score:.4f}")
    return start_epoch, best_val_score


# ─── Scheduler fabrikası ──────────────────────────────────────────────────────

def build_scheduler(optimizer, cfg: Config):
    warmup = LinearLR(
        optimizer,
        start_factor=0.01,
        end_factor=1.0,
        total_iters=cfg.train.warmup_epochs,
    )
    cosine = CosineAnnealingLR(
        optimizer,
        T_max=cfg.train.epochs - cfg.train.warmup_epochs,
        eta_min=cfg.train.eta_min,
    )
    return SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[cfg.train.warmup_epochs],
    )


# ─── Validation ───────────────────────────────────────────────────────────────

@torch.no_grad()
def validate(model, loss_fn, val_loader, device, weights, writer, epoch):
    model.eval()
    total_losses: Dict[str, float] = {}
    n_batches = 0

    for embeddings, real_imgs in val_loader:
        embeddings = embeddings.to(device)
        real_imgs  = real_imgs.to(device)

        generated = model(embeddings)
        losses    = loss_fn(generated, real_imgs, weights)

        for k, v in losses.items():
            total_losses[k] = total_losses.get(k, 0.0) + v.item()
        n_batches += 1

    avg = {k: v / n_batches for k, v in total_losses.items()}

    if writer:
        for k, v in avg.items():
            writer.add_scalar(f"val/{k}", v, epoch)

        # İlk batch'ten görselleştirme
        real_grid = make_grid(denormalize(real_imgs[:8]), nrow=4)
        gen_grid  = make_grid(denormalize(generated[:8]), nrow=4)
        writer.add_image("val/real",      real_grid, epoch)
        writer.add_image("val/generated", gen_grid,  epoch)

    model.train()
    return avg


# ─── Ana eğitim fonksiyonu ────────────────────────────────────────────────────

def train(
    cfg: Config = DEFAULT_CONFIG,
    manifest_path: Optional[str] = None,
    resume_from: Optional[str] = None,
):
    """
    Ana eğitim fonksiyonu.

    Args:
        cfg:           Config nesnesi. Varsayılan: DEFAULT_CONFIG.
        manifest_path: processed/manifest.txt yolu. Yoksa cfg.data.processed_dir kullanılır.
        resume_from:   Devam edilecek checkpoint dosyası.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Cihaz: {device}")

    # ── DataLoader ─────────────────────────────────────────────
    if manifest_path is None:
        manifest_path = str(Path(cfg.data.processed_dir) / "manifest.txt")

    train_loader, val_loader = build_dataloaders(
        manifest_path=manifest_path,
        image_size=cfg.data.image_size,
        batch_size=cfg.train.batch_size,
        val_split=cfg.data.val_split,
        num_workers=cfg.data.num_workers,
    )
    log.info(f"Train: {len(train_loader.dataset):,}  |  Val: {len(val_loader.dataset):,}")

    # ── Model ──────────────────────────────────────────────────
    model = FaceDecoder(
        embedding_dim=cfg.model.embedding_dim,
        initial_spatial=cfg.model.initial_spatial,
        initial_channels=cfg.model.initial_channels,
        decoder_channels=cfg.model.decoder_channels,
    ).to(device)

    info = model.count_parameters()
    log.info(f"Model: {info['total_params']:,} parametre | {info['float32_mb']} MB (float32)")

    # ── Loss ───────────────────────────────────────────────────
    loss_fn = ReconstructionLoss(
        vgg_layer         = cfg.loss.vgg_layer,
        facenet_input_size = cfg.loss.facenet_input_size,
        arcface_r50_path  = cfg.loss.arcface_r50_path,
    ).to(device)

    # ── Optimizer & Scheduler ──────────────────────────────────
    optimizer = AdamW(
        model.parameters(),
        lr=cfg.train.learning_rate,
        betas=cfg.train.adam_betas,
        eps=cfg.train.adam_eps,
        weight_decay=cfg.train.weight_decay,
    )
    scheduler = build_scheduler(optimizer, cfg)
    scaler    = GradScaler(enabled=cfg.train.use_amp)

    # ── Resume ─────────────────────────────────────────────────
    start_epoch    = 0
    best_val_score = 0.0   # Identity cosine similarity (yüksek = iyi)
    patience_ctr   = 0

    if resume_from and Path(resume_from).exists():
        start_epoch, best_val_score = load_checkpoint(
            model, optimizer, scheduler, resume_from, device
        )

    # ── TensorBoard ────────────────────────────────────────────
    writer = SummaryWriter(log_dir=cfg.train.log_dir)

    # ── Eğitim döngüsü ─────────────────────────────────────────
    log.info(f"Eğitim başlıyor: {cfg.train.epochs} epoch")

    for epoch in range(start_epoch, cfg.train.epochs):
        model.train()
        weights     = cfg.get_loss_weights(epoch)
        epoch_loss  = 0.0
        step_count  = 0

        for step, (embeddings, real_imgs) in enumerate(train_loader):
            embeddings = embeddings.to(device, non_blocking=True)
            real_imgs  = real_imgs.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=cfg.train.use_amp):
                generated = model(embeddings)
                losses    = loss_fn(generated, real_imgs, weights)

            scaler.scale(losses["total"]).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += losses["total"].item()
            step_count += 1

            global_step = epoch * len(train_loader) + step
            if step % cfg.train.log_every_steps == 0:
                lr = optimizer.param_groups[0]["lr"]
                log.info(
                    f"Epoch {epoch+1:3d}/{cfg.train.epochs} "
                    f"Step {step:5d}/{len(train_loader)} "
                    f"loss={losses['total'].item():.4f}  "
                    f"id={losses['identity'].item():.4f}  "
                    f"perc={losses['perceptual'].item():.4f}  "
                    f"lr={lr:.2e}"
                )
                for k, v in losses.items():
                    writer.add_scalar(f"train/{k}", v.item(), global_step)
                writer.add_scalar("train/lr", lr, global_step)

        scheduler.step()

        # ── Validation ─────────────────────────────────────────
        val_losses = validate(model, loss_fn, val_loader, device, weights, writer, epoch)
        val_id_score = 1.0 - val_losses.get("identity", 1.0)  # yüksek = iyi

        log.info(
            f"[Val] epoch={epoch+1}  "
            f"total={val_losses['total']:.4f}  "
            f"identity_similarity={val_id_score:.4f}  "
            f"best={best_val_score:.4f}"
        )

        # ── Checkpoint ─────────────────────────────────────────
        is_best = val_id_score > best_val_score + cfg.train.min_delta
        if is_best:
            best_val_score = val_id_score
            patience_ctr   = 0
        else:
            patience_ctr  += 1

        if (epoch + 1) % cfg.train.save_every_epochs == 0 or is_best:
            save_checkpoint(
                state={
                    "epoch":          epoch,
                    "model":          model.state_dict(),
                    "optimizer":      optimizer.state_dict(),
                    "scheduler":      scheduler.state_dict(),
                    "best_val_score": best_val_score,
                    "config":         cfg,
                },
                save_dir=cfg.train.save_dir,
                epoch=epoch + 1,
                is_best=is_best,
                keep_last_n=cfg.train.keep_last_n,
            )

        # ── Early stopping ─────────────────────────────────────
        if patience_ctr >= cfg.train.patience:
            log.info(
                f"Early stopping: {cfg.train.patience} epoch boyunca "
                f"val identity score artmadı. Son score: {best_val_score:.4f}"
            )
            break

    writer.close()
    log.info(f"Eğitim tamamlandı. En iyi val identity score: {best_val_score:.4f}")

    # Best model yolunu döndür
    return str(Path(cfg.train.save_dir) / "best_model.pt")


if __name__ == "__main__":
    train()
