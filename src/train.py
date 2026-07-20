"""
Ana eğitim döngüsü.

Colab'da doğrudan import edilebilir veya bağımsız çalıştırılabilir.
"""

import os
import logging
import shutil
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

from .config import Config, DEFAULT_CONFIG
from .data.dataset import build_dataloaders, denormalize
from .models.decoder import FaceDecoder
from .models.discriminator import PatchDiscriminator
from .models.losses import ReconstructionLoss
from .models.evaluator import IndependentEvaluator

log = logging.getLogger(__name__)

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


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

    ckpts = sorted(save_path.glob("checkpoint_epoch*.pt"))
    for old in ckpts[:-keep_last_n]:
        old.unlink()


def load_checkpoint(
    model: nn.Module,
    optimizer,
    scheduler,
    ckpt_path: str,
    device: torch.device,
    discriminator: Optional[nn.Module] = None,
    disc_optimizer=None,
):
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    if scheduler and "scheduler" in state:
        scheduler.load_state_dict(state["scheduler"])

    if discriminator is not None:
        if state.get("discriminator") is not None:
            discriminator.load_state_dict(state["discriminator"])
            log.info("[GAN] Discriminator checkpoint'ten yuklendi.")
        else:
            log.info("[GAN] Checkpoint'te discriminator yok - sifirdan baslatiliyor.")

    if disc_optimizer is not None and state.get("disc_optimizer") is not None:
        disc_optimizer.load_state_dict(state["disc_optimizer"])

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


# ─── GAN yardımcıları ──────────────────────────────────────────────────────────

def _current_adv_weight(cfg: Config, epoch: int) -> float:
    """GAN devreye girdikten sonra adv_weight'i lineer olarak 0'dan tavana çıkar."""
    start = cfg.train.gan_start_epoch
    ramp = max(1, cfg.train.adv_weight_ramp_epochs)
    if epoch < start:
        return 0.0
    progress = min(1.0, (epoch - start + 1) / ramp)
    return cfg.train.adv_weight * progress


# ─── Transport Simulation (INT8 Roundtrip) ────────────────────────────────────

def transport_simulate(embeddings: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        e_min = embeddings.min(dim=1, keepdim=True).values
        e_max = embeddings.max(dim=1, keepdim=True).values
        scale = (e_max - e_min).clamp(min=1e-8) / 255.0
        quantized   = torch.round((embeddings - e_min) / scale).clamp(0, 255)
        dequantized = quantized * scale + e_min

    return embeddings + (dequantized - embeddings).detach()


# ─── Validation ───────────────────────────────────────────────────────────────

def _compute_psnr(generated: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
    """generated, real: [-1,1] araliginda. PSNR dB cinsinden dondurur (yuksek=iyi)."""
    gen_01  = generated * 0.5 + 0.5
    real_01 = real * 0.5 + 0.5
    mse = torch.mean((gen_01 - real_01) ** 2, dim=[1, 2, 3])
    mse = mse.clamp(min=1e-10)  # log(0) onlemi
    psnr = 10 * torch.log10(1.0 / mse)
    return psnr.mean()


@torch.no_grad()
def validate(model, loss_fn, val_loader, device, weights, writer, epoch):
    """
    NOT: checkpoint karari icin kullanilan identity skoru, egitim
    fazindan BAGIMSIZ sabit bir agirlikla (identity=1.0) ayrica
    hesaplaniyor. Aksi halde faz1'de weights['identity']=0 oldugu icin
    identity loss hep 0 cikar ve en iyi model secimi faz1 sonunda
    kalici olarak kilitlenir.

    psnr_db ve ssim_raw SADECE raporlama icindir; toplam loss'a
    (backward'a) hicbir etkisi yoktur.
    """
    model.eval()
    total_losses: Dict[str, float] = {}
    identity_raw_sum = 0.0
    psnr_sum = 0.0
    n_batches = 0

    eval_weights = dict(weights)
    eval_weights["identity"] = 1.0

    generated = None
    real_imgs = None
    for embeddings, real_imgs in val_loader:
        embeddings = embeddings.to(device, non_blocking=True)
        real_imgs  = real_imgs.to(device, non_blocking=True)

        generated = model(embeddings)
        losses      = loss_fn(generated, real_imgs, weights)
        eval_losses = loss_fn(generated, real_imgs, eval_weights)

        for k, v in losses.items():
            total_losses[k] = total_losses.get(k, 0.0) + v.item()
        identity_raw_sum += eval_losses["identity"].item()
        psnr_sum += _compute_psnr(generated, real_imgs).item()
        n_batches += 1

    avg = {k: v / n_batches for k, v in total_losses.items()}
    avg["identity_score"] = identity_raw_sum / n_batches
    avg["psnr_db"] = psnr_sum / n_batches
    # ssim loss'u zaten "1 - SSIM" olarak tutuluyor; ham SSIM'i buradan turetiyoruz
    avg["ssim_raw"] = 1.0 - avg["ssim"]

    if writer:
        for k, v in avg.items():
            writer.add_scalar(f"val/{k}", v, epoch)

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
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Cihaz: {device}")
    if device.type == "cuda":
        log.info(f"GPU: {torch.cuda.get_device_name(0)}")

    # ── DataLoader ─────────────────────────────────────────────
    if manifest_path is None:
        manifest_path = str(Path(cfg.data.processed_dir) / "manifest.txt")

    train_loader, val_loader, test_loader = build_dataloaders(
        manifest_path=manifest_path,
        image_size=cfg.data.image_size,
        batch_size=cfg.train.batch_size,
        val_split=cfg.data.val_split,
        test_split=cfg.data.test_split,
        num_workers=cfg.data.num_workers,
    )
    log.info(
        f"Train: {len(train_loader.dataset):,}  |  "
        f"Val: {len(val_loader.dataset):,}  |  "
        f"Test: {len(test_loader.dataset):,} (eğitimde kullanılmaz, nihai değerlendirme için)"
    )

    # ── Model ──────────────────────────────────────────────────
    model = FaceDecoder(
        embedding_dim=cfg.model.embedding_dim,
        initial_spatial=cfg.model.initial_spatial,
        initial_channels=cfg.model.initial_channels,
        decoder_channels=cfg.model.decoder_channels,
    ).to(device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    info = model.count_parameters()
    log.info(f"Model: {info['total_params']:,} parametre | {info['float32_mb']} MB (float32)")

    # ── Discriminator (SADECE egitimde; export'a dahil edilmez) ──
    discriminator = None
    disc_optimizer = None
    disc_scaler = None
    if cfg.train.use_gan:
        discriminator = PatchDiscriminator().to(device)
        disc_optimizer = AdamW(
            discriminator.parameters(),
            lr=cfg.train.disc_lr,
            betas=cfg.train.adam_betas,
        )
        disc_scaler = GradScaler("cuda", enabled=cfg.train.use_amp)
        d_info = discriminator.count_parameters()
        log.info(
            f"[GAN] Discriminator: {d_info['total_params']:,} parametre "
            f"({d_info['float32_mb']} MB) — SADECE egitimde kullanilir, export edilmez."
        )
        log.info(
            f"[GAN] gan_start_epoch={cfg.train.gan_start_epoch}, "
            f"adv_weight={cfg.train.adv_weight}, ramp={cfg.train.adv_weight_ramp_epochs} epoch"
        )

    # ── Loss ───────────────────────────────────────────────────
    loss_fn = ReconstructionLoss(
        vgg_layer         = cfg.loss.vgg_layer,
        facenet_input_size = cfg.loss.facenet_input_size,
        arcface_r50_path  = cfg.loss.arcface_r50_path,
    ).to(device)

    # ── Bağımsız Evaluator (PDF v2.0 §7) ───────────────────────
    evaluator = None
    if cfg.train.use_independent_evaluator:
        try:
            evaluator = IndependentEvaluator().to(device)
            log.info("[Evaluator] FaceNet bağımsız değerlendirici yüklendi.")
        except Exception as exc:
            log.warning(f"[Evaluator] Yüklenemedi: {exc} — atlanıyor.")

    # ── Optimizer & Scheduler ──────────────────────────────────
    optimizer = AdamW(
        model.parameters(),
        lr=cfg.train.learning_rate,
        betas=cfg.train.adam_betas,
        eps=cfg.train.adam_eps,
        weight_decay=cfg.train.weight_decay,
    )
    scheduler = build_scheduler(optimizer, cfg)
    scaler    = GradScaler("cuda", enabled=cfg.train.use_amp)

    # ── Resume ─────────────────────────────────────────────────
    start_epoch    = 0
    best_val_score = float("inf")  # identity RAW LOSS - dusuk = iyi
    patience_ctr   = 0

    if resume_from and Path(resume_from).exists():
        start_epoch, best_val_score = load_checkpoint(
            model, optimizer, scheduler, resume_from, device,
            discriminator=discriminator, disc_optimizer=disc_optimizer,
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

        use_gan_this_epoch = (discriminator is not None) and (epoch >= cfg.train.gan_start_epoch)
        adv_w = _current_adv_weight(cfg, epoch) if use_gan_this_epoch else 0.0

        for step, (embeddings, real_imgs) in enumerate(train_loader):
            embeddings = embeddings.to(device, non_blocking=True)
            real_imgs  = real_imgs.to(device, non_blocking=True)
            if device.type == "cuda":
                real_imgs = real_imgs.contiguous(memory_format=torch.channels_last)

            if cfg.train.transport_simulate:
                embeddings = transport_simulate(embeddings)

            # ---- Generator adimi ----
            optimizer.zero_grad(set_to_none=True)
            with autocast("cuda", enabled=cfg.train.use_amp):
                generated = model(embeddings)
                losses    = loss_fn(generated, real_imgs, weights)
                g_total   = losses["total"]

                if use_gan_this_epoch and adv_w > 0:
                    fake_logits = discriminator(generated)
                    adv_loss_g = F.binary_cross_entropy_with_logits(
                        fake_logits, torch.ones_like(fake_logits)
                    )
                    g_total = g_total + adv_w * adv_loss_g
                else:
                    adv_loss_g = torch.zeros((), device=device)

            scaler.scale(g_total).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            # ---- Discriminator adimi ----
            if use_gan_this_epoch:
                disc_optimizer.zero_grad(set_to_none=True)
                with autocast("cuda", enabled=cfg.train.use_amp):
                    real_logits = discriminator(real_imgs)
                    fake_logits_d = discriminator(generated.detach())
                    d_loss_real = F.binary_cross_entropy_with_logits(
                        real_logits, torch.full_like(real_logits, 0.9)  # label smoothing
                    )
                    d_loss_fake = F.binary_cross_entropy_with_logits(
                        fake_logits_d, torch.zeros_like(fake_logits_d)
                    )
                    d_loss = 0.5 * (d_loss_real + d_loss_fake)
                disc_scaler.scale(d_loss).backward()
                disc_scaler.step(disc_optimizer)
                disc_scaler.update()
            else:
                d_loss = torch.zeros((), device=device)

            epoch_loss += g_total.item()
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
                    f"adv_g={adv_loss_g.item():.4f}  "
                    f"d_loss={d_loss.item():.4f}  "
                    f"lr={lr:.2e}"
                )
                for k, v in losses.items():
                    writer.add_scalar(f"train/{k}", v.item(), global_step)
                if use_gan_this_epoch:
                    writer.add_scalar("train/adv_g", adv_loss_g.item(), global_step)
                    writer.add_scalar("train/d_loss", d_loss.item(), global_step)
                    writer.add_scalar("train/adv_weight", adv_w, global_step)
                writer.add_scalar("train/lr", lr, global_step)

        scheduler.step()

        # ── Validation ─────────────────────────────────────────
        val_losses = validate(model, loss_fn, val_loader, device, weights, writer, epoch)
        val_id_score = val_losses["identity_score"]

        log.info(
            f"[Val] epoch={epoch+1}  "
            f"total={val_losses['total']:.4f}  "
            f"identity_score(raw_loss, dusuk_iyi)={val_id_score:.4f}  "
            f"PSNR={val_losses['psnr_db']:.2f}dB  "
            f"SSIM={val_losses['ssim_raw']:.4f}  "
            f"best={best_val_score:.4f}  "
            f"gan_aktif={use_gan_this_epoch}"
        )

        # ── Bağımsız Evaluator (PDF v2.0 §7) ───────────────────
        if (evaluator is not None
                and (epoch + 1) % cfg.train.eval_every_epochs == 0):
            model.eval()
            indep_scores = []
            with torch.no_grad():
                for emb_b, real_b in val_loader:
                    emb_b  = emb_b.to(device)
                    real_b = real_b.to(device)
                    gen_b  = model(emb_b)
                    score  = evaluator.cosine_sim(gen_b, real_b)
                    indep_scores.append(score.item())
            indep_mean = sum(indep_scores) / len(indep_scores)
            log.info(f"[Bağımsız Eval] epoch={epoch+1}  FaceNet cosine_sim={indep_mean:.4f}")
            if writer:
                writer.add_scalar("eval/independent_cosine_sim", indep_mean, epoch)
            model.train()

        # ── Checkpoint ─────────────────────────────────────────
        is_best = val_id_score < best_val_score - cfg.train.min_delta
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
                    "config":         asdict(cfg) if is_dataclass(cfg) else cfg,
                    "discriminator":  discriminator.state_dict() if discriminator is not None else None,
                    "disc_optimizer": disc_optimizer.state_dict() if disc_optimizer is not None else None,
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
                f"val identity score iyilesmedi. Son score: {best_val_score:.4f}"
            )
            break

    # ── Nihai Test Değerlendirmesi ──
    best_ckpt = Path(cfg.train.save_dir) / "best_model.pt"
    if best_ckpt.exists():
        state = torch.load(best_ckpt, map_location=device)
        model.load_state_dict(state["model"])
    model.eval()

    test_id_scores = []
    with torch.no_grad():
        for emb_b, real_b in test_loader:
            emb_b  = emb_b.to(device)
            real_b = real_b.to(device)
            gen_b  = model(emb_b)
            losses = loss_fn(gen_b, real_b, cfg.loss.phase3_weights)
            test_id_scores.append(1.0 - losses["identity"].item())
            if evaluator is not None:
                indep_score = evaluator.cosine_sim(gen_b, real_b)
                writer.add_scalar("test/independent_cosine_sim", indep_score.item())

    if test_id_scores:
        test_mean = sum(test_id_scores) / len(test_id_scores)
        log.info(
            f"[NİHAİ TEST] {len(test_loader.dataset):,} görülmemiş görüntü — "
            f"identity_similarity={test_mean:.4f}"
        )
        writer.add_scalar("test/identity_similarity", test_mean, 0)

    writer.close()
    log.info(f"Eğitim tamamlandı. En iyi val identity score (raw loss): {best_val_score:.4f}")

    return str(best_ckpt)


if __name__ == "__main__":
    train()