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
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

from .config import Config, DEFAULT_CONFIG
from .data.dataset import build_dataloaders, denormalize
from .models.decoder import FaceDecoder
from .models.losses import ReconstructionLoss
from .models.evaluator import IndependentEvaluator

log = logging.getLogger(__name__)

# GPU'yu maksimum verimle kullanmak icin: sabit input boyutlarinda
# cuDNN en hizli algoritmayi arayip cache'ler (ilk birkac step yavas,
# sonrasi hizlanir). image_size sabit oldugu icin bu guvenli.
torch.backends.cudnn.benchmark = True
# TF32 matmul/conv - T4 (Turing, sm_75) icin gecerli, hassasiyet kaybi
# ihmal edilebilir duzeyde, throughput'u belirgin artirir.
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


# ─── Transport Simulation (INT8 Roundtrip) ────────────────────────────────────

def transport_simulate(embeddings: torch.Tensor) -> torch.Tensor:
    """
    INT8 quantization-dequantization roundtrip'ini simüle eder (PDF v2.0 §6.3).

    Deployment sırasında embedding QR koddan uint8 olarak gelir ve
    float32'ye dönüştürülür. Bu dönüşüm, küçük bir quantization gürültüsü
    ekler. Eğitimde bu gürültüyü simüle ederek domain gap'i kapatırız.

    Yalnızca embedding giriş tensörüne uygulanır; decoder parametrelerine değil.
    Gradient geçirmez (detach sonra yeniden attach → Straight-Through Estimator).
    """
    with torch.no_grad():
        e_min = embeddings.min(dim=1, keepdim=True).values
        e_max = embeddings.max(dim=1, keepdim=True).values
        scale = (e_max - e_min).clamp(min=1e-8) / 255.0
        quantized   = torch.round((embeddings - e_min) / scale).clamp(0, 255)
        dequantized = quantized * scale + e_min

    # Straight-Through Estimator: gradient'i orijinal tensörden geçir
    return embeddings + (dequantized - embeddings).detach()


# ─── Validation ───────────────────────────────────────────────────────────────

@torch.no_grad()
def validate(model, loss_fn, val_loader, device, weights, writer, epoch):
    """
    NOT: checkpoint/"en iyi model" karari icin kullanilan identity skoru,
    egitim fazindan BAGIMSIZ sabit bir agirlikla (identity=1.0) ayrica
    hesaplaniyor. Aksi halde faz1'de weights['identity']=0 oldugu icin
    "1 - agirlikli_identity_loss" her zaman 1.0 cikar ve en iyi model
    secimi faz1 sonunda kalici olarak kilitlenir (faz2/3'te asilamaz).

    UYARI: loss_fn(...).identity'nin AGIRLIKLI mi HAM mi dondugu
    losses.py'ye bakilmadan %100 teyit edilemedi — losses.py paylasilinca
    bu fonksiyon gerekirse ince ayar yapilacak.
    """
    model.eval()
    total_losses: Dict[str, float] = {}
    identity_raw_sum = 0.0
    n_batches = 0

    eval_weights = dict(weights)
    eval_weights["identity"] = 1.0  # checkpoint karari icin sabit agirlik

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
        n_batches += 1

    avg = {k: v / n_batches for k, v in total_losses.items()}
    avg["identity_score"] = identity_raw_sum / n_batches  # dusuk = iyi (loss)

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
    # channels_last, conv-agirlikli modellerde Tensor Core kullanimini
    # artirip GPU throughput'unu yukseltir (T4/Turing destekliyor).
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    info = model.count_parameters()
    log.info(f"Model: {info['total_params']:,} parametre | {info['float32_mb']} MB (float32)")

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
    best_val_score = float("inf")  # identity RAW LOSS - dusuk = iyi (bkz. validate() notu)
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
            if device.type == "cuda":
                embeddings = embeddings.contiguous(memory_format=torch.channels_last) \
                    if embeddings.dim() == 4 else embeddings
                real_imgs  = real_imgs.contiguous(memory_format=torch.channels_last)

            # Transport simulation: INT8 roundtrip (PDF v2.0 §6.3)
            if cfg.train.transport_simulate:
                embeddings = transport_simulate(embeddings)

            optimizer.zero_grad(set_to_none=True)

            with autocast("cuda", enabled=cfg.train.use_amp):
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
        val_id_score = val_losses["identity_score"]  # RAW loss, dusuk = iyi

        log.info(
            f"[Val] epoch={epoch+1}  "
            f"total={val_losses['total']:.4f}  "
            f"identity_score(raw_loss, dusuk_iyi)={val_id_score:.4f}  "
            f"best={best_val_score:.4f}"
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
        # RAW identity LOSS kullanildigi icin kriter tersine dondu: dusuk = iyi.
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
                    # cfg dataclass'i ham haliyle pickle edilirse, modul
                    # yeniden yuklendiginde (git clone tekrar calisinca)
                    # sinif kimligi degisip PicklingError verir. dict'e
                    # cevirerek bu riski tamamen ortadan kaldiriyoruz.
                    "config": asdict(cfg) if is_dataclass(cfg) else cfg,
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

    # ── Nihai Test Değerlendirmesi (tek seferlik, hiç görülmemiş kimlikler) ──
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

    # Best model yolunu döndür
    return str(best_ckpt)


if __name__ == "__main__":
    train()