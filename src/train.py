"""
Ana eğitim döngüsü.

Colab'da doğrudan import edilebilir veya bağımsız çalıştırılabilir.
Eğitim: L1 + perceptual + identity + SSIM (+ opsiyonel LPIPS/cycle/diversity).
"""

import copy
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
    ckpt_path: str,
    device: torch.device,
):
    """
    NOT: scheduler KASITLI olarak yuklenmiyor. Checkpoint'teki scheduler
    state'i farkli bir cfg.train.epochs (farkli T_max) ile hesaplanmis
    olabilir; bunu yeni bir scheduler'a yuklemek LR'nin beklenmedik
    sekilde artmasina/tutarsiz olmasina yol aciyordu. Bunun yerine
    train() icinde scheduler, start_epoch kadar .step() ile "ileri
    sarilarak" her zaman GUNCEL cfg.train.epochs'a gore konumlandiriliyor.

    Eski GAN'li checkpoint'lerdeki discriminator/disc_optimizer anahtarlari
    sessizce yoksayilir (model agirliklari yuklenir).
    """
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])

    start_epoch    = state["epoch"] + 1
    best_val_score = state.get("best_val_score", 0.0)
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


def _augment_weights(cfg: Config, base_weights: dict) -> dict:
    """Opsiyonel loss terimlerini (lpips, cycle_identity) faz agirliklarina enjekte eder."""
    weights = dict(base_weights)
    if cfg.loss.use_lpips:
        weights.setdefault("lpips", cfg.loss.lpips_weight)
    if cfg.loss.cycle_identity_weight > 0:
        weights.setdefault("cycle_identity", cfg.loss.cycle_identity_weight)
    return weights


def _diversity_loss(
    generated_a: torch.Tensor, generated_b: torch.Tensor,
    noise_a: torch.Tensor, noise_b: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """
    Mode-seeking regularization (noise injection icin):
        loss = -||G(e,z1)-G(e,z2)||_1 / ||z1-z2||_1
    """
    img_dist = (generated_a - generated_b).abs().mean(dim=[1, 2, 3])
    z_dist = (noise_a - noise_b).abs().mean(dim=1)
    return -(img_dist / (z_dist + eps)).mean()


class ModelEMA:
    """Decoder agirliklarinin EMA golge kopyasi — daha stabil/keskin sonuc."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module):
        for ema_p, p in zip(self.shadow.parameters(), model.parameters()):
            ema_p.mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)
        for ema_b, b in zip(self.shadow.buffers(), model.buffers()):
            ema_b.copy_(b)

    def state_dict(self):
        return self.shadow.state_dict()

    def load_state_dict(self, state_dict):
        self.shadow.load_state_dict(state_dict)


def _recalibrate_bn(model: nn.Module, loader, device: torch.device, max_batches: int = 100):
    """
    EMA gölge modelinin BatchNorm running_mean/var'ını KENDİ (yumuşatılmış)
    ağırlıklarına göre yeniden hesaplar — torch.optim.swa_utils.update_bn
    ile aynı yöntem: running stats sıfırlanır, momentum=None (kümülatif
    hareketli ortalama) yapılır, train() modunda gradyansız birkaç batch
    ileri geçirilir, sonra momentum eski haline döner.

    NEDEN: ModelEMA.update() buffer'ları (running_mean/var) CANLI modelden
    doğrudan kopyalar — bu istatistikler CANLI ağırlıkların ürettiği
    aktivasyon dağılımını yansıtır, ama golge modelin YUMUŞATILMIŞ
    (decay≈0.999 → ~1000 adımlık gecikme) ağırlıklarına uygulanır. Bu
    uyumsuzluk düşük-frekans (renk) sinyalini az etkiler ama yüksek-frekans
    (doku) detayında gözle görülür bozulmaya yol açar (bkz. proje notları).

    Sadece nn.BatchNorm2d etkilenir — norm_type="instance" ile no-op'tur
    (InstanceNorm2d'de running stats yok).
    """
    bn_modules = [m for m in model.modules() if isinstance(m, nn.BatchNorm2d)]
    if not bn_modules or max_batches <= 0:
        return

    original_momenta = {m: m.momentum for m in bn_modules}
    for m in bn_modules:
        m.reset_running_stats()
        m.momentum = None  # kümülatif hareketli ortalama

    was_training = model.training
    model.train()
    with torch.no_grad():
        for i, (embeddings, _real_imgs) in enumerate(loader):
            if i >= max_batches:
                break
            embeddings = embeddings.to(device, non_blocking=True)
            model(embeddings)

    for m in bn_modules:
        m.momentum = original_momenta[m]
    model.train(was_training)


def _selection_score(identity_score: float, sharpness_ratio: float, weight: float) -> float:
    """Best checkpoint skoru (düşük = iyi)."""
    sharp_pen = 1.0 - min(float(sharpness_ratio), 1.0)
    return float(identity_score) + weight * sharp_pen


# ─── Transport Simulation (INT8 Roundtrip) ────────────────────────────────────

def transport_simulate(embeddings: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        e_min = embeddings.min(dim=1, keepdim=True).values
        e_max = embeddings.max(dim=1, keepdim=True).values
        scale = (e_max - e_min).clamp(min=1e-8) / 255.0
        quantized   = torch.round((embeddings - e_min) / scale).clamp(0, 255)
        dequantized = quantized * scale + e_min

    return embeddings + (dequantized - embeddings).detach()


# ─── Ek Değerlendirme Metrikleri ─────────────────────────────────────────────

def _compute_psnr(generated: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
    gen_01  = generated * 0.5 + 0.5
    real_01 = real * 0.5 + 0.5
    mse = torch.mean((gen_01 - real_01) ** 2, dim=[1, 2, 3]).clamp(min=1e-10)
    return (10 * torch.log10(1.0 / mse)).mean()


_LAPLACIAN_KERNEL = torch.tensor(
    [[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]], dtype=torch.float32
).view(1, 1, 3, 3)


def _laplacian_variance(img: torch.Tensor) -> torch.Tensor:
    kernel = _LAPLACIAN_KERNEL.to(img.device)
    gray = img.mean(dim=1, keepdim=True)
    lap = F.conv2d(gray, kernel, padding=1)
    return lap.var(dim=[1, 2, 3]).mean()


@torch.no_grad()
def _batch_identity_retrieval_accuracy(
    evaluator: "IndependentEvaluator",
    generated: torch.Tensor,
    real: torch.Tensor,
) -> float:
    gen_emb = evaluator.encode(generated)
    real_emb = evaluator.encode(real)
    sim_matrix = gen_emb @ real_emb.T
    predicted = sim_matrix.argmax(dim=1)
    correct = torch.arange(generated.size(0), device=generated.device)
    return (predicted == correct).float().mean().item()


# ─── Validation ───────────────────────────────────────────────────────────────

@torch.no_grad()
def validate(model, loss_fn, val_loader, device, weights, writer, epoch, evaluator=None):
    model.eval()
    total_losses: Dict[str, float] = {}
    identity_raw_sum = 0.0
    psnr_sum = 0.0
    sharpness_gen_sum = 0.0
    sharpness_real_sum = 0.0
    retrieval_acc_sum = 0.0
    n_retrieval_batches = 0
    n_batches = 0

    eval_weights = dict(weights)
    eval_weights["identity"] = 1.0

    generated = None
    real_imgs = None
    for embeddings, real_imgs in val_loader:
        embeddings = embeddings.to(device, non_blocking=True)
        real_imgs  = real_imgs.to(device, non_blocking=True)

        generated = model(embeddings)
        losses      = loss_fn(generated, real_imgs, weights, input_embedding=embeddings)
        eval_losses = loss_fn(generated, real_imgs, eval_weights, input_embedding=embeddings)

        for k, v in losses.items():
            total_losses[k] = total_losses.get(k, 0.0) + v.item()
        identity_raw_sum += eval_losses["identity"].item()
        psnr_sum += _compute_psnr(generated, real_imgs).item()
        sharpness_gen_sum  += _laplacian_variance(generated).item()
        sharpness_real_sum += _laplacian_variance(real_imgs).item()

        if evaluator is not None and generated.size(0) > 1:
            retrieval_acc_sum += _batch_identity_retrieval_accuracy(evaluator, generated, real_imgs)
            n_retrieval_batches += 1

        n_batches += 1

    avg = {k: v / n_batches for k, v in total_losses.items()}
    avg["identity_score"] = identity_raw_sum / n_batches
    avg["psnr_db"] = psnr_sum / n_batches
    avg["ssim_raw"] = 1.0 - avg["ssim"]
    avg["sharpness_gen"] = sharpness_gen_sum / n_batches
    avg["sharpness_real"] = sharpness_real_sum / n_batches
    avg["sharpness_ratio"] = avg["sharpness_gen"] / max(avg["sharpness_real"], 1e-8)
    avg["identity_retrieval_acc"] = (
        retrieval_acc_sum / n_retrieval_batches if n_retrieval_batches > 0 else 0.0
    )

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

    if manifest_path is None:
        manifest_path = str(Path(cfg.data.processed_dir) / "manifest.txt")

    train_loader, val_loader, test_loader = build_dataloaders(
        manifest_path=manifest_path,
        image_size=cfg.data.image_size,
        batch_size=cfg.train.batch_size,
        val_split=cfg.data.val_split,
        test_split=cfg.data.test_split,
        num_workers=cfg.data.num_workers,
        max_samples=cfg.data.max_samples,
    )
    log.info(
        f"Train: {len(train_loader.dataset):,}  |  "
        f"Val: {len(val_loader.dataset):,}  |  "
        f"Test: {len(test_loader.dataset):,} (eğitimde kullanılmaz, nihai değerlendirme için)"
    )

    model = FaceDecoder(
        embedding_dim=cfg.model.embedding_dim,
        initial_spatial=cfg.model.initial_spatial,
        initial_channels=cfg.model.initial_channels,
        decoder_channels=cfg.model.decoder_channels,
        norm_type=cfg.model.norm_type,
        use_noise_injection=cfg.model.use_noise_injection,
        noise_dim=cfg.model.noise_dim,
        use_cascade_skip=cfg.model.use_cascade_skip,
        cascade_skip_last_n_blocks=cfg.model.cascade_skip_last_n_blocks,
    ).to(device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    info = model.count_parameters()
    log.info(f"Model: {info['total_params']:,} parametre | {info['float32_mb']} MB (float32)")
    log.info("Eğitim: L1 + perceptual + identity + SSIM")
    if cfg.model.use_cascade_skip:
        log.info(f"[CascadeSkip] aktif: last_n_blocks={cfg.model.cascade_skip_last_n_blocks}")
    if cfg.model.use_noise_injection:
        log.info(
            f"[Noise] aktif: noise_dim={cfg.model.noise_dim}, "
            f"diversity_weight={cfg.loss.diversity_weight}"
        )
    if cfg.loss.cycle_identity_weight > 0:
        log.info(f"[CycleID] cycle_identity_weight={cfg.loss.cycle_identity_weight}")

    loss_fn = ReconstructionLoss(
        vgg_layer          = cfg.loss.vgg_layer,
        facenet_input_size = cfg.loss.facenet_input_size,
        arcface_r50_path   = cfg.loss.arcface_r50_path,
        use_lpips          = cfg.loss.use_lpips,
        lpips_net          = cfg.loss.lpips_net,
    ).to(device)

    evaluator = None
    if cfg.train.use_independent_evaluator:
        try:
            evaluator = IndependentEvaluator().to(device)
            log.info("[Evaluator] FaceNet bağımsız değerlendirici yüklendi.")
        except Exception as exc:
            log.warning(f"[Evaluator] Yüklenemedi: {exc} — atlanıyor.")

    optimizer = AdamW(
        model.parameters(),
        lr=cfg.train.learning_rate,
        betas=cfg.train.adam_betas,
        eps=cfg.train.adam_eps,
        weight_decay=cfg.train.weight_decay,
    )
    scheduler = build_scheduler(optimizer, cfg)
    scaler    = GradScaler("cuda", enabled=cfg.train.use_amp)

    start_epoch    = 0
    best_val_score = float("inf")
    patience_ctr   = 0

    if resume_from and Path(resume_from).exists():
        start_epoch, best_val_score = load_checkpoint(
            model, optimizer, resume_from, device,
        )
        for _ in range(start_epoch):
            scheduler.step()
        log.info(
            f"[Scheduler] {start_epoch} epoch ileri sarildi, "
            f"lr={optimizer.param_groups[0]['lr']:.2e}"
        )

    ema = None
    if cfg.train.use_ema:
        ema = ModelEMA(model, decay=cfg.train.ema_decay)
        if resume_from and Path(resume_from).exists():
            ckpt_state = torch.load(resume_from, map_location=device, weights_only=False)
            if ckpt_state.get("model_ema") is not None:
                ema.load_state_dict(ckpt_state["model_ema"])
                log.info("[EMA] EMA golge agirliklari checkpoint'ten yuklendi.")
            else:
                log.info("[EMA] Checkpoint'te EMA yok - guncel model agirliklarindan baslatiliyor.")
            del ckpt_state
        log.info(
            f"[EMA] Aktif: decay={cfg.train.ema_decay}, eval_use_ema={cfg.train.eval_use_ema}, "
            f"bn_recalib_batches={cfg.train.ema_bn_recalib_batches}"
        )

    writer = SummaryWriter(log_dir=cfg.train.log_dir)
    log.info(f"Eğitim başlıyor: {cfg.train.epochs} epoch")

    for epoch in range(start_epoch, cfg.train.epochs):
        model.train()
        weights = _augment_weights(cfg, cfg.get_loss_weights(epoch))

        for step, (embeddings, real_imgs) in enumerate(train_loader):
            embeddings = embeddings.to(device, non_blocking=True)
            real_imgs  = real_imgs.to(device, non_blocking=True)
            if device.type == "cuda":
                real_imgs = real_imgs.contiguous(memory_format=torch.channels_last)

            if (
                cfg.train.transport_simulate
                and epoch >= cfg.train.transport_simulate_start_epoch
            ):
                embeddings = transport_simulate(embeddings)

            use_diversity = cfg.model.use_noise_injection and cfg.loss.diversity_weight > 0

            optimizer.zero_grad(set_to_none=True)
            with autocast("cuda", enabled=cfg.train.use_amp):
                noise1 = None
                if cfg.model.use_noise_injection:
                    noise1 = torch.randn(embeddings.size(0), model.noise_dim, device=device)
                generated = model(embeddings, noise=noise1)
                losses    = loss_fn(generated, real_imgs, weights, input_embedding=embeddings)
                g_total   = losses["total"]

                if use_diversity:
                    noise2 = torch.randn(embeddings.size(0), model.noise_dim, device=device)
                    generated_b = model(embeddings, noise=noise2)
                    div_loss = _diversity_loss(generated, generated_b, noise1, noise2)
                    g_total = g_total + cfg.loss.diversity_weight * div_loss
                else:
                    div_loss = torch.zeros((), device=device)

            scaler.scale(g_total).backward()
            scaler.unscale_(optimizer)
            # clip_grad_norm_ CLIP'TEN ONCEKI toplam normu dondurur - bu deger
            # zaten hesaplaniyordu ama loglanmiyordu. Buyuk sicramalari (patlama)
            # tespit etmek icin loglaniyor (bkz. proje notlari - epoch gecisi
            # instabilite teshisi).
            raw_grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            if ema is not None:
                ema.update(model)

            global_step = epoch * len(train_loader) + step
            # grad_norm HER adimda (log_every_steps'ten bagimsiz) TB'ye yaziliyor
            # ki kisa surели bir sicrama (patlama) log-sampling'de kacirilmasin.
            writer.add_scalar("train/grad_norm_preclip", float(raw_grad_norm), global_step)

            if step % cfg.train.log_every_steps == 0:
                lr = optimizer.param_groups[0]["lr"]
                lpips_str = f"lpips={losses['lpips'].item():.4f}  " if cfg.loss.use_lpips else ""
                cycle_str = (
                    f"cyc_id={losses['cycle_identity'].item():.4f}  "
                    if cfg.loss.cycle_identity_weight > 0 else ""
                )
                div_str = f"div={div_loss.item():.4f}  " if use_diversity else ""
                log.info(
                    f"Epoch {epoch+1:3d}/{cfg.train.epochs} "
                    f"Step {step:5d}/{len(train_loader)} "
                    f"loss={losses['total'].item():.4f}  "
                    f"id={losses['identity'].item():.4f}  "
                    f"perc={losses['perceptual'].item():.4f}  "
                    f"{lpips_str}{cycle_str}{div_str}"
                    f"grad_norm={float(raw_grad_norm):.3f}  "
                    f"lr={lr:.2e}"
                )
                for k, v in losses.items():
                    writer.add_scalar(f"train/{k}", v.item(), global_step)
                if use_diversity:
                    writer.add_scalar("train/diversity", div_loss.item(), global_step)
                writer.add_scalar("train/lr", lr, global_step)

        scheduler.step()

        use_ema_for_eval = ema is not None and cfg.train.eval_use_ema
        eval_model = ema.shadow if use_ema_for_eval else model
        if use_ema_for_eval and cfg.train.ema_bn_recalib_batches > 0:
            _recalibrate_bn(
                eval_model, train_loader, device,
                max_batches=cfg.train.ema_bn_recalib_batches,
            )
        val_losses = validate(
            eval_model, loss_fn, val_loader, device, weights, writer, epoch,
            evaluator=evaluator,
        )
        val_id_score = val_losses["identity_score"]
        val_selection = _selection_score(
            val_id_score,
            val_losses["sharpness_ratio"],
            cfg.train.sharpness_ckpt_weight,
        )

        log.info(
            f"[Val] epoch={epoch+1}  "
            f"total={val_losses['total']:.4f}  "
            f"identity_score(raw_loss, dusuk_iyi)={val_id_score:.4f}  "
            f"selection={val_selection:.4f}  "
            f"PSNR={val_losses['psnr_db']:.2f}dB  "
            f"SSIM={val_losses['ssim_raw']:.4f}  "
            f"sharpness_ratio={val_losses['sharpness_ratio']:.3f}  "
            f"id_retrieval_acc={val_losses['identity_retrieval_acc']:.3f}  "
            f"best={best_val_score:.4f}"
        )
        if writer:
            writer.add_scalar("val/selection_score", val_selection, epoch)

        if evaluator is not None and (epoch + 1) % cfg.train.eval_every_epochs == 0:
            eval_model.eval()
            indep_scores = []
            with torch.no_grad():
                for emb_b, real_b in val_loader:
                    emb_b  = emb_b.to(device)
                    real_b = real_b.to(device)
                    gen_b  = eval_model(emb_b)
                    indep_scores.append(evaluator.cosine_sim(gen_b, real_b).item())
            indep_mean = sum(indep_scores) / len(indep_scores)
            log.info(f"[Bağımsız Eval] epoch={epoch+1}  FaceNet cosine_sim={indep_mean:.4f}")
            if writer:
                writer.add_scalar("eval/independent_cosine_sim", indep_mean, epoch)
            eval_model.train()

        is_best = val_selection < best_val_score - cfg.train.min_delta
        if is_best:
            best_val_score = val_selection
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
                    "model_ema":      ema.state_dict() if ema is not None else None,
                },
                save_dir=cfg.train.save_dir,
                epoch=epoch + 1,
                is_best=is_best,
                keep_last_n=cfg.train.keep_last_n,
            )

        if patience_ctr >= cfg.train.patience:
            log.info(
                f"Early stopping: {cfg.train.patience} epoch boyunca "
                f"val selection score iyilesmedi. Son score: {best_val_score:.4f}"
            )
            break

    best_ckpt = Path(cfg.train.save_dir) / "best_model.pt"
    if best_ckpt.exists():
        state = torch.load(best_ckpt, map_location=device, weights_only=False)
        if cfg.train.use_ema and cfg.train.eval_use_ema and state.get("model_ema") is not None:
            model.load_state_dict(state["model_ema"])
            log.info("[EMA] Nihai test icin EMA agirliklari yuklendi.")
        else:
            model.load_state_dict(state["model"])
    model.eval()

    test_id_scores = []
    with torch.no_grad():
        for emb_b, real_b in test_loader:
            emb_b  = emb_b.to(device)
            real_b = real_b.to(device)
            gen_b  = model(emb_b)
            losses = loss_fn(
                gen_b, real_b, _augment_weights(cfg, cfg.loss.phase3_weights),
                input_embedding=emb_b,
            )
            test_id_scores.append(1.0 - losses["identity"].item())
            if evaluator is not None:
                writer.add_scalar(
                    "test/independent_cosine_sim",
                    evaluator.cosine_sim(gen_b, real_b).item(),
                )

    if test_id_scores:
        test_mean = sum(test_id_scores) / len(test_id_scores)
        log.info(
            f"[NİHAİ TEST] {len(test_loader.dataset):,} görülmemiş görüntü — "
            f"identity_similarity={test_mean:.4f}"
        )
        writer.add_scalar("test/identity_similarity", test_mean, 0)

    writer.close()
    log.info(
        f"Eğitim tamamlandı. En iyi val selection score "
        f"(identity + sharpness): {best_val_score:.4f}"
    )

    return str(best_ckpt)


if __name__ == "__main__":
    train()
