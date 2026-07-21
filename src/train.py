"""
Ana eğitim döngüsü.

Colab'da doğrudan import edilebilir veya bağımsız çalıştırılabilir.
"""

import copy
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
    ckpt_path: str,
    device: torch.device,
    discriminator: Optional[nn.Module] = None,
    disc_optimizer=None,
):
    """
    NOT: scheduler KASITLI olarak yuklenmiyor. Checkpoint'teki scheduler
    state'i farkli bir cfg.train.epochs (farkli T_max) ile hesaplanmis
    olabilir; bunu yeni bir scheduler'a yuklemek LR'nin beklenmedik
    sekilde artmasina/tutarsiz olmasina yol aciyordu. Bunun yerine
    train() icinde scheduler, start_epoch kadar .step() ile "ileri
    sarilarak" her zaman GUNCEL cfg.train.epochs'a gore konumlandiriliyor.
    """
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])

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


def _feature_matching_loss(fake_feats, real_feats) -> torch.Tensor:
    """PatchGAN ara katman L1 farkı (generator keskinlik sinyali)."""
    loss = fake_feats[0].new_zeros(())
    for f_fake, f_real in zip(fake_feats, real_feats):
        loss = loss + F.l1_loss(f_fake, f_real.detach())
    return loss / max(len(fake_feats), 1)


def _adv_loss_generator(fake_logits: torch.Tensor, gan_loss_type: str) -> torch.Tensor:
    """
    Generator adversarial loss. gan_loss_type: "bce" | "lsgan" | "hinge".

    "bce" (vanilla GAN) D cok emin oldugunda gradyan sifira yakinsayabilir
    (=> zayif keskinlik sinyali). "lsgan" ve "hinge" bu acidan daha stabildir
    ve genelde daha keskin sonuclara yol acar.
    """
    if gan_loss_type == "bce":
        return F.binary_cross_entropy_with_logits(
            fake_logits, torch.ones_like(fake_logits)
        )
    elif gan_loss_type == "lsgan":
        return F.mse_loss(fake_logits, torch.ones_like(fake_logits))
    elif gan_loss_type == "hinge":
        return -fake_logits.mean()
    raise ValueError(f"Bilinmeyen gan_loss_type: {gan_loss_type}")


def _adv_loss_discriminator(
    real_logits: torch.Tensor, fake_logits: torch.Tensor, gan_loss_type: str
) -> torch.Tensor:
    """Discriminator adversarial loss. Real icin 0.9 label smoothing (bce/lsgan)."""
    if gan_loss_type == "bce":
        d_real = F.binary_cross_entropy_with_logits(
            real_logits, torch.full_like(real_logits, 0.9)
        )
        d_fake = F.binary_cross_entropy_with_logits(
            fake_logits, torch.zeros_like(fake_logits)
        )
        return 0.5 * (d_real + d_fake)
    elif gan_loss_type == "lsgan":
        d_real = F.mse_loss(real_logits, torch.full_like(real_logits, 0.9))
        d_fake = F.mse_loss(fake_logits, torch.zeros_like(fake_logits))
        return 0.5 * (d_real + d_fake)
    elif gan_loss_type == "hinge":
        d_real = F.relu(1.0 - real_logits).mean()
        d_fake = F.relu(1.0 + fake_logits).mean()
        return 0.5 * (d_real + d_fake)
    raise ValueError(f"Bilinmeyen gan_loss_type: {gan_loss_type}")


def _augment_weights(cfg: Config, base_weights: dict) -> dict:
    """
    Opsiyonel loss terimlerinin (lpips, cycle_identity) agirliklarini
    faz-bazli agirlik sozlugune enjekte eder. Ikisi de kapaliyken
    (varsayilan) sozluk degismeden doner.
    """
    weights = dict(base_weights)
    if cfg.loss.use_lpips:
        weights.setdefault("lpips", cfg.loss.lpips_weight)
    if cfg.loss.cycle_identity_weight > 0:
        weights.setdefault("cycle_identity", cfg.loss.cycle_identity_weight)
    return weights


def _r1_penalty(discriminator: nn.Module, real_imgs: torch.Tensor) -> torch.Tensor:
    """
    StyleGAN2 R1 gradient penalty: D(real)'in gercek piksellere gore
    gradyaninin L2 normunun karesi. Discriminator'i gercek goruntuler
    etrafinda asiri keskin/emin karar sinirlari olusturmaktan alikoyar;
    bu da G'ye daha anlamli/surekli bir gradyan (=> keskinlik sinyali)
    saglar. Sayisal kararlilik icin autocast/AMP DISINDA (tam hassasiyet)
    cagrilmalidir — cift backward (create_graph=True) AMP ile guvenilir
    calismaz.
    """
    real_imgs = real_imgs.detach().requires_grad_(True)
    real_logits = discriminator(real_imgs)
    grad_real, = torch.autograd.grad(
        outputs=real_logits.sum(), inputs=real_imgs, create_graph=True,
    )
    return grad_real.pow(2).reshape(grad_real.size(0), -1).sum(1).mean()


def _diversity_loss(
    generated_a: torch.Tensor, generated_b: torch.Tensor,
    noise_a: torch.Tensor, noise_b: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """
    MSGAN-tarzi mode-seeking regularization:
        loss = -lambda * ||G(e,z1)-G(e,z2)||_1 / ||z1-z2||_1
    Ayni embedding'den farkli gurultulerle uretilen ciktilar birbirine ne
    kadar yakinsa, o kadar cezalandirilir (negatif isaret = mesafeyi
    ARTIRMAYA tesvik) — decoder'in gurultuyu yoksayip tek bir "ortalama"
    ciktiya cokmesini (mode collapse) engeller. Sadece
    model.use_noise_injection=True VE loss.diversity_weight>0 iken cagrilir.
    """
    img_dist = (generated_a - generated_b).abs().mean(dim=[1, 2, 3])
    z_dist = (noise_a - noise_b).abs().mean(dim=1)
    return -(img_dist / (z_dist + eps)).mean()


class ModelEMA:
    """
    Generator agirliklarinin EMA (exponential moving average) golge kopyasi.

    Egitim adimlarindaki anlik gradyan gurultusunu/GAN salinimlarini
    ortalar; StyleGAN/BigGAN gibi calismalarda hem daha keskin hem daha
    stabil sonuc verdigi gozlemlenmistir. BatchNorm/InstanceNorm running
    stats (buffer) EMA'lanmaz, dogrudan canli modelden kopyalanir (EMA'lanan
    running stats egitim erken donemlerinde yanlis/kararsiz olabiliyor).
    """

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


def _selection_score(identity_score: float, sharpness_ratio: float, weight: float) -> float:
    """
    Best checkpoint skoru (düşük = iyi).
    identity_score + weight * (1 - min(sharpness_ratio, 1))
    """
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


# ─── Ek Değerlendirme Metrikleri (PDF v2.0 §2 - metrik ailesi) ────────────────

def _compute_psnr(generated: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
    """generated, real: [-1,1] araliginda. PSNR dB cinsinden dondurur (yuksek=iyi)."""
    gen_01  = generated * 0.5 + 0.5
    real_01 = real * 0.5 + 0.5
    mse = torch.mean((gen_01 - real_01) ** 2, dim=[1, 2, 3])
    mse = mse.clamp(min=1e-10)
    psnr = 10 * torch.log10(1.0 / mse)
    return psnr.mean()


_LAPLACIAN_KERNEL = torch.tensor(
    [[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]], dtype=torch.float32
).view(1, 1, 3, 3)


def _laplacian_variance(img: torch.Tensor) -> torch.Tensor:
    """
    Sharpness/detail metrigi. img: (B,3,H,W) [-1,1].
    Yuksek deger = daha keskin/detayli goruntu (bulaniklik dusuk).
    Sadece raporlama icindir, backward'a girmez.
    """
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
    """
    Batch-ici kimlik retrieval dogrulugu (tam galeri-bazli top-1 retrieval'in
    hafif bir proxy'si). Her uretilen goruntunun FaceNet embedding'i, AYNI
    batch icindeki tum gercek goruntularin embedding'leriyle kiyaslanir;
    en yakin eslesme kendi cifti (dogru satir) ise basarili sayilir.

    NOT: Bu, gercek kimlik etiketi kullanan tam bir galeri-retrieval testi
    DEGILDIR (mevcut DataLoader kimlik etiketi dondurmuyor) — sadece
    kimlik ayirt edilebilirligini batch icinde olcen bir yaklasik gostergedir.
    Batch buyuklugu arttikca test biraz daha zorlasir (daha fazla "yanlis
    aday" olur), bu yuzden batch_size sabit tutulmalidir ki epoch'lar arasi
    karsilastirilabilir olsun.
    """
    gen_emb = evaluator.encode(generated)   # (B, 512) L2-normalized
    real_emb = evaluator.encode(real)       # (B, 512) L2-normalized

    sim_matrix = gen_emb @ real_emb.T       # (B, B) cosine similarity
    predicted = sim_matrix.argmax(dim=1)
    correct = torch.arange(generated.size(0), device=generated.device)
    accuracy = (predicted == correct).float().mean()
    return accuracy.item()


# ─── Validation ───────────────────────────────────────────────────────────────

@torch.no_grad()
def validate(model, loss_fn, val_loader, device, weights, writer, epoch, evaluator=None):
    """
    NOT: checkpoint karari icin kullanilan identity skoru, egitim
    fazindan BAGIMSIZ sabit bir agirlikla (identity=1.0) ayrica
    hesaplaniyor.

    psnr_db, ssim_raw, sharpness (laplacian variance) ve
    batch_identity_retrieval_acc SADECE raporlama icindir; toplam
    loss'a (backward'a) hicbir etkisi yoktur.
    """
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
    if n_retrieval_batches > 0:
        avg["identity_retrieval_acc"] = retrieval_acc_sum / n_retrieval_batches
    else:
        avg["identity_retrieval_acc"] = 0.0

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
        max_samples=cfg.data.max_samples,
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
        norm_type=cfg.model.norm_type,
        use_noise_injection=cfg.model.use_noise_injection,
        noise_dim=cfg.model.noise_dim,
    ).to(device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    info = model.count_parameters()
    log.info(f"Model: {info['total_params']:,} parametre | {info['float32_mb']} MB (float32)")
    if cfg.model.use_noise_injection:
        log.info(
            f"[Noise] Noise injection aktif: noise_dim={cfg.model.noise_dim}, "
            f"diversity_weight={cfg.loss.diversity_weight}"
        )
    if cfg.loss.cycle_identity_weight > 0:
        log.info(
            f"[CycleID] cycle_identity_weight={cfg.loss.cycle_identity_weight} — "
            f"YALNIZCA manifest embedding'leri identity encoder ile AYNI model "
            f"ise anlamli (bkz. config.py yorumu)."
        )

    # ── Discriminator (SADECE egitimde; export'a dahil edilmez) ──
    discriminator = None
    disc_optimizer = None
    disc_scaler = None
    if cfg.train.use_gan:
        discriminator = PatchDiscriminator(
            base_channels=cfg.train.disc_base_channels,
        ).to(device)
        disc_optimizer = AdamW(
            discriminator.parameters(),
            lr=cfg.train.disc_lr,
            betas=cfg.train.disc_betas,
        )
        disc_scaler = GradScaler("cuda", enabled=cfg.train.use_amp)
        d_info = discriminator.count_parameters()
        log.info(
            f"[GAN] Discriminator: {d_info['total_params']:,} parametre "
            f"({d_info['float32_mb']} MB) — SADECE egitimde kullanilir, export edilmez."
        )
        log.info(
            f"[GAN] gan_start_epoch={cfg.train.gan_start_epoch}, "
            f"adv_weight={cfg.train.adv_weight}, ramp={cfg.train.adv_weight_ramp_epochs} epoch, "
            f"feat_match_weight={cfg.train.feat_match_weight}, "
            f"gan_loss_type={cfg.train.gan_loss_type}, disc_betas={cfg.train.disc_betas}"
        )

    # ── Loss ───────────────────────────────────────────────────
    loss_fn = ReconstructionLoss(
        vgg_layer         = cfg.loss.vgg_layer,
        facenet_input_size = cfg.loss.facenet_input_size,
        arcface_r50_path  = cfg.loss.arcface_r50_path,
        use_lpips         = cfg.loss.use_lpips,
        lpips_net         = cfg.loss.lpips_net,
    ).to(device)

    # ── Bağımsız Evaluator (PDF v2.0 §7) ───────────────────────
    # Ayni evaluator hem "Bagimsiz Eval" cosine sim raporu hem de
    # yeni identity_retrieval_acc metrigi icin kullanilir.
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
    best_val_score = float("inf")
    patience_ctr   = 0

    if resume_from and Path(resume_from).exists():
        start_epoch, best_val_score = load_checkpoint(
            model, optimizer, resume_from, device,
            discriminator=discriminator, disc_optimizer=disc_optimizer,
        )
        for _ in range(start_epoch):
            scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
        log.info(f"[Scheduler] {start_epoch} epoch ileri sarildi, guncel lr={current_lr:.2e}")

    # ── Generator EMA ──────────────────────────────────────────
    # Resume SONRASI olusturuluyor ki golge kopya, taze rastgele-init
    # agirliklar yerine dogrudan (varsa resume edilmis) guncel model
    # agirliklarindan baslasin.
    ema = None
    if cfg.train.use_ema:
        ema = ModelEMA(model, decay=cfg.train.ema_decay)
        if resume_from and Path(resume_from).exists():
            ckpt_state = torch.load(resume_from, map_location=device)
            if ckpt_state.get("model_ema") is not None:
                ema.load_state_dict(ckpt_state["model_ema"])
                log.info("[EMA] EMA golge agirliklari checkpoint'ten yuklendi.")
            else:
                log.info("[EMA] Checkpoint'te EMA yok - guncel model agirliklarindan baslatiliyor.")
            del ckpt_state
        log.info(f"[EMA] Aktif: decay={cfg.train.ema_decay}, eval_use_ema={cfg.train.eval_use_ema}")

    # ── TensorBoard ────────────────────────────────────────────
    writer = SummaryWriter(log_dir=cfg.train.log_dir)

    # ── Eğitim döngüsü ─────────────────────────────────────────
    log.info(f"Eğitim başlıyor: {cfg.train.epochs} epoch")

    for epoch in range(start_epoch, cfg.train.epochs):
        model.train()
        weights     = _augment_weights(cfg, cfg.get_loss_weights(epoch))
        epoch_loss  = 0.0
        step_count  = 0

        use_gan_this_epoch = (discriminator is not None) and (epoch >= cfg.train.gan_start_epoch)
        adv_w = _current_adv_weight(cfg, epoch) if use_gan_this_epoch else 0.0

        for step, (embeddings, real_imgs) in enumerate(train_loader):
            embeddings = embeddings.to(device, non_blocking=True)
            real_imgs  = real_imgs.to(device, non_blocking=True)
            if device.type == "cuda":
                real_imgs = real_imgs.contiguous(memory_format=torch.channels_last)

            use_transport = (
                cfg.train.transport_simulate
                and epoch >= cfg.train.transport_simulate_start_epoch
            )
            if use_transport:
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

                if use_gan_this_epoch and adv_w > 0:
                    fake_logits, fake_feats = discriminator.forward_features(generated)
                    with torch.no_grad():
                        _, real_feats = discriminator.forward_features(real_imgs)
                    adv_loss_g = _adv_loss_generator(fake_logits, cfg.train.gan_loss_type)
                    fm_loss = _feature_matching_loss(fake_feats, real_feats)
                    g_total = (
                        g_total
                        + adv_w * adv_loss_g
                        + cfg.train.feat_match_weight * fm_loss
                    )
                else:
                    adv_loss_g = torch.zeros((), device=device)
                    fm_loss = torch.zeros((), device=device)

                # ── Diversity / mode-seeking (MSGAN) ────────────────
                # Ayni z ile ikinci bir gurultu ornegi kullanarak decoder'i
                # tekrar calistirir; gurultuyu yoksayan bir decoder burada
                # cezalandirilir (bkz. _diversity_loss docstring).
                if use_diversity:
                    noise2 = torch.randn(embeddings.size(0), model.noise_dim, device=device)
                    generated_b = model(embeddings, noise=noise2)
                    div_loss = _diversity_loss(generated, generated_b, noise1, noise2)
                    g_total = g_total + cfg.loss.diversity_weight * div_loss
                else:
                    div_loss = torch.zeros((), device=device)

            scaler.scale(g_total).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            if ema is not None:
                ema.update(model)

            if use_gan_this_epoch:
                disc_optimizer.zero_grad(set_to_none=True)
                with autocast("cuda", enabled=cfg.train.use_amp):
                    real_logits = discriminator(real_imgs)
                    fake_logits_d = discriminator(generated.detach())
                    d_loss = _adv_loss_discriminator(
                        real_logits, fake_logits_d, cfg.train.gan_loss_type
                    )

                if cfg.train.r1_gamma > 0:
                    # R1: tam hassasiyet, autocast DISINDA (cift backward AMP
                    # altinda guvenilir degil).
                    r1_pen = _r1_penalty(discriminator, real_imgs)
                    d_loss_total = d_loss + 0.5 * cfg.train.r1_gamma * r1_pen
                else:
                    d_loss_total = d_loss
                    r1_pen = torch.zeros((), device=device)

                disc_scaler.scale(d_loss_total).backward()
                disc_scaler.step(disc_optimizer)
                disc_scaler.update()
            else:
                d_loss = torch.zeros((), device=device)
                r1_pen = torch.zeros((), device=device)

            epoch_loss += g_total.item()
            step_count += 1

            global_step = epoch * len(train_loader) + step
            if step % cfg.train.log_every_steps == 0:
                lr = optimizer.param_groups[0]["lr"]
                lpips_str = f"lpips={losses['lpips'].item():.4f}  " if cfg.loss.use_lpips else ""
                cycle_str = (
                    f"cyc_id={losses['cycle_identity'].item():.4f}  "
                    if cfg.loss.cycle_identity_weight > 0 else ""
                )
                div_str = f"div={div_loss.item():.4f}  " if use_diversity else ""
                r1_str = f"r1={r1_pen.item():.4f}  " if cfg.train.r1_gamma > 0 else ""
                log.info(
                    f"Epoch {epoch+1:3d}/{cfg.train.epochs} "
                    f"Step {step:5d}/{len(train_loader)} "
                    f"loss={losses['total'].item():.4f}  "
                    f"id={losses['identity'].item():.4f}  "
                    f"perc={losses['perceptual'].item():.4f}  "
                    f"{lpips_str}{cycle_str}{div_str}"
                    f"adv_g={adv_loss_g.item():.4f}  "
                    f"fm={fm_loss.item():.4f}  "
                    f"d_loss={d_loss.item():.4f}  "
                    f"{r1_str}"
                    f"lr={lr:.2e}"
                )
                for k, v in losses.items():
                    writer.add_scalar(f"train/{k}", v.item(), global_step)
                if use_gan_this_epoch:
                    writer.add_scalar("train/adv_g", adv_loss_g.item(), global_step)
                    writer.add_scalar("train/feat_match", fm_loss.item(), global_step)
                    writer.add_scalar("train/d_loss", d_loss.item(), global_step)
                    writer.add_scalar("train/adv_weight", adv_w, global_step)
                    if cfg.train.r1_gamma > 0:
                        writer.add_scalar("train/r1_penalty", r1_pen.item(), global_step)
                if use_diversity:
                    writer.add_scalar("train/diversity", div_loss.item(), global_step)
                writer.add_scalar("train/lr", lr, global_step)

        scheduler.step()

        # ── Validation ─────────────────────────────────────────
        # eval_use_ema aktifse validation/checkpoint-secimi EMA golge
        # kopyasi uzerinden yapilir (genellikle daha keskin/stabil).
        eval_model = ema.shadow if (ema is not None and cfg.train.eval_use_ema) else model
        val_losses = validate(eval_model, loss_fn, val_loader, device, weights, writer, epoch, evaluator=evaluator)
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
            f"best={best_val_score:.4f}  "
            f"gan_aktif={use_gan_this_epoch}"
        )
        if writer:
            writer.add_scalar("val/selection_score", val_selection, epoch)

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

        # ── Checkpoint (identity + sharpness composite) ────────
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
                    "discriminator":  discriminator.state_dict() if discriminator is not None else None,
                    "disc_optimizer": disc_optimizer.state_dict() if disc_optimizer is not None else None,
                    "model_ema":      ema.state_dict() if ema is not None else None,
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
                f"val selection score iyilesmedi. Son score: {best_val_score:.4f}"
            )
            break

    # ── Nihai Test Değerlendirmesi ──
    best_ckpt = Path(cfg.train.save_dir) / "best_model.pt"
    if best_ckpt.exists():
        state = torch.load(best_ckpt, map_location=device)
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
    log.info(
        f"Eğitim tamamlandı. En iyi val selection score "
        f"(identity + sharpness): {best_val_score:.4f}"
    )

    return str(best_ckpt)


if __name__ == "__main__":
    train()