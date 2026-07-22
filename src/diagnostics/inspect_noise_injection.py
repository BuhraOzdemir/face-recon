"""
Noise Injection Diagnostik Karşılaştırması.

AMAÇ: checkpoint_epoch0010.pt sonrasında TensorBoard'da görülen
moiré/waffle dokulu, kimlikten bağımsız (mode-collapse) çıktıların
NEDENİNİN noise injection modülasyon katmanlarından (src/models/decoder.py
NoiseModulation) kaynaklanıp kaynaklanmadığını izole etmek için aynı
embedding'lerle ÜÇ farklı inference modu karşılaştırır:

    1. normal        : noise injection AÇIK, sabit seed'li gürültü
                        (yeniden üretilebilir; TensorBoard'daki rastgele
                        gürültülü çıktıyla AYNI KARAKTERDE olması beklenir,
                        piksel-eşit değil — orada seed sabitlenmemişti).
    2. zero_noise     : aynı checkpoint, ama gürültü GİRDİSİ sıfır vektör
                        (torch.zeros) — talep edilen karşılaştırma.
                        DİKKAT: bu, eğitilmiş noise_mlp/modülasyon
                        katmanlarından HÂLÂ geçer; bias'lar eğitimle
                        sıfırdan kaymışsa çıktı yine de sıfır-olmayan bir
                        w üretebilir (bkz. aşağıdaki init/drift raporu).
    3. hard_disabled  : (EK, daha kesin ablasyon) NoiseModulation.forward
                        çağrısı monkeypatch ile TAMAMEN bypass edilir —
                        öğrenilmiş ağırlıklardan bağımsız, "noise injection
                        hiç eklenmemiş olsaydı decoder ne üretirdi"
                        sorusuna kesin cevap verir.

Ayrıca NoiseModulation'ın init kodunu ve checkpoint SONRASI (eğitimle ne
kadar sürüklendiğini) raporlar.

Kullanım (Kaggle/Colab hücresinde):
    from src.inspect_noise_injection import run_noise_injection_check
    run_noise_injection_check(
        cfg, manifest_path, "/kaggle/working/checkpoints/checkpoint_epoch0010.pt"
    )
"""

import logging
from pathlib import Path

import torch
import matplotlib.pyplot as plt

from .config import Config
from .data.dataset import build_dataloaders, denormalize
from .models.decoder import FaceDecoder, NoiseModulation

log = logging.getLogger(__name__)


# ─── NoiseModulation init / drift raporu ───────────────────────────────────────

def _report_noise_modulation_init(model: FaceDecoder) -> bool:
    """
    decoder.py'deki NoiseModulation.reset_parameters() init kodu:
        to_scale.weight = 0, to_scale.bias = 1   -> başlangıçta scale(w) SABİT 1
        to_shift.weight = 0, to_shift.bias = 0   -> başlangıçta shift(w) SABİT 0
    Yani modülasyon RASTGELE değil, TAM KİMLİK dönüşümüyle başlar (girdi
    hangi w olursa olsun x değişmeden geçer). Aşağıdaki rapor, checkpoint
    YÜKLENDİKTEN SONRAKİ (eğitimle sürüklenmiş) gerçek değerleri bu ideal
    init ile kıyaslar — büyük sapma, modülasyonun eğitim sırasında
    agresif şekilde "öğrenildiğini" (ve olası bir başıboş kalma riskini)
    gösterir.
    """
    log.info("=" * 78)
    log.info("[NoiseModulation init kodu] (src/models/decoder.py, reset_parameters):")
    log.info("    to_scale.weight = 0, to_scale.bias = 1  -> init'te scale(w) SABİT 1")
    log.info("    to_shift.weight = 0, to_shift.bias = 0  -> init'te shift(w) SABİT 0")
    log.info("    => Noise injection RASTGELE DEĞİL, başlangıçta TAM KİMLİK dönüşümüdür.")
    log.info("-" * 78)

    found = False
    for name, m in model.named_modules():
        if isinstance(m, NoiseModulation):
            found = True
            sw = m.to_scale.weight.detach()
            sb = m.to_scale.bias.detach()
            hw = m.to_shift.weight.detach()
            hb = m.to_shift.bias.detach()
            log.info(
                f"[{name}] CKPT-SONRASI değerler (init ile kıyasla):\n"
                f"    to_scale.weight: |mean|={sw.abs().mean():.5f}  max|.|={sw.abs().max():.5f}   (init: 0)\n"
                f"    to_scale.bias  : mean={sb.mean():.5f}  std={sb.std():.5f}                     (init: 1 sabit)\n"
                f"    to_shift.weight: |mean|={hw.abs().mean():.5f}  max|.|={hw.abs().max():.5f}   (init: 0)\n"
                f"    to_shift.bias  : mean={hb.mean():.5f}  std={hb.std():.5f}                     (init: 0 sabit)"
            )
    if not found:
        log.warning(
            "[NoiseModulation] Modelde hiç NoiseModulation katmanı yok "
            "(use_noise_injection=False ile eğitilmiş olabilir)."
        )
    log.info("=" * 78)
    return found


# ─── Hard-disable: modülasyonu forward'dan tamamen bypass et ─────────────────

class _DisableNoiseModulation:
    """
    Context manager: NoiseModulation.forward'i geçici olarak "x'i olduğu
    gibi döndür" (w'yi tamamen yoksay) şeklinde monkeypatch'ler. Eğitilmiş
    scale/shift ağırlıklarından BAĞIMSIZ en kesin ablasyondur.
    """

    def __enter__(self):
        self._orig_forward = NoiseModulation.forward

        def _identity_forward(self_mod, x, w):
            return x

        NoiseModulation.forward = _identity_forward
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        NoiseModulation.forward = self._orig_forward


# ─── Checkpoint'ten, checkpoint'in KENDİ config'ine göre model kur ────────────

def _build_model_from_checkpoint(checkpoint_path: str, cfg: Config, device: torch.device):
    """
    Mimariyi ÖNCELİKLE checkpoint'in kendi kaydettiği config'ten kurar
    (train.py save_checkpoint -> state["config"]). cfg parametresiyle
    checkpoint'in gerçekte eğitildiği mimari arasında bir uyuşmazlık varsa
    (örn. farklı noise_dim) sessizce yanlış/anlamsız sonuç üretmek yerine
    doğru mimariyi garanti eder. Checkpoint'te config yoksa (eski
    checkpoint) verilen cfg.model'e düşer ve açıkça uyarır.
    """
    state = torch.load(checkpoint_path, map_location=device)
    ckpt_cfg = state.get("config")

    if isinstance(ckpt_cfg, dict) and "model" in ckpt_cfg:
        m = ckpt_cfg["model"]
        log.info("[Mimari] checkpoint'in KENDİ kaydettiği config kullanılıyor.")
        model = FaceDecoder(
            embedding_dim=m["embedding_dim"],
            initial_spatial=m["initial_spatial"],
            initial_channels=m["initial_channels"],
            decoder_channels=tuple(m["decoder_channels"]),
            norm_type=m.get("norm_type", "batch"),
            use_noise_injection=m.get("use_noise_injection", False),
            noise_dim=m.get("noise_dim", 64),
            use_cascade_skip=m.get("use_cascade_skip", False),
            cascade_skip_last_n_blocks=m.get("cascade_skip_last_n_blocks", 2),
        ).to(device)
    else:
        log.warning(
            "[Mimari] checkpoint'te config yok — verilen cfg.model kullanılıyor "
            "(MİMARİ UYUŞMAZLIĞI RİSKİ, muhtemelen eski bir checkpoint)."
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

    model.load_state_dict(state["model"])
    return model, state


# ─── Ana fonksiyon ──────────────────────────────────────────────────────────

def run_noise_injection_check(
    cfg: Config,
    manifest_path: str,
    checkpoint_path: str,
    n_samples: int = 8,
    seed: int = 0,
    save_path: str = "/kaggle/working/noise_injection_check.png",
):
    """
    checkpoint'ten normal / zero-noise / hard-disabled inference
    karşılaştırması üretir + NoiseModulation init/drift raporu yazar.

    Returns:
        dict: {"real","normal","zero_noise","hard_disabled"} -> CPU tensörler [-1,1]
        (use_noise_injection=False ile eğitilmiş bir checkpoint verilirse None döner)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"[Noise Check] Cihaz: {device}")

    model, state = _build_model_from_checkpoint(checkpoint_path, cfg, device)
    model.eval()
    log.info(
        f"[Noise Check] Checkpoint yüklendi: epoch={state.get('epoch')}, "
        f"use_noise_injection={model.use_noise_injection}, noise_dim={model.noise_dim}"
    )

    has_noise_mod = _report_noise_modulation_init(model)
    if not model.use_noise_injection or not has_noise_mod:
        log.warning(
            "[Noise Check] Bu checkpoint noise injection OLMADAN eğitilmiş "
            "(use_noise_injection=False) — zero/hard-disabled karşılaştırması "
            "anlamsız olur (hiçbiri fark yaratmaz). Script burada duruyor."
        )
        return None

    # ── validate()'in TB'ye yazdığı GERÇEK görüntülerle aynı batch ─────
    # (dataset.py val_loader'ı shuffle=False; validate() döngünün SON
    # batch'ini TB'ye yazıyor — aynı davranışı burada tekrarlıyoruz.)
    _, val_loader, _ = build_dataloaders(
        manifest_path=manifest_path,
        image_size=cfg.data.image_size,
        batch_size=cfg.train.batch_size,
        val_split=cfg.data.val_split,
        test_split=cfg.data.test_split,
        num_workers=0,
        max_samples=cfg.data.max_samples,
    )
    embeddings, real_imgs = None, None
    for embeddings, real_imgs in val_loader:
        pass
    if embeddings is None:
        raise RuntimeError("val_loader boş — manifest/split ayarlarını kontrol edin.")

    embeddings = embeddings[:n_samples].to(device)
    real_imgs = real_imgs[:n_samples].to(device)
    B = embeddings.size(0)

    with torch.no_grad():
        # 1) Normal: noise injection AÇIK, sabit seed (yeniden üretilebilir)
        generated_normal = model(embeddings, noise_seed=seed)

        # 2) Zero-noise: gürültü GİRDİSİ sıfır vektör — talep edilen karşılaştırma
        zero_noise = torch.zeros(B, model.noise_dim, device=device)
        generated_zero = model(embeddings, noise=zero_noise)

        # 3) Hard-disabled (ek): modülasyon katmanları forward'dan bypass
        with _DisableNoiseModulation():
            generated_hard = model(embeddings, noise_seed=seed)

    diff_zero = (generated_normal - generated_zero).abs().mean().item()
    diff_hard = (generated_normal - generated_hard).abs().mean().item()
    log.info(
        f"[Noise Check] Ortalama mutlak fark — normal vs zero_noise: {diff_zero:.5f}  |  "
        f"normal vs hard_disabled: {diff_hard:.5f}"
    )
    if diff_zero < 1e-4 and diff_hard < 1e-4:
        log.info(
            "[Noise Check] SONUÇ: Noise injection'ın çıktıya PRATİKTE hiçbir "
            "etkisi yok gibi görünüyor — moiré deseni BAŞKA bir kaynaktan "
            "geliyor olmalı (discriminator/PatchGAN, PixelShuffle checkerboard, vb.)."
        )
    else:
        log.info(
            "[Noise Check] SONUÇ: Noise injection çıktıda GÖZLE GÖRÜLÜR fark "
            "yaratıyor — moiré/mode-collapse deseninin (bir kısmının/tamamının) "
            "noise modulation katmanlarından kaynaklanma ihtimali YÜKSEK."
        )

    # ── Görselleştirme ───────────────────────────────────────────────
    real_np = denormalize(real_imgs.cpu()).permute(0, 2, 3, 1).numpy()
    normal_np = denormalize(generated_normal.cpu()).permute(0, 2, 3, 1).numpy()
    zero_np = denormalize(generated_zero.cpu()).permute(0, 2, 3, 1).numpy()
    hard_np = denormalize(generated_hard.cpu()).permute(0, 2, 3, 1).numpy()

    rows = [
        ("Gerçek", real_np),
        ("Normal (noise açık)", normal_np),
        ("Zero-noise (z=0)", zero_np),
        ("Hard-disabled (bypass)", hard_np),
    ]
    fig, axes = plt.subplots(len(rows), B, figsize=(2.4 * B, 2.6 * len(rows)), squeeze=False)
    for r, (title, imgs) in enumerate(rows):
        for c in range(B):
            axes[r, c].imshow(imgs[c].clip(0, 1))
            axes[r, c].axis("off")
        axes[r, 0].text(
            -0.25, 0.5, title, fontsize=9, ha="right", va="center",
            transform=axes[r, 0].transAxes,
        )
    plt.suptitle(
        f"Noise Injection Ablasyonu — checkpoint epoch={state.get('epoch')}  "
        f"|normal-zero|={diff_zero:.4f}  |normal-hard|={diff_hard:.4f}",
        fontsize=11,
    )
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    log.info(f"[Noise Check] Karşılaştırma kaydedildi: {save_path}")

    return {
        "real": real_imgs.cpu(),
        "normal": generated_normal.cpu(),
        "zero_noise": generated_zero.cpu(),
        "hard_disabled": generated_hard.cpu(),
    }
