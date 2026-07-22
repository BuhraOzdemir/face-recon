"""
Tüm hiperparametreler tek bir yerde.
Colab'da çalıştırırken sadece bu dosyayı düzenlemek yeterli.
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DataConfig:
    data_dir: str = "/content/drive/MyDrive/face_data/raw"
    processed_dir: str = "/content/drive/MyDrive/face_data/processed"
    image_size: int = 128
    align_size: int = 128  # doğrudan 128 align; 112→128 soft blur yok
    embedding_dim: int = 512
    val_split: float = 0.05
    test_split: float = 0.05
    num_workers: int = 4
    # Toplam örnek üst sınırı (train+val+test). 0 = sınırsız.
    max_samples: int = 100_000


@dataclass
class ModelConfig:
    embedding_dim: int = 512
    initial_spatial: int = 4
    # bilinear + 2x DWRes; INT8 <5MB hedefi: 192→(192,128,96,64,32)
    initial_channels: int = 192
    decoder_channels: tuple = (192, 128, 96, 64, 32)
    # "batch" (varsayılan) veya "instance".
    # state_dict anahtarları değiştiği için ESKİ checkpoint'ler norm_type
    # değiştirilince YÜKLENEMEZ — değiştirirken sıfırdan eğitim gerekir.
    norm_type: str = "batch"

    # ── Noise Injection (many-to-one embedding inversion sorununa çözüm) ──
    use_noise_injection: bool = False
    noise_dim: int = 64

    # ── Cascade skip (düşük-frekans kısayolu) ───────────────────────────
    # Varsayılan kapalı: nearest/erken skip + zayıf decoder ızgara üretebilir.
    # Açmak sıfırdan eğitim gerektirir.
    use_cascade_skip: bool = False
    cascade_skip_last_n_blocks: int = 2


@dataclass
class LossConfig:
    phase1_epochs: int = 10
    phase1_weights: dict = field(default_factory=lambda: {
        "l1": 1.0, "perceptual": 1.0, "identity": 0.0, "ssim": 0.0,
    })

    phase2_epochs: int = 50
    phase2_weights: dict = field(default_factory=lambda: {
        "l1": 0.3, "perceptual": 1.0, "identity": 5.0, "ssim": 0.2,
    })

    phase3_epochs: int = 40
    phase3_weights: dict = field(default_factory=lambda: {
        "l1": 0.1, "perceptual": 1.0, "identity": 6.0, "ssim": 0.2,
    })

    # Multi-layer VGG: relu2_2 + relu3_3 (virgülle ayrılmış)
    vgg_layer: str = "relu2_2,relu3_3"
    facenet_input_size: int = 160
    arcface_r50_path: str = None

    use_lpips: bool = False
    lpips_weight: float = 0.8
    lpips_net: str = "alex"

    # SADECE model.use_noise_injection=True iken anlamlıdır.
    diversity_weight: float = 0.0

    cycle_identity_weight: float = 0.0


@dataclass
class TrainConfig:
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4

    adam_betas: tuple = (0.9, 0.999)
    adam_eps: float = 1e-8

    warmup_epochs: int = 5
    eta_min: float = 1e-6

    save_dir: str = "/content/drive/MyDrive/face_recon/checkpoints"
    save_every_epochs: int = 5
    keep_last_n: int = 3

    patience: int = 20
    min_delta: float = 1e-5

    use_amp: bool = True
    transport_simulate: bool = True
    transport_simulate_start_epoch: int = 20

    use_independent_evaluator: bool = True
    eval_every_epochs: int = 5

    log_dir: str = "/content/drive/MyDrive/face_recon/logs"
    log_every_steps: int = 50

    # Checkpoint seçimi: identity + sharpness composite
    sharpness_ckpt_weight: float = 0.5

    # Generator EMA — gradyan gürültüsünü yumuşatır, daha stabil/keskin sonuç
    use_ema: bool = True
    ema_decay: float = 0.999
    eval_use_ema: bool = True

    # EMA + BatchNorm uyumsuzluğu düzeltmesi: ModelEMA.update() running_mean/
    # var'ı CANLI modelden doğrudan kopyalar (buffer'ları EMA'lamak DAHA
    # yanlış olurdu), ama bu istatistikler golge modelin YUMUŞATILMIŞ
    # (decay=0.999 → ~1000 adımlık gecikme) ağırlıklarına uygulanır — düşük-
    # frekans (renk) sinyalini az etkiler, yüksek-frekans (doku) detayında
    # gözle görülür bozulmaya yol açar. Her epoch validation'dan ÖNCE, EMA
    # gölge modelinin BatchNorm istatistiklerini KENDİ ağırlıklarına göre
    # yeniden hesaplar (torch.optim.swa_utils.update_bn ile aynı yöntem).
    # 0 = kapalı. norm_type="instance" iken no-op'tur (running stats yok).
    ema_bn_recalib_batches: int = 100


@dataclass
class ExportConfig:
    export_dir: str = "/content/drive/MyDrive/face_recon/export"
    model_name: str = "face_decoder"
    quantize_int8: bool = True
    quantize_float16: bool = True


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    export: ExportConfig = field(default_factory=ExportConfig)

    def total_epochs(self) -> int:
        return (
            self.loss.phase1_epochs
            + self.loss.phase2_epochs
            + self.loss.phase3_epochs
        )

    def get_loss_weights(self, epoch: int) -> dict:
        p1 = self.loss.phase1_epochs
        p2 = p1 + self.loss.phase2_epochs
        if epoch < p1:
            return self.loss.phase1_weights
        elif epoch < p2:
            return self.loss.phase2_weights
        else:
            return self.loss.phase3_weights


# ── Preset Config'ler ──────────────────────────────────────────────────────────

DEFAULT_CONFIG = Config()

# Eski küçük decoder (bilinear dönem) — geriye dönük karşılaştırma için
SMALL_CONFIG = Config(
    model=ModelConfig(
        embedding_dim=512,
        initial_spatial=4,
        initial_channels=128,
        decoder_channels=(128, 128, 64, 32, 16),
        use_cascade_skip=False,
    ),
    loss=LossConfig(
        phase1_epochs=10,
        phase2_epochs=50,
        phase3_epochs=40,
        phase2_weights={
            "l1": 0.5, "perceptual": 1.0, "identity": 5.0, "ssim": 0.1,
        },
        phase3_weights={
            "l1": 0.2, "perceptual": 1.0, "identity": 8.0, "ssim": 0.1,
        },
        vgg_layer="relu3_3",
    ),
    train=TrainConfig(
        epochs=100,
        batch_size=64,
        learning_rate=1e-4,
        use_ema=False,
        eval_use_ema=False,
    ),
)

MEDIUM_CONFIG = DEFAULT_CONFIG

# ── Kısa izole test config'i (init-fix DOĞRULAMASI) ────────────────────────────
# Amaç: "önceki 15-epoch GAN'sız/cascade'siz/EMA'lı testinizle" BİREBİR AYNI
# ayarlar — TEK FARK decoder.py'nin artık düzeltilmiş output_conv init'i
# kullanması (bu otomatik gelir, flag gerekmez). Faz zamanlaması KASITLI
# olarak DEĞİŞTİRİLMEDİ ki "sadece init değişikliği neyi düzeltti" sorusuna
# net cevap alınabilsin.
#
# UYARI: phase1_epochs=10 varsayılanı, bu 15-epoch'luk bütçenin 10'unu
# (%67'sini) identity=0 olan salt L1+perceptual "ısınma" fazında tüketiyor —
# yani identity-ayırt edici sinyal ancak epoch 10'da başlıyor ve sadece 5
# epoch çalışıyor. Bu, "farklı embedding'ler neredeyse özdeş çıktı üretiyor"
# gözleminizin (init'ten BAĞIMSIZ) ikinci bir olası nedeni — bkz. QUICK_TEST_PHASE_FIX_CONFIG.
QUICK_TEST_CONFIG = Config(
    data=DataConfig(max_samples=100_000),
    model=ModelConfig(use_cascade_skip=False, use_noise_injection=False),
    train=TrainConfig(epochs=15, use_ema=True, eval_use_ema=True),
)

# ── Kısa izole test config'i + faz zamanlaması düzeltmesi ──────────────────────
# QUICK_TEST_CONFIG ile AYNI, TEK FARK: identity-ayırt edici sinyal 15 epoch'un
# çoğunu değil küçük bir kısmını "kör" geçiriyor. 10/50/40 oranı (100 epoch'luk
# TAM eğitim için tasarlanmış) 15 epoch'a orantılı ölçeklendi: 2/7/6.
# İki config'i AYRI AYRI çalıştırıp karşılaştırın — hangisi "tek değişken"
# ilkesini bozmadan asıl faktörü izole eder.
QUICK_TEST_PHASE_FIX_CONFIG = Config(
    data=DataConfig(max_samples=100_000),
    model=ModelConfig(use_cascade_skip=False, use_noise_injection=False),
    loss=LossConfig(phase1_epochs=2, phase2_epochs=7, phase3_epochs=6),
    train=TrainConfig(epochs=15, use_ema=True, eval_use_ema=True),
)
