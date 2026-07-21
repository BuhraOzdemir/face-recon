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
    # PixelShuffle + 2x DWRes ile INT8 <5MB: 192→(192,128,96,64,32) ≈4.94MB
    initial_channels: int = 192
    decoder_channels: tuple = (192, 128, 96, 64, 32)
    # "batch" (varsayılan, mevcut checkpoint'lerle uyumlu) veya "instance".
    # GAN eğitiminde instance norm bazen daha keskin doku üretir, ama
    # state_dict anahtarları değiştiği için ESKİ checkpoint'ler bu modla
    # YÜKLENEMEZ — değiştirirken sıfırdan eğitim gerekir.
    norm_type: str = "batch"

    # ── Noise Injection (many-to-one embedding inversion sorununa çözüm) ──
    # ArcFace/MobileFaceNet embedding'i poz/ifade/aydınlatma-invaryanttır;
    # aynı kişinin farklı fotoğrafları neredeyse aynı z'ye düşer ama piksel
    # içerikleri (poz, ifade, arka plan) farklıdır. Deterministik decoder
    # (z → tek çıktı) bu belirsizliği "ortalama yüz" öğrenerek çözer =
    # bulanıklık. AdaIN-tarzı per-block noise modulation, decoder'a z'nin
    # açıklayamadığı yüksek-frekans detayı üretebileceği ekstra bir
    # stokastik serbestlik derecesi tanır. Girdi sözleşmesi BOZULMAZ:
    # forward(z) hâlâ çalışır, gürültü forward içinde otomatik örneklenir.
    # Kapalıyken (varsayılan) mimari eskisiyle BİREBİR aynıdır, checkpoint
    # uyumluluğu korunur. Açmak sıfırdan eğitim gerektirir (yeni parametreler).
    use_noise_injection: bool = False
    noise_dim: int = 64


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

    # ── LPIPS (opsiyonel, VGG perceptual'a EK olarak) ──────────────────
    # VGG feature-MSE de "yumuşak" olabilir; LPIPS öğrenilmiş bir algısal
    # mesafe metriği olduğu için ince doku/keskinlik farklarına daha
    # duyarlıdır. Kapalıyken davranış hiç değişmez (varsayılan False).
    # Açmak için: use_lpips=True ve phaseX_weights sözlüğüne "lpips" anahtarı
    # eklemeye gerek yok — lpips_weight otomatik olarak enjekte edilir.
    use_lpips: bool = False
    lpips_weight: float = 0.8
    lpips_net: str = "alex"  # "alex" (hafif/hızlı) veya "vgg"

    # ── Diversity / mode-seeking loss (MSGAN-tarzı) ─────────────────────
    # SADECE model.use_noise_injection=True iken anlamlıdır. Aynı embedding
    # (e) için iki farklı gürültüyle (z1,z2) üretilen çıktılar arasındaki
    # mesafeyi z-mesafesine oranla ödüllendirir — decoder'ın gürültüyü
    # yoksayıp yine tek bir "ortalama" çıktıya çökmesini (mode collapse)
    # engeller. Kapalıyken (0.0) davranış hiç değişmez.
    diversity_weight: float = 0.0

    # ── Cycle/identity-embedding loss (opsiyonel) ───────────────────────
    # Üretilen görüntüyü identity encoder'dan (ArcFace R50 veya FaceNet,
    # yukarıdaki IdentityLoss ile AYNI dondurulmuş ağ) tekrar geçirip
    # çıkan embedding'i decoder'ın GİRDİSİ olan ham z ile cosine mesafeyle
    # karşılaştırır: encode(decode(z)) ≈ z.
    # ÖNEMLİ UYARI: Bu SADECE, preprocess.py'de manifest embedding'lerini
    # üreten model (şu an insightface buffalo_s) identity encoder ile
    # (arcface_r50_path verilmezse FaceNet, verilirse IResNet50-ArcFace)
    # AYNI ağırlıklara sahipse anlamlıdır. buffalo_s ile ArcFace-R50/FaceNet
    # FARKLI embedding uzaylarıdır — bu durumda cosine mesafe rastgele
    # gürültüden ibarettir ve loss'a ZARARLI olabilir. Sadece manifest
    # embedding'leri arcface_r50_path ile AYNI R50 checkpoint'iyle
    # üretildiyse (yani preprocess'i de ArcFace R50'ye çevirdiyseniz) açın.
    # Varsayılan 0.0 = kapalı, davranış değişmez.
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
    # Erken epoch'larda INT8 noise kapalı — soft reconstruction azalır
    transport_simulate_start_epoch: int = 20

    use_independent_evaluator: bool = True
    eval_every_epochs: int = 5

    log_dir: str = "/content/drive/MyDrive/face_recon/logs"
    log_every_steps: int = 50

    # Checkpoint seçimi: identity + sharpness composite
    # selection = identity_score + sharpness_ckpt_weight * (1 - min(sharpness_ratio, 1))
    sharpness_ckpt_weight: float = 0.5

    # ── GAN (adversarial) ayarları ──────────────────────────────────
    # Discriminator SADECE eğitimde kullanılır, export/deploy edilen
    # FaceDecoder modeline hiç dahil edilmez (boyut/QR kısıtını etkilemez).
    use_gan: bool = True
    gan_start_epoch: int = 3
    disc_lr: float = 1e-4
    disc_base_channels: int = 64
    adv_weight: float = 1.0
    adv_weight_ramp_epochs: int = 5
    feat_match_weight: float = 10.0

    # Discriminator icin ayri Adam beta1/beta2. Generator ile ayni
    # (0.9, 0.999) kullanmak GAN egitiminde momentum'un discriminator'i
    # generator'in gerisinde birakmasina, zayif/gecikmeli adversarial
    # sinyale (=> bulanik/ortalama yuz egilimi) yol acabiliyor.
    # (0.5, 0.999) DCGAN'dan beri standart GAN pratigidir.
    disc_betas: tuple = (0.5, 0.999)

    # "bce" (mevcut varsayilan, vanishing-gradient riski yuksek confident
    # D'de), "lsgan" (MSE tabanli, daha yumusak/stabil gradyan) veya
    # "hinge" (modern GAN'larda standart, en keskin sonuclari verme
    # egiliminde). Degistirmek checkpoint uyumlulugunu ETKILEMEZ (sadece
    # loss formulu degisir, model agirliklari ayni kalir).
    gan_loss_type: str = "bce"

    # R1 gradient penalty (Mescheder et al., StyleGAN2'de standart).
    # Discriminator'in gercek goruntuler etrafinda asiri keskin/emin karar
    # sinirlari olusturmasini (=> G'ye zayif/gurultulu gradyan =>
    # bulaniklik) engeller. 0.0 = kapali (varsayilan, davranis degismez).
    # Denenecek deger: 10.0. Her adimda ekstra bir D forward+double-backward
    # gerektirdigi icin GAN aktifken egitim adimini yavaslatir.
    r1_gamma: float = 0.0

    # ── Generator EMA (Exponential Moving Average) ──────────────────────
    # Egitim boyunca generator agirliklarinin "yumusatilmis" bir golge
    # kopyasi tutulur (StyleGAN/BigGAN'da standart). Anlik gradyan
    # gurultusunu/GAN salinimlarini ortalayarak genelde daha keskin VE
    # daha stabil sonuc verir. use_ema=False iken hicbir ek parametre/
    # maliyet yoktur, checkpoint yapisi degismez (model_ema alani None).
    use_ema: bool = False
    ema_decay: float = 0.999
    # True ise validation/final-test/checkpoint-secimi EMA agirliklariyla
    # yapilir (genellikle use_ema ile BIRLIKTE True yapilir).
    eval_use_ema: bool = False


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
        adv_weight=0.4,
        gan_start_epoch=5,
        disc_base_channels=32,
        feat_match_weight=0.0,
    ),
)

# Alias: DEFAULT mobil kalite konfigi
MEDIUM_CONFIG = DEFAULT_CONFIG
