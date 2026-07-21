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
    align_size: int = 112
    embedding_dim: int = 512
    val_split: float = 0.05
    test_split: float = 0.05
    num_workers: int = 4


@dataclass
class ModelConfig:
    embedding_dim: int = 512
    initial_spatial: int = 4
    initial_channels: int = 128
    decoder_channels: tuple = (128, 128, 64, 32, 16)


@dataclass
class LossConfig:
    phase1_epochs: int = 10
    phase1_weights: dict = field(default_factory=lambda: {
        "l1": 1.0, "perceptual": 1.0, "identity": 0.0, "ssim": 0.0,
    })

    phase2_epochs: int = 50
    phase2_weights: dict = field(default_factory=lambda: {
        "l1": 0.5, "perceptual": 1.0, "identity": 5.0, "ssim": 0.1,
    })

    phase3_epochs: int = 40
    phase3_weights: dict = field(default_factory=lambda: {
        "l1": 0.2, "perceptual": 1.0, "identity": 8.0, "ssim": 0.1,
    })

    vgg_layer: str = "relu3_3"
    facenet_input_size: int = 160
    arcface_r50_path: str = None


@dataclass
class TrainConfig:
    epochs: int = 100
    batch_size: int = 64
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

    use_independent_evaluator: bool = True
    eval_every_epochs: int = 5

    log_dir: str = "/content/drive/MyDrive/face_recon/logs"
    log_every_steps: int = 50

    # ── GAN (adversarial) ayarları ──────────────────────────────────
    # Discriminator SADECE eğitimde kullanılır, export/deploy edilen
    # FaceDecoder modeline hiç dahil edilmez (boyut/QR kısıtını etkilemez).
    use_gan: bool = True
    gan_start_epoch: int = 5        # identity/perceptual biraz oturduktan sonra GAN devreye girsin
    disc_lr: float = 1e-4
    adv_weight: float = 0.4         # generator loss'una eklenen adversarial ağırlık (tavan değer)
    adv_weight_ramp_epochs: int = 3 # 0'dan adv_weight'e bu kadar epoch'ta lineer yükselt


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

MEDIUM_CONFIG = Config(
    model=ModelConfig(
        embedding_dim=512,
        initial_spatial=4,
        initial_channels=256,
        decoder_channels=(256, 192, 128, 64, 32),
    ),
    loss=LossConfig(
        phase1_epochs=10,
        phase2_epochs=60,
        phase3_epochs=30,
        phase2_weights={
            "l1": 0.5, "perceptual": 1.0, "identity": 6.0, "ssim": 0.1,
        },
        phase3_weights={
            "l1": 0.2, "perceptual": 1.0, "identity": 10.0, "ssim": 0.1,
        },
        vgg_layer="relu3_3",
    ),
    train=TrainConfig(
        epochs=100,
        batch_size=32,
        learning_rate=5e-5,
    ),
)