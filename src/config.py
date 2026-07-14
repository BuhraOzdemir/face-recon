"""
Tüm hiperparametreler tek bir yerde.
Colab'da çalıştırırken sadece bu dosyayı düzenlemek yeterli.
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DataConfig:
    # Ham veri klasörü: her alt klasör bir kimliği temsil eder
    # Örnek: data_dir/person_001/img001.jpg
    data_dir: str = "/content/drive/MyDrive/face_data/raw"

    # Ön işlem sonucu hizalanmış yüzler + embedding'ler buraya kaydedilir
    processed_dir: str = "/content/drive/MyDrive/face_data/processed"

    image_size: int = 128          # Decoder çıktı boyutu
    align_size: int = 112          # insightface hizalama boyutu (sonra resize edilir)
    embedding_dim: int = 512       # ArcFace embedding boyutu

    # Eğitim/val split oranı (kimlik bazlı)
    val_split: float = 0.05        # Kimliklerin %5'i validasyon için ayrılır
    num_workers: int = 4


@dataclass
class ModelConfig:
    embedding_dim: int = 512

    # MLP Head çıktısı → spatial feature map boyutu
    # 4×4×128 → sonraki 5 up-block ile 128×128 olur
    initial_spatial: int = 4       # Başlangıç feature map boyutu (4×4)
    initial_channels: int = 128    # Başlangıç kanal sayısı

    # Her UpsampleBlock'un çıktı kanal sayıları (5 blok, 4→8→16→32→64→128)
    # [128, 128, 64, 32, 16] → toplam ~1.1M parametre
    decoder_channels: tuple = (128, 128, 64, 32, 16)


@dataclass
class LossConfig:
    # Eğitim aşamalarına göre loss ağırlıkları
    # Aşama 1 (warm-up): yalnızca temel loss'lar
    phase1_epochs: int = 10
    phase1_weights: dict = field(default_factory=lambda: {
        "l1": 1.0,
        "perceptual": 1.0,
        "identity": 0.0,
        "ssim": 0.0,
    })

    # Aşama 2 (ana eğitim): identity loss devreye girer
    phase2_epochs: int = 50
    phase2_weights: dict = field(default_factory=lambda: {
        "l1": 0.5,
        "perceptual": 1.0,
        "identity": 5.0,
        "ssim": 0.1,
    })

    # Aşama 3 (ince ayar): identity ağırlığı artırılır
    phase3_epochs: int = 40  # Kalan epoch'lar
    phase3_weights: dict = field(default_factory=lambda: {
        "l1": 0.2,
        "perceptual": 1.0,
        "identity": 8.0,
        "ssim": 0.1,
    })

    # VGG perceptual loss için hangi katman kullanılacak
    # relu3_3 → iyi doku + yapı dengesi
    vgg_layer: str = "relu3_3"

    # Identity loss için FaceNet input boyutu
    facenet_input_size: int = 160


@dataclass
class TrainConfig:
    epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4

    # Optimizer: AdamW
    adam_betas: tuple = (0.9, 0.999)
    adam_eps: float = 1e-8

    # Scheduler: Linear warmup + CosineAnnealingLR
    warmup_epochs: int = 5
    eta_min: float = 1e-6

    # Checkpointing
    save_dir: str = "/content/drive/MyDrive/face_recon/checkpoints"
    save_every_epochs: int = 5
    keep_last_n: int = 3           # Sadece son N checkpoint saklanır

    # Early stopping
    patience: int = 10             # Val identity score 10 epoch artmazsa dur
    min_delta: float = 1e-4        # Minimum iyileşme eşiği

    # Mixed precision (Colab T4/A100 için)
    use_amp: bool = True

    # Logging
    log_dir: str = "/content/drive/MyDrive/face_recon/logs"
    log_every_steps: int = 50


@dataclass
class ExportConfig:
    export_dir: str = "/content/drive/MyDrive/face_recon/export"
    model_name: str = "face_decoder"

    # Quantization
    quantize_int8: bool = True
    quantize_float16: bool = True

    # Test input için örnek embedding (sıfır vektör)
    # Export ve doğrulama sırasında kullanılır


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
        """Epoch numarasına göre doğru loss ağırlıklarını döndür."""
        p1 = self.loss.phase1_epochs
        p2 = p1 + self.loss.phase2_epochs
        if epoch < p1:
            return self.loss.phase1_weights
        elif epoch < p2:
            return self.loss.phase2_weights
        else:
            return self.loss.phase3_weights


# Varsayılan config — notebook'ta override edilebilir
DEFAULT_CONFIG = Config()
