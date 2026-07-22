"""
Teşhis scriptleri için ortak, GÜVENLİ checkpoint/mimari yükleme yardımcısı.

inspect_noise_injection.py'deki desenle AYNI: mimari ÖNCELİKLE checkpoint'in
kendi kaydettiği config'ten kurulur (train.py save_checkpoint -> state["config"]).
cfg parametresiyle checkpoint'in gerçekte eğitildiği mimari arasında bir
uyuşmazlık varsa sessizce yanlış sonuç üretmek yerine doğru mimariyi garanti
eder. Bilerek inspect_noise_injection.py'nin kendi kopyasına DOKUNULMADI
(regresyon riski) — bu modül SADECE bu oturumda eklenen yeni teşhis
scriptleri için ortak kod.
"""

import logging

import torch

from ..config import Config
from ..models.decoder import FaceDecoder

log = logging.getLogger(__name__)


def build_model_from_checkpoint(checkpoint_path: str, cfg: Config, device: torch.device):
    """Checkpoint'in kendi config'ine göre FaceDecoder kurar + state_dict yükler."""
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
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


def sample_val_batch(manifest_path: str, cfg: Config, n_samples: int = 32, device=None):
    """
    val split'ten (train'de görülmemiş) n_samples kadar (embeddings, real_imgs,
    ) döndürür — validate()'in kullandığı AYNI split mantığıyla (build_dataloaders).
    """
    from ..data.dataset import build_dataloaders

    _, val_loader, _ = build_dataloaders(
        manifest_path=manifest_path,
        image_size=cfg.data.image_size,
        batch_size=max(n_samples, cfg.train.batch_size),
        val_split=cfg.data.val_split,
        test_split=cfg.data.test_split,
        num_workers=0,
        max_samples=cfg.data.max_samples,
    )
    embeddings, real_imgs = next(iter(val_loader))
    embeddings, real_imgs = embeddings[:n_samples], real_imgs[:n_samples]
    if device is not None:
        embeddings, real_imgs = embeddings.to(device), real_imgs.to(device)
    return embeddings, real_imgs
