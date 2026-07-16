"""
FaceDataset: Ön işlenmiş (aligned image + embedding) çiftlerini yükler.

Ön koşul: src/data/preprocess.py çalıştırılmış ve manifest.txt oluşturulmuş olmalı.
"""

import logging
import random
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

log = logging.getLogger(__name__)


# ─── Görüntü dönüşümleri ──────────────────────────────────────────────────────

def build_train_transform(image_size: int = 128) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
        transforms.ToTensor(),                          # [0, 1]
        transforms.Normalize(mean=[0.5, 0.5, 0.5],     # [-1, 1]
                             std=[0.5, 0.5, 0.5]),
    ])


def build_val_transform(image_size: int = 128) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5]),
    ])


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """[-1,1] → [0,1] dönüşümü (görselleştirme için)."""
    return (tensor * 0.5 + 0.5).clamp(0, 1)


# ─── Manifest okuma ────────────────────────────────────────────────────────────

def _load_manifest_samples(manifest_path: str) -> List[Tuple[str, str, str]]:
    """
    manifest.txt'ten (img_path, emb_path, identity) üçlülerini okur.

    identity: img_path'in üst klasör adı (örn. '.../id_048247/0007.jpg' → 'id_048247').
    preprocess.py çıktısı bu yapıyı garanti eder — her kimlik kendi klasöründe.
    """
    manifest = Path(manifest_path)
    if not manifest.exists():
        raise FileNotFoundError(
            f"Manifest bulunamadı: {manifest}\n"
            "Önce 'src/data/preprocess.py' çalıştırın."
        )

    samples: List[Tuple[str, str, str]] = []
    with open(manifest) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                continue
            img_path, emb_path = parts
            if Path(img_path).exists() and Path(emb_path).exists():
                identity = Path(img_path).parent.name
                samples.append((img_path, emb_path, identity))

    if not samples:
        raise ValueError(f"Manifest'te geçerli örnek bulunamadı: {manifest_path}")

    return samples


# ─── Dataset ──────────────────────────────────────────────────────────────────

class FaceDataset(Dataset):
    """
    (img_path, emb_path, identity) listesinden yükleyen dataset.

    __getitem__ dönüşü:
        embedding: Tensor (512,)   float32
        image:     Tensor (3, H, W) float32 — [-1, 1] normalize
    """

    def __init__(
        self,
        samples: List[Tuple[str, str, str]],
        transform: Optional[transforms.Compose] = None,
        image_size: int = 128,
    ):
        self.samples   = samples
        self.transform = transform or build_train_transform(image_size)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path, emb_path, _ = self.samples[idx]

        embedding = torch.from_numpy(np.load(emb_path)).float()

        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        return embedding, image


# ─── DataLoader fabrikası ─────────────────────────────────────────────────────

def build_dataloaders(
    manifest_path: str,
    image_size: int = 128,
    batch_size: int = 64,
    val_split: float = 0.05,
    test_split: float = 0.05,
    num_workers: int = 4,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Train / Validation / Test DataLoader'larını KİMLİK BAZLI olarak oluşturur.

    ÖNEMLİ: split örnek (görüntü) bazlı DEĞİL, kimlik bazlıdır — bir kimliğin
    tüm görüntüleri yalnızca TEK bir split'te bulunur. Bu, model
    değerlendirmesinde veri sızıntısını (leakage) önler: validation/test'te
    görülen kişiler train'de asla görülmemiş olur (gerçek "görülmemiş yüz"
    değerlendirmesi mümkün olur — Type-II identity skoru için gerekli).

    Returns:
        (train_loader, val_loader, test_loader)
        test_loader eğitim döngüsünde KULLANILMAZ — yalnızca eğitim bittikten
        sonra tek seferlik, nihai/bağımsız değerlendirme için ayrılmıştır.
    """
    all_samples = _load_manifest_samples(manifest_path)

    identities = sorted(set(s[2] for s in all_samples))
    rng = random.Random(seed)
    rng.shuffle(identities)

    n_total = len(identities)
    n_test  = max(1, int(n_total * test_split))
    n_val   = max(1, int(n_total * val_split))
    n_train = n_total - n_val - n_test

    if n_train <= 0:
        raise ValueError(
            f"val_split ({val_split}) + test_split ({test_split}) çok yüksek: "
            f"{n_total} kimlik için train'e hiç kimlik kalmadı. Oranları düşür."
        )

    train_ids = set(identities[:n_train])
    val_ids   = set(identities[n_train:n_train + n_val])
    test_ids  = set(identities[n_train + n_val:])

    train_samples = [s for s in all_samples if s[2] in train_ids]
    val_samples   = [s for s in all_samples if s[2] in val_ids]
    test_samples  = [s for s in all_samples if s[2] in test_ids]

    log.info(
        f"Kimlik bazlı split — "
        f"train: {len(train_ids):,} kimlik / {len(train_samples):,} görüntü | "
        f"val: {len(val_ids):,} kimlik / {len(val_samples):,} görüntü | "
        f"test: {len(test_ids):,} kimlik / {len(test_samples):,} görüntü"
    )

    train_ds = FaceDataset(train_samples, build_train_transform(image_size), image_size)
    val_ds   = FaceDataset(val_samples,   build_val_transform(image_size),   image_size)
    test_ds  = FaceDataset(test_samples,  build_val_transform(image_size),  image_size)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=(num_workers > 0),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=(num_workers > 0),
    )

    return train_loader, val_loader, test_loader
