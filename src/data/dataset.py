"""
FaceDataset: Ön işlenmiş (aligned image + embedding) çiftlerini yükler.

Ön koşul: src/data/preprocess.py çalıştırılmış ve manifest.txt oluşturulmuş olmalı.
"""

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms


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


# ─── Dataset ──────────────────────────────────────────────────────────────────

class FaceDataset(Dataset):
    """
    manifest.txt'ten (img_path, emb_path) çiftlerini okur.

    __getitem__ dönüşü:
        embedding: Tensor (512,)   float32
        image:     Tensor (3, H, W) float32 — [-1, 1] normalize
    """

    def __init__(
        self,
        manifest_path: str,
        transform: Optional[transforms.Compose] = None,
        image_size: int = 128,
    ):
        self.transform = transform or build_train_transform(image_size)
        self.samples: list[Tuple[str, str]] = []

        manifest = Path(manifest_path)
        if not manifest.exists():
            raise FileNotFoundError(
                f"Manifest bulunamadı: {manifest}\n"
                "Önce 'src/data/preprocess.py' çalıştırın."
            )

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
                    self.samples.append((img_path, emb_path))

        if not self.samples:
            raise ValueError(f"Manifest'te geçerli örnek bulunamadı: {manifest_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path, emb_path = self.samples[idx]

        # Embedding yükle
        embedding = torch.from_numpy(np.load(emb_path)).float()

        # Görüntü yükle ve dönüştür
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        return embedding, image


# ─── DataLoader fabrikası ─────────────────────────────────────────────────────

def build_dataloaders(
    manifest_path: str,
    image_size: int = 128,
    batch_size: int = 64,
    val_split: float = 0.05,
    num_workers: int = 4,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """
    Train ve validation DataLoader'larını oluşturur.

    Val split kimlik bazlı değil örnek bazlıdır (basitlik için).
    Küçük veri setlerinde yeterli; büyük veri setlerinde kimlik bazlı split önerilir.
    """
    full_dataset = FaceDataset(
        manifest_path=manifest_path,
        transform=build_train_transform(image_size),
        image_size=image_size,
    )

    n_total = len(full_dataset)
    n_val   = max(1, int(n_total * val_split))
    n_train = n_total - n_val

    train_ds, val_ds = random_split(
        full_dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(seed),
    )

    # Val için augmentation kapalı transform
    val_ds.dataset = FaceDataset(
        manifest_path=manifest_path,
        transform=build_val_transform(image_size),
        image_size=image_size,
    )
    # NOT: random_split, dataset'i sarmalıyor, sadece indeksleri böler.
    # Val transform'u doğrudan uygulamak için Subset wrapper kullanıyoruz:
    train_ds = _TransformSubset(full_dataset, train_ds.indices,
                                 build_train_transform(image_size))
    val_ds   = _TransformSubset(full_dataset, val_ds.indices,
                                 build_val_transform(image_size))

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

    return train_loader, val_loader


class _TransformSubset(Dataset):
    """Subset'e özel transform uygulayan yardımcı sınıf."""

    def __init__(self, source_dataset: FaceDataset, indices: list, transform: transforms.Compose):
        self.source    = source_dataset
        self.indices   = indices
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        real_idx = self.indices[idx]
        img_path, emb_path = self.source.samples[real_idx]

        embedding = torch.from_numpy(np.load(emb_path)).float()
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        return embedding, image
