"""
FaceDataset: Ön işlenmiş (aligned image + embedding) çiftlerini yükler.

Manifest satırı:
    img_uri\\temb_uri

URI düz dosya yolu olabilir veya tar member:
    /path/shard_000001.tar::000042/00003.jpg
"""

from __future__ import annotations

import io
import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from concurrent.futures import ThreadPoolExecutor

from .tar_shards import (
    TarMemberCache,
    identity_from_member,
    parse_uri,
    read_uri_bytes,
    uri_exists,
)

log = logging.getLogger(__name__)


# ─── Görüntü dönüşümleri ──────────────────────────────────────────────────────

def build_train_transform(image_size: int = 128) -> transforms.Compose:
    # NOT: RandomHorizontalFlip KASITLI OLARAK YOK. embedding, preprocess
    # sırasında orijinal (aynalanmamış) görüntüden hesaplanıp .npy'den SABİT
    # yükleniyor (FaceDataset.__getitem__) — ama transform SADECE image'e
    # uygulanıyor. Flip açıksa, eğitim adımlarının ~%50'sinde AYNI z'ye
    # yüzün aynalanmış hali hedef gösteriliyordu; ArcFace-ailesi embedding'ler
    # yön/poz bilgisini güvenilir taşımadığı için decoder bu iki çelişen
    # hedef arasında uzlaşmaya (sol/sağ tutarsız, bölünmüş yüz çıktısı)
    # zorlanıyordu. Kaldırılması sıfırdan eğitim gerektirir.
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


def build_val_transform(image_size: int = 128) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """[-1,1] → [0,1] dönüşümü (görselleştirme için)."""
    return (tensor * 0.5 + 0.5).clamp(0, 1)


# ─── Manifest okuma ────────────────────────────────────────────────────────────

def _identity_from_uri(uri: str) -> str:
    tar_path, member = parse_uri(uri)
    if tar_path is not None:
        return identity_from_member(member)
    return Path(member).parent.name


def _load_manifest_samples(manifest_path: str) -> List[Tuple[str, str, str]]:
    """
    manifest.txt'ten (img_uri, emb_uri, identity) üçlülerini okur.

    identity: tar member klasöründen veya dosya yolunun üst klasöründen.

    Not: exists() kontrolu /kaggle/input gibi FUSE-tabanli salt-okunur
    mount'larda dosya basina birkaç ms surebiliyor; yuzbinlerce satirda
    bu tek-thread'li kontrolu onlarca dakikaya cikariyor. ThreadPoolExecutor
    ile I/O-bound bu kontrolleri paralellestiriyoruz (GIL, I/O bekleme
    sirasinda serbest kaldigi icin bu gercek bir hizlanma saglar).
    """
    manifest = Path(manifest_path)
    if not manifest.exists():
        raise FileNotFoundError(
            f"Manifest bulunamadı: {manifest}\n"
            "Önce 'src/data/preprocess.py' çalıştırın."
        )

    lines: List[Tuple[str, str]] = []
    with open(manifest) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                continue
            lines.append((parts[0], parts[1]))

    def _check(pair):
        img_uri, emb_uri = pair
        if uri_exists(img_uri) and uri_exists(emb_uri):
            identity = _identity_from_uri(img_uri)
            return (img_uri, emb_uri, identity)
        return None

    samples: List[Tuple[str, str, str]] = []
    with ThreadPoolExecutor(max_workers=32) as executor:
        for result in executor.map(_check, lines):
            if result is not None:
                samples.append(result)

    if not samples:
        raise ValueError(f"Manifest'te geçerli örnek bulunamadı: {manifest_path}")

    return samples


def _cap_samples_by_identity(
    samples: List[Tuple[str, str, str]],
    max_samples: Optional[int],
    seed: int = 42,
) -> List[Tuple[str, str, str]]:
    """
    Toplam örnek sayısını max_samples ile sınırlar.
    Kimlikleri karıştırıp kimlik kimlik ekler; son kimlikte gerekirse kırpılır.
    """
    if max_samples is None or max_samples <= 0 or len(samples) <= max_samples:
        return samples

    by_id: dict = defaultdict(list)
    for s in samples:
        by_id[s[2]].append(s)

    rng = random.Random(seed)
    identities = list(by_id.keys())
    rng.shuffle(identities)

    selected: List[Tuple[str, str, str]] = []
    for iid in identities:
        imgs = by_id[iid]
        rng.shuffle(imgs)
        remaining = max_samples - len(selected)
        if remaining <= 0:
            break
        selected.extend(imgs[:remaining])

    log.info(
        f"max_samples={max_samples:,}: {len(samples):,} → {len(selected):,} örnek "
        f"({len(set(s[2] for s in selected)):,} kimlik)"
    )
    return selected


# ─── Dataset ──────────────────────────────────────────────────────────────────

class FaceDataset(Dataset):
    """
    (img_uri, emb_uri, identity) listesinden yükleyen dataset.

    __getitem__ dönüşü:
        embedding: Tensor (512,)
        image:     Tensor (3, H, W) — [-1, 1]
    """

    def __init__(
        self,
        samples: List[Tuple[str, str, str]],
        transform: Optional[transforms.Compose] = None,
        image_size: int = 128,
    ):
        self.samples = samples
        self.transform = transform or build_train_transform(image_size)
        # DataLoader worker'larında lazy init (process-local)
        self._cache: Optional[TarMemberCache] = None

    def _get_cache(self) -> TarMemberCache:
        if self._cache is None:
            self._cache = TarMemberCache()
        return self._cache

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_uri, emb_uri, _ = self.samples[idx]
        cache = self._get_cache()

        emb_bytes = read_uri_bytes(emb_uri, cache=cache)
        embedding = torch.from_numpy(
            np.load(io.BytesIO(emb_bytes), allow_pickle=False)
        ).float()

        img_bytes = read_uri_bytes(img_uri, cache=cache)
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
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
    max_samples: int = 100_000,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Train / Validation / Test DataLoader'larını KİMLİK BAZLI olarak oluşturur.

    Split örnek bazlı DEĞİL, kimlik bazlıdır — bir kimliğin tüm görüntüleri
    yalnızca TEK bir split'te bulunur.

    max_samples: toplam (train+val+test) üst sınırı; aşılırsa kimlik bazlı kırpılır.
    """
    all_samples = _load_manifest_samples(manifest_path)
    all_samples = _cap_samples_by_identity(all_samples, max_samples, seed=seed)

    identities = sorted(set(s[2] for s in all_samples))
    rng = random.Random(seed)
    rng.shuffle(identities)

    n_total = len(identities)
    n_test = max(1, int(n_total * test_split))
    n_val = max(1, int(n_total * val_split))
    n_train = n_total - n_val - n_test

    if n_train <= 0:
        raise ValueError(
            f"val_split ({val_split}) + test_split ({test_split}) çok yüksek: "
            f"{n_total} kimlik için train'e hiç kimlik kalmadı. Oranları düşür."
        )

    train_ids = set(identities[:n_train])
    val_ids = set(identities[n_train:n_train + n_val])
    test_ids = set(identities[n_train + n_val:])

    train_samples = [s for s in all_samples if s[2] in train_ids]
    val_samples = [s for s in all_samples if s[2] in val_ids]
    test_samples = [s for s in all_samples if s[2] in test_ids]

    log.info(
        f"Kimlik bazlı split — "
        f"train: {len(train_ids):,} kimlik / {len(train_samples):,} görüntü | "
        f"val: {len(val_ids):,} kimlik / {len(val_samples):,} görüntü | "
        f"test: {len(test_ids):,} kimlik / {len(test_samples):,} görüntü | "
        f"toplam: {len(all_samples):,} (max_samples={max_samples})"
    )

    train_ds = FaceDataset(train_samples, build_train_transform(image_size), image_size)
    val_ds = FaceDataset(val_samples, build_val_transform(image_size), image_size)
    test_ds = FaceDataset(test_samples, build_val_transform(image_size), image_size)

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
