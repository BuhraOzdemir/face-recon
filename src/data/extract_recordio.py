"""
MS1MV3 (ve genel InsightFace) RecordIO → WebDataset tarzı tar shard çıkarımı.

Kaggle /kaggle/working inode limiti (~1.3M) nedeniyle her görüntü ayrı
JPEG dosyası olarak YAZILMAZ. Bunun yerine sabit sayıda shard'a yazılır:

    extracted_dir/
        shard_000000.tar
        shard_000001.tar
        ...
    her tar içi: {label_id:06d}/{img_idx:05d}.jpg

RecordIO Kayıt Yapısı (MXNet / InsightFace, flag>0 durumu dahil):
    Prefix   8B : magic(4) + length(4)
    IRHeader 24B: flag(4) + base_lbl(4) + id(8) + id2(8)
    Extras  flag*4 B: extra_lbl0(4) ...
    JPEG      ?B

Sequential okuma kullanılır (seek YOK).
"""

from __future__ import annotations

import os
import struct
from io import BytesIO
from pathlib import Path

from PIL import Image
from tqdm.auto import tqdm

from .tar_shards import (
    DEFAULT_IMAGES_PER_SHARD,
    DEFAULT_MIN_FREE_INODES,
    ShardWriter,
    count_shard_stats,
    list_shard_paths,
    print_disk_inode_stats,
    resource_ok,
)

_PREFIX_SZ = 8
_IRHDR_SZ = 24
_RESOURCE_CHECK_EVERY = 500


def extract_ms1mv3(
    rec_path: str,
    extracted_dir: str,
    max_per_id: int = 15,
    max_ids: int = 93000,
    max_samples: int = 100_000,
    skip_if_exists: bool = True,
    min_free_gb: float = 2.0,
    min_free_inodes: int = DEFAULT_MIN_FREE_INODES,
    jpeg_quality: int = 90,
    images_per_shard: int = DEFAULT_IMAGES_PER_SHARD,
) -> dict:
    """
    MS1MV3 .rec dosyasını tar shard'lara çıkarır (ayrı JPEG dosyası YOK).

    Args:
        rec_path:          .rec yolu
        extracted_dir:     çıktı klasörü (örn. /kaggle/working/ms1mv3_shards)
        max_per_id:        kimlik başına max görüntü
        max_ids:           işlenecek max kimlik (class_id üst sınırı)
        max_samples:       toplam görüntü üst sınırı (0=sınırsız)
        skip_if_exists:    shard_*.tar zaten varsa atla
        min_free_gb:       disk eşiği — altında güvenli dur
        min_free_inodes:   inode eşiği — altında güvenli dur
        jpeg_quality:      JPEG kalitesi
        images_per_shard:  shard başına görüntü sayısı (~5000-10000)

    Returns:
        dict: n_ids, n_images, n_errors, n_shards, data_dir, stopped_early
    """
    existing = list_shard_paths(extracted_dir)
    if skip_if_exists and existing:
        stats = count_shard_stats(extracted_dir)
        print(
            f"Zaten cikarilmis: {extracted_dir}  "
            f"({stats['n_shards']} shard, {stats['n_images']:,} goruntu, "
            f"{stats['n_ids']:,} kimlik)"
        )
        return {
            "n_ids": stats["n_ids"],
            "n_images": stats["n_images"],
            "n_errors": 0,
            "n_shards": stats["n_shards"],
            "data_dir": extracted_dir,
            "stopped_early": False,
        }

    os.makedirs(extracted_dir, exist_ok=True)
    writer = ShardWriter(extracted_dir, images_per_shard=images_per_shard)

    id_counts: dict = {}
    errors = 0
    rec_idx = 0
    stopped_early = False
    stop_reason = ""

    file_size = os.path.getsize(rec_path)
    print(f"Rec dosyasi: {file_size / 1e9:.1f} GB — sequential okuma basliyor...")
    print(
        f"Cikti: tar shard (her biri ~{images_per_shard} goruntu) → {extracted_dir}"
    )
    print(
        f"Guvenlik: disk<{min_free_gb:.0f}GB veya inode<{min_free_inodes} "
        f"kalinca otomatik duracak."
    )

    try:
        with open(rec_path, "rb") as f:
            with tqdm(
                total=file_size,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc="Okunan",
            ) as pbar:

                while True:
                    pos_start = f.tell()

                    prefix = f.read(_PREFIX_SZ)
                    if len(prefix) < _PREFIX_SZ:
                        break
                    _, length = struct.unpack("<II", prefix)
                    if length == 0 or length > 5_000_000:
                        break

                    data = f.read(length)
                    if len(data) < length:
                        break
                    pad = (4 - (length % 4)) % 4
                    if pad:
                        f.seek(pad, 1)

                    pbar.update(f.tell() - pos_start)
                    rec_idx += 1

                    if rec_idx == 1:
                        continue
                    if len(data) < _IRHDR_SZ:
                        continue

                    flag = struct.unpack("<i", data[0:4])[0]

                    if 0 < flag <= 16:
                        extra_sz = flag * 4
                        if len(data) < _IRHDR_SZ + extra_sz:
                            continue
                        label_id = int(
                            struct.unpack("<f", data[_IRHDR_SZ:_IRHDR_SZ + 4])[0]
                        )
                        img_start = _IRHDR_SZ + extra_sz
                    else:
                        label_id = int(struct.unpack("<f", data[4:8])[0])
                        img_start = _IRHDR_SZ

                    if label_id < 0 or label_id >= max_ids:
                        continue
                    cnt = id_counts.get(label_id, 0)
                    if cnt >= max_per_id:
                        continue

                    img_bytes = data[img_start:]
                    if len(img_bytes) < 50:
                        continue

                    if writer.n_written % _RESOURCE_CHECK_EVERY == 0:
                        ok, reason = resource_ok(
                            extracted_dir, min_free_gb, min_free_inodes
                        )
                        if not ok:
                            stopped_early = True
                            stop_reason = reason
                            print(
                                f"\n[DUR] Kaynak azaldi ({reason}). "
                                f"Shard kapatilip guvenli cikiliyor."
                            )
                            break

                    try:
                        img = Image.open(BytesIO(img_bytes)).convert("RGB")
                        member = f"{label_id:06d}/{cnt:05d}.jpg"
                        writer.add_jpeg_image(
                            member, img, jpeg_quality=jpeg_quality
                        )
                        id_counts[label_id] = cnt + 1
                    except Exception:
                        errors += 1
                        continue

                    if max_samples and max_samples > 0 and writer.n_written >= max_samples:
                        stopped_early = True
                        stop_reason = f"max_samples={max_samples}"
                        print(
                            f"\n[DUR] max_samples={max_samples:,} ulasildi. "
                            f"Shard kapatilip guvenli cikiliyor."
                        )
                        break
    finally:
        writer.close()

    if stopped_early:
        print(f"\nErken durdu ({stop_reason}) — yazilan shard'lar gecerli.")
    else:
        print("\nTamamlandi!")

    print(f"  Kimlik  : {len(id_counts):,}")
    print(f"  Goruntu : {writer.n_written:,}")
    print(f"  Shard   : {len(writer.shard_paths):,}")
    print(f"  Hata    : {errors:,}")
    print_disk_inode_stats(str(Path(extracted_dir).parent))

    return {
        "n_ids": len(id_counts),
        "n_images": writer.n_written,
        "n_errors": errors,
        "n_shards": len(writer.shard_paths),
        "data_dir": extracted_dir,
        "stopped_early": stopped_early,
    }


def extract_ms1mv3_flat(
    rec_path: str,
    extracted_dir: str,
    max_per_id: int = 15,
    max_ids: int = 46500,
    skip_if_exists: bool = True,
    min_free_gb: float = 2.0,
    min_free_inodes: int = DEFAULT_MIN_FREE_INODES,
    jpeg_quality: int = 90,
) -> dict:
    """
    MS1MV3 .rec dosyasini DUZ KLASOR/JPG olarak cikarir (tar YOK).
    extracted_dir/{label_id:06d}/{img_idx:05d}.jpg

    Inode-guvenli kullanim icin max_ids * max_per_id + max_ids
    (klasor sayisi) toplami, min_free_inodes payi birakacak sekilde
    Kaggle'in inode limitinin (~1.3M) altinda tutulmalidir.
    """
    out_dir = Path(extracted_dir)

    if skip_if_exists and out_dir.exists():
        existing_ids = [d for d in out_dir.iterdir() if d.is_dir()]
        if existing_ids:
            n_images = sum(len(list(d.glob("*.jpg"))) for d in existing_ids)
            print(
                f"Zaten cikarilmis: {extracted_dir}  "
                f"({len(existing_ids):,} kimlik, {n_images:,} goruntu)"
            )
            return {
                "n_ids": len(existing_ids),
                "n_images": n_images,
                "n_errors": 0,
                "n_shards": 0,
                "data_dir": extracted_dir,
                "stopped_early": False,
            }

    out_dir.mkdir(parents=True, exist_ok=True)

    id_counts: dict = {}
    errors = 0
    n_written = 0
    rec_idx = 0
    stopped_early = False
    stop_reason = ""

    file_size = os.path.getsize(rec_path)
    print(f"Rec dosyasi: {file_size / 1e9:.1f} GB — sequential okuma basliyor...")
    print(f"Cikti: duz klasor/jpg → {extracted_dir}")
    print(
        f"Guvenlik: disk<{min_free_gb:.0f}GB veya inode<{min_free_inodes} "
        f"kalinca otomatik duracak."
    )

    try:
        with open(rec_path, "rb") as f:
            with tqdm(
                total=file_size,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc="Okunan",
            ) as pbar:

                while True:
                    pos_start = f.tell()

                    prefix = f.read(_PREFIX_SZ)
                    if len(prefix) < _PREFIX_SZ:
                        break
                    _, length = struct.unpack("<II", prefix)
                    if length == 0 or length > 5_000_000:
                        break

                    data = f.read(length)
                    if len(data) < length:
                        break
                    pad = (4 - (length % 4)) % 4
                    if pad:
                        f.seek(pad, 1)

                    pbar.update(f.tell() - pos_start)
                    rec_idx += 1

                    if rec_idx == 1:
                        continue
                    if len(data) < _IRHDR_SZ:
                        continue

                    flag = struct.unpack("<i", data[0:4])[0]

                    if 0 < flag <= 16:
                        extra_sz = flag * 4
                        if len(data) < _IRHDR_SZ + extra_sz:
                            continue
                        label_id = int(
                            struct.unpack("<f", data[_IRHDR_SZ:_IRHDR_SZ + 4])[0]
                        )
                        img_start = _IRHDR_SZ + extra_sz
                    else:
                        label_id = int(struct.unpack("<f", data[4:8])[0])
                        img_start = _IRHDR_SZ

                    if label_id < 0 or label_id >= max_ids:
                        continue
                    cnt = id_counts.get(label_id, 0)
                    if cnt >= max_per_id:
                        continue

                    img_bytes = data[img_start:]
                    if len(img_bytes) < 50:
                        continue

                    if n_written % _RESOURCE_CHECK_EVERY == 0:
                        ok, reason = resource_ok(
                            extracted_dir, min_free_gb, min_free_inodes
                        )
                        if not ok:
                            stopped_early = True
                            stop_reason = reason
                            print(
                                f"\n[DUR] Kaynak azaldi ({reason}). "
                                f"Guvenli cikiliyor."
                            )
                            break

                    try:
                        img = Image.open(BytesIO(img_bytes)).convert("RGB")
                        id_dir = out_dir / f"{label_id:06d}"
                        id_dir.mkdir(exist_ok=True)
                        img_path = id_dir / f"{cnt:05d}.jpg"
                        img.save(img_path, "JPEG", quality=jpeg_quality)
                        id_counts[label_id] = cnt + 1
                        n_written += 1
                    except Exception:
                        errors += 1
                        continue
    finally:
        pass

    if stopped_early:
        print(f"\nErken durdu ({stop_reason}) — yazilan goruntuler gecerli.")
    else:
        print("\nTamamlandi!")

    print(f"  Kimlik  : {len(id_counts):,}")
    print(f"  Goruntu : {n_written:,}")
    print(f"  Hata    : {errors:,}")
    print_disk_inode_stats(str(out_dir.parent))

    return {
        "n_ids": len(id_counts),
        "n_images": n_written,
        "n_errors": errors,
        "n_shards": 0,
        "data_dir": extracted_dir,
        "stopped_early": stopped_early,
    }