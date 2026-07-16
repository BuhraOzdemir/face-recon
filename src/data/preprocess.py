"""
Veri ön işleme: yüz tespiti, hizalama ve ArcFace embedding üretimi.

Girdi: id_*/ klasörleri VEYA shard_*.tar (WebDataset tarzı)
Çıktı: processed shard'lar + manifest.txt (tar::member URI'leri)

Kullanım:
    python -m src.data.preprocess \
        --raw_dir /path/to/ms1mv3_shards \
        --out_dir /path/to/processed
"""

from __future__ import annotations

import argparse
import logging
import os
from collections import defaultdict
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from .tar_shards import (
    DEFAULT_IMAGES_PER_SHARD,
    DEFAULT_MIN_FREE_INODES,
    ShardWriter,
    TarMemberCache,
    identity_from_member,
    iter_shard_members,
    list_shard_paths,
    make_uri,
    print_disk_inode_stats,
    read_uri_bytes,
    resource_ok,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def load_insightface_app(det_size=(320, 320)):
    """
    insightface FaceAnalysis uygulamasını yükle.

    buffalo_s: MobileFaceNet backbone, 512-dim embedding.
    """
    import insightface
    from insightface.app import FaceAnalysis

    app = FaceAnalysis(
        name="buffalo_s",
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    app.prepare(ctx_id=0, det_size=det_size)
    return app


def align_and_embed(app, img_bgr: np.ndarray, output_size: int = 128):
    """
    BGR görüntüden yüzü tespit et, hizala ve ArcFace embedding üret.

    Returns:
        aligned_rgb: (output_size, output_size, 3) uint8 RGB
        embedding:   (512,) float32
        None, None:  yüz tespit edilemezse
    """
    faces = app.get(img_bgr)
    if not faces:
        return None, None

    face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    embedding = face.normed_embedding.astype(np.float32)

    aligned_bgr = _manual_align(app, img_bgr, face)
    if aligned_bgr is None:
        return None, None

    aligned_rgb = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB)
    if aligned_rgb.shape[0] != output_size:
        aligned_rgb = cv2.resize(
            aligned_rgb, (output_size, output_size), interpolation=cv2.INTER_LANCZOS4
        )

    return aligned_rgb, embedding


def _manual_align(app, img_bgr: np.ndarray, face) -> Optional[np.ndarray]:
    from insightface.utils import face_align

    kps = face.kps
    if kps is None:
        return None
    return face_align.norm_crop(img_bgr, landmark=kps, image_size=112)


def _is_shards_dir(path: Path) -> bool:
    return bool(list_shard_paths(str(path)))


def _collect_folder_samples(
    raw_path: Path, max_per_id: int
) -> List[Tuple[str, str, bytes]]:
    """
    Klasör yapısından örnekler.
    Returns list of (identity, member_stem, jpeg_bytes) — jpeg_bytes placeholder
    as path encoded... actually better return (identity, stem, source_kind, source).

    Simpler: return list of dicts.
    """
    samples = []
    identity_dirs = sorted([d for d in raw_path.iterdir() if d.is_dir()])
    for id_dir in identity_dirs:
        imgs = sorted(
            f for f in id_dir.iterdir()
            if f.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )[:max_per_id]
        for img_file in imgs:
            # member stem: label/idx without extension for processed naming
            if id_dir.name.startswith("id_"):
                try:
                    label = int(id_dir.name.split("_", 1)[1])
                except ValueError:
                    label = abs(hash(id_dir.name)) % 1_000_000
            else:
                try:
                    label = int(id_dir.name)
                except ValueError:
                    label = abs(hash(id_dir.name)) % 1_000_000
            # use original stem index if numeric else enumerate later
            samples.append(("folder", str(img_file), f"{label:06d}/{img_file.stem}"))
    return samples


def _collect_shard_samples(
    raw_path: Path, max_per_id: int
) -> List[Tuple[str, str, str]]:
    """Returns list of ('shard', uri, member_stem_without_ext)."""
    per_id: Dict[str, int] = defaultdict(int)
    samples = []
    for tar_path, member, identity in iter_shard_members(str(raw_path), suffix=".jpg"):
        if per_id[identity] >= max_per_id:
            continue
        per_id[identity] += 1
        stem = member.rsplit(".", 1)[0]  # 000042/00003
        uri = make_uri(tar_path, member)
        samples.append(("shard", uri, stem))
    return samples


def preprocess_dataset(
    raw_dir: str,
    out_dir: str,
    output_size: int = 128,
    max_per_id: int = 100,
    skip_existing: bool = True,
    images_per_shard: int = DEFAULT_IMAGES_PER_SHARD,
    min_free_gb: float = 2.0,
    min_free_inodes: int = DEFAULT_MIN_FREE_INODES,
    jpeg_quality: int = 95,
):
    """
    Ham veriyi işle → aligned images + embeddings (tar shard'lara).

    raw_dir:
        - shard_*.tar içeren klasör, VEYA
        - id_XXXXXX/ klasör yapısı (geriye dönük uyumluluk)

    out_dir:
        out_dir/shards/shard_XXXXXX.tar   # her sample: {id}/{idx}.jpg + .npy
        out_dir/manifest.txt              # img_uri\\temb_uri
    """
    raw_path = Path(raw_dir)
    out_path = Path(out_dir)
    shards_out = out_path / "shards"
    out_path.mkdir(parents=True, exist_ok=True)
    shards_out.mkdir(parents=True, exist_ok=True)

    manifest_path = out_path / "manifest.txt"
    if skip_existing and manifest_path.exists() and list_shard_paths(str(shards_out)):
        with open(manifest_path) as f:
            n = sum(1 for line in f if line.strip())
        log.info(f"Islenmis veri zaten mevcut: {manifest_path} ({n:,} ornek)")
        return str(manifest_path)

    use_shards = _is_shards_dir(raw_path)
    if use_shards:
        log.info(f"Girdi formati: tar shard ({raw_path})")
        samples = _collect_shard_samples(raw_path, max_per_id)
    else:
        log.info(f"Girdi formati: klasor ({raw_path})")
        samples = _collect_folder_samples(raw_path, max_per_id)

    log.info(f"{len(samples):,} goruntu islenecek.")
    log.info("insightface yukleniyor...")
    app = load_insightface_app()

    writer = ShardWriter(str(shards_out), images_per_shard=images_per_shard)
    cache = TarMemberCache()
    manifest_lines: List[str] = []
    total_ok, total_fail = 0, 0
    stopped_early = False

    try:
        for kind, source, stem in tqdm(samples, desc="On isleme"):
            ok, reason = resource_ok(str(shards_out), min_free_gb, min_free_inodes)
            if not ok:
                stopped_early = True
                log.warning(f"Kaynak azaldi ({reason}) — preprocess durduruldu.")
                break

            try:
                if kind == "shard":
                    raw_bytes = read_uri_bytes(source, cache=cache)
                    pil = Image.open(BytesIO(raw_bytes)).convert("RGB")
                else:
                    pil = Image.open(source).convert("RGB")
                img_bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
            except Exception:
                total_fail += 1
                continue

            aligned, emb = align_and_embed(app, img_bgr, output_size)
            if aligned is None:
                total_fail += 1
                continue

            jpg_member = f"{stem}.jpg"
            npy_member = f"{stem}.npy"

            img_uri = writer.add_jpeg_image(
                jpg_member, Image.fromarray(aligned), jpeg_quality=jpeg_quality
            )
            # npy aynı shard'a, görüntü sayacına eklenmeden
            # add_npy count_as_image=False — ama aynı shard'da kalmalı.
            # ShardWriter count_in_shard sadece jpg ile artıyor; npy aynı open tar'a gider.
            npy_uri = writer.add_npy(npy_member, emb)

            manifest_lines.append(f"{img_uri}\t{npy_uri}")
            total_ok += 1
    finally:
        writer.close()
        cache.close()

    with open(manifest_path, "w") as f:
        f.write("\n".join(manifest_lines))

    log.info(
        f"Tamamlandi: {total_ok} basarili, {total_fail} basarisiz"
        + (" (erken durdu)" if stopped_early else "")
    )
    log.info(f"Manifest: {manifest_path}")
    log.info(f"Shard sayisi: {len(writer.shard_paths)}")
    print_disk_inode_stats(str(out_path.parent if out_path.parent.exists() else out_path))
    return str(manifest_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Yüz veri seti ön işleme")
    parser.add_argument("--raw_dir", required=True, help="Ham görüntü / shard klasörü")
    parser.add_argument("--out_dir", required=True, help="Çıktı klasörü")
    parser.add_argument("--output_size", default=128, type=int)
    parser.add_argument("--max_per_id", default=100, type=int)
    parser.add_argument("--no_skip", action="store_true")
    args = parser.parse_args()

    preprocess_dataset(
        raw_dir=args.raw_dir,
        out_dir=args.out_dir,
        output_size=args.output_size,
        max_per_id=args.max_per_id,
        skip_existing=not args.no_skip,
    )
