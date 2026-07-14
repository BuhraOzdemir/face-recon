"""
Veri ön işleme: yüz tespiti, hizalama ve ArcFace embedding üretimi.

Kullanım:
    python -m src.data.preprocess \
        --raw_dir /path/to/raw \
        --out_dir /path/to/processed \
        --max_per_id 100
"""

import os
import argparse
import logging
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def load_insightface_app(det_size=(320, 320)):
    """
    insightface FaceAnalysis uygulamasını yükle.

    buffalo_s: MobileFaceNet backbone, 512-dim embedding.
    Bu model hem eğitimde hem telefonda kullanılır → embedding uzayı tutarlı.
    (buffalo_l ArcFace R50 ile eğitilirdi ama telefonda MobileFaceNet çalışır
     → embedding dağılımı uyuşmazdı.)
    """
    import insightface
    from insightface.app import FaceAnalysis

    app = FaceAnalysis(
        name="buffalo_s",   # MobileFaceNet — telefon ile aynı encoder
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    app.prepare(ctx_id=0, det_size=det_size)
    return app


def align_and_embed(app, img_bgr: np.ndarray, output_size: int = 128):
    """
    BGR görüntüden yüzü tespit et, hizala ve ArcFace embedding üret.

    Returns:
        aligned_rgb: (output_size, output_size, 3) uint8 RGB görüntü
        embedding:   (512,) float32 ArcFace embedding
        None, None:  yüz tespit edilemezse
    """
    faces = app.get(img_bgr)
    if not faces:
        return None, None

    # En büyük yüzü seç
    face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))

    embedding = face.normed_embedding.astype(np.float32)  # (512,) L2-normalized

    # insightface 112×112 kırpılmış yüz döndürür
    # norm_crop → aligned BGR 112×112
    aligned_bgr = face.get_aligned_face() if hasattr(face, "get_aligned_face") else _manual_align(app, img_bgr, face)

    if aligned_bgr is None:
        return None, None

    # BGR → RGB, 112→output_size resize
    aligned_rgb = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB)
    if aligned_rgb.shape[0] != output_size:
        aligned_rgb = cv2.resize(aligned_rgb, (output_size, output_size), interpolation=cv2.INTER_LANCZOS4)

    return aligned_rgb, embedding


def _manual_align(app, img_bgr: np.ndarray, face) -> np.ndarray:
    """insightface norm_crop yardımcı fonksiyonu."""
    from insightface.utils import face_align
    kps = face.kps
    if kps is None:
        return None
    aligned = face_align.norm_crop(img_bgr, landmark=kps, image_size=112)
    return aligned


def preprocess_dataset(
    raw_dir: str,
    out_dir: str,
    output_size: int = 128,
    max_per_id: int = 100,
    skip_existing: bool = True,
):
    """
    Ham veri klasörünü işle → aligned images + embeddings kaydet.

    Beklenen raw_dir yapısı:
        raw_dir/
            identity_001/
                img001.jpg
                img002.jpg
            identity_002/
                ...

    Çıktı out_dir yapısı:
        out_dir/
            identity_001/
                img001.npy   # (embedding_dim,) float32
                img001.jpg   # aligned RGB image
            ...
        out_dir/manifest.txt  # başarıyla işlenmiş (img_path, emb_path) çiftleri
    """
    raw_path = Path(raw_dir)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    log.info("insightface yükleniyor...")
    app = load_insightface_app()

    identity_dirs = sorted([d for d in raw_path.iterdir() if d.is_dir()])
    log.info(f"{len(identity_dirs)} kimlik bulundu.")

    manifest_lines = []
    total_ok, total_fail = 0, 0

    for id_dir in tqdm(identity_dirs, desc="Kimlikler"):
        img_files = sorted(
            [f for f in id_dir.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        )[:max_per_id]

        if not img_files:
            continue

        out_id_dir = out_path / id_dir.name
        out_id_dir.mkdir(exist_ok=True)

        count = 0
        for img_file in img_files:
            out_img = out_id_dir / img_file.name
            out_emb = out_id_dir / (img_file.stem + ".npy")

            if skip_existing and out_img.exists() and out_emb.exists():
                manifest_lines.append(f"{out_img}\t{out_emb}")
                count += 1
                continue

            img_bgr = cv2.imread(str(img_file))
            if img_bgr is None:
                total_fail += 1
                continue

            aligned, emb = align_and_embed(app, img_bgr, output_size)
            if aligned is None:
                total_fail += 1
                continue

            # Kaydet
            Image.fromarray(aligned).save(str(out_img), quality=95)
            np.save(str(out_emb), emb)

            manifest_lines.append(f"{out_img}\t{out_emb}")
            count += 1
            total_ok += 1

    # Manifest yaz
    manifest_path = out_path / "manifest.txt"
    with open(manifest_path, "w") as f:
        f.write("\n".join(manifest_lines))

    log.info(f"Tamamlandı: {total_ok} başarılı, {total_fail} başarısız.")
    log.info(f"Manifest: {manifest_path}")
    return str(manifest_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Yüz veri seti ön işleme")
    parser.add_argument("--raw_dir",     required=True,   help="Ham görüntü klasörü")
    parser.add_argument("--out_dir",     required=True,   help="Çıktı klasörü")
    parser.add_argument("--output_size", default=128,     type=int)
    parser.add_argument("--max_per_id",  default=100,     type=int)
    parser.add_argument("--no_skip",     action="store_true")
    args = parser.parse_args()

    preprocess_dataset(
        raw_dir=args.raw_dir,
        out_dir=args.out_dir,
        output_size=args.output_size,
        max_per_id=args.max_per_id,
        skip_existing=not args.no_skip,
    )
