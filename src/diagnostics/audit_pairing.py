"""
1/5 — audit_pairing.py: Veri eşleşme (img↔embedding) denetimi.

Manifest.txt'teki her satır (img_uri, emb_uri) bir çift oluşturur.
Bu script, kayıtlı .npy embedding'in GERÇEKTEN o JPG'ye ait olup olmadığını
doğrular: aligned JPG'yi insightface recognition pipeline'ından TEKRAR
geçirip, yeni embedding ile kayıtlı embedding arasındaki cosine similarity'yi
ölçer.

ÖNEMLİ METODOLOJİK NOT: preprocess.py'de kayıtlı embedding, orijinal HAM
görüntü üzerinde `app.get()`'in kendi iç 112×112 hizalamasıyla üretiliyor;
manifest'e YAZILAN JPG ise AYRI bir çağrıyla (`_manual_align`, aynı 5-nokta
şablonu, image_size=128) üretilmiş 128×128 crop. Bu crop'u yeniden
`app.get()`'ten geçirmek küçük bir yeniden-tespit/yeniden-hizalama sapması
getirir — bu yüzden beklenen eşik ~1.0 DEĞİL, >0.90 gibi YÜKSEK bir
cosine'dir. Buna göre eşik seçildi (SUSPECT_THRESHOLD).

Kullanım (Kaggle/Colab, insightface kurulu ortamda):
    from src.audit_pairing import run_pairing_audit
    run_pairing_audit(manifest_path="/path/manifest.txt")
"""

import io
import logging
import random
import tarfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import List, Tuple

import numpy as np

from ..data.tar_shards import parse_uri, read_uri_bytes, list_shard_paths

log = logging.getLogger(__name__)

SUSPECT_THRESHOLD = 0.90  # bkz. dosya başındaki metodolojik not


def _read_manifest_lines(manifest_path: str) -> List[Tuple[str, str]]:
    lines = []
    with open(manifest_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) == 2:
                lines.append((parts[0], parts[1]))
    return lines


def _check_duplicate_manifest_lines(lines: List[Tuple[str, str]]) -> dict:
    counts = Counter(lines)
    dupes = {k: v for k, v in counts.items() if v > 1}
    return {"n_duplicate_lines": len(dupes), "examples": list(dupes.items())[:10]}


def _check_embedding_reuse(lines: List[Tuple[str, str]]) -> dict:
    """Bir emb_uri, birden fazla FARKLI img_uri'ye bağlıysa şüpheli (embedding kopyası/yanlış eşleşme)."""
    by_emb = defaultdict(set)
    for img_uri, emb_uri in lines:
        by_emb[emb_uri].add(img_uri)
    reused = {emb: imgs for emb, imgs in by_emb.items() if len(imgs) > 1}
    return {"n_reused_embeddings": len(reused), "examples": list(reused.items())[:10]}


def _check_duplicate_tar_keys(lines: List[Tuple[str, str]]) -> dict:
    """Manifestte referans verilen her tar shard içinde tekrar eden member adı var mı?"""
    tar_paths = set()
    for img_uri, emb_uri in lines:
        for uri in (img_uri, emb_uri):
            tp, _ = parse_uri(uri)
            if tp is not None:
                tar_paths.add(tp)

    result = {}
    for tp in sorted(tar_paths):
        if not Path(tp).exists():
            result[tp] = {"error": "tar dosyası yok"}
            continue
        try:
            with tarfile.open(tp, "r") as tf:
                names = tf.getnames()
            counts = Counter(names)
            dupes = {n: c for n, c in counts.items() if c > 1}
            result[tp] = {"n_members": len(names), "n_duplicate_members": len(dupes),
                          "examples": list(dupes.items())[:5]}
        except (tarfile.TarError, OSError) as e:
            result[tp] = {"error": str(e)}
    return result


def _check_missing_files(sample_lines: List[Tuple[str, str]]) -> dict:
    """SADECE örneklenen alt küme için 'deep' (tar içi member) kontrolü — pahalı, tüm manifest için değil."""
    missing_img, missing_emb = [], []
    for img_uri, emb_uri in sample_lines:
        try:
            read_uri_bytes(img_uri)
        except (FileNotFoundError, tarfile.TarError, OSError, KeyError):
            missing_img.append(img_uri)
        try:
            read_uri_bytes(emb_uri)
        except (FileNotFoundError, tarfile.TarError, OSError, KeyError):
            missing_emb.append(emb_uri)
    return {"missing_img": missing_img, "missing_emb": missing_emb}


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-8)
    b = b / (np.linalg.norm(b) + 1e-8)
    return float(np.dot(a, b))


def _reembed_and_compare(app, sample_lines: List[Tuple[str, str]]) -> List[dict]:
    """Her (img_uri, emb_uri) için: JPG'yi yeniden embed et, kayıtlı .npy ile cosine kıyasla."""
    import cv2
    from PIL import Image

    rows = []
    for img_uri, emb_uri in sample_lines:
        row = {"img_uri": img_uri, "emb_uri": emb_uri}
        try:
            img_bytes = read_uri_bytes(img_uri)
            pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            img_bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

            emb_bytes = read_uri_bytes(emb_uri)
            stored_emb = np.load(io.BytesIO(emb_bytes), allow_pickle=False)

            faces = app.get(img_bgr)
            if not faces:
                row["status"] = "TESPIT_HATASI"
                row["cosine"] = None
                rows.append(row)
                continue

            face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
            new_emb = face.normed_embedding.astype(np.float32)

            cos = _cosine(stored_emb, new_emb)
            row["cosine"] = cos
            row["status"] = "OK" if cos > SUSPECT_THRESHOLD else "SUPHELI"
        except Exception as e:
            row["status"] = f"HATA: {e}"
            row["cosine"] = None
        rows.append(row)
    return rows


def run_pairing_audit(manifest_path: str, n_samples: int = 100, seed: int = 0) -> dict:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    random.seed(seed)

    log.info("=" * 90)
    log.info("AUDIT_PAIRING — Veri eşleşme denetimi")
    log.info("=" * 90)

    all_lines = _read_manifest_lines(manifest_path)
    log.info(f"Manifest toplam satır: {len(all_lines):,}")

    dup_report = _check_duplicate_manifest_lines(all_lines)
    log.info(f"[Duplicate manifest satırı] {dup_report['n_duplicate_lines']} tekrar eden satır")

    reuse_report = _check_embedding_reuse(all_lines)
    log.info(
        f"[Embedding yeniden kullanımı] {reuse_report['n_reused_embeddings']} embedding, "
        f"birden fazla FARKLI img_uri'ye bağlı"
    )

    tar_report = _check_duplicate_tar_keys(all_lines)
    n_tar_with_dupes = sum(1 for v in tar_report.values() if v.get("n_duplicate_members", 0) > 0)
    n_tar_errors = sum(1 for v in tar_report.values() if "error" in v)
    log.info(
        f"[Tar shard bütünlüğü] {len(tar_report)} shard tarandı, "
        f"{n_tar_with_dupes} shard'da duplicate member adı, {n_tar_errors} shard açılamadı"
    )

    sample = random.sample(all_lines, min(n_samples, len(all_lines)))
    missing_report = _check_missing_files(sample)
    log.info(
        f"[Eksik dosya — {len(sample)} örnek] "
        f"{len(missing_report['missing_img'])} eksik JPG, {len(missing_report['missing_emb'])} eksik NPY"
    )

    reembed_rows = None
    try:
        from ..data.preprocess import load_insightface_app
        app = load_insightface_app()
        usable_sample = [
            (i, e) for i, e in sample
            if i not in missing_report["missing_img"] and e not in missing_report["missing_emb"]
        ]
        reembed_rows = _reembed_and_compare(app, usable_sample)
    except Exception as e:
        log.warning(
            f"[Re-embed] insightface yüklenemedi/çalışmadı ({e}) — cosine karşılaştırması ATLANDI. "
            f"Bu adım Kaggle/Colab'da (insightface kurulu) çalıştırılmalı."
        )

    summary = {
        "n_manifest_lines": len(all_lines),
        "duplicate_lines": dup_report,
        "embedding_reuse": reuse_report,
        "tar_integrity": tar_report,
        "missing_files": missing_report,
    }

    if reembed_rows is not None:
        cosines = [r["cosine"] for r in reembed_rows if r["cosine"] is not None]
        n_ok = sum(1 for r in reembed_rows if r["status"] == "OK")
        n_suspect = sum(1 for r in reembed_rows if r["status"] == "SUPHELI")
        n_detect_fail = sum(1 for r in reembed_rows if r["status"] == "TESPIT_HATASI")
        n_error = sum(1 for r in reembed_rows if r["status"].startswith("HATA"))

        log.info("-" * 90)
        log.info(f"[Re-embed cosine] {len(reembed_rows)} kayıt test edildi")
        log.info(f"  cosine>{SUSPECT_THRESHOLD} (OK)      : {n_ok}")
        log.info(f"  cosine<={SUSPECT_THRESHOLD} (ŞÜPHELİ) : {n_suspect}")
        log.info(f"  yüz tespit edilemedi         : {n_detect_fail}")
        log.info(f"  diğer hata                   : {n_error}")
        if cosines:
            log.info(f"  ortalama cosine: {np.mean(cosines):.4f}  min: {np.min(cosines):.4f}")
        for r in reembed_rows:
            if r["status"] not in ("OK",):
                log.info(f"    [{r['status']}] {r['img_uri']} cosine={r['cosine']}")

        summary["reembed"] = {
            "n_tested": len(reembed_rows), "n_ok": n_ok, "n_suspect": n_suspect,
            "n_detect_fail": n_detect_fail, "n_error": n_error,
            "mean_cosine": float(np.mean(cosines)) if cosines else None,
        }

    log.info("=" * 90)
    return summary


if __name__ == "__main__":
    import sys
    run_pairing_audit(sys.argv[1] if len(sys.argv) > 1 else "manifest.txt")
