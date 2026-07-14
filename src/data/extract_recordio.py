"""
MS1MV3 (ve genel InsightFace) RecordIO formatından görüntü çıkarımı.

Notebook hücrelerine mantık gömmek yerine bu modülü kullan — böylece
kod değişince yalnızca `git clone` / `git pull` yeterli olur, notebook
hücresi elle güncellenmesine gerek kalmaz.

RecordIO Kayıt Yapısı (MXNet / InsightFace, flag>0 durumu dahil):
    Prefix   8B : magic(4) + length(4)          — length: header+extras+jpg boyutu
    IRHeader 24B: flag(4) + base_lbl(4) + id(8) + id2(8)
    Extras  flag*4 B: extra_lbl0(4) [+ extra_lbl1(4) ...]   (yalnızca flag>0)
    JPEG      ?B: görüntü verisi

    base_lbl her zaman 0.0 olabilir (YANLIŞ, class ID değildir).
    flag>0 ise gerçek class ID, extras bloğunun ilk float'ındadır (extra_lbl0).

Sequential okuma kullanılır (seek YOK) — .idx dosyasındaki offsetler MS1MV3
gibi büyük datasetlerde tutarsız/yanlış olabildiği için idx dosyasına
güvenilmez; dosya baştan sona sırayla okunur.
"""

import os
import shutil
import struct
from io import BytesIO
from pathlib import Path

from PIL import Image
from tqdm.auto import tqdm

_PREFIX_SZ = 8    # magic(4) + length(4)
_IRHDR_SZ  = 24   # flag(4) + base_lbl(4) + id(8) + id2(8)

# Disk dolmadan ÖNCE güvenli şekilde durmak için kaç görüntüde bir
# `shutil.disk_usage` kontrolü yapılacağı (performans için her görüntüde değil).
_DISK_CHECK_EVERY = 500


def extract_ms1mv3(
    rec_path: str,
    extracted_dir: str,
    max_per_id: int = 30,
    max_ids: int = 93000,
    skip_if_exists: bool = True,
    min_free_gb: float = 2.0,
) -> dict:
    """
    MS1MV3 .rec dosyasını kimlik başına klasörlere ayrılmış JPEG'lere çıkarır.

    Args:
        rec_path:       .rec dosyasının yolu (örn. /kaggle/input/.../train.rec)
        extracted_dir:  çıktı klasörü (örn. /kaggle/working/ms1mv3_images)
        max_per_id:     kimlik başına maksimum görüntü sayısı
        max_ids:        işlenecek maksimum kimlik sayısı (class_id sınırı)
        skip_if_exists: extracted_dir zaten dolu ise atla
        min_free_gb:    diskte bu kadar GB boş alan kalınca çıkarımı DURDUR
                         (OSError: No space left on device çökmesini önler)

    Returns:
        dict: {"n_ids": int, "n_images": int, "n_errors": int, "data_dir": str,
               "stopped_early": bool}
    """
    if skip_if_exists and os.path.exists(extracted_dir) and len(list(Path(extracted_dir).iterdir())) > 0:
        n_ids = len(list(Path(extracted_dir).iterdir()))
        print(f"Zaten cikarilmis: {extracted_dir}  ({n_ids:,} kimlik)")
        return {"n_ids": n_ids, "n_images": None, "n_errors": None,
                "data_dir": extracted_dir, "stopped_early": False}

    os.makedirs(extracted_dir, exist_ok=True)
    saved_total   = 0
    id_counts     = {}
    errors        = 0
    rec_idx       = 0
    stopped_early = False
    min_free_bytes = min_free_gb * (1024 ** 3)

    file_size = os.path.getsize(rec_path)
    print(f"Rec dosyasi: {file_size / 1e9:.1f} GB — sequential okuma basliyor...")
    print(f"Guvenlik: diskte {min_free_gb:.0f}GB kalinca otomatik duracak (crash yok).")
    print("Tahmini sure: ~3-5 dakika")

    with open(rec_path, "rb") as f:
        with tqdm(total=file_size, unit="B", unit_scale=True,
                  unit_divisor=1024, desc="Okunan") as pbar:

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
                    label_id  = int(struct.unpack("<f", data[_IRHDR_SZ:_IRHDR_SZ + 4])[0])
                    img_start = _IRHDR_SZ + extra_sz
                else:
                    label_id  = int(struct.unpack("<f", data[4:8])[0])
                    img_start = _IRHDR_SZ

                if label_id < 0 or label_id >= max_ids:
                    continue
                cnt = id_counts.get(label_id, 0)
                if cnt >= max_per_id:
                    continue

                img_bytes = data[img_start:]
                if len(img_bytes) < 50:
                    continue

                # Disk alanı kontrolü (her görüntüde değil, N'de bir — hızlı)
                if saved_total % _DISK_CHECK_EVERY == 0:
                    free_bytes = shutil.disk_usage(extracted_dir).free
                    if free_bytes < min_free_bytes:
                        stopped_early = True
                        print(
                            f"\n[DUR] Disk alani azaldi ({free_bytes / 1e9:.1f}GB kaldi, "
                            f"esik={min_free_gb:.0f}GB). Cikarim guvenli sekilde durduruldu."
                        )
                        break

                id_dir = os.path.join(extracted_dir, f"id_{label_id:06d}")
                os.makedirs(id_dir, exist_ok=True)
                try:
                    img = Image.open(BytesIO(img_bytes))
                    img.save(os.path.join(id_dir, f"{cnt:04d}.jpg"), quality=95)
                    id_counts[label_id] = cnt + 1
                    saved_total += 1
                except Exception:
                    errors += 1
                    continue

    print("\nTamamlandi!" if not stopped_early else "\nDisk limiti nedeniyle erken durdu (veri kaybı yok).")
    print(f"  Kimlik  : {len(id_counts):,}")
    print(f"  Goruntu : {saved_total:,}")
    print(f"  Hata    : {errors:,}")

    return {
        "n_ids": len(id_counts),
        "n_images": saved_total,
        "n_errors": errors,
        "data_dir": extracted_dir,
        "stopped_early": stopped_early,
    }
