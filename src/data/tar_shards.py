"""
WebDataset-benzeri tar shard yardımcıları.

Amaç: milyonlarca ayrı JPEG/NPY dosyası yerine az sayıda .tar shard
kullanarak Kaggle /kaggle/working inode limitini aşmamak.

Shard adı  : shard_%06d.tar
Member adı : {label_id:06d}/{img_idx:05d}.jpg  (+ opsiyonel .npy)

Manifest URI formatı (iki sütun, mevcut sözleşmeyle uyumlu):
    /path/shard_000001.tar::000042/00003.jpg\\t/path/shard_000001.tar::000042/00003.npy
"""

from __future__ import annotations

import io
import os
import shutil
import tarfile
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

from PIL import Image
from tqdm.auto import tqdm

URI_SEP = "::"
DEFAULT_IMAGES_PER_SHARD = 8000
DEFAULT_MIN_FREE_INODES = 5000


# ─── Disk / inode yardımcıları ────────────────────────────────────────────────

def free_bytes(path: str) -> int:
    return shutil.disk_usage(path).free


def free_inodes(path: str) -> Optional[int]:
    """Linux'ta boş inode sayısı; Windows'ta None (kontrol atlanır)."""
    if not hasattr(os, "statvfs"):
        return None
    try:
        st = os.statvfs(path)
        return int(st.f_ffree)
    except (OSError, AttributeError):
        return None


def print_disk_inode_stats(path: str = "/kaggle/working") -> None:
    """df -i / du -sh benzeri özet (Kaggle doğrulama için)."""
    p = Path(path)
    if not p.exists():
        print(f"[stats] Yol yok: {path}")
        return

    usage = shutil.disk_usage(path)
    print(f"=== Disk: {path} ===")
    print(f"  Toplam : {usage.total / 1e9:.2f} GB")
    print(f"  Kullan : {usage.used / 1e9:.2f} GB")
    print(f"  Bos    : {usage.free / 1e9:.2f} GB")

    fi = free_inodes(path)
    if fi is not None and hasattr(os, "statvfs"):
        st = os.statvfs(path)
        total_i = int(st.f_files)
        used_i = total_i - fi
        pct = (100.0 * used_i / total_i) if total_i else 0.0
        print(f"=== Inode: {path} ===")
        print(f"  Toplam : {total_i:,}")
        print(f"  Kullan : {used_i:,}  ({pct:.1f}%)")
        print(f"  Bos    : {fi:,}")
    else:
        print("=== Inode: (bu platformda desteklenmiyor) ===")

    print(f"=== Icerik: {path} ===")
    for child in sorted(p.iterdir()):
        if child.is_dir():
            # Yalnızca üst seviye boyut — hızlı yaklaşık
            size = sum(f.stat().st_size for f in child.rglob("*") if f.is_file())
            nfiles = sum(1 for f in child.rglob("*") if f.is_file())
            print(f"  {child.name}/  {size / 1e9:.2f} GB  ({nfiles:,} dosya)")
        else:
            print(f"  {child.name}  {child.stat().st_size / 1e6:.1f} MB")


def resource_ok(
    path: str,
    min_free_gb: float = 2.0,
    min_free_inodes: int = DEFAULT_MIN_FREE_INODES,
) -> Tuple[bool, str]:
    """Disk ve inode eşiği kontrolü. (ok, reason)"""
    if free_bytes(path) < min_free_gb * (1024 ** 3):
        return False, f"disk<{min_free_gb:.0f}GB"
    fi = free_inodes(path)
    if fi is not None and fi < min_free_inodes:
        return False, f"inode<{min_free_inodes} (kalan={fi})"
    return True, ""


# ─── URI yardımcıları ─────────────────────────────────────────────────────────

def make_uri(tar_path: str, member: str) -> str:
    return f"{tar_path}{URI_SEP}{member}"


def parse_uri(uri: str) -> Tuple[Optional[str], str]:
    """'tar::member' → (tar_path, member); düz dosya yolu → (None, path)."""
    if URI_SEP in uri:
        tar_path, member = uri.split(URI_SEP, 1)
        return tar_path, member
    return None, uri


def identity_from_member(member: str) -> str:
    """'000042/00003.jpg' → 'id_000042'."""
    parent = Path(member).parent.name
    if parent.startswith("id_"):
        return parent
    # Sadece rakam klasör adı
    try:
        return f"id_{int(parent):06d}"
    except ValueError:
        return parent or "unknown"


# ─── ShardWriter ──────────────────────────────────────────────────────────────

class ShardWriter:
    """Sırayla tar shard'larına member yazar; ara klasör/dosya oluşturmaz."""

    def __init__(
        self,
        out_dir: str,
        images_per_shard: int = DEFAULT_IMAGES_PER_SHARD,
        prefix: str = "shard",
    ):
        self.out_dir = out_dir
        self.images_per_shard = images_per_shard
        self.prefix = prefix
        os.makedirs(out_dir, exist_ok=True)
        self.shard_idx = 0
        self.count_in_shard = 0
        self.n_written = 0
        self._tar: Optional[tarfile.TarFile] = None
        self.current_path: Optional[str] = None
        self.shard_paths: List[str] = []

    def _open_next(self) -> None:
        self.close()
        self.current_path = os.path.join(
            self.out_dir, f"{self.prefix}_{self.shard_idx:06d}.tar"
        )
        self._tar = tarfile.open(self.current_path, "w")
        self.shard_paths.append(self.current_path)
        self.shard_idx += 1
        self.count_in_shard = 0

    def add_bytes(self, member_name: str, data: bytes, count_as_image: bool = True) -> str:
        """
        Member'ı mevcut shard'a ekle; URI döndür.

        count_as_image=True  → shard doluluk sayacına eklenir; gerekirse yeni shard açılır.
        count_as_image=False → yan dosya (.npy); ASLA yeni shard açmaz (jpg ile aynı tar'da kalır).
        """
        if count_as_image:
            if self._tar is None or self.count_in_shard >= self.images_per_shard:
                self._open_next()
        else:
            if self._tar is None:
                self._open_next()

        assert self._tar is not None and self.current_path is not None

        info = tarfile.TarInfo(name=member_name)
        info.size = len(data)
        self._tar.addfile(info, io.BytesIO(data))

        if count_as_image:
            self.count_in_shard += 1
            self.n_written += 1

        return make_uri(self.current_path, member_name)

    def add_jpeg_image(
        self,
        member_name: str,
        img: Image.Image,
        jpeg_quality: int = 90,
    ) -> str:
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=jpeg_quality)
        return self.add_bytes(member_name, buf.getvalue(), count_as_image=True)

    def add_npy(self, member_name: str, array) -> str:
        import numpy as np
        buf = io.BytesIO()
        np.save(buf, array)
        # np.save yazdığı buffer'ın başı .npy formatında
        return self.add_bytes(member_name, buf.getvalue(), count_as_image=False)

    def close(self) -> None:
        if self._tar is not None:
            self._tar.close()
            self._tar = None


# ─── Okuma ────────────────────────────────────────────────────────────────────

class TarMemberCache:
    """
    Worker-safe lazy tar cache. Her process kendi instance'ını tutar;
    DataLoader num_workers>0 iken __getitem__ içinde ilk erişimde açılır.
    """

    def __init__(self):
        self._tars: Dict[str, tarfile.TarFile] = {}

    def read_bytes(self, tar_path: str, member: str) -> bytes:
        tf = self._tars.get(tar_path)
        if tf is None:
            tf = tarfile.open(tar_path, "r")
            self._tars[tar_path] = tf
        f = tf.extractfile(member)
        if f is None:
            raise FileNotFoundError(f"Member yok: {tar_path}::{member}")
        return f.read()

    def close(self) -> None:
        for tf in self._tars.values():
            try:
                tf.close()
            except Exception:
                pass
        self._tars.clear()


_THREAD_CACHE = TarMemberCache()


def read_uri_bytes(uri: str, cache: Optional[TarMemberCache] = None) -> bytes:
    tar_path, member = parse_uri(uri)
    if tar_path is None:
        with open(member, "rb") as f:
            return f.read()
    c = cache or _THREAD_CACHE
    return c.read_bytes(tar_path, member)


def uri_exists(uri: str, deep: bool = False) -> bool:
    """
    URI erişilebilir mi?

    deep=False (varsayılan, hızlı): tar dosyasının varlığı yeterli.
    deep=True: tar içinde member gerçekten var mı diye açıp bakar (yavaş).
    """
    tar_path, member = parse_uri(uri)
    if tar_path is None:
        return Path(member).exists()
    if not Path(tar_path).exists():
        return False
    if not deep:
        return True
    try:
        with tarfile.open(tar_path, "r") as tf:
            try:
                tf.getmember(member)
                return True
            except KeyError:
                return False
    except (tarfile.TarError, OSError):
        return False


def list_shard_paths(shards_dir: str) -> List[str]:
    p = Path(shards_dir)
    if not p.exists():
        return []
    return sorted(str(x) for x in p.glob("shard_*.tar"))


def iter_shard_members(
    shards_dir: str,
    suffix: str = ".jpg",
) -> Iterator[Tuple[str, str, str]]:
    """
    Yields: (tar_path, member_name, identity)
    identity: id_XXXXXX
    """
    for tar_path in list_shard_paths(shards_dir):
        with tarfile.open(tar_path, "r") as tf:
            for m in tf.getmembers():
                if not m.isfile():
                    continue
                if not m.name.lower().endswith(suffix):
                    continue
                yield tar_path, m.name, identity_from_member(m.name)


def count_shard_stats(shards_dir: str) -> dict:
    n_images = 0
    identities = set()
    for _tar_path, _member, ident in iter_shard_members(shards_dir):
        n_images += 1
        identities.add(ident)
    return {
        "n_shards": len(list_shard_paths(shards_dir)),
        "n_images": n_images,
        "n_ids": len(identities),
        "shards_dir": shards_dir,
    }


# ─── Klasör → shard dönüşümü (inode kurtarma) ─────────────────────────────────

def repack_folder_to_shards(
    src_dir: str,
    dest_dir: str,
    images_per_shard: int = DEFAULT_IMAGES_PER_SHARD,
    jpeg_quality: int = 90,
    min_free_gb: float = 1.0,
    min_free_inodes: int = DEFAULT_MIN_FREE_INODES,
    delete_src: bool = False,
) -> dict:
    """
    id_XXXXXX/*.jpg klasör yapısını shard_*.tar formatına dönüştürür.

    Inode doluysa önce src'yi /kaggle/temp altına tek tar olarak taşıyıp
    silmek gerekir; bu fonksiyon doğrudan klasörden okur.
    """
    src = Path(src_dir)
    if not src.exists():
        raise FileNotFoundError(src_dir)

    os.makedirs(dest_dir, exist_ok=True)
    writer = ShardWriter(dest_dir, images_per_shard=images_per_shard)
    id_dirs = sorted([d for d in src.iterdir() if d.is_dir()])
    n_errors = 0
    stopped_early = False

    try:
        for id_dir in tqdm(id_dirs, desc="Repack"):
            # id_000042 veya düz sayı
            name = id_dir.name
            if name.startswith("id_"):
                try:
                    label_id = int(name.split("_", 1)[1])
                except ValueError:
                    label_id = abs(hash(name)) % 1_000_000
            else:
                try:
                    label_id = int(name)
                except ValueError:
                    label_id = abs(hash(name)) % 1_000_000

            imgs = sorted(
                f for f in id_dir.iterdir()
                if f.suffix.lower() in {".jpg", ".jpeg", ".png"}
            )
            for img_idx, img_path in enumerate(imgs):
                ok, reason = resource_ok(dest_dir, min_free_gb, min_free_inodes)
                if not ok:
                    stopped_early = True
                    print(f"\n[DUR] Kaynak yetersiz ({reason}). Repack durduruldu.")
                    break
                try:
                    img = Image.open(img_path).convert("RGB")
                    member = f"{label_id:06d}/{img_idx:05d}.jpg"
                    writer.add_jpeg_image(member, img, jpeg_quality=jpeg_quality)
                except Exception:
                    n_errors += 1
            if stopped_early:
                break
    finally:
        writer.close()

    if delete_src and not stopped_early:
        print(f"Kaynak klasor siliniyor: {src_dir}")
        shutil.rmtree(src_dir, ignore_errors=True)

    stats = count_shard_stats(dest_dir)
    stats.update({"n_errors": n_errors, "stopped_early": stopped_early})
    print(
        f"Repack tamam: {stats['n_shards']} shard, "
        f"{stats['n_images']:,} goruntu, {stats['n_ids']:,} kimlik"
    )
    return stats


def migrate_working_images_via_temp(
    src_dir: str = "/kaggle/working/ms1mv3_images",
    dest_dir: str = "/kaggle/working/ms1mv3_shards",
    temp_tar: str = "/kaggle/temp/ms1mv3_images.tar",
    images_per_shard: int = DEFAULT_IMAGES_PER_SHARD,
    jpeg_quality: int = 90,
) -> dict:
    """
    Inode dolu /kaggle/working için kurtarma akışı:
      1) src'yi /kaggle/temp altına tek tar olarak paketle
      2) src klasörünü sil (inode boşalsın)
      3) temp tar'dan STREAM okuyup dest_dir'e shard'lara yaz
         (tekrar klasöre extract YOK → inode tekrar dolmaz)
      4) temp tar'ı sil
    """
    src = Path(src_dir)
    if not src.exists():
        raise FileNotFoundError(src_dir)

    # Zaten shard varsa atla
    if list_shard_paths(dest_dir):
        stats = count_shard_stats(dest_dir)
        print(f"Hedef zaten shard iceriyor: {dest_dir} ({stats})")
        return stats

    temp_tar_path = Path(temp_tar)
    temp_tar_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] Paketleniyor: {src_dir} -> {temp_tar}")
    with tarfile.open(temp_tar_path, "w") as tf:
        for id_dir in tqdm(sorted([d for d in src.iterdir() if d.is_dir()]), desc="Pack"):
            for img in id_dir.iterdir():
                if img.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    arc = f"{src.name}/{id_dir.name}/{img.name}"
                    tf.add(str(img), arcname=arc)

    print(f"[2/4] Kaynak siliniyor (inode kurtarma): {src_dir}")
    shutil.rmtree(src_dir, ignore_errors=True)
    print_disk_inode_stats(str(src.parent))

    print(f"[3/4] Temp tar stream -> shard'lar: {dest_dir}")
    os.makedirs(dest_dir, exist_ok=True)
    writer = ShardWriter(dest_dir, images_per_shard=images_per_shard)
    id_counters: Dict[int, int] = {}
    n_errors = 0

    try:
        with tarfile.open(temp_tar_path, "r") as tf:
            members = [m for m in tf.getmembers() if m.isfile()]
            for m in tqdm(members, desc="Reshard"):
                name = m.name.replace("\\", "/")
                parts = name.split("/")
                # .../id_XXXXXX/file.jpg
                id_part = None
                for p in parts:
                    if p.startswith("id_"):
                        id_part = p
                        break
                if id_part is None:
                    continue
                try:
                    label_id = int(id_part.split("_", 1)[1])
                except ValueError:
                    continue

                fobj = tf.extractfile(m)
                if fobj is None:
                    n_errors += 1
                    continue
                try:
                    img = Image.open(io.BytesIO(fobj.read())).convert("RGB")
                    idx = id_counters.get(label_id, 0)
                    member = f"{label_id:06d}/{idx:05d}.jpg"
                    writer.add_jpeg_image(member, img, jpeg_quality=jpeg_quality)
                    id_counters[label_id] = idx + 1
                except Exception:
                    n_errors += 1
    finally:
        writer.close()

    print(f"[4/4] Temp tar siliniyor: {temp_tar}")
    try:
        temp_tar_path.unlink()
    except OSError:
        pass

    stats = count_shard_stats(dest_dir)
    stats.update({"n_errors": n_errors, "stopped_early": False})
    print(
        f"Migrasyon tamam: {stats['n_shards']} shard, "
        f"{stats['n_images']:,} goruntu, {stats['n_ids']:,} kimlik"
    )
    print_disk_inode_stats(str(Path(dest_dir).parent))
    return stats
