"""
Model export: PyTorch → ONNX → TFLite (float32, float16, INT8).

Tercih edilen yol: ai-edge-torch (Google, 2024) — PyTorch'tan doğrudan TFLite.
Yedek yol: ONNX ara adımı (ai-edge-torch yoksa).

Kullanım:
    from src.export_tflite import export_model
    export_model(checkpoint_path="best_model.pt", export_dir="export/")
"""

import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from .config import Config, DEFAULT_CONFIG
from .models.decoder import FaceDecoder

log = logging.getLogger(__name__)


# ─── Yardımcı: Model yükle ────────────────────────────────────────────────────

def load_model_from_checkpoint(ckpt_path: str, cfg: Config) -> FaceDecoder:
    device = torch.device("cpu")
    state  = torch.load(ckpt_path, map_location=device)
    model  = FaceDecoder(
        embedding_dim=cfg.model.embedding_dim,
        initial_spatial=cfg.model.initial_spatial,
        initial_channels=cfg.model.initial_channels,
        decoder_channels=cfg.model.decoder_channels,
    )
    model.load_state_dict(state["model"])
    model.eval()
    return model


# ─── PyTorch → ONNX ───────────────────────────────────────────────────────────

def export_onnx(model: FaceDecoder, export_dir: str, embedding_dim: int = 512) -> str:
    out = Path(export_dir)
    out.mkdir(parents=True, exist_ok=True)

    onnx_path = str(out / "face_decoder.onnx")
    dummy_z   = torch.randn(1, embedding_dim)

    torch.onnx.export(
        model,
        dummy_z,
        onnx_path,
        export_params=True,
        opset_version=17,
        input_names=["embedding"],
        output_names=["face_image"],
        dynamic_axes={
            "embedding":  {0: "batch_size"},
            "face_image": {0: "batch_size"},
        },
        do_constant_folding=True,
    )
    log.info(f"ONNX export tamamlandı: {onnx_path}")

    # ONNX model doğrulama
    try:
        import onnx
        onnx.checker.check_model(onnx_path)
        log.info("ONNX model doğrulama: PASSED")
    except ImportError:
        log.warning("onnx paketi bulunamadı, doğrulama atlandı.")

    return onnx_path


# ─── ONNX → TFLite (yedek yol) ───────────────────────────────────────────────

def _onnx_to_tflite(onnx_path: str, export_dir: str) -> dict:
    """
    ONNX → TensorFlow → TFLite dönüşüm zinciri.
    ai-edge-torch yoksa kullanılır.
    """
    import subprocess
    out      = Path(export_dir)
    tf_path  = str(out / "tf_saved_model")
    results  = {}

    # ONNX → TF SavedModel (onnx-tf gerekli)
    log.info("ONNX → TF SavedModel dönüştürülüyor...")
    try:
        from onnx_tf.backend import prepare
        import onnx
        onnx_model = onnx.load(onnx_path)
        tf_rep     = prepare(onnx_model)
        tf_rep.export_graph(tf_path)
        log.info(f"TF SavedModel: {tf_path}")
    except ImportError:
        log.error("onnx-tf bulunamadı. 'pip install onnx-tf tensorflow' ile kurun.")
        return results

    # TF SavedModel → TFLite (float32)
    try:
        import tensorflow as tf

        # float32
        converter = tf.lite.TFLiteConverter.from_saved_model(tf_path)
        tflite_f32 = converter.convert()
        path_f32 = str(out / "face_decoder_f32.tflite")
        with open(path_f32, "wb") as f:
            f.write(tflite_f32)
        results["float32"] = path_f32
        log.info(f"TFLite float32: {path_f32}  ({len(tflite_f32)/1e6:.2f} MB)")

        # float16
        converter = tf.lite.TFLiteConverter.from_saved_model(tf_path)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        tflite_f16 = converter.convert()
        path_f16 = str(out / "face_decoder_f16.tflite")
        with open(path_f16, "wb") as f:
            f.write(tflite_f16)
        results["float16"] = path_f16
        log.info(f"TFLite float16: {path_f16}  ({len(tflite_f16)/1e6:.2f} MB)")

        # INT8 dynamic range quantization
        converter = tf.lite.TFLiteConverter.from_saved_model(tf_path)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_int8 = converter.convert()
        path_int8 = str(out / "face_decoder_int8.tflite")
        with open(path_int8, "wb") as f:
            f.write(tflite_int8)
        results["int8"] = path_int8
        log.info(f"TFLite INT8:    {path_int8}  ({len(tflite_int8)/1e6:.2f} MB)")

    except ImportError:
        log.error("tensorflow bulunamadı. 'pip install tensorflow' ile kurun.")

    return results


# ─── ai-edge-torch yolu (tercih edilen) ──────────────────────────────────────

def _export_via_ai_edge_torch(model: FaceDecoder, export_dir: str, embedding_dim: int = 512) -> dict:
    """
    ai-edge-torch ile doğrudan PyTorch → TFLite dönüşümü.
    ONNX ara adımına gerek yok.
    """
    import ai_edge_torch

    out    = Path(export_dir)
    out.mkdir(parents=True, exist_ok=True)
    sample = (torch.randn(1, embedding_dim),)
    results = {}

    log.info("ai-edge-torch ile TFLite export başlatılıyor...")

    # float32
    tfl = ai_edge_torch.convert(model, sample)
    path_f32 = str(out / "face_decoder_f32.tflite")
    tfl.export(path_f32)
    results["float32"] = path_f32
    log.info(f"TFLite float32: {path_f32}")

    # INT8 dynamic range quantization
    try:
        import ai_edge_torch.quantize as aq
        q_config = aq.DynamicRangeQuantizationConfig()
        tfl_q    = ai_edge_torch.convert(model, sample, quant_config=q_config)
        path_int8 = str(out / "face_decoder_int8.tflite")
        tfl_q.export(path_int8)
        results["int8"] = path_int8
        log.info(f"TFLite INT8:  {path_int8}")
    except Exception as e:
        log.warning(f"INT8 quantization başarısız: {e}. float32 versiyonu kullanılabilir.")

    return results


# ─── TFLite doğrulama ─────────────────────────────────────────────────────────

def validate_tflite(tflite_path: str, embedding_dim: int = 512) -> bool:
    """TFLite modeli yükle ve dummy input ile doğrula."""
    try:
        import tensorflow as tf
    except ImportError:
        log.warning("tensorflow bulunamadı, TFLite doğrulama atlandı.")
        return False

    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    in_details  = interpreter.get_input_details()
    out_details = interpreter.get_output_details()

    dummy_emb = np.random.randn(1, embedding_dim).astype(np.float32)
    interpreter.set_tensor(in_details[0]["index"], dummy_emb)
    interpreter.invoke()
    output = interpreter.get_tensor(out_details[0]["index"])

    ok = output.shape == (1, 3, 128, 128) or output.shape == (1, 128, 128, 3)
    log.info(f"TFLite doğrulama: {'PASSED' if ok else 'FAILED'}  çıktı boyutu: {output.shape}")
    return ok


# ─── Ana export fonksiyonu ────────────────────────────────────────────────────

def export_model(
    checkpoint_path: str,
    export_dir: str,
    cfg: Optional[Config] = None,
) -> dict:
    """
    Eğitilmiş modeli PyTorch checkpoint'inden TFLite'a export eder.

    Args:
        checkpoint_path: best_model.pt yolu
        export_dir:      Export klasörü
        cfg:             Config nesnesi (None → DEFAULT_CONFIG)

    Returns:
        {"onnx": path, "float32": path, "int8": path, ...}
    """
    if cfg is None:
        cfg = DEFAULT_CONFIG

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    log.info(f"Model yükleniyor: {checkpoint_path}")
    model = load_model_from_checkpoint(checkpoint_path, cfg)
    info  = model.count_parameters()
    log.info(f"Parametre: {info['total_params']:,}  |  {info['float32_mb']} MB")

    results = {}

    # Önce ONNX export (her iki yolda da kullanışlı)
    log.info("── ONNX Export ──")
    onnx_path = export_onnx(model, export_dir, cfg.model.embedding_dim)
    results["onnx"] = onnx_path

    # TFLite export: ai-edge-torch tercih edilir, yoksa ONNX zinciri
    log.info("── TFLite Export ──")
    try:
        import ai_edge_torch  # noqa: F401
        tfl_results = _export_via_ai_edge_torch(model, export_dir, cfg.model.embedding_dim)
    except ImportError:
        log.info("ai-edge-torch bulunamadı, ONNX→TF→TFLite zinciri kullanılıyor.")
        log.info("'pip install ai-edge-torch' ile daha iyi sonuç alınabilir.")
        tfl_results = _onnx_to_tflite(onnx_path, export_dir)

    results.update(tfl_results)

    # Doğrulama
    if "int8" in results:
        validate_tflite(results["int8"], cfg.model.embedding_dim)
    elif "float32" in results:
        validate_tflite(results["float32"], cfg.model.embedding_dim)

    # Özet
    log.info("\n── Export Özeti ──")
    for fmt, path in results.items():
        if Path(path).exists():
            size_mb = Path(path).stat().st_size / 1e6
            log.info(f"  {fmt:12s}: {path}  ({size_mb:.2f} MB)")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="best_model.pt yolu")
    parser.add_argument("--export_dir", default="export", help="Çıktı klasörü")
    args = parser.parse_args()

    export_model(
        checkpoint_path=args.checkpoint,
        export_dir=args.export_dir,
    )
