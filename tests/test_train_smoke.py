"""
Eğitim ve mimari smoke testleri (GAN kaldırıldıktan sonra).

Calistirma: python tests/test_train_smoke.py
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import DEFAULT_CONFIG, Config, ModelConfig, TrainConfig  # noqa: E402
from src.models.decoder import FaceDecoder  # noqa: E402
from src.train import save_checkpoint  # noqa: E402


class TestDefaultConfig(unittest.TestCase):
    def test_no_gan_fields(self):
        cfg = DEFAULT_CONFIG
        # Varsayılan KAPALI (bkz. config.py yorumu: nearest/erken skip zayıf
        # decoder'da grid artifact üretebilir; açmak sıfırdan eğitim ister).
        self.assertFalse(cfg.model.use_cascade_skip)
        self.assertEqual(cfg.model.cascade_skip_last_n_blocks, 2)
        self.assertTrue(cfg.train.use_ema)
        self.assertTrue(cfg.train.eval_use_ema)
        self.assertFalse(hasattr(cfg.train, "use_gan"))

    def test_cascade_decoder_forward(self):
        model = FaceDecoder(
            use_cascade_skip=True,
            cascade_skip_last_n_blocks=2,
        )
        out = model(torch.randn(2, 512))
        self.assertEqual(out.shape, (2, 3, 128, 128))
        flags = [b.use_cascade_skip for b in model.up_blocks]
        self.assertEqual(flags, [False, False, False, True, True])


class TestCheckpointFormat(unittest.TestCase):
    def test_checkpoint_has_no_discriminator_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = {
                "epoch": 0,
                "model": FaceDecoder().state_dict(),
                "optimizer": {},
                "scheduler": {},
                "best_val_score": 1.0,
                "config": {},
                "model_ema": None,
            }
            save_checkpoint(state, tmp, epoch=1, is_best=True, keep_last_n=1)
            loaded = torch.load(Path(tmp) / "best_model.pt", map_location="cpu", weights_only=False)
            self.assertNotIn("discriminator", loaded)
            self.assertNotIn("disc_optimizer", loaded)
            self.assertIn("model", loaded)


if __name__ == "__main__":
    unittest.main(verbosity=2)
