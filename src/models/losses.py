"""
ReconstructionLoss: Identity + Perceptual + L1 + SSIM kombinasyonu.

Loss akışı (her batch için):
    z_mobile (MobileFaceNet emb) → Decoder → I_gen
    loss_id    = 1 - cosine_sim( IResNet50(I_gen), IResNet50(I_real) )
    loss_perc  = || VGG(I_gen) - VGG(I_real) ||_2
    loss_l1    = || I_gen - I_real ||_1
    loss_ssim  = 1 - SSIM(I_gen, I_real)

Identity supervisor: IResNet50-ArcFace (frozen, yalnızca eğitimde).
Gradient akışı   : IResNet50(I_gen) → I_gen → Decoder (FaceNet fallback da aynı).
Fallback         : IResNet50 ağırlık dosyası yoksa FaceNet kullanılır.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from pytorch_msssim import ssim
from typing import Dict


# ─── Görüntü dönüşüm yardımcıları ────────────────────────────────────────────

# Model giriş normalizasyonları
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

_FACENET_MEAN  = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
_FACENET_STD   = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)


def _to_imagenet(x: torch.Tensor) -> torch.Tensor:
    """[-1,1] → ImageNet normalize ([0,1] → ImageNet std)."""
    x01 = x * 0.5 + 0.5
    mean = _IMAGENET_MEAN.to(x.device)
    std  = _IMAGENET_STD.to(x.device)
    return (x01 - mean) / std


def _to_facenet(x: torch.Tensor, size: int = 160) -> torch.Tensor:
    """[-1,1] → FaceNet normalize, resize to 160×160."""
    if x.shape[-1] != size:
        x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
    return x  # [-1,1] FaceNet'in kabul ettiği aralık


# ─── VGG Perceptual Loss ──────────────────────────────────────────────────────

class VGGPerceptualLoss(nn.Module):
    """
    VGG16 relu3_3 feature map farkı.
    Yüz yapısını, göz-burun oranlarını ve genel dokuyu korur.
    """

    # VGG16 katman indeksleri: relu3_3 = layer 15
    LAYER_MAP = {
        "relu1_2": 4,
        "relu2_2": 9,
        "relu3_3": 16,   # önerilen
        "relu4_3": 23,
    }

    def __init__(self, layer: str = "relu3_3"):
        super().__init__()
        if layer not in self.LAYER_MAP:
            raise ValueError(f"Bilinmeyen VGG katmanı: {layer}. Seçenekler: {list(self.LAYER_MAP)}")

        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        cutoff = self.LAYER_MAP[layer] + 1
        self.features = nn.Sequential(*list(vgg.features.children())[:cutoff])
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

    def forward(self, generated: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
        """
        Args:
            generated: (B, 3, H, W) [-1, 1]
            real:      (B, 3, H, W) [-1, 1]
        Returns:
            scalar loss
        """
        gen_feat  = self.features(_to_imagenet(generated))
        real_feat = self.features(_to_imagenet(real))
        return F.mse_loss(gen_feat, real_feat.detach())


# ─── Identity Loss ─────────────────────────────────────────────────────────────

class IdentityLoss(nn.Module):
    """
    Kimlik korunumu loss'u.

    Öncelik sırası:
      1. IResNet50-ArcFace (arcface_r50_path verilirse) — önerilir
      2. FaceNet fallback (her zaman kullanılabilir)

    Gradient akışı:
      - identity_model(generated) → NO no_grad → gradient decoder'a akar  ✓
      - identity_model(real).detach() → gerçek görüntü gradient üretmez   ✓

    Args:
        arcface_r50_path: IResNet50 backbone.pth dosya yolu.
                          None veya dosya yoksa FaceNet kullanılır.
        facenet_input_size: FaceNet için resize boyutu (sadece fallback).
    """

    def __init__(self, arcface_r50_path: str = None, facenet_input_size: int = 160):
        super().__init__()
        self.input_size = 112   # IResNet50 standart girişi

        self.use_arcface = False
        self.identity_model = None

        # IResNet50-ArcFace dene
        if arcface_r50_path is not None:
            try:
                from src.models.iresnet import iresnet50
                self.identity_model = iresnet50(pretrained_path=arcface_r50_path)
                self.use_arcface    = True
                self.input_size     = 112
                print("[IdentityLoss] IResNet50-ArcFace yüklendi.")
            except Exception as e:
                print(f"[IdentityLoss] IResNet50 yüklenemedi ({e}), FaceNet'e geçiliyor.")

        # FaceNet fallback
        if not self.use_arcface:
            from facenet_pytorch import InceptionResnetV1
            self.identity_model = InceptionResnetV1(pretrained="vggface2").eval()
            for p in self.identity_model.parameters():
                p.requires_grad_(False)
            self.input_size = facenet_input_size
            print("[IdentityLoss] FaceNet (InceptionResnetV1-VGGFace2) kullanılıyor.")

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """[-1,1] görüntüyü identity embedding'ine çevir."""
        x_resized = _to_facenet(x, size=self.input_size)
        emb = self.identity_model(x_resized)
        return F.normalize(emb, dim=1)

    def forward(self, generated: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
        """
        Args:
            generated: (B, 3, H, W) [-1, 1]  ← gradient akar
            real:      (B, 3, H, W) [-1, 1]  ← detach edilir
        Returns:
            scalar loss (düşük = iyi kimlik korunumu)
        """
        gen_emb  = self._encode(generated)           # gradient ✓
        real_emb = self._encode(real).detach()        # no gradient ✓
        cosine   = F.cosine_similarity(gen_emb, real_emb, dim=1)
        return 1.0 - cosine.mean()


# ─── SSIM Loss ────────────────────────────────────────────────────────────────

class SSIMLoss(nn.Module):
    """1 - SSIM(generated, real). pytorch-msssim kullanır."""

    def forward(self, generated: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
        # ssim [0,1] aralığında görüntü bekler
        gen_01  = generated * 0.5 + 0.5
        real_01 = real * 0.5 + 0.5
        return 1.0 - ssim(gen_01, real_01, data_range=1.0, size_average=True)


# ─── Birleşik Loss ────────────────────────────────────────────────────────────

class ReconstructionLoss(nn.Module):
    """
    Nihai loss fonksiyonu.

    total = w_id   × loss_identity
          + w_perc × loss_perceptual
          + w_l1   × loss_l1
          + w_ssim × loss_ssim

    Ağırlıklar config.py'deki aşamalı eğitim stratejisine göre dışarıdan verilir.
    """

    def __init__(self, vgg_layer: str = "relu3_3", facenet_input_size: int = 160,
                 arcface_r50_path: str = None):
        super().__init__()
        self.perceptual = VGGPerceptualLoss(layer=vgg_layer)
        self.identity   = IdentityLoss(
            arcface_r50_path  = arcface_r50_path,
            facenet_input_size = facenet_input_size,
        )
        self.ssim_loss  = SSIMLoss()

    def forward(
        self,
        generated:  torch.Tensor,
        real:       torch.Tensor,
        weights:    Dict[str, float],
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            generated: (B, 3, H, W) decoder çıktısı, [-1, 1]
            real:      (B, 3, H, W) gerçek görüntü, [-1, 1]
            weights:   {"l1": ..., "perceptual": ..., "identity": ..., "ssim": ...}

        Returns:
            dict:
                "total":       toplam loss (backward için)
                "l1":          L1 loss değeri
                "perceptual":  perceptual loss değeri
                "identity":    identity loss değeri
                "ssim":        ssim loss değeri
        """
        losses: Dict[str, torch.Tensor] = {}

        # L1
        losses["l1"] = F.l1_loss(generated, real)

        # Perceptual
        losses["perceptual"] = (
            self.perceptual(generated, real)
            if weights.get("perceptual", 0.0) > 0
            else torch.zeros(1, device=generated.device)
        )

        # Identity
        losses["identity"] = (
            self.identity(generated, real)
            if weights.get("identity", 0.0) > 0
            else torch.zeros(1, device=generated.device)
        )

        # SSIM
        losses["ssim"] = (
            self.ssim_loss(generated, real)
            if weights.get("ssim", 0.0) > 0
            else torch.zeros(1, device=generated.device)
        )

        # Toplam
        losses["total"] = (
            weights.get("l1",          0.0) * losses["l1"]
            + weights.get("perceptual", 0.0) * losses["perceptual"]
            + weights.get("identity",   0.0) * losses["identity"]
            + weights.get("ssim",       0.0) * losses["ssim"]
        )

        return losses
