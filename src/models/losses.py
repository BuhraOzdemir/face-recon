"""
ReconstructionLoss: Identity + Perceptual + L1 + SSIM kombinasyonu.

Loss akışı (her batch için):
    z_mobile (MobileFaceNet emb) → Decoder → I_gen
    loss_id    = 1 - cosine_sim( IResNet50(I_gen), IResNet50(I_real) )
    loss_perc  = || VGG(I_gen) - VGG(I_real) ||_2  (multi-layer)
    loss_l1    = || I_gen - I_real ||_1
    loss_ssim  = 1 - SSIM(I_gen, I_real)

Identity supervisor: IResNet50-ArcFace (frozen, yalnızca eğitimde).
Gradient akışı   : IResNet50(I_gen) → I_gen → Decoder (FaceNet fallback da aynı).
Fallback         : IResNet50 ağırlık dosyası yoksa FaceNet kullanılır.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from pytorch_msssim import ssim
from typing import Dict, List, Union


# ─── Görüntü dönüşüm yardımcıları ────────────────────────────────────────────

_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


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
    return x


# ─── VGG Perceptual Loss ──────────────────────────────────────────────────────

class VGGPerceptualLoss(nn.Module):
    """
    VGG16 multi-layer feature map farkı.
    Varsayılan: relu2_2 + relu3_3 (doku + yapı).
    """

    LAYER_MAP = {
        "relu1_2": 4,
        "relu2_2": 9,
        "relu3_3": 16,
        "relu4_3": 23,
    }

    def __init__(self, layer: Union[str, List[str]] = "relu2_2,relu3_3"):
        super().__init__()
        if isinstance(layer, str):
            layers = [s.strip() for s in layer.split(",") if s.strip()]
        else:
            layers = list(layer)

        if not layers:
            raise ValueError("En az bir VGG katmanı gerekli.")
        for name in layers:
            if name not in self.LAYER_MAP:
                raise ValueError(
                    f"Bilinmeyen VGG katmanı: {name}. Seçenekler: {list(self.LAYER_MAP)}"
                )

        cutoffs = sorted({self.LAYER_MAP[name] for name in layers})
        max_cutoff = max(cutoffs) + 1

        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(*list(vgg.features.children())[:max_cutoff])
        self.target_indices = cutoffs
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

    def forward(self, generated: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
        """
        Args:
            generated: (B, 3, H, W) [-1, 1]
            real:      (B, 3, H, W) [-1, 1]
        Returns:
            scalar loss (katman ortalaması)
        """
        gen_x = _to_imagenet(generated)
        real_x = _to_imagenet(real)

        loss = torch.zeros((), device=generated.device)
        n = 0
        for i, layer in enumerate(self.features):
            gen_x = layer(gen_x)
            real_x = layer(real_x)
            if i in self.target_indices:
                loss = loss + F.mse_loss(gen_x, real_x.detach())
                n += 1
        return loss / max(n, 1)


# ─── Identity Loss ─────────────────────────────────────────────────────────────

class IdentityLoss(nn.Module):
    """
    Kimlik korunumu loss'u.

    Öncelik sırası:
      1. IResNet50-ArcFace (arcface_r50_path verilirse) — önerilir
      2. FaceNet fallback (her zaman kullanılabilir)
    """

    def __init__(self, arcface_r50_path: str = None, facenet_input_size: int = 160):
        super().__init__()
        self.input_size = 112

        self.use_arcface = False
        self.identity_model = None

        if arcface_r50_path is not None:
            try:
                from src.models.iresnet import iresnet50
                self.identity_model = iresnet50(pretrained_path=arcface_r50_path)
                self.use_arcface    = True
                self.input_size     = 112
                print("[IdentityLoss] IResNet50-ArcFace yüklendi.")
            except Exception as e:
                print(f"[IdentityLoss] IResNet50 yüklenemedi ({e}), FaceNet'e geçiliyor.")

        if not self.use_arcface:
            from facenet_pytorch import InceptionResnetV1
            self.identity_model = InceptionResnetV1(pretrained="vggface2").eval()
            for p in self.identity_model.parameters():
                p.requires_grad_(False)
            self.input_size = facenet_input_size
            print("[IdentityLoss] FaceNet (InceptionResnetV1-VGGFace2) kullanılıyor.")

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        x_resized = _to_facenet(x, size=self.input_size)
        emb = self.identity_model(x_resized)
        return F.normalize(emb, dim=1)

    def forward(self, generated: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
        gen_emb  = self._encode(generated)
        real_emb = self._encode(real).detach()
        cosine   = F.cosine_similarity(gen_emb, real_emb, dim=1)
        return 1.0 - cosine.mean()


# ─── LPIPS Perceptual Loss (opsiyonel) ─────────────────────────────────────────

class LPIPSPerceptualLoss(nn.Module):
    """
    Öğrenilmiş algısal mesafe (Zhang et al.). VGG feature-MSE'ye EK bir
    keskinlik/doku sinyali sağlar; VGGPerceptualLoss'un yerine değil,
    yanına kullanılır. `lpips` paketi kurulu değilse açık ImportError fırlatır
    (bu yüzden yalnızca cfg.loss.use_lpips=True iken import edilir).

    Girdi: [-1, 1] aralığında (lpips kütüphanesinin beklediği format,
    ekstra normalize gerekmez).
    """

    def __init__(self, net: str = "alex"):
        super().__init__()
        import lpips as lpips_lib
        self.model = lpips_lib.LPIPS(net=net)
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.eval()

    def forward(self, generated: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
        return self.model(generated, real).mean()


# ─── SSIM Loss ────────────────────────────────────────────────────────────────

class SSIMLoss(nn.Module):
    """1 - SSIM(generated, real). pytorch-msssim kullanır."""

    def forward(self, generated: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
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
    """

    def __init__(self, vgg_layer: str = "relu2_2,relu3_3", facenet_input_size: int = 160,
                 arcface_r50_path: str = None, use_lpips: bool = False, lpips_net: str = "alex"):
        super().__init__()
        self.perceptual = VGGPerceptualLoss(layer=vgg_layer)
        self.identity   = IdentityLoss(
            arcface_r50_path  = arcface_r50_path,
            facenet_input_size = facenet_input_size,
        )
        self.ssim_loss  = SSIMLoss()

        self.lpips_loss = None
        if use_lpips:
            try:
                self.lpips_loss = LPIPSPerceptualLoss(net=lpips_net)
                print(f"[ReconstructionLoss] LPIPS ({lpips_net}) perceptual loss aktif.")
            except ImportError as e:
                print(f"[ReconstructionLoss] lpips paketi bulunamadi ({e}), "
                      f"LPIPS atlaniyor. `pip install lpips` ile kurun.")

    def forward(
        self,
        generated:  torch.Tensor,
        real:       torch.Tensor,
        weights:    Dict[str, float],
        input_embedding: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        """
        input_embedding: (B, embedding_dim) decoder'a verilen HAM z (opsiyonel).
            Sadece weights["cycle_identity"] > 0 iken kullanılır — bkz.
            cfg.loss.cycle_identity_weight yorumundaki embedding-uzayı uyarısı.
        """
        losses: Dict[str, torch.Tensor] = {}

        losses["l1"] = F.l1_loss(generated, real)

        losses["perceptual"] = (
            self.perceptual(generated, real)
            if weights.get("perceptual", 0.0) > 0
            else torch.zeros((), device=generated.device)
        )

        need_identity = weights.get("identity", 0.0) > 0
        need_cycle = (input_embedding is not None) and (weights.get("cycle_identity", 0.0) > 0)

        # encode(generated) her iki loss icin de ayni dondurulmus agi
        # kullanir — gereksiz ikinci forward'i onlemek icin BIR kez hesapla.
        gen_id_emb = self.identity._encode(generated) if (need_identity or need_cycle) else None

        losses["identity"] = (
            (1.0 - F.cosine_similarity(
                gen_id_emb, self.identity._encode(real).detach(), dim=1
            ).mean())
            if need_identity
            else torch.zeros((), device=generated.device)
        )

        losses["cycle_identity"] = (
            (1.0 - F.cosine_similarity(
                gen_id_emb, F.normalize(input_embedding, dim=1), dim=1
            ).mean())
            if need_cycle
            else torch.zeros((), device=generated.device)
        )

        losses["ssim"] = (
            self.ssim_loss(generated, real)
            if weights.get("ssim", 0.0) > 0
            else torch.zeros((), device=generated.device)
        )

        losses["lpips"] = (
            self.lpips_loss(generated, real)
            if (self.lpips_loss is not None and weights.get("lpips", 0.0) > 0)
            else torch.zeros((), device=generated.device)
        )

        losses["total"] = (
            weights.get("l1",             0.0) * losses["l1"]
            + weights.get("perceptual",    0.0) * losses["perceptual"]
            + weights.get("identity",      0.0) * losses["identity"]
            + weights.get("cycle_identity", 0.0) * losses["cycle_identity"]
            + weights.get("ssim",          0.0) * losses["ssim"]
            + weights.get("lpips",         0.0) * losses["lpips"]
        )

        return losses
