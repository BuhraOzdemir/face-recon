"""
Bağımsız Identity Evaluator — eğitim lossuna girmez, yalnızca ölçüm yapar.

Mimari kural (PDF v2.0 §7):
    ArcFace eğitimde supervisor olduğu için nihai hakem olamaz.
    Bu modül eğitimde hiç kullanılmayan FaceNet veya AdaFace ile
    Type-I / Type-II kimlik skorunu ölçer.

Kullanım:
    evaluator = IndependentEvaluator()
    score = evaluator.cosine_sim(generated_imgs, real_imgs)  # 0-1 arası
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class IndependentEvaluator(nn.Module):
    """
    Eğitimde hiç kullanılmayan bağımsız kimlik değerlendirici.

    FaceNet (InceptionResnetV1, VGGFace2) kullanır.
    ArcFace/IResNet50 (training supervisor) ile aynı model değildir.

    Kullanım alanları:
        - Validation sırasında diagnostic ölçüm (loss'a girmez)
        - Test setinde Type-I ve Type-II kimlik skoru
        - Bağımsız TAR/SAR hesabı
    """

    def __init__(self, input_size: int = 160):
        super().__init__()
        from facenet_pytorch import InceptionResnetV1

        self.facenet    = InceptionResnetV1(pretrained="vggface2").eval()
        self.input_size = input_size

        for p in self.facenet.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        [-1,1] görüntüyü L2-normalized FaceNet embedding'ine çevir.

        Args:
            x: (B, 3, H, W) float tensor, [-1,1]
        Returns:
            emb: (B, 512) L2-normalized
        """
        if x.shape[-1] != self.input_size:
            x = F.interpolate(x, size=(self.input_size, self.input_size),
                              mode="bilinear", align_corners=False)
        emb = self.facenet(x)
        return F.normalize(emb, dim=1)

    @torch.no_grad()
    def cosine_sim(
        self,
        generated: torch.Tensor,
        real:      torch.Tensor,
    ) -> torch.Tensor:
        """
        Üretilen ve gerçek görüntüler arasındaki ortalama cosine similarity.

        Returns:
            scalar: ortalama cosine similarity (yüksek = iyi kimlik korunumu)
        """
        gen_emb  = self.encode(generated)
        real_emb = self.encode(real)
        return F.cosine_similarity(gen_emb, real_emb, dim=1).mean()

    @torch.no_grad()
    def type2_score(
        self,
        generated:  torch.Tensor,
        probe_imgs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Type-II değerlendirme: üretilen yüz vs aynı kişinin GÖRÜLMEMIŞ fotoğrafı.

        Args:
            generated:  (B, 3, H, W) decoder çıktısı
            probe_imgs: (B, 3, H, W) aynı kimliğin eğitimde görülmemiş fotoğrafı
        Returns:
            per-sample cosine similarity: (B,)
        """
        gen_emb   = self.encode(generated)
        probe_emb = self.encode(probe_imgs)
        return F.cosine_similarity(gen_emb, probe_emb, dim=1)

    @torch.no_grad()
    def tar_at_far(
        self,
        genuine_scores: torch.Tensor,
        impostor_scores: torch.Tensor,
        far_target: float = 0.01,
    ) -> Tuple[float, float]:
        """
        TAR @ FAR threshold hesabı.

        Args:
            genuine_scores:  (N,) aynı kişi çiftleri cosine similarity
            impostor_scores: (M,) farklı kişi çiftleri cosine similarity
            far_target: hedef FAR (0.01 = %1)
        Returns:
            (tar, threshold): TAR değeri ve eşik
        """
        threshold = float(torch.quantile(impostor_scores,
                                         1.0 - far_target).item())
        tar = float((genuine_scores >= threshold).float().mean().item())
        return tar, threshold
