"""
3/5 — audit_loss_preprocessing.py: Loss giriş sözleşmeleri denetimi.

İki bölüm:
    A) STATİK kod denetimi (losses.py'yi okuyup her bileşenin decoder
       çıktısını [-1,1] -> hedef aralığa nasıl dönüştürdüğünü raporlar).
    B) DİNAMİK ölçüm (checkpoint + gerçek veri gerekir): her loss bileşeni
       için ham değer, config ağırlığı, ağırlıklı değer VE decoder
       parametreleri üzerindeki gradyan normu (sum of abs grad) — TEK
       BAŞINA (izole, diğer bileşenler backward'a katılmadan).

Kullanım:
    from src.audit_loss_preprocessing import run_static_audit, run_dynamic_audit
    run_static_audit()
    run_dynamic_audit(cfg, manifest_path, "/path/checkpoint.pt")
"""

import logging

import torch

from ..config import Config
from ..models.losses import ReconstructionLoss
from .common import build_model_from_checkpoint, sample_val_batch

log = logging.getLogger(__name__)


# ─── A) Statik kod denetimi ────────────────────────────────────────────────────

_STATIC_FINDINGS = """
STATİK DENETİM — src/models/losses.py (satır referansları bu oturumdaki
kod haline göre; içerik son okunduğunda değişmemişti):

[L1]  (losses.py: 'losses["l1"] = F.l1_loss(generated, real)')
  Dönüşüm: YOK — doğrudan [-1,1] üzerinde. DOĞRU (L1 skala-bağımsız,
  ekstra dönüşüm gerekmez, target de aynı [-1,1] aralığında).

[SSIM]  (SSIMLoss.forward)
  gen_01 = generated*0.5+0.5; real_01 = real*0.5+0.5
  ssim(gen_01, real_01, data_range=1.0)
  Dönüşüm [-1,1]->[0,1] UYGULANIYOR, data_range=1.0 bu aralıkla TUTARLI.
  DOĞRU — önceki oturumda da doğrulanmıştı, regresyon yok.

[VGG16 Perceptual]  (_to_imagenet fonksiyonu)
  x01 = x*0.5+0.5              # [-1,1] -> [0,1]
  (x01 - IMAGENET_MEAN) / IMAGENET_STD
  [-1,1] -> [0,1] -> ImageNet mean/std dönüşümü UYGULANIYOR. DOĞRU —
  kullanıcının şüphelendiği "eksik dönüşüm" senaryosu BURADA GEÇERLİ DEĞİL,
  bu adım mevcut ve doğru.

[Identity — ArcFace/FaceNet]  (_to_facenet fonksiyonu + IdentityLoss._encode)
  x_resized = F.interpolate(x, size=(input_size,input_size), mode="bilinear")
  emb = identity_model(x_resized)         # DÖNÜŞÜM YOK, [-1,1] DOĞRUDAN VERİLİYOR
  emb = F.normalize(emb, dim=1)           # L2 norm SONRA yapılıyor
  Girdi boyutu : 112 (IResNet50-ArcFace) veya 160 (FaceNet fallback) —
                 arcface_r50_path verilip verilmemesine göre otomatik seçilir.
  [-1,1] aralığı: ArcFace-ailesi modeller (deepinsight/insightface) ve
  facenet-pytorch'un InceptionResnetV1'i GELENEKSEL OLARAK [-1,1]
  normalize girdi bekler (ör. (img/255-0.5)/0.5) — ImageNet mean/std
  normalizasyonu KULLANMAZLAR. Yani ImageNet-normalize UYGULANMAMASI
  burada BEKLENEN/DOĞRU davranıştır (VGG'den farklı bir sözleşme).
  RGB/BGR: kodda AÇIK bir kanal-çevirme YOK — girdi tensörünün kanal
  sırası (pipeline boyunca RGB, preprocess.py+dataset.py ile doğrulandı)
  DEĞİŞTİRİLMEDEN identity_model'e veriliyor.
  ⚠ DOĞRULANAMAYAN NOKTA: bu kod, hem facenet-pytorch (RGB eğitilmiş,
  bilinen/güvenilir) hem de `arcface_r50_path` ile yüklenen HARİCİ bir
  IResNet50 checkpoint'i için AYNI RGB varsayımını yapıyor. IResNet50
  ağırlıklarının orijinal eğitiminde RGB mi BGR mi kullanıldığı, bu
  checkpoint'in KENDİ kaynağına (hangi script/repo ile üretildiğine) bağlı
  — koddan kesin olarak doğrulanamaz. Eğer arcface_r50_path kullanıyorsanız
  ve o ağırlıklar BGR-eğitilmiş bir pipeline'dan geliyorsa, identity loss
  YANLIŞ kanal sırasıyla çalışıyor olabilir. Bu TEK gerçek şüpheli nokta.
  Alignment template: preprocess.py insightface'in standart 5-nokta
  norm_crop şablonunu kullanıyor — ArcFace-ailesi modellerin çoğu bu
  şablonla eğitildiği için genelde tutarlıdır, ama arcface_r50_path'in
  KENDİ orijinal eğitim alignment'ıyla aynı olduğu kesin değildir.
"""


def run_static_audit() -> str:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    log.info(_STATIC_FINDINGS)
    return _STATIC_FINDINGS


# ─── B) Dinamik ölçüm: ham/ağırlıklı loss + gradyan normu ─────────────────────

def _grad_norm(model: torch.nn.Module) -> float:
    return sum(p.grad.abs().sum().item() for p in model.parameters() if p.grad is not None)


def run_dynamic_audit(
    cfg: Config,
    manifest_path: str,
    checkpoint_path: str,
    n_samples: int = 16,
    weights_override: dict = None,
) -> dict:
    """
    Her loss bileşenini İZOLE ederek (weights={component: 1.0}, diğerleri 0)
    backward alır ve decoder parametreleri üzerindeki toplam |grad| normunu
    ölçer. weights_override verilmezse cfg.loss.phase3_weights kullanılır
    (üretim ağırlıkları — "config ağırlığı" sütunu için).
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, state = build_model_from_checkpoint(checkpoint_path, cfg, device)
    model.train()  # BN istatistikleri değil, sadece backward icin gradyan gerekli

    loss_fn = ReconstructionLoss(
        vgg_layer=cfg.loss.vgg_layer,
        facenet_input_size=cfg.loss.facenet_input_size,
        arcface_r50_path=cfg.loss.arcface_r50_path,
        use_lpips=cfg.loss.use_lpips,
        lpips_net=cfg.loss.lpips_net,
    ).to(device)

    config_weights = weights_override or cfg.loss.phase3_weights
    z, target = sample_val_batch(manifest_path, cfg, n_samples=n_samples, device=device)

    components = ["l1", "perceptual", "identity", "ssim"]
    if cfg.loss.use_lpips:
        components.append("lpips")
    if cfg.loss.cycle_identity_weight > 0:
        components.append("cycle_identity")

    rows = []
    for comp in components:
        model.zero_grad(set_to_none=True)
        generated = model(z)
        isolate_weights = {comp: 1.0}
        losses = loss_fn(generated, target, isolate_weights, input_embedding=z)
        raw = losses[comp]
        cfg_w = config_weights.get(comp, cfg.loss.lpips_weight if comp == "lpips"
                                    else cfg.loss.cycle_identity_weight if comp == "cycle_identity" else 0.0)
        weighted = cfg_w * raw
        if weighted.requires_grad:
            weighted.backward()
        gnorm = _grad_norm(model)
        rows.append({
            "component": comp, "raw_loss": raw.item(), "config_weight": cfg_w,
            "weighted_loss": weighted.item(), "grad_norm": gnorm,
        })

    log.info("=" * 100)
    log.info("AUDIT_LOSS_PREPROCESSING — Dinamik ölçüm (izole gradyan normu)")
    log.info(f"{'Bileşen':<16}{'Ham loss':>12}{'Config ağırlığı':>18}{'Ağırlıklı loss':>16}{'|Grad| normu':>16}")
    for r in rows:
        log.info(
            f"{r['component']:<16}{r['raw_loss']:>12.5f}{r['config_weight']:>18.4f}"
            f"{r['weighted_loss']:>16.5f}{r['grad_norm']:>16.3f}"
        )
    log.info("-" * 100)

    max_gnorm = max(r["grad_norm"] for r in rows) or 1.0
    for r in rows:
        rel = r["grad_norm"] / max_gnorm
        verdict = "SAĞLIKLI" if rel > 0.01 else "ŞÜPHELİ (diğerlerine göre ihmal edilebilir gradyan etkisi)"
        log.info(f"  [{r['component']}] göreli gradyan etkisi: {rel:.4f}  -> {verdict}")
    log.info("=" * 100)

    return {"rows": rows}


if __name__ == "__main__":
    run_static_audit()
