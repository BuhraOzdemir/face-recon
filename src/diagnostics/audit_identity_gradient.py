"""
4/5 — audit_identity_gradient.py: Identity loss gradyan akışı birim testi.

optimizer.zero_grad() -> loss_identity.backward() -> decoder parametreleri
üzerindeki gradyan normunu KATMAN KATMAN (MLP head'den output_conv'a kadar)
raporlar. Sadece çıkış katmanında gradyan olması YETERLİ DEĞİL — embedding
projeksiyonuna (mlp, decoder'ın EN BAŞI) kadar ulaştığı doğrulanmalı;
aksi halde identity loss decoder'ın erken katmanlarını hiç etkilemiyor
olabilir (örn. bir .detach() sızıntısı veya kopuk bir hesaplama grafiği).

Kullanım:
    from src.audit_identity_gradient import run_identity_gradient_test
    run_identity_gradient_test(cfg, manifest_path, "/path/checkpoint.pt")
"""

import logging

import torch

from ..config import Config
from ..models.losses import ReconstructionLoss
from .common import build_model_from_checkpoint, sample_val_batch

log = logging.getLogger(__name__)


def _layer_group_grad_norms(model: torch.nn.Module) -> dict:
    """decoder'ı mantıksal katman gruplarına ayırıp her grubun |grad| normunu döndürür."""
    groups = {"mlp_head (EN BAŞ)": model.mlp}
    for i, block in enumerate(model.up_blocks):
        groups[f"up_block_{i}"] = block
    groups["output_conv (EN SON)"] = model.output_conv

    result = {}
    for name, module in groups.items():
        grads = [p.grad.abs().sum().item() for p in module.parameters() if p.grad is not None]
        n_params_with_grad = sum(1 for p in module.parameters() if p.grad is not None)
        n_params_total = sum(1 for _ in module.parameters())
        result[name] = {
            "grad_norm": sum(grads),
            "n_params_with_grad": n_params_with_grad,
            "n_params_total": n_params_total,
        }
    return result


def run_identity_gradient_test(
    cfg: Config,
    manifest_path: str,
    checkpoint_path: str,
    n_samples: int = 16,
) -> dict:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, state = build_model_from_checkpoint(checkpoint_path, cfg, device)
    model.train()

    loss_fn = ReconstructionLoss(
        vgg_layer=cfg.loss.vgg_layer,
        facenet_input_size=cfg.loss.facenet_input_size,
        arcface_r50_path=cfg.loss.arcface_r50_path,
    ).to(device)

    z, target = sample_val_batch(manifest_path, cfg, n_samples=n_samples, device=device)

    model.zero_grad(set_to_none=True)
    generated = model(z)
    losses = loss_fn(generated, target, {"identity": 1.0}, input_embedding=z)
    loss_identity = losses["identity"]
    log.info(f"[Identity Gradient Test] loss_identity (ham) = {loss_identity.item():.5f}")

    if not loss_identity.requires_grad:
        log.error(
            "[HATA] loss_identity.requires_grad=False — hesaplama grafiği "
            "generated'e hiç bağlı değil, backward ANLAMSIZ. identity ağırlığı "
            "0 olabilir mi kontrol et (weights.get('identity',0)>0 gerekir)."
        )
        return {"ok": False, "reason": "requires_grad=False"}

    loss_identity.backward()

    groups = _layer_group_grad_norms(model)

    log.info("=" * 90)
    log.info("AUDIT_IDENTITY_GRADIENT — Katman katman gradyan akışı")
    log.info(f"{'Katman grubu':<24}{'|Grad| normu':>16}{'Grad''li param':>16}{'Toplam param':>14}")
    for name, r in groups.items():
        log.info(
            f"{name:<24}{r['grad_norm']:>16.5f}{r['n_params_with_grad']:>16d}{r['n_params_total']:>14d}"
        )
    log.info("-" * 90)

    total_grad_norm = sum(r["grad_norm"] for r in groups.values())
    mlp_grad_norm = groups["mlp_head (EN BAŞ)"]["grad_norm"]
    mlp_params_with_grad = groups["mlp_head (EN BAŞ)"]["n_params_with_grad"]
    mlp_params_total = groups["mlp_head (EN BAŞ)"]["n_params_total"]

    decoder_has_grad = total_grad_norm > 0
    mlp_has_grad = mlp_grad_norm > 0 and mlp_params_with_grad == mlp_params_total

    verdict_decoder = "SAĞLIKLI" if decoder_has_grad else "ŞÜPHELİ (decoder_grad=0, gradyan HİÇ akmıyor)"
    verdict_mlp = (
        "SAĞLIKLI (gradyan en başa kadar ulaşıyor)" if mlp_has_grad
        else "ŞÜPHELİ (MLP head'e gradyan ulaşmıyor — sadece geç katmanlar etkileniyor)"
    )

    log.info(f"  Toplam decoder |grad| normu : {total_grad_norm:.5f}  -> {verdict_decoder}")
    log.info(f"  MLP head |grad| normu       : {mlp_grad_norm:.5f}  -> {verdict_mlp}")
    log.info("=" * 90)

    return {
        "ok": True, "total_grad_norm": total_grad_norm, "mlp_grad_norm": mlp_grad_norm,
        "groups": groups, "verdict_decoder": verdict_decoder, "verdict_mlp": verdict_mlp,
    }


if __name__ == "__main__":
    import sys
    from ..config import DEFAULT_CONFIG
    run_identity_gradient_test(DEFAULT_CONFIG, sys.argv[1], sys.argv[2])
