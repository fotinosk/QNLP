from datetime import datetime

import numpy as np
import torch

from qnlp.discoclip2.dataset.aro_dataloader import get_aro_dataloader
from qnlp.discoclip2.models.einsum_model import EinsumModel
from qnlp.discoclip2.models.image_model import TTNImageModel
from qnlp.utils.logging import setup_logger
from qnlp.utils.torch_utils import get_device

EXPERIMENT_NAME = "diagnose_full_model_improved"
ts_string = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
logger = setup_logger(log_name=EXPERIMENT_NAME, ts_string=ts_string)

DEVICE = get_device()


def run_post_mortem(checkpoint_path: str):
    logger.info("--- Diagnostic: Bilinear TTN & Gated Position ---")

    # 1. Load Model
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    image_model = TTNImageModel(embedding_dim=512).to(DEVICE)
    image_model.load_state_dict(checkpoint["image_model_state_dict"])
    image_model.eval()

    text_model = EinsumModel().to(DEVICE)
    text_model.load_state_dict(checkpoint["text_model_state_dict"])
    text_model.eval()

    loaders, _ = get_aro_dataloader(batch_size=64, return_images=True)
    test_loader = loaders[2]

    full_metrics = {}

    # 2. Updated Hook Logic
    def make_universal_hook(name):
        def hook(module, input, output):
            out_data = output.detach()
            if name not in full_metrics:
                full_metrics[name] = {"mag": [], "var": [], "sparsity": []}

            # Record health metrics
            full_metrics[name]["mag"].append(out_data.abs().mean().item())
            full_metrics[name]["var"].append(out_data.var().item())
            full_metrics[name]["sparsity"].append((out_data.abs() < 1e-4).float().mean().item())

        return hook

    handles = []
    # We now target the layers and the final components
    for name, module in image_model.named_modules():
        if any(target in name for target in ["layers", "norm", "head"]):
            handles.append(module.register_forward_hook(make_universal_hook(name)))

    # 3. Execution Pass
    logger.info("Running diagnostic pass...")
    with torch.no_grad():
        for batch in test_loader:
            _ = image_model(batch["images"].to(DEVICE))

    for h in handles:
        h.remove()

    # 4. Bilinear & Positional Analysis (Direct Param Access)
    logger.info("\n--- Bilinear & Positional Health ---")

    # Check the "loudness" of the gated position
    pos_scale_val = image_model.pos_scale.item()
    pos_mag = (image_model.positional_embedding.detach() * pos_scale_val).abs().mean().item()

    # Color vs Pixel factor energy
    c_factor_mag = image_model.color_factor.detach().abs().mean().item()
    p_factor_mag = image_model.pixel_factor.detach().abs().mean().item()

    logger.info(f"Learnable Position Scale: {pos_scale_val:.6f}")
    logger.info(f"Gated Position Magnitude: {pos_mag:.4f}")
    logger.info(f"Color Factor Magnitude:   {c_factor_mag:.4f}")
    logger.info(f"Pixel Factor Magnitude:   {p_factor_mag:.4f}")

    # 5. Tabular Component Report
    logger.info("\n--- Component Health Report ---")
    logger.info(f"{'Module Name':<30} | {'Mag':<8} | {'Var':<8} | {'Sparse %':<8}")
    logger.info("-" * 65)

    names = sorted(full_metrics.keys())
    for name in names:
        m = full_metrics[name]
        logger.info(
            f"{name:<30} | {np.mean(m['mag']):.4f} | {np.mean(m['var']):.4f} | {np.mean(m['sparsity']) * 100:>6.1f}%"
        )

    # 6. Final Recommendation Logic
    # We want to see if the SNR (Image Signal / Gated Position) is finally > 1.0
    # Since we don't have a 'patch_embed' hook anymore, we look at layers.0 input mag
    l0_mag = np.mean(full_metrics["layers.0"]["mag"]) if "layers.0" in full_metrics else 0
    snr = l0_mag / (pos_mag + 1e-8)

    logger.info(f"\nCalculated SNR (L0 Signal / Gated Position): {snr:.4f}")

    if snr < 1.0:
        logger.warning("ALERT: Position is still drowning out image signal. SNR < 1.0.")
    else:
        logger.info("HEALTHY: Image signal is now dominant over position coordinates.")


if __name__ == "__main__":
    PATH = "runs/checkpoints/train_vlm_on_aro/imported_model/best_model.pt"
    run_post_mortem(PATH)
