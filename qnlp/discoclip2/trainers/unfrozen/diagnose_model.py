from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch

from qnlp.discoclip2.dataset.aro_dataloader import get_aro_dataloader
from qnlp.discoclip2.models.einsum_model import EinsumModel
from qnlp.discoclip2.models.image_model import TTNImageModel
from qnlp.utils.logging import setup_logger
from qnlp.utils.torch_utils import get_device

EXPERIMENT_NAME = "diagnose_full_model"
ts_string = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
logger = setup_logger(log_name=EXPERIMENT_NAME, ts_string=ts_string)

DEVICE = get_device()


def run_post_mortem(checkpoint_path: str):
    logger.info(f"--- Starting Holistic Model Diagnostic: {checkpoint_path} ---")

    # 1. Load best checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)

    # Reconstruct Models
    image_model = TTNImageModel(embedding_dim=512).to(DEVICE)
    image_model.load_state_dict(checkpoint["image_model_state_dict"])
    image_model.eval()

    text_model = EinsumModel().to(DEVICE)
    text_model.load_state_dict(checkpoint["text_model_state_dict"])
    text_model.eval()

    # 2. Setup Data (ARO Test Set)
    loaders, _ = get_aro_dataloader(batch_size=64, return_images=True)
    test_loader = loaders[2]

    # 3. Global Metrics Dictionary
    # We track every named module to find hidden bottlenecks
    full_metrics = {}

    def make_universal_hook(name):
        def hook(module, input, output):
            # Handle cases where output might be a tuple
            out_data = output[0] if isinstance(output, tuple) else output
            out_data = out_data.detach()

            if name not in full_metrics:
                full_metrics[name] = {"mag": [], "var": [], "sparsity": []}

            full_metrics[name]["mag"].append(out_data.abs().mean().item())
            full_metrics[name]["var"].append(out_data.var().item())
            full_metrics[name]["sparsity"].append((out_data.abs() < 1e-4).float().mean().item())

        return hook

    # Register hooks for all major components
    handles = []
    for name, module in image_model.named_modules():
        # Filter to only log significant layers to avoid clutter
        if any(target in name for target in ["patch_embed", "layers", "norm", "head"]):
            handles.append(module.register_forward_hook(make_universal_hook(name)))

    # 4. Diagnostic Pass
    logger.info("Executing diagnostic pass...")
    with torch.no_grad():
        for batch in test_loader:
            _ = image_model(batch["images"].to(DEVICE))

    for h in handles:
        h.remove()

    # 5. Analysis of Static Weights (The "Energy" of the network)
    logger.info("\n--- Static Parameter Analysis ---")
    pos_emb_mag = image_model.positional_embedding.detach().abs().mean().item()
    logger.info(f"Positional Embedding Mean Magnitude: {pos_emb_mag:.4f}")

    # Check if Positional Embedding dominates Layer 0
    patch_out_mag = np.mean(full_metrics["patch_embed"]["mag"])
    snr = patch_out_mag / pos_emb_mag
    logger.info(f"Signal-to-Position Ratio (SNR): {snr:.4f} (Low SNR means position masks color)")

    # 6. Tabular Health Report
    logger.info("\n--- Component Health Report ---")
    logger.info(f"{'Module Name':<30} | {'Mag':<8} | {'Var':<8} | {'Sparse %':<8}")
    logger.info("-" * 65)

    names = sorted(full_metrics.keys())
    for name in names:
        m = full_metrics[name]
        logger.info(
            f"{name:<30} | {np.mean(m['mag']):.4f} | {np.mean(m['var']):.4f} | {np.mean(m['sparsity']) * 100:>6.1f}%"
        )

    # 7. Multi-Pane Visualization
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Plot Magnitudes
    comp_names = [n.replace("layers.", "L") for n in names]
    ax1.bar(comp_names, [np.mean(full_metrics[n]["mag"]) for n in names], color="royalblue")
    ax1.set_title("Signal Magnitude per Component")
    ax1.set_ylabel("Mean Absolute Value")
    ax1.tick_params(axis="x", rotation=45)

    # Plot Variance (Feature Diversity)
    ax2.bar(comp_names, [np.mean(full_metrics[n]["var"]) for n in names], color="seagreen")
    ax2.set_title("Feature Variance per Component")
    ax2.set_ylabel("Variance")
    ax2.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    # plot_path = f"full_model_diagnostic_{ts_string}.png"
    # plt.savefig(plot_path)
    # logger.info(f"\nFull diagnostic plot saved to {plot_path}")
    plt.show()


if __name__ == "__main__":
    PATH = "runs/checkpoints/train_vlm_on_aro/imported_model/best_model.pt"
    run_post_mortem(PATH)
