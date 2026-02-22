from datetime import datetime

import lambeq
import matplotlib.pyplot as plt
import numpy as np
import torch

from qnlp.discoclip2.dataset.aro_dataloader import get_aro_dataloader
from qnlp.discoclip2.models.image_model import TTNImageModel
from qnlp.utils.logging import setup_logger

# --- SECURITY & SETUP ---
torch.serialization.add_safe_globals([lambeq.backend.symbol.Symbol])
EXPERIMENT_NAME = "ttn_master_diagnostic"
ts_string = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
logger = setup_logger(log_name=EXPERIMENT_NAME, ts_string=ts_string)
DEVICE = "cpu"  # some ops not supported in mps


def analyze_effective_rank(tensor):
    """
    Calculates the Participation Ratio (Effective Rank).
    Uses a more robust singular value calculation.
    """
    # [Batch, Nodes, Features] or [Batch, Features]
    # We must flatten to [Samples, Features]
    if tensor.dim() == 3:
        flat = tensor.reshape(-1, tensor.shape[-1]).float()
    else:
        flat = tensor.float()

    # Center the data to look at variance/covariance rank
    flat = flat - flat.mean(dim=0)

    try:
        # We use the Eigenvalues of the Covariance matrix for speed and stability
        # Cov = (X^T * X) / (N-1)
        cov = (flat.T @ flat) / (flat.shape[0] - 1)
        eigvals = torch.linalg.eigvalsh(cov)
        eigvals = torch.clamp(eigvals, min=1e-9)  # Prevent division by zero

        # Participation Ratio: (sum(L))^2 / sum(L^2)
        rank = (eigvals.sum() ** 2) / (eigvals**2).sum()
        return rank.item()
    except Exception:
        # Log the error so we know WHY it failed
        return 0.01


def run_diagnostic(checkpoint_path: str):
    logger.info("--- Running Master Health Check ---")

    # 1. Load Model
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    model = TTNImageModel(embedding_dim=512).to(DEVICE)
    model.load_state_dict(checkpoint["image_model_state_dict"])
    model.eval()

    loaders, _ = get_aro_dataloader(batch_size=64, return_images=True)
    test_loader = loaders[2]

    full_metrics = {}

    def make_universal_hook(name):
        def hook(module, input, output):
            # Handle cases where output might be a tuple
            out_data = output[0].detach() if isinstance(output, tuple) else output.detach()

            if name not in full_metrics:
                full_metrics[name] = {"mag": [], "var": [], "rank": [], "sparse": [], "dim": 0}

            # 1. Magnitude & Variance
            mag = out_data.abs().mean().item()
            var = out_data.var().item()

            # 2. Dynamic Sparsity:
            # Instead of fixed 1e-4, use 10% of the mean magnitude as the 'noise floor'
            threshold = mag * 0.1
            sparse = (out_data.abs() < threshold).float().mean().item()

            full_metrics[name]["mag"].append(mag)
            full_metrics[name]["var"].append(var)
            full_metrics[name]["sparse"].append(sparse)
            full_metrics[name]["dim"] = out_data.shape[-1]
            full_metrics[name]["rank"].append(analyze_effective_rank(out_data))

        return hook

    handles = []
    for name, module in model.named_modules():
        if any(target in name for target in ["layers", "head", "final_norm"]):
            if "res_proj" not in name:
                handles.append(module.register_forward_hook(make_universal_hook(name)))

    # 3. Execution Pass
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            _ = model(batch["images"].to(DEVICE))
            if i >= 10:
                break

    for h in handles:
        h.remove()

    # 4. SNR Analysis
    pos_scale_val = model.pos_scale.item()
    pos_mag = (model.positional_embedding.detach() * pos_scale_val).abs().mean().item()
    l0_mag = np.mean(full_metrics["layers.0"]["mag"])
    snr = l0_mag / (pos_mag + 1e-8)

    logger.info(f"\nSNR Check: {snr:.4f} (L0/Position)")

    # 5. Master Table
    logger.info("\n" + "=" * 95)
    header = f"{'Module Name':<22} | {'Mag':<7} | {'Var':<7} | {'Rank':<7} | {'Usage %':<7} | {'Sparse %':<7}"
    logger.info(header)
    logger.info("-" * 95)

    plot_names, plot_mags, plot_usage, plot_sparse = [], [], [], []

    for name in sorted(full_metrics.keys()):
        m = full_metrics[name]
        avg_rank = np.mean(m["rank"])
        usage = (avg_rank / m["dim"]) * 100 if m["dim"] > 0 else 0
        avg_sparse = np.mean(m["sparse"]) * 100

        logger.info(
            f"{name:<22} | {np.mean(m['mag']):.4f} | {np.mean(m['var']):.4f} | "
            f"{avg_rank:>7.1f} | {usage:>7.1f}% | {avg_sparse:>7.1f}%"
        )

        plot_names.append(name)
        plot_mags.append(np.mean(m["mag"]))
        plot_usage.append(usage)
        plot_sparse.append(avg_sparse)

    # 6. Comprehensive Visualization
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

    ax1.plot(plot_names, plot_mags, marker="o", color="teal")
    ax1.set_title(f"Magnitude (SNR: {snr:.2f})")
    ax1.tick_params(axis="x", rotation=45)

    ax2.bar(plot_names, plot_usage, color="salmon", alpha=0.7)
    ax2.set_title("Effective Rank Usage %")
    ax2.set_ylim(0, 100)
    ax2.tick_params(axis="x", rotation=45)

    ax3.bar(plot_names, plot_sparse, color="plum", alpha=0.7)
    ax3.set_title("Sparsity %")
    ax3.set_ylim(0, 100)
    ax3.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_diagnostic("runs/checkpoints/train_vlm_on_aro/2026-02-22_19-26-17/best_model.pt")
