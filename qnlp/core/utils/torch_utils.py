import os

import torch


def get_device():
    """
    Returns the best available device: mps, cuda, or cpu.
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def create_checkpoint_path(experiment_name: str, ts_string: str) -> str:
    checkpoint_path = os.path.join(f"./runs/checkpoints/{experiment_name}", f"{ts_string}/best_model.pt")
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    return checkpoint_path
