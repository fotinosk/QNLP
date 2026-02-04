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
