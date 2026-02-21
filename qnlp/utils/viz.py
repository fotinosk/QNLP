import matplotlib.pyplot as plt


def show_tensor(tensor):
    """
    Visualizes a PyTorch tensor with matplotlib.
    Handles shapes: (C, H, W), (1, C, H, W), or (H, W).
    """
    img = tensor.detach().cpu()
    # Remove batch dimension if present
    if img.dim() == 4:
        img = img[0]
    # Permute dimensions if needed
    if img.dim() == 3 and img.size(0) in [1, 3]:
        img = img.permute(1, 2, 0)
    # Handle grayscale/RGB
    if img.dim() == 3 and img.size(-1) == 1:
        img = img.squeeze(-1)
        plt.imshow(img, cmap="gray")
    else:
        plt.imshow(img.clamp(0, 1))
    plt.axis("off")
    plt.show()
