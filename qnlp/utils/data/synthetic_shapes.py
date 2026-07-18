import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class SyntheticShapesDataset(Dataset):
    """
    A programmatic dataset that generates 2D shape images on the fly.
    Supports three modes to test color/shape representation learning:
    - mode="color_only": Correlated unique shape/color combinations (Circle=Red, Square=Green, Triangle=Blue, Cross=Yellow).
    - mode="shape_only": Grayscale images of the 4 shapes. Color info is removed.
    - mode="overlapping": Overlapping shapes/colors (Red Circle, Red Square, Green Circle, Green Square).
    """

    def __init__(self, num_samples=1000, img_size=16, mode="overlapping", transform=None, seed=42):
        self.num_samples = num_samples
        self.img_size = img_size
        self.mode = mode
        self.transform = transform

        # Set seed for reproducibility
        self.rng = np.random.default_rng(seed)

        # Pre-generate labels so they are fixed
        self.labels = self.rng.choice(4, size=num_samples)

    def __len__(self):
        return self.num_samples

    def _draw_circle(self, grid_x, grid_y, cx, cy, r=4.0):
        return (grid_x - cx) ** 2 + (grid_y - cy) ** 2 <= r**2

    def _draw_square(self, grid_x, grid_y, cx, cy, hw=3.5):
        return (np.abs(grid_x - cx) <= hw) & (np.abs(grid_y - cy) <= hw)

    def _draw_triangle(self, grid_x, grid_y, cx, cy, h=4.0):
        # A simple upward pointing triangle
        y_cond = (grid_y >= cy - h / 2) & (grid_y <= cy + h / 2)
        x_cond = np.abs(grid_x - cx) <= (cy + h / 2 - grid_y) * 0.8
        return y_cond & x_cond

    def _draw_cross(self, grid_x, grid_y, cx, cy, l=4.0, w=1.0):
        v_bar = (np.abs(grid_x - cx) <= w) & (np.abs(grid_y - cy) <= l)
        h_bar = (np.abs(grid_y - cy) <= w) & (np.abs(grid_x - cx) <= l)
        return v_bar | h_bar

    def __getitem__(self, idx):
        label = self.labels[idx]

        # Initialize background: dark gray [0.05, 0.05, 0.05]
        img = np.zeros((3, self.img_size, self.img_size), dtype=np.float32) + 0.05

        # Create grid coordinates centered around middle of the image
        mid = (self.img_size - 1) / 2.0

        scale = self.img_size / 16.0

        # Add random spatial offset (jitter) to prevent absolute coordinate memorization
        offset_x = self.rng.uniform(-1.5, 1.5) * scale
        offset_y = self.rng.uniform(-1.5, 1.5) * scale
        cx, cy = mid + offset_x, mid + offset_y

        # Create coordinate grids
        x = np.arange(self.img_size)
        y = np.arange(self.img_size)
        grid_x, grid_y = np.meshgrid(x, y)

        # Draw the appropriate shape and assign color based on mode
        if self.mode == "color_only":
            if label == 0:  # Red Circle
                mask = self._draw_circle(grid_x, grid_y, cx, cy, r=self.rng.uniform(3.2, 4.0) * scale)
                img[0, mask] = self.rng.uniform(0.85, 1.0)
            elif label == 1:  # Green Square
                mask = self._draw_square(grid_x, grid_y, cx, cy, hw=self.rng.uniform(3.0, 3.8) * scale)
                img[1, mask] = self.rng.uniform(0.85, 1.0)
            elif label == 2:  # Blue Triangle
                mask = self._draw_triangle(grid_x, grid_y, cx, cy, h=self.rng.uniform(7.0, 8.5) * scale)
                img[2, mask] = self.rng.uniform(0.85, 1.0)
            elif label == 3:  # Yellow Cross
                mask = self._draw_cross(grid_x, grid_y, cx, cy, l=self.rng.uniform(3.5, 4.5) * scale, w=0.8 * scale)
                img[0, mask] = self.rng.uniform(0.85, 1.0)
                img[1, mask] = self.rng.uniform(0.85, 1.0)

        elif self.mode == "shape_only":
            val = self.rng.uniform(0.85, 1.0)
            if label == 0:  # Circle
                mask = self._draw_circle(grid_x, grid_y, cx, cy, r=self.rng.uniform(3.2, 4.0) * scale)
            elif label == 1:  # Square
                mask = self._draw_square(grid_x, grid_y, cx, cy, hw=self.rng.uniform(3.0, 3.8) * scale)
            elif label == 2:  # Triangle
                mask = self._draw_triangle(grid_x, grid_y, cx, cy, h=self.rng.uniform(7.0, 8.5) * scale)
            elif label == 3:  # Cross
                mask = self._draw_cross(grid_x, grid_y, cx, cy, l=self.rng.uniform(3.5, 4.5) * scale, w=0.8 * scale)
            img[:, mask] = val  # Grayscale (White)

        elif self.mode == "overlapping":
            # Shape Selection
            if label in [0, 2]:  # Circle
                mask = self._draw_circle(grid_x, grid_y, cx, cy, r=self.rng.uniform(3.2, 4.0) * scale)
            else:  # Square
                mask = self._draw_square(grid_x, grid_y, cx, cy, hw=self.rng.uniform(3.0, 3.8) * scale)

            # Color Selection
            if label in [0, 1]:  # Red
                img[0, mask] = self.rng.uniform(0.85, 1.0)
            else:  # Green
                img[1, mask] = self.rng.uniform(0.85, 1.0)

        # Add tiny Gaussian noise to represent real-world channel fluctuations
        noise = self.rng.normal(0, 0.03, size=img.shape).astype(np.float32)
        img = np.clip(img + noise, 0.0, 1.0)

        # Convert to torch tensor
        img_tensor = torch.tensor(img)

        if self.transform:
            img_tensor = self.transform(img_tensor)

        return img_tensor, torch.tensor(label, dtype=torch.long)


def get_synthetic_shapes_loaders(
    batch_size=64, train_samples=1000, test_samples=200, img_size=16, mode="overlapping", seed=42
):
    """
    Returns train and test dataloaders for the synthetic shapes dataset.
    """
    train_dataset = SyntheticShapesDataset(num_samples=train_samples, img_size=img_size, mode=mode, seed=seed)
    test_dataset = SyntheticShapesDataset(num_samples=test_samples, img_size=img_size, mode=mode, seed=seed + 1000)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


if __name__ == "__main__":
    # Small self-check script
    print("Checking Synthetic Shapes Dataset configurations...")
    train_loader, _ = get_synthetic_shapes_loaders(batch_size=4, train_samples=8, img_size=32, mode="overlapping")
    for imgs, labels in train_loader:
        print(f"Overlapping Batch image shape: {imgs.shape} (expect [4, 3, 32, 32])")
        print(f"Labels:                         {labels.tolist()}")
        break
