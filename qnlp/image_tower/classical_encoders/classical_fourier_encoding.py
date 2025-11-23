"""
This is the classical fourier encoding - DFT 
"""
from tqdm import tqdm
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from qnlp.image_tower.base import Encoder


class DFTEncoder(Encoder):
    def __init__(self, batch_size: int = 16, grayscale: bool = False):
        super().__init__()
        self.N = batch_size
        self.grayscale = grayscale
        
    def encode(self, image_path: Path):
        image_array = self.image_path_to_np_array(image_path, grayscale=self.grayscale)
        N = image_array.shape[0]
        M = image_array.shape[1]
        
        n, m = np.mgrid[0:N, 0:M]
        
        dft_base = np.zeros((N, M), dtype=np.complex128)
        
        for u_val in tqdm(range(N)):
            for v_val in range(M):
                kernel = np.exp(-2j * np.pi * (u_val * n / N + v_val * m / M))
                dft_base[u_val, v_val] = np.sum(image_array * kernel)
                
        return dft_base
        
    def encode_batch(self, directory_path):
        # Save the kernel to be reused
        pass
    
    def visualize(self, image_path: Path, sort_by_amplitude=True):
        # Get the original image and its DFT
        original_img = self.image_path_to_np_array(image_path, grayscale=self.grayscale)
        dft_coeffs = self.encode(image_path)
        
        N, M = dft_coeffs.shape
        
        # Create spatial coordinate grids
        n, m = np.mgrid[0:N, 0:M]
        
        # Determine order to add components
        if sort_by_amplitude:
            # Sort by magnitude (descending order)
            magnitudes = np.abs(dft_coeffs)
            u_indices, v_indices = np.mgrid[0:N, 0:M]
            u_flat = u_indices.flatten()
            v_flat = v_indices.flatten()
            mag_flat = magnitudes.flatten()
            
            # Get indices that would sort magnitudes in descending order
            sort_idx = np.argsort(mag_flat)[::-1]  # Descending order
            
            # Reorder frequency indices by magnitude
            u_sorted = u_flat[sort_idx]
            v_sorted = v_flat[sort_idx]
            coeffs_sorted = dft_coeffs.flatten()[sort_idx]
        else:
            # Standard order (u,v) in row-major order
            u_indices, v_indices = np.mgrid[0:N, 0:M]
            u_sorted = u_indices.flatten()
            v_sorted = v_indices.flatten()
            coeffs_sorted = dft_coeffs.flatten()
        
        total_components = len(u_sorted)
        
        # Set up the plot
        fig, ax = plt.subplots(figsize=(10, 8))
        plt.subplots_adjust(bottom=0.15)
        
        # Initial empty reconstruction
        current_reconstruction = np.zeros((N, M), dtype=np.complex128)
        im = ax.imshow(current_reconstruction.real, cmap='gray' if self.grayscale else None)
        ax.set_title(f'DFT Reconstruction: 0 components (sorted by {"amplitude" if sort_by_amplitude else "position"})')
        ax.axis('off')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        
        # Store current component count
        current_component_count = [0]  # Use list to make it mutable in nested function
        
        def update_display():
            """Update the display with current reconstruction."""
            im.set_array(current_reconstruction.real)
            im.set_clim(vmin=current_reconstruction.real.min(), vmax=current_reconstruction.real.max())
            ax.set_title(f'DFT Reconstruction: {current_component_count[0]} components (sorted by {"amplitude" if sort_by_amplitude else "position"})')
            fig.canvas.draw()
        
        def on_plus_clicked(event):
            """Add one more component."""
            nonlocal current_reconstruction  # Declare nonlocal to modify outer variable
            
            if current_component_count[0] < total_components:
                u, v = u_sorted[current_component_count[0]], v_sorted[current_component_count[0]]
                
                if sort_by_amplitude:
                    coeff_val = coeffs_sorted[current_component_count[0]]
                    kernel = np.exp(2j * np.pi * (u*n/N + v*m/M))
                    current_reconstruction += coeff_val * kernel
                else:
                    kernel = np.exp(2j * np.pi * (u*n/N + v*m/M))
                    current_reconstruction += dft_coeffs[u, v] * kernel
                
                current_component_count[0] += 1
                update_display()
        
        def on_minus_clicked(event):
            """Remove the last added component."""
            nonlocal current_reconstruction  # Declare nonlocal to modify outer variable
            
            if current_component_count[0] > 0:
                current_component_count[0] -= 1
                u, v = u_sorted[current_component_count[0]], v_sorted[current_component_count[0]]
                
                if sort_by_amplitude:
                    coeff_val = coeffs_sorted[current_component_count[0]]
                    kernel = np.exp(2j * np.pi * (u*n/N + v*m/M))
                    current_reconstruction -= coeff_val * kernel
                else:
                    kernel = np.exp(2j * np.pi * (u*n/N + v*m/M))
                    current_reconstruction -= dft_coeffs[u, v] * kernel
                
                update_display()
        
        def on_reset_clicked(event):
            """Reset to zero components."""
            nonlocal current_reconstruction  # Declare nonlocal to modify outer variable
            current_reconstruction.fill(0)
            current_component_count[0] = 0
            update_display()
        
        # Create buttons
        ax_plus = plt.axes([0.7, 0.05, 0.1, 0.04])
        ax_minus = plt.axes([0.55, 0.05, 0.1, 0.04])
        ax_reset = plt.axes([0.4, 0.05, 0.1, 0.04])
        
        btn_plus = Button(ax_plus, '+')
        btn_minus = Button(ax_minus, '-')
        btn_reset = Button(ax_reset, 'Reset')
        
        # Connect buttons to functions
        btn_plus.on_clicked(on_plus_clicked)
        btn_minus.on_clicked(on_minus_clicked)
        btn_reset.on_clicked(on_reset_clicked)
        
        plt.show()

if __name__ == "__main__":
    image_path = Path(__file__).parent / "test_images/valid/air hockey/1.jpg"
    dft = DFTEncoder(16, grayscale=True)
    # print(dft.encode(image_path))
    dft.visualize(image_path, sort_by_amplitude=True)
