import numpy as np
from pathlib import Path
from qnlp.image_tower.classical_fourier_encoding import DFTEncoder


def test_dft_vs_numpy_fft():
    """Test that DFT implementation matches NumPy's FFT using the example image."""
    # Use the actual image from the example
    image_path = Path(__file__).parent.parent.parent / "qnlp/image_encoding/test_images/valid/air hockey/1.jpg"
    
    encoder = DFTEncoder(batch_size=16, grayscale=True)
    
    # Get result from encoder
    result_encoder = encoder.encode(image_path)
    
    # Load image and compute reference with NumPy
    img_array = encoder.image_path_to_np_array(image_path, grayscale=True)
    result_numpy = np.fft.fft2(img_array)
    
    # Verify results are close (allowing for floating point precision)
    assert np.allclose(result_encoder, result_numpy, atol=1e-8), \
        f"Results differ by more than tolerance. Max diff: {np.max(np.abs(result_encoder - result_numpy))}"
    
    # Verify shapes match
    assert result_encoder.shape == result_numpy.shape, \
        f"Shape mismatch: {result_encoder.shape} vs {result_numpy.shape}"
