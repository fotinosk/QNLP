from PIL import Image
from pathlib import Path
import numpy as np
from torchvision import transforms

def preprocess_clip_image(image_path: Path) -> np.array:
    """
    Applies CLIP-style preprocessing to an image from a file path.
    
    Args:
        image_path (str): Path to the input image file.

    Returns:
        torch.Tensor: Preprocessed image tensor of shape (C, H, W) normalized for CLIP.
    """
    # Define the preprocessing pipeline
    preprocess = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.Lambda(lambda image: image.convert('RGB')),  # Ensure RGB
        transforms.ToTensor(),  # Convert to tensor (values in [0, 1])
        transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711)
        )
    ])
    
    # Load and preprocess the image
    image = Image.open(image_path).convert("RGB")
    image_tensor = preprocess(image)
    
    return image_tensor


if __name__ == "__main__":
    print(
        preprocess_clip_image(
            Path(__file__).parent.parent / "test_images/valid/air hockey/1.jpg"
        )
    )