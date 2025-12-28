from qnlp.image_tower.reconstruction.ttn_reconstruction_model import QuadTreeAutoencoder

import time
import wandb
import torch.optim as optim
import wandb
import time
import torch
from torch import nn
from qnlp.utils.data import get_mnist_loaders
from qnlp.image_tower.classification.ttn_quadtree_classification_model import DEVICE, LEARNING_RATE, EPOCHS, BOND_DIM, CP_RANK, PATCH_SIZE, BATCH_SIZE
import os
from torchvision.utils import save_image


def save_reconstruction_results(original, reconstructed, directory, filename="reconstruction.jpg"):
    """
    Saves the original and reconstructed tensors as a single comparison JPEG.
    
    Args:
        original (torch.Tensor): Original image tensor [C, H, W] or [1, C, H, W]
        reconstructed (torch.Tensor): Output tensor from the decoder
        directory (str): Path to the folder where image will be saved
        filename (str): Name of the file (should end in .jpg or .png)
    """
    # Create directory if it doesn't exist
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # Ensure tensors are on CPU and detached from the graph
    orig = original.detach().cpu()
    reco = reconstructed.detach().cpu()
    
    # If tensors are in a batch [B, C, H, W], take the first one
    if orig.dim() == 4:
        orig = orig[0]
    if reco.dim() == 4:
        reco = reco[0]
        
    # Clip reconstructed values to valid range [0, 1] 
    reco = torch.clamp(reco, 0, 1)
    
    # Stack images side-by-side (dim=2 is Width)
    comparison = torch.cat([orig, reco], dim=2)
    
    # Full path
    save_path = os.path.join(directory, filename)
    
    # save_image handles the 0-255 scaling and JPEG encoding
    save_image(comparison, save_path)
    print(f"Saved reconstruction comparison to {save_path}")

def train_reconstruction(log_run: bool = True):
    # --- Setup ---
    if log_run:
        run = wandb.init(project="quad-image-reconstruction", name="autoencoder-v1", save_code=True)
    
    train_loader, test_loader = get_mnist_loaders(batch_size=BATCH_SIZE)
    
    # Use the combined Autoencoder class from the previous response
    model = QuadTreeAutoencoder(
        img_size=32,
        patch_size=PATCH_SIZE,
        bond_dim=BOND_DIM,
        cp_rank=CP_RANK,
        in_channels=1,
        dropout=0.0
    ).to(DEVICE)
    
    if log_run:
        wandb.watch(model, log="all", log_freq=100)
    
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LEARNING_RATE, 
                                              steps_per_epoch=len(train_loader), epochs=EPOCHS)
    
    # RECONSTRUCTION LOSS: MSE measures pixel-to-pixel difference
    criterion = nn.MSELoss()

    print(f"Parameters: {sum(p.numel() for p in model.parameters())}")
    
    for epoch in range(EPOCHS):
        model.train()
        loss_epoch = 0.0
        t0 = time.time()
        
        for imgs, _ in train_loader: # Labels are ignored in reconstruction
            imgs = imgs.to(DEVICE)
            
            optimizer.zero_grad()
            reconstructed = model(imgs)
            loss = criterion(reconstructed, imgs) # Target is the original image
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            if log_run:
                wandb.log({"loss": loss.item()})
            
            loss_epoch += loss.item()
            
        # --- Evaluation & Visualization ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            is_first = True
            for val_imgs, _ in test_loader:
                val_imgs = val_imgs.to(DEVICE)
                val_out = model(val_imgs)
                val_loss += criterion(val_out, val_imgs).item()
                if is_first:
                    save_reconstruction_results(val_imgs, val_out, directory="test_reconstructions",filename=f"reconstructed-epoch{epoch}.jpg")
                    is_first = False

            # Log some visual results to W&B
            if log_run:
                wandb.log({
                    "train_loss": loss_epoch / len(train_loader),
                    "val_loss": val_loss / len(test_loader),
                })
                
        dt = time.time() - t0
        print(f"Epoch {epoch+1:02} | Time: {dt:.1f}s | "
              f"Train MSE: {loss_epoch/len(train_loader):.6f} | Val MSE: {val_loss/len(test_loader):.6f}")

    # --- SAVE MODEL ---
    save_path = "reconstruction_ttn_model.pth"
    torch.save(model.state_dict(), save_path)
    if log_run:
        artifact = wandb.Artifact(name="reconstruction-ttn-weights", type="model")
        artifact.add_file(save_path)
        run.log_artifact(artifact)
        
        wandb.finish()
    print(f"\nModel saved to {save_path}")

if __name__ == "__main__":
    train_reconstruction(log_run=True)