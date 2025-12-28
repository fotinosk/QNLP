import wandb
import time
import torch
from torch import nn
from qnlp.utils.data import get_mnist_loaders
from qnlp.image_tower.classification.ttn_for_image_classification_model import PatchTTN, DEVICE, LEARNING_RATE, EPOCHS, BOND_DIM, CP_RANK, DROPOUT, PATCH_SIZE, BATCH_SIZE
import torch.optim as optim



def train(log_run: bool = True):
    with_phase_embedding = False
    if log_run:
        run = wandb.init(project="image-classifier", name="larger-batches", save_code=True)
    train_loader, test_loader = get_mnist_loaders(batch_size=BATCH_SIZE, with_phase_embedding=with_phase_embedding)
    
    model = PatchTTN(
        img_size=32,
        patch_size=PATCH_SIZE,
        bond_dim=BOND_DIM,
        cp_rank=CP_RANK,
        dropout=DROPOUT,
        in_channels=2 if with_phase_embedding else 1
    ).to(DEVICE)
    
    if log_run:
        wandb.watch(models=model, log="all", log_freq=100, log_graph=True)
    
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LEARNING_RATE, 
                                              steps_per_epoch=len(train_loader), epochs=EPOCHS)
    
    criterion = nn.CrossEntropyLoss()

    print(f"Parameters: {sum(p.numel() for p in model.parameters())}")
    
    for epoch in range(EPOCHS):
        model.train()
        loss_epoch = 0.0
        correct = 0
        total = 0
        t0 = time.time()
        
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            output = model(imgs)
            loss = criterion(output, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            scheduler.step()
            if log_run:
                wandb.log({"loss": loss.item()})
            
            loss_epoch += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(labels).sum().item()
            total += labels.size(0)
            
        # Validation
        model.eval()
        val_correct = 0
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                output = model(imgs)
                val_correct += output.argmax(dim=1).eq(labels).sum().item()
                
        train_acc = 100 * correct / total
        val_acc = 100 * val_correct / len(test_loader.dataset)
        dt = time.time() - t0
        
        print(f"Epoch {epoch+1:02} | Time: {dt:.1f}s | "
              f"Loss: {loss_epoch/len(train_loader):.4f} | "
              f"Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

    # --- SAVE MODEL ---
    save_path = "mnist_ttn_model.pth"
    torch.save(model.state_dict(), save_path)
    if log_run:
        artifact = wandb.Artifact(name="mnist-ttn-weights", type="model")
        artifact.add_file(save_path)
        run.log_artifact(artifact)
        
        wandb.finish()
    print(f"\nModel saved to {save_path}")

if __name__ == "__main__":
    train(log_run=False)