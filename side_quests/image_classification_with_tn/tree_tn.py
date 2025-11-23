import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time

# ==========================================
# 1. Configuration & Hyperparameters
# ==========================================
BATCH_SIZE = 256
LEARNING_RATE = 0.002
EPOCHS = 20
BOND_DIM = 64          
CP_RANK = 16           
DROPOUT = 0.1          
PATCH_SIZE = 4         
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Fallback for Apple Silicon
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")

print(f"Running on device: {DEVICE}")

class CPLowRankLayer(nn.Module):
    """
    Represents a layer of tree nodes using CP Decomposition.
    Compresses a dense tensor of shape (Out, Left, Right) into 3 matrices.
    """
    def __init__(self, num_nodes, in_dim, out_dim, rank, dropout_p=0.0):
        super().__init__()
        self.num_nodes = num_nodes
        self.rank = rank
        self.dropout_p = dropout_p
        
        self.factor_left = nn.Parameter(torch.randn(num_nodes, rank, in_dim))
        self.factor_right = nn.Parameter(torch.randn(num_nodes, rank, in_dim))
        self.factor_out = nn.Parameter(torch.randn(num_nodes, rank, out_dim))
        self.norm = nn.LayerNorm(out_dim)
        
        self.scale = nn.Parameter(torch.ones(num_nodes, rank))

        self._initialize()

    def _initialize(self):
        with torch.no_grad():
            self.factor_left.data /= (self.factor_left.data.norm(dim=-1, keepdim=True) + 1e-8)
            self.factor_right.data /= (self.factor_right.data.norm(dim=-1, keepdim=True) + 1e-8)
            self.factor_out.data /= (self.factor_out.data.norm(dim=-1, keepdim=True) + 1e-8)
            self.scale.data.normal_(1.0, 0.02)

    def forward(self, x):
        B = x.size(0) 
        
        x = x.view(B, self.num_nodes, 2, -1)
        x_l = x[:, :, 0, :] # (B, N, In)
        x_r = x[:, :, 1, :] # (B, N, In)

        proj_l = torch.einsum('bni,nri->bnr', x_l, self.factor_left)
        proj_r = torch.einsum('bni,nri->bnr', x_r, self.factor_right)
        
        merged = self.scale.unsqueeze(0) * proj_l * proj_r
        
        if self.training and self.dropout_p > 0:
            mask = torch.bernoulli(torch.full_like(merged, 1 - self.dropout_p))
            merged = merged * mask / (1 - self.dropout_p)

        out = torch.einsum('bnr,nro->bno', merged, self.factor_out)
        out = out + x_l + x_r
        out = self.norm(out)
        
        return out

class PatchTTN(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_channels=1, 
                 num_classes=10, bond_dim=64, cp_rank=16, dropout=0.1):
        super().__init__()
        
        num_patches = (img_size // patch_size) ** 2
        self.patch_embed = nn.Conv2d(in_channels, bond_dim, 
                                     kernel_size=patch_size, stride=patch_size)
        
        self.depth = int(torch.log2(torch.tensor(float(num_patches))))
        
        print(f"Tree Structure: {num_patches} leaves -> Depth {self.depth}")
        
        self.layers = nn.ModuleList()
        current_nodes = num_patches // 2
        
        for _ in range(self.depth):
            layer = CPLowRankLayer(
                num_nodes=current_nodes,
                in_dim=bond_dim,
                out_dim=bond_dim,
                rank=cp_rank,
                dropout_p=dropout
            )
            self.layers.append(layer)
            current_nodes //= 2
            
        self.norm = nn.LayerNorm(bond_dim)
        self.head = nn.Linear(bond_dim, num_classes)

    def forward(self, x):
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        for layer in self.layers:
            x = layer(x)
        x = x.squeeze(1)
        x = self.norm(x)
        return self.head(x)

def get_mnist_loaders():
    transform = transforms.Compose([
        transforms.Resize((32,32)), 
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)) # Added normalization for stability
    ])
    
    train_set = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_set = datasets.MNIST('./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    return train_loader, test_loader

def train():
    train_loader, test_loader = get_mnist_loaders()
    
    model = PatchTTN(
        img_size=32,
        patch_size=PATCH_SIZE,
        bond_dim=BOND_DIM,
        cp_rank=CP_RANK,
        dropout=DROPOUT
    ).to(DEVICE)
    
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

if __name__ == "__main__":
    train()