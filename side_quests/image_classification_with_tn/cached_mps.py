import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import tensornetwork as tn
from torch.utils.tensorboard import SummaryWriter
from torchviz import make_dot
import torch.onnx
import os
import glob
from datetime import datetime
import sys

sys.setrecursionlimit(5000)

# --- 0. SETUP ---
tn.set_default_backend("pytorch")
device = torch.device("mps")

# --- 1. HYPERPARAMETERS ---
# We use 10x10 images for this demo so the visualization graphs 
# remain readable. 28x28 works but produces massive PDF graphs.
IMG_SIZE = 10
N_PIXELS = IMG_SIZE * IMG_SIZE
FEATURE_DIM = 2     # Maps pixel -> [cos, sin]
BOND_DIM = 16       # "Memory" capacity of the MPS
NUM_CLASSES = 10    # MNIST Digits
BATCH_SIZE = 64
STEPS_PER_CORE = 1  # Gradient steps per core per sweep
EPOCHS = 20

# --- 2. HELPER CLASSES ---

class FeatureMap(nn.Module):
    """Maps scalar pixel values to a 2D quantum feature vector."""
    def __init__(self):
        super().__init__()
        self.register_buffer("factor", torch.tensor(torch.pi / 2.0))

    def forward(self, x):
        x = x.unsqueeze(-1)
        return torch.cat([torch.cos(self.factor * x), torch.sin(self.factor * x)], dim=-1)

class CachedMPS(nn.Module):
    """
    Matrix Product State Classifier with Environment Caching.
    """
    def __init__(self, input_dim, feature_dim, bond_dim, num_classes):
        super().__init__()
        self.input_dim = input_dim
        self.feature_map = FeatureMap()
        
        # Initialize MPS Cores
        self.cores = nn.ParameterList()
        
        # Left Edge: [Feature, 1, Bond] (Dummy dimension '1' used for consistency)
        self.cores.append(nn.Parameter(torch.randn(feature_dim, 1, bond_dim) * 0.1))
        
        # Middle Cores: [Feature, Bond, Bond]
        # Init close to Identity to allow gradient flow through chain
        for _ in range(1, input_dim - 1):
            base = torch.eye(bond_dim).unsqueeze(0).repeat(feature_dim, 1, 1)
            noise = torch.randn(feature_dim, bond_dim, bond_dim) * 0.01
            self.cores.append(nn.Parameter(base + noise))
            
        # Right Edge: [Feature, Bond, Classes]
        self.cores.append(nn.Parameter(torch.randn(feature_dim, bond_dim, num_classes) * 0.1))

    def forward(self, x):
        """
        Standard Forward Pass (Left -> Right contraction).
        Used for Inference, Validation, and Graph Visualization.
        """
        x_flat = x.view(-1, self.input_dim)
        features = self.feature_map(x_flat)
        
        v_0 = features[:, 0, :]
        A_0 = self.cores[0]
        
        # Squeeze dummy dim on first core
        M = tn.ncon([v_0, A_0.squeeze(1)], [[-1, 1], [1, -2]])

        for i in range(1, self.input_dim - 1):
            v_i = features[:, i, :]
            A_i = self.cores[i]
            # Contract Feature + Core
            core_contracted = tn.ncon([v_i, A_i], [[-1, 1], [1, -2, -3]])
            # Contract with Accumulator M
            M = tn.ncon([M, core_contracted], [[-1, 1], [-1, 1, -2]])
            # Normalize
            M = M / (torch.norm(M, dim=1, keepdim=True) + 1e-8)

        v_N = features[:, -1, :]
        A_N = self.cores[-1]
        final = tn.ncon([v_N, A_N], [[-1, 1], [1, -2, -3]])
        logits = tn.ncon([M, final], [[-1, 1], [-1, 1, -2]])
        return logits

    # --- CACHING LOGIC (TRAINING ONLY) ---
    
    def precompute_right_env(self, features):
        """Compute R[i] for the whole chain based on current batch."""
        self.R_cache = [None] * self.input_dim
        v_last = features[:, -1, :]
        A_last = self.cores[-1]
        
        # R[-1] is effectively the contraction of the last site
        curr = tn.ncon([v_last, A_last], [[-1, 1], [1, -2, -3]])
        self.R_cache[-1] = curr
        
        # Sweep backwards from N-2 down to 0
        for i in range(self.input_dim - 2, -1, -1):
            v_i = features[:, i, :]
            A_i = self.cores[i]
            prev_R = self.R_cache[i+1]
            
            if i == self.input_dim - 2:
                curr = tn.ncon([v_i, A_i, prev_R], [[-1, 1], [1, -2, 3], [-1, 3, -3]])
            else:
                curr = tn.ncon([v_i, A_i, prev_R], [[-1, 1], [1, -2, 3], [-1, 3, -3]])
                
            curr = curr / (torch.norm(curr, dim=1, keepdim=True) + 1e-8)
            self.R_cache[i] = curr

    def update_L_cache(self, index, features):
        """Update L[index] using the newly optimized core."""
        v_i = features[:, index, :]
        core = self.cores[index]
        
        if index == 0:
             new_L = tn.ncon([v_i, core.squeeze(1)], [[-1, 1], [1, -2]])
        else:
             prev_L = self.L_cache[index-1]
             new_L = tn.ncon([prev_L, v_i, core], [[-1, 1], [-1, 2], [2, 1, -2]])
             
        new_L = new_L / (torch.norm(new_L, dim=1, keepdim=True) + 1e-8)
        
        if not hasattr(self, 'L_cache'): 
            self.L_cache = [None] * self.input_dim
        self.L_cache[index] = new_L

    def get_loss_at_site(self, index, features, labels, criterion):
        """Calculate loss locally at Core[index] using Environments."""
        v_i = features[:, index, :]
        core = self.cores[index]
        
        parts = [v_i, core]
        con_schemes = [[-1, 1]] 
        
        # 1. Connect Left
        if index == 0:
            parts[1] = core.squeeze(1) # Squeeze dummy dim
            con_schemes.append([1, 2])
        else:
            L = self.L_cache[index-1] 
            parts.append(L)
            con_schemes.append([1, 2, 3])
            con_schemes.append([-1, 2])   
            
        # 2. Connect Right
        if index == self.input_dim - 1:
            # Last core handling
            if index == 0: con_schemes[-1] = [1, -2]
            else: con_schemes[1] = [1, 2, -2]
        else:
            R = self.R_cache[index+1]
            parts.append(R)
            if index == 0: con_schemes.append([-1, 2, -2])
            else: con_schemes.append([-1, 3, -2])

        # 3. Contract to get Logits
        logits = tn.ncon(parts, con_schemes)
        return criterion(logits, labels)

# --- 3. DIRECTORY & CHECKPOINT UTILS ---

def get_run_dir(model_name, base_path="checkpoints"):
    """Generates a unique directory name based on timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{model_name}_{timestamp}"
    return os.path.join(base_path, run_name), run_name

def get_latest_run_dir(model_name, base_path="checkpoints"):
    """Finds the most recent directory for this model to resume."""
    search_pattern = os.path.join(base_path, f"{model_name}_*")
    dirs = sorted(glob.glob(search_pattern))
    if not dirs: return None
    return dirs[-1]

def save_checkpoint(model, directory, epoch, loss, filename="checkpoint.pth"):
    if not os.path.exists(directory): os.makedirs(directory)
    filepath = os.path.join(directory, filename)
    print(f"   [Save] Saving to {filepath}")
    torch.save({
        'epoch': epoch, 'model_state_dict': model.state_dict(), 'loss': loss,
    }, filepath)

def load_checkpoint(model, directory, filename="checkpoint.pth"):
    filepath = os.path.join(directory, filename)
    if os.path.isfile(filepath):
        print(f"   [Load] Found checkpoint at {filepath}. Loading...")
        checkpoint = torch.load(filepath)
        model.load_state_dict(checkpoint['model_state_dict'])
        return checkpoint['epoch'], checkpoint['loss']
    else:
        print(f"   [Init] No checkpoint found at {filepath}.")
        return 0, float('inf')

def create_visualizations(model, sample_input, save_dir):
    print("\n--- Generating Visualizations ---")
    
    # Move model and input to CPU for visualization to avoid device conflicts
    # We switch back to the original device afterwards
    original_device = next(model.parameters()).device
    model.to("cpu")
    sample_input = sample_input.to("cpu")

    # 1. TorchViz
    try:
        print("1. Creating TorchViz Graph...")
        # We run a forward pass to trace the graph
        y = model(sample_input)
        
        # Create the dot object
        dot = make_dot(y, params=dict(model.named_parameters()))
        dot.format = 'png'
        
        # Save into the run directory
        path = os.path.join(save_dir, "mps_structure")
        dot.render(path)
        print(f"   -> Saved {path}.png")
        
    except RecursionError:
        print("   -> Skipped TorchViz: Graph is still too deep even with increased limit.")
    except Exception as e:
        print(f"   -> Skipped TorchViz: {e}")

    # 2. ONNX
    try:
        print("2. Exporting ONNX model...")
        path = os.path.join(save_dir, "mps_model.onnx")
        
        # Opset version 11 or higher is usually safer for custom ops
        torch.onnx.export(model, sample_input, path,
                          input_names=['input_image'], 
                          output_names=['class_logits'],
                          opset_version=14) 
        print(f"   -> Saved {path} (View at https://netron.app)")
    except ImportError:
        print("   -> Skipped ONNX: Please run 'pip install onnx onnxscript'")
    except Exception as e:
        print(f"   -> Skipped ONNX: {e}")

    # Restore model to original device for training
    model.to(original_device)

# --- 4. MAIN EXECUTION ---

def main():
    # CONFIGURATION
    MODEL_NAME = "MPS_MNIST"
    RESUME_LATEST = False  # Set to True to resume previous run
    
    print(f"Running on device: {device}")

    # A. Setup Directory
    if RESUME_LATEST:
        run_dir = get_latest_run_dir(MODEL_NAME)
        if run_dir:
            print(f"--> Resuming from: {run_dir}")
            run_name = os.path.basename(run_dir)
        else:
            print("--> No previous run found. Starting fresh.")
            run_dir, run_name = get_run_dir(MODEL_NAME)
    else:
        run_dir, run_name = get_run_dir(MODEL_NAME)
        print(f"--> Starting new run: {run_dir}")
    
    os.makedirs(run_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=f"runs/{run_name}")

    # B. Data & Model
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)), 
        transforms.ToTensor(), 
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    
    mps = CachedMPS(N_PIXELS, FEATURE_DIM, BOND_DIM, NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()

    # C. Load & Viz
    start_epoch, best_loss = load_checkpoint(mps, run_dir, "mps_latest.pth")
    if not RESUME_LATEST: best_loss = float('inf')
    
    # Generate graphs (using dummy input on CPU to avoid graphviz device errors)
    dummy_input = torch.randn(BATCH_SIZE, IMG_SIZE, IMG_SIZE).to(device)
    create_visualizations(mps, dummy_input, run_dir)

    # D. Training Loop
    print("\n--- Starting Training Loop ---")
    global_step = start_epoch * len(train_loader) * N_PIXELS

    for epoch in range(start_epoch, start_epoch + EPOCHS):
        mps.train()
        total_loss = 0
        num_batches = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device).long() # Fix for CrossEntropy
            
            x_flat = images.view(-1, N_PIXELS)
            features = mps.feature_map(x_flat)
            
            # --- SWEEPING PHASE ---
            # 1. Prepare Caches
            with torch.no_grad():
                mps.precompute_right_env(features)
                mps.L_cache = [None] * N_PIXELS

            # 2. Sweep Left -> Right
            batch_loss_accum = 0
            for i in range(N_PIXELS - 1):
                current_core = mps.cores[i]
                optimizer = optim.Adam([current_core], lr=0.01)
                
                for _ in range(STEPS_PER_CORE):
                    optimizer.zero_grad()
                    loss = mps.get_loss_at_site(i, features, labels, criterion)
                    loss.backward()
                    
                    if global_step % 100 == 0:
                        writer.add_scalar('Loss/train', loss.item(), global_step)
                        grad_norm = current_core.grad.norm().item()
                        writer.add_scalar('GradNorm/core', grad_norm, global_step)
                    
                    torch.nn.utils.clip_grad_norm_([current_core], 1.0)
                    optimizer.step()
                    global_step += 1
                
                with torch.no_grad():
                    mps.update_L_cache(i, features)
                
                batch_loss_accum = loss.item()

            total_loss += batch_loss_accum
            num_batches += 1
            
            if batch_idx % 50 == 0:
                print(f"Epoch {epoch+1} | Batch {batch_idx}/{len(train_loader)} | Loss: {batch_loss_accum:.4f}")

        # --- SAVE & LOG EPOCH ---
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1} Finished. Avg Loss: {avg_loss:.4f}")
        
        if avg_loss < best_loss:
            print(f"   >>> New Best Loss! ({best_loss:.4f} -> {avg_loss:.4f})")
            best_loss = avg_loss
            save_checkpoint(mps, run_dir, epoch+1, best_loss, "mps_best_model.pth")
            
        save_checkpoint(mps, run_dir, epoch+1, avg_loss, "mps_latest.pth")

    writer.close()
    print(f"\nDone. Outputs saved in: {run_dir}")

if __name__ == '__main__':
    main()