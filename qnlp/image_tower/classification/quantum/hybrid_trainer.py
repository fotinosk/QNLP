import torch
import torch.nn as nn
import pennylane as qml
from pennylane import qnn
from einops import rearrange

class QuantumQuadLayer(nn.Module):
    def __init__(self, num_qubits=4):
        super().__init__()
        self.n = num_qubits
        dev = qml.device("default.qubit", wires=num_qubits)

        @qml.qnode(dev, interface="torch")
        def circuit(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(num_qubits))
            qml.StronglyEntanglingLayers(weights, wires=range(num_qubits))
            # We measure one qubit to represent the 'parent' node output
            return qml.expval(qml.PauliZ(0))

        # Weight shape: (L layers, N qubits, 3 rotations)
        weight_shapes = {"weights": (2, num_qubits, 3)}
        self.q_layer = qnn.TorchLayer(circuit, weight_shapes)

    def forward(self, x):
        # x input: [Batch, Nodes, 4]
        batch, nodes, _ = x.shape
        # Flatten nodes and batches to process through the QNode
        x_flat = x.view(-1, self.n) 
        out = self.q_layer(x_flat)
        return out.view(batch, nodes, 1) # Output: [Batch, Nodes, 1]

class HybridQuadTTN(nn.Module):
    def __init__(self, img_size=32, patch_size=4):
        super().__init__()
        self.grid_dim = img_size // patch_size # e.g., 8
        
        # Initial classical embedding to get to 4 features per patch
        # (This keeps the quantum circuit small and fast)
        self.embed = nn.Linear(patch_size**2, 4)
        
        # Quantum Layers
        self.layer1 = QuantumQuadLayer(num_qubits=4) # 8x8 grid -> 4x4 grid
        self.layer2 = QuantumQuadLayer(num_qubits=4) # 4x4 grid -> 2x2 grid
        self.layer3 = QuantumQuadLayer(num_qubits=4) # 2x2 grid -> 1x1 grid
        
        self.head = nn.Linear(1, 10)

    def forward(self, x):
        # 1. Patching
        x = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=4, p2=4)
        x = torch.tanh(self.embed(x)) # [B, 64, 4]

        # 2. Quad-Tree Hierarchy
        curr_h = self.grid_dim
        for layer in [self.layer1, self.layer2, self.layer3]:
            # Spatial grouping into 2x2 blocks
            x = rearrange(x, 'b (h w) c -> b c h w', h=curr_h)
            x = rearrange(x, 'b c (h h2) (w w2) -> b (h w) (h2 w2 c)', h2=2, w2=2)
            # x is now [Batch, New_Nodes, 4]
            x = layer(x)
            curr_h //= 2
            
        return self.head(x.squeeze(-1))

# Initialize
model = HybridQuadTTN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
print("Hybrid Model Initialized. Ready for training.")
