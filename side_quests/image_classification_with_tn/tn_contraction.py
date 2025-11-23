import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

device = torch.device("mps")

BATCH_SIZE = 4
NUM_LABELS = 10

class FeatureMap(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("factor", torch.tensor(torch.pi / 2.0))

    def forward(self, x):
        x = x.unsqueeze(-1)
        return torch.cat([torch.cos(self.factor * x), torch.sin(self.factor * x)], dim=-1)
    
def loss(log_probs, y):
    d = y_pred.shape[0] # batch size
    loss = - 1/d * log_probs[torch.arange(d),y].sum()
    return loss
    
    
if __name__ == "__main__":
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    data_iter = iter(train_loader)
        
    f_map = FeatureMap()
    images, labels = next(data_iter)
    embedded_images = f_map.forward(images.to(device))
    
    # f(x) = W * phi.  -> f has shape [batch_size, label_size]
    W = torch.randn([1, 28,28,2, NUM_LABELS]).to(device)
    res = torch.einsum('cwhfl, bcwhf -> bl', W, embedded_images)
    normalization_factor = res.sum(dim=1, keepdim=True).to(device)
    probs = res.div(normalization_factor).to(device)
    y_pred = probs.max(dim=1).indices
    print(loss(res, labels))