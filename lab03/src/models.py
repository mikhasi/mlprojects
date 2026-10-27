import torch
import torch.nn as nn

# 2

class FashionMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.net(x)


class FashionCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 10)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class FashionAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_dim = 7 * 7
        self.num_patches = 16
        
        self.proj = nn.Linear(self.patch_dim, 64)
        
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, 64) * 0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=64, 
            nhead=4, 
            dim_feedforward=128, 
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        self.head = nn.Linear(64, 10)

    def forward(self, x):
        B = x.size(0)
        x = x.unfold(2, 7, 7).unfold(3, 7, 7)
        x = x.contiguous().view(B, 16, 49)
        
        x = self.proj(x) + self.pos_embedding
        
        x = self.transformer(x)
        
        x = x.mean(dim=1)
        
        return self.head(x)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    mlp = FashionMLP()
    cnn = FashionCNN()
    attn = FashionAttention()
    
    print(f"MLP Parameters:    {count_parameters(mlp):,}")
    print(f"CNN Parameters:    {count_parameters(cnn):,}")
    print(f"Attention Params:  {count_parameters(attn):,}")