import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn import functional as F

class ImageEmbedding(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()
        # Load pretrained ResNet18 and modify for embedding
        resnet = models.resnet18(pretrained=True)
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])  # Remove final FC layer
        self.projection = nn.Linear(512, embedding_dim)
        
    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = self.projection(x)
        return F.normalize(x, p=2, dim=1)  # L2 normalize embeddings

class Decoder(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()
        self.initial_size = 8  # Starting size for upsampling
        
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim, 512 * self.initial_size * self.initial_size),
            nn.ReLU()
        )
        
        self.decoder = nn.Sequential(
            # 8x8 -> 16x16
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            # 16x16 -> 32x32
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            # 32x32 -> 64x64
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # 64x64 -> 128x128
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            # 128x128 -> 256x256
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            
            # Final layer
            nn.Conv2d(16, 3, kernel_size=3, padding=1),
            nn.Tanh()
        )
        
    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 512, self.initial_size, self.initial_size)
        return self.decoder(x)

class ImageEmbeddingModel(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()
        self.encoder = ImageEmbedding(embedding_dim)
        self.decoder = Decoder(embedding_dim)
        
    def forward(self, x):
        embedding = self.encoder(x)
        reconstruction = self.decoder(embedding)
        return embedding, reconstruction 