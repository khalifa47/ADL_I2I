import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import wandb
from model import ImageEmbeddingModel
from loss import CombinedLoss

class HorseZebraDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform or transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        
        self.horse_dir = os.path.join(root_dir, 'horse')
        self.zebra_dir = os.path.join(root_dir, 'zebra')
        
        self.horse_files = [f for f in os.listdir(self.horse_dir) if f.endswith(('.jpg', '.png'))]
        self.zebra_files = [f for f in os.listdir(self.zebra_dir) if f.endswith(('.jpg', '.png'))]
        
        # Create labels: 0 for horse, 1 for zebra
        self.images = []
        self.labels = []
        
        for f in self.horse_files:
            self.images.append(os.path.join(self.horse_dir, f))
            self.labels.append(0)
            
        for f in self.zebra_files:
            self.images.append(os.path.join(self.zebra_dir, f))
            self.labels.append(1)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            img = self.transform(img)
            
        return img, label

def train(config):
    # Initialize wandb
    wandb.init(project="horse2zebra-embedding", config=config)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = ImageEmbeddingModel(embedding_dim=config.embedding_dim).to(device)
    
    # Create loss function
    criterion = CombinedLoss(
        embedding_weight=config.embedding_weight,
        reconstruction_weight=config.reconstruction_weight,
        symmetric_weight=config.symmetric_weight
    ).to(device)
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    
    # Create dataset and dataloader
    dataset = HorseZebraDataset(config.data_dir)
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers
    )
    
    # Training loop
    for epoch in range(config.num_epochs):
        model.train()
        total_loss = 0
        
        progress_bar = tqdm(dataloader)
        for images, labels in progress_bar:
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            embeddings, reconstructions = model(images)
            
            # Calculate loss
            loss_dict = criterion(embeddings, reconstructions, images, labels, model.decoder)
            loss = loss_dict['total_loss']
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update progress
            total_loss += loss.item()
            progress_bar.set_description(f"Epoch {epoch+1}/{config.num_epochs}")
            progress_bar.set_postfix(loss=loss.item())
            
            # Log to wandb
            wandb.log({
                "loss": loss.item(),
                "embedding_loss": loss_dict['embedding_loss'].item(),
                "reconstruction_loss": loss_dict['reconstruction_loss'].item(),
                "symmetric_loss": loss_dict['symmetric_loss'].item()
            })
        
        # Save checkpoint
        if (epoch + 1) % config.save_every == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': total_loss / len(dataloader)
            }, os.path.join(config.output_dir, f'checkpoint-{epoch+1}.pt'))

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Path to horse-zebra dataset")
    parser.add_argument("--output_dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--embedding_dim", type=int, default=128, help="Dimension of image embeddings")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of dataloader workers")
    parser.add_argument("--save_every", type=int, default=10, help="Save checkpoint every N epochs")
    parser.add_argument("--embedding_weight", type=float, default=1.0, help="Weight for embedding loss")
    parser.add_argument("--reconstruction_weight", type=float, default=1.0, help="Weight for reconstruction loss")
    parser.add_argument("--symmetric_weight", type=float, default=0.5, help="Weight for symmetric loss")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    train(args) 