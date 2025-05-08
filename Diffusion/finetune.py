import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline, DDPMScheduler
from accelerate import Accelerator
from tqdm import tqdm
import wandb

class HorseZebraDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform or transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        
        self.horse_dir = os.path.join(root_dir, 'horse')
        self.zebra_dir = os.path.join(root_dir, 'zebra')
        
        self.horse_files = [f for f in os.listdir(self.horse_dir) if f.endswith(('.jpg', '.png'))]
        self.zebra_files = [f for f in os.listdir(self.zebra_dir) if f.endswith(('.jpg', '.png'))]
        
    def __len__(self):
        return min(len(self.horse_files), len(self.zebra_files))
    
    def __getitem__(self, idx):
        horse_img = Image.open(os.path.join(self.horse_dir, self.horse_files[idx])).convert('RGB')
        zebra_img = Image.open(os.path.join(self.zebra_dir, self.zebra_files[idx])).convert('RGB')
        
        if self.transform:
            horse_img = self.transform(horse_img)
            zebra_img = self.transform(zebra_img)
            
        return {'horse': horse_img, 'zebra': zebra_img}

def train(config):
    # Initialize accelerator
    accelerator = Accelerator()
    
    # Initialize wandb
    if accelerator.is_main_process:
        wandb.init(project="horse2zebra", config=config)
    
    # Load model
    model = StableDiffusionImg2ImgPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16 if config.use_fp16 else torch.float32
    )
    
    # Move model to device
    model = model.to(accelerator.device)
    
    # Freeze all parameters except the attention layers
    for param in model.unet.parameters():
        param.requires_grad = False
    
    for name, param in model.unet.named_parameters():
        if "attn" in name:
            param.requires_grad = True
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(
        model.unet.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    # Load dataset
    dataset = HorseZebraDataset(config.data_dir)
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers
    )
    
    # Prepare for distributed training
    model, optimizer, dataloader = accelerator.prepare(
        model, optimizer, dataloader
    )
    
    # Training loop
    for epoch in range(config.num_epochs):
        model.train()
        total_loss = 0
        
        progress_bar = tqdm(dataloader, disable=not accelerator.is_main_process)
        for batch in progress_bar:
            optimizer.zero_grad()
            
            # Get images
            horse_images = batch['horse']
            zebra_images = batch['zebra']
            
            # Generate noise
            noise = torch.randn_like(zebra_images)
            timesteps = torch.randint(0, 1000, (zebra_images.shape[0],), device=accelerator.device)
            
            # Add noise to zebra images
            noisy_zebra = model.scheduler.add_noise(zebra_images, noise, timesteps)
            
            # Predict noise
            noise_pred = model.unet(noisy_zebra, timesteps, encoder_hidden_states=model.text_encoder("zebra")[0])
            
            # Calculate loss
            loss = F.mse_loss(noise_pred, noise)
            
            # Backward pass
            accelerator.backward(loss)
            optimizer.step()
            
            total_loss += loss.item()
            
            # Update progress bar
            progress_bar.set_description(f"Epoch {epoch+1}/{config.num_epochs}")
            progress_bar.set_postfix(loss=loss.item())
            
            # Log to wandb
            if accelerator.is_main_process:
                wandb.log({
                    "loss": loss.item(),
                    "epoch": epoch
                })
        
        # Save checkpoint
        if accelerator.is_main_process and (epoch + 1) % config.save_every == 0:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(
                os.path.join(config.output_dir, f"checkpoint-{epoch+1}")
            )

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Path to horse-zebra dataset")
    parser.add_argument("--output_dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of dataloader workers")
    parser.add_argument("--save_every", type=int, default=10, help="Save checkpoint every N epochs")
    parser.add_argument("--use_fp16", action="store_true", help="Use mixed precision training")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    train(args)