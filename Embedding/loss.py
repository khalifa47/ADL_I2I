import torch
import torch.nn as nn
import torch.nn.functional as F

class EmbeddingLoss(nn.Module):
    def __init__(self, margin=0.5):
        super().__init__()
        self.margin = margin
        
    def forward(self, embeddings, labels):
        # Calculate cosine similarity matrix
        similarity_matrix = torch.matmul(embeddings, embeddings.t())
        
        # Create label matrix (1 for same label, 0 for different)
        label_matrix = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        
        # Positive pairs (same label)
        positive_mask = label_matrix
        positive_loss = (1 - similarity_matrix) * positive_mask
        
        # Negative pairs (different label)
        negative_mask = 1 - label_matrix
        negative_loss = torch.clamp(similarity_matrix - self.margin, min=0) * negative_mask
        
        # Combine losses
        total_loss = positive_loss.sum() + negative_loss.sum()
        num_pairs = positive_mask.sum() + negative_mask.sum()
        
        return total_loss / (num_pairs + 1e-8)

class ReconstructionLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        
    def forward(self, original, reconstructed):
        return self.mse(original, reconstructed)

class SymmetricLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        
    def forward(self, embedding, decoder):
        # Get reconstruction of original embedding
        reconstruction1 = decoder(embedding)
        
        # Get reconstruction of negative embedding
        reconstruction2 = decoder(-embedding)
        
        # Calculate similarity between reconstructions
        return self.mse(reconstruction1, reconstruction2)

class CombinedLoss(nn.Module):
    def __init__(self, embedding_weight=1.0, reconstruction_weight=1.0, symmetric_weight=0.5):
        super().__init__()
        self.embedding_weight = embedding_weight
        self.reconstruction_weight = reconstruction_weight
        self.symmetric_weight = symmetric_weight
        
    def forward(self, embeddings, reconstructions, original_images, labels, decoder):
        batch_size = embeddings.size(0)
        
        # 1. Embedding Loss (Contrastive Loss)
        # Create label matrix (1 for same class, 0 for different)
        label_matrix = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        
        # Calculate cosine similarity matrix
        similarity_matrix = torch.matmul(embeddings, embeddings.t())
        
        # Calculate contrastive loss
        positive_pairs = similarity_matrix * label_matrix
        negative_pairs = similarity_matrix * (1 - label_matrix)
        
        # We want high similarity for positive pairs and low for negative
        embedding_loss = (
            -torch.mean(positive_pairs) +  # Maximize similarity for same class
            torch.mean(negative_pairs)     # Minimize similarity for different classes
        )
        
        # 2. Reconstruction Loss (MSE)
        reconstruction_loss = F.mse_loss(reconstructions, original_images)
        
        # 3. Symmetric Loss
        # Create negative embeddings
        negative_embeddings = -embeddings
        
        # Reconstruct from negative embeddings
        negative_reconstructions = decoder(negative_embeddings)
        
        # Calculate MSE between original and negative reconstructions
        symmetric_loss = F.mse_loss(negative_reconstructions, original_images)
        
        # Combine losses
        total_loss = (
            self.embedding_weight * embedding_loss +
            self.reconstruction_weight * reconstruction_loss +
            self.symmetric_weight * symmetric_loss
        )
        
        return {
            'total_loss': total_loss,
            'embedding_loss': embedding_loss,
            'reconstruction_loss': reconstruction_loss,
            'symmetric_loss': symmetric_loss
        } 