import torch
import torch.nn as nn

from data_loader import TinyImageNetDataLoader

class PatchEmbedding(nn.Module):
    def __init__(self, img_size: int = 224, patch_size: int = 16, channels: int = 3, embd_dim: int = 384):
        super().__init__()
        """
        This assumes that the image is square
        """
        assert (img_size % patch_size) == 0, f"Image must be a square image with size that is a mutliple of the patch_size {patch_size}" 
        self.img_size = img_size
        self.patch_size = patch_size
        self.embd_dim = embd_dim
        self.num_patches = (img_size // patch_size) ** 2
        
        # projection of flattened patch to token embedding
        self.proj = nn.Linear(self.patch_size**2*channels, self.embd_dim)

        # positional embedding
        self.pos_embd = nn.Parameter(torch.zeros(1, self.num_patches + 1, self.embd_dim))  # +1 for cls token

        # class token
        self.cls_token = nn.Parameter(torch.zeros(1,1, self.embd_dim))
        

    def forward(self, x):
        B = x.size(0)  # Batch size

        # Extract patches
        patches = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        # (B, C, num_patches_h, num_patches_w, patch_size, patch_size)

        patches = patches.contiguous().view(B, 3, -1, self.patch_size, self.patch_size)
        # (B, C, num_patches, patch_size, patch_size)
    
        patches = patches.permute(0, 2, 1, 3, 4)  
        # (B, num_patches, channels, patch_size, patch_size)

        # Flatten patches
        patches = patches.contiguous().view(B, self.num_patches, -1)  
        # (B, num_patches, channels * patch_size * patch_size)

        # Linear projection
        x_tok = self.proj(patches)  
        # (B, num_patches, embd_dim)

        # Expand class token
        cls_tokens = self.cls_token.expand(B, -1, -1)  
        # (B, 1, embd_dim)

        # Concatenate class token and patch embeddings
        x = torch.cat((cls_tokens, x_tok), dim=1)  
        # (B, num_patches + 1, embd_dim)

        # Add positional embeddings
        x = x + self.pos_embd  
        # (B, num_patches + 1, embd_dim)

        return x


"""
TESTING
"""
# Initialize data loader
data_loader = TinyImageNetDataLoader(batch_size=64, num_workers=4)

# Create an instance of PatchEmbedding
emd = PatchEmbedding()

# Access the training DataLoader
for images, labels in data_loader.train_loader:
    output = emd(images)  # Pass images to the model's forward method
    print(output.shape)
    break