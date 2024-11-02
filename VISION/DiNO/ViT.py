import torch
from model import ViT
from config import config
from data_loader import TinyImageNetDataLoader

# Initialize the model
model = ViT(config)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)
torch.compile(model)

# Print the model architecture
print(model)

# Initialize data loader
data_loader = TinyImageNetDataLoader(batch_size=64, num_workers=4)

# Configure the optimizer
optim = model.configure_optimizers(weight_decay=config.weight_decay, learning_rate=0.001, betas=config.betas, device=device)

# Print the optimizer
print(optim)

# Test the model on a batch of images

for images, labels in data_loader.train_loader:
    images = images.to(device)
    output = model(images)  # Pass images to the model's forward method
    print(output.shape)
    logits = model(images)
    print(logits.shape)
    print(logits)
    break