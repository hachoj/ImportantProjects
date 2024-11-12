import torch
import torch.nn as nn
from config import config
from data_loader import TinyImageNetDataLoader
from model import ViT
from torch.optim import optimizer

# Initialize the model
model = ViT(config)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
torch.compile(model)

# Initialize data loader
data_loader = TinyImageNetDataLoader(batch_size=128, num_workers=4)

# Configure the optimizer
optim = model.configure_optimizers(
    weight_decay=config.weight_decay,
    learning_rate=0.001,
    betas=config.betas,
    device=device,
)


# Test the model on a batch of images
# very basic test to see if the device even trains

loss = nn.CrossEntropyLoss()

losses = []

i = 0
for images, labels in data_loader.train_loader:
    optim.zero_grad()
    images, labels = images.to(device), labels.to(device)
    logits = model(images)  # Pass images to the model's forward method
    cls_logit = logits[:, -1, :]
    output = loss(cls_logit, labels)
    losses.append(output.item())
    print(f"step: {i}, loss: {output.item()}")
    output.backward()
    optim.step()
    i += 1
    if i == 720:
        break
