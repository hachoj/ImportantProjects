import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import torch
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from typing import Callable
import ViT

# These augmentations are defined exactly as proposed in the paper
# Define the transforms
def global_augment(images):
    global_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.4, 1.0)),  # Larger crops
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),  # Color jittering
        transforms.RandomGrayscale(p=0.2),
        transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return torch.stack([global_transform(img) for img in images])

def multiple_local_augments(images, num_crops=6):
    size = 96  # Smaller crops for local
    local_transform = transforms.Compose([
        transforms.RandomResizedCrop(size, scale=(0.05, 0.4)),  # Smaller, more concentrated crops
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),  # Same level of jittering
        transforms.RandomGrayscale(p=0.2),
        transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    # Apply the transformation multiple times to the same image
    return torch.stack([local_transform(img) for img in images])

class DINO(nn.Module):
    def __init__(self, student: nn.Module, teacher: nn.Module, device: torch.device):
        """
        Args:
            student_arch (nn.Module): ViT Network for student_arch
            teacher_arch (nn.Module): ViT Network for teacher_arch
            device: torch.device ('cuda' or 'cpu')
        """
        super(DINO, self).__init__()

        self.student = student.to(device)
        self.teacher = teacher.to(device)
        self.teacher.load_state_dict(self.student.state_dict())

        # Initialize center as nuffer to avoid backpropagation
        self.register_buffer('center', torch.zeros(1, student.ff.out_features))

        for param in self.teacher.parameters():
            param.requires_grad = False

    @staticmethod
    def distillication_loss(student_logits, teacher_logits, center, tau_s, tau_t):
        """
        Creating the centered and sharpened loss function to evaluate the student's performance

        NOTE:
        """
        # Detatching teacher logits to stop gradients from flowing back into the teacher
        teacher_logits = teacher_logits.detach()

        # Center and sharpen the teacher's logits
        teacher_probs = F.softmax((teacher_logits - center) / tau_t, dim=1)

        # Sharpen the student's logits
        student_probs = F.log_softmax(student_logits / tau_s, dim=1)

        # Calculate cross-entropy loss between the student's and teacher's probs
        loss = - (teacher_probs * student_probs).sum(dim=1).mean()
        return loss

    def teacher_update(self, beta: float):
        for teacher_params, student_params in zip(self.teacher.parameters(), self.student.parameters()):
            teacher_params.data.mul_(beta).add_(student_params.data, alpha=(1 - beta))

student = ViT.VissionTransformer(num_layers=8, img_size=96, emb_size=768, patch_size=8, num_head=6, num_class=10)
teacher = ViT.VissionTransformer(num_layers=8, img_size=96, emb_size=768, patch_size=8, num_head=6, num_class=10)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dino = DINO(student, teacher, device)

def train_dino(dino: DINO,
               data_loader: DataLoader,
               optimizer: Optimizer,
               device: torch.device,
               num_epochs,
               tps=0.9,
               tpt= 0.04,
               beta= 0.9,
               m= 0.9,
               ):
        """
        Args:
        dino: DINO Module
        data_loader (nn.Module): Dataloader for training
        optimizer (nn.optimizer): Optimizer for optimization (SGD etc.)
        defice (torch.device): 'cuda', 'cpu'
        num_epochs: Number of Epochs
        tps (float): tau for sharpening student logits
        tpt: for sharpening teacher logits
        beta (float): moving average decay 
        m (float): center moveing average decay
        """
    
        for epoch in range(num_epochs):
            print(f"Epoch: {epoch+1}/{num_epochs}")
            for x in data_loader:

                x1, x2 = global_augment(x), multiple_local_augments(x)  

                student_output1, student_output2 = dino.student(x1.to(device)), dino.student(x2.to(device))
                with torch.no_grad():
                    teacher_output1, teacher_output2 = dino.teacher(x1.to(device)), dino.teacher(x2.to(device))

                # Compute distillation loss
                loss = (dino.distillation_loss(teacher_output1, student_output2, dino.center, tps, tpt) +
                        dino.distillation_loss(teacher_output2, student_output1, dino.center, tps, tpt)) / 2

                print(f"Loss: {loss.item()}")
                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Update the teacher network parameters
                dino.teacher_update(beta)
                
                # Update the center
                with torch.no_grad():
                    dino.center = m * dino.center + (1 - m) * torch.cat([teacher_output1, teacher_output2], dim=0).mean(dim=0)

# Create your own CustomDataset and dataloader
from torchvision import datasets

class STL10(datasets.STL10):
    def __init__(self, root, split='train', download=True):
        super().__init__(root=root, split=split, download=download)
    
    def __getitem__(self, idx):
        img, label = super().__getitem__(idx)
        return img, label  # Return PIL Image

def collate_fn(batch):
    images = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    return images, labels

train_data = STL10(root='./data', split='train', download=True)

dataloader = DataLoader(train_data, batch_size=32, shuffle=True, collate_fn=collate_fn)
optimizer = torch.optim.AdamW(dino.parameters(), lr=1e-4)
train_dino(dino,
           data_loader=dataloader,
           optimizer=optimizer,
           device=device,
           num_epochs=300,
           tps=0.9,
           tpt= 0.04,
           beta= 0.9,
           m= 0.9)