import torch
import torch.nn as nn
import torch.nn.functional as F
from config import config
from data_loader import TinyImageNetDataLoader
from vit_model import ViT
from train import train_dino
import inspect
import math


class DINO(nn.Module):
    def __init__(self, student_arch, teacher_arch, device):
        """
        Args:
            student_arch (nn.Module): ViT Network for student_arch
            teacher_arch (nn.Module): ViT Network for teacher_arch
            device: torch.device
        """
        super(DINO, self).__init__()

        self.student = student_arch.to(device)
        self.teacher = teacher_arch.to(device)
        self.teacher.load_state_dict(self.student.state_dict())

        # Initialize center as buffer and move it to the correct device
        self.register_buffer("center", torch.zeros(1, student_arch.output_dim).to(device))

        # Ensure the teacher parameters do not get updated during backprop
        for param in self.teacher.parameters():
            param.requires_grad = False

    def update_teacher(self, momentum_teacher):
        for param_student, param_teacher in zip(
            self.student.parameters(), self.teacher.parameters()
        ):
            param_teacher.data = (
                param_teacher.data * momentum_teacher
                + param_student.data * (1.0 - momentum_teacher)
            )

# learning rate + optim settings
base_lr = 3e-4  # Starting learning rate after warmup
min_lr = 1e-6   # Minimum learning rate
warmup_epochs = 15
betas = (0.9, 0.999)
# number of epochs
num_epochs = 100

# Actual Training
student = ViT(config)
teacher = ViT(config)
# try cuda, then mps, then cpu
device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "mps" if torch.mps.is_available() and "cuda" not in device else "cpu"

dino_model = DINO(student, teacher, device)
data_loader = TinyImageNetDataLoader(batch_size=16, num_workers=4)

# configure optimizer
# create a fused adamW optimizer if possible
fused_availabel = "fused" in inspect.signature(torch.optim.AdamW).parameters  # type: ignore
use_fused = fused_availabel and "cuda" in device
print(f"using fused AdamW: {use_fused}")

weight_decay = 0.04  # Typical value used in DiNO
decay_params = []
no_decay_params = []

for name, param in dino_model.named_parameters():
    if param.requires_grad:
        if len(param.shape) > 1:
            decay_params.append(param)  # Apply weight decay
        else:
            no_decay_params.append(param)  # Biases and LayerNorm weights

optimizer = torch.optim.AdamW(
    [
        {"params": decay_params, "weight_decay": config.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ],
    lr=config.min_lr,  # Start with warmup learning rate
    betas=config.betas,
    fused=use_fused,
)

train_dino(
    dino_model,
    data_loader,
    optimizer,
    device,
    num_epochs,
    base_lr=base_lr,
    min_lr=min_lr,
    warmup_epochs=warmup_epochs,
    save_dir='./checkpoints',
    tps=0.1,
    tpt=0.04,
    beta=0.9996,
    m=0.996
)
