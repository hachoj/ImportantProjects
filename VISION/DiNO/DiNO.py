import torch
import torch.nn as nn
import torch.nn.functional as F
from config import config
from data_loader import TinyImageNetDataLoader
from model import ViT
from train import train_dino
import inspect


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

        # Initialize center as buffer to avoid backpropagation
        self.register_buffer("center", torch.zeros(1, student_arch.output_dim))

        # Ensure the teacher parameters do not get updated during backprop
        for param in self.teacher.parameters():
            param.requires_grad = False

    def update_teacher(self, student_model, teacher_model, momentum_teacher):
        for param_student, param_teacher in zip(
            student_model.parameters(), teacher_model.parameters()
        ):
            param_teacher.data = (
                param_teacher.data * momentum_teacher
                + param_student.data * (1.0 - momentum_teacher)
            )

    def distillation_loss(self, student_outputs, teacher_outputs, center, tau_s, tau_t):
        """
        Compute the distillation loss between student and teacher outputs.
        """
        total_loss = 0
        n_loss_terms = 0
        for student_output in student_outputs:
            # Student probabilities
            student_logits = (student_output - center) / tau_s
            student_probs = F.log_softmax(student_logits, dim=-1)

            for teacher_output in teacher_outputs:
                # Teacher probabilities
                teacher_logits = (teacher_output - center) / tau_t
                teacher_probs = F.softmax(teacher_logits, dim=-1)

                # Cross-entropy loss
                loss = torch.sum(-teacher_probs * student_probs, dim=-1).mean()
                total_loss += loss
                n_loss_terms += 1

        total_loss /= n_loss_terms
        return total_loss

    # def configure_optimizers(self, weight_decay, learning_rate, betas, device):
    #     # SOME OPTIMIZATION THAT WORKS FOR ViT
    #
    #     # Separate parameters into those with and without weight decay
    #     decay_params = []
    #     no_decay_params = []
    #     for name, param in self.named_parameters():
    #         if param.requires_grad:
    #             if len(param.shape) > 1:
    #                 decay_params.append(param)
    #             else:
    #                 no_decay_params.append(param)
    #
    #     # create a fused adamW optimizer if possible
    #     fused_availabel = 'fused' in inspect.signature(torch.optim.AdamW).parameters # type: ignore
    #     use_fused = fused_availabel and 'cuda' in device
    #     print(f"using fused AdamW: {use_fused}")
    #
    #     optimizer = torch.optim.AdamW([
    #         {'params': decay_params, 'weight_decay': weight_decay},
    #         {'params': no_decay_params, 'weight_decay': 0.0}
    #     ], lr=learning_rate, betas=betas, fused=use_fused)
    #     return optimizer


# learning rate + optim settings
min_lr = 1e-5
max_lr = 1e-3
betas = (0.9, 0.999)
# number of epochs
num_epochs = 100

# Actual Training
student = ViT(config)
teacher = ViT(config)
# try cuda, then mps, then cpu
device = "cuda" if torch.cuda.is_available() else "cpu"
device = "mps" if torch.mps.is_available() and "cuda" not in device else "cpu"

dino_model = DINO(student, teacher, device)
data_loader = TinyImageNetDataLoader()

# configure optimizer
# create a fused adamW optimizer if possible
fused_availabel = "fused" in inspect.signature(torch.optim.AdamW).parameters  # type: ignore
use_fused = fused_availabel and "cuda" in device
print(f"using fused AdamW: {use_fused}")

optimizer = torch.optim.AdamW(
    # [
    #     {"params": decay_params, "weight_decay": weight_decay},
    #     {"params": no_decay_params, "weight_decay": 0.0},
    # ],
    params=dino_model.parameters(),
    lr=min_lr,
    betas=betas,
    fused=use_fused,
)
train_dino(
    dino_model,
    data_loader,
    optimizer,
    device,
    num_epochs,
    config.student_temp,
    config.teacher_temp,
    config.beta,
    config.m,
)
