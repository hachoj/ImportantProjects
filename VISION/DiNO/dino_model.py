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

    def forward(self, x, is_student=True):
        if is_student:
            return self.student(x)
        else:
            return self.teacher(x)
        
    def get_teacher_params(self):
        return self.teacher.parameters()
    
    def get_student_params(self):
        return self.student.parameters()
