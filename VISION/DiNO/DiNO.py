import torch
import torch.nn as nn
from loss import distilled_loss
from data_loader import TinyImageNetDataLoader

class DINO(nn.Module):
    def __init__(self, student_arch, teacher_arch, device):
        """
        Args:
            student_arch (nn.Module): ViT Network for student_arch
            teacher_arch (nn.Module): ViT Network for teacher_arch
            device: torch.device
        """
        super(DINO, self).__init__()
    
        self.student = student_arch().to(device)
        self.teacher = teacher_arch().to(device)
        self.teacher.load_state_dict(self.student.state_dict())

        # Initialize center as buffer to avoid backpropagation
        self.register_buffer('center', torch.zeros(1, student_arch().output_dim)).to(device)

        # Ensure the teacher parameters do not get updated during backprop
        for param in self.teacher.parameters():
            param.requires_grad = False

    def update_teacher(student_model, teacher_model, momentum_teacher):
        for param_student, param_teacher in zip(student_model.parameters(), teacher_model.parameters()):
            param_teacher.data = param_teacher.data * momentum_teacher + param_student.data * (1.0 - momentum_teacher)

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

