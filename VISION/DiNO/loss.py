import torch.nn.functional as F
import json
import matplotlib.pyplot as plt
import math
def compute_dino_loss(student_outputs, teacher_outputs, center, tau_s, tau_t, global_step, total_steps):
    """
    Args:
        student_outputs: Tensor [batch_size, 8, embedding_dim] (all views)
        teacher_outputs: Tensor [batch_size, 2, embedding_dim] (only global views)
        center: Center for loss computation [1, embedding_dim]
        tau_s: Student temperature
        tau_t: Teacher temperature
        global_step: Current training step
        total_steps: Total number of training steps
    """
    # Apply sharpening to temperatures
    tau_s = tau_s * (1 + math.cos(math.pi * global_step / total_steps)) / 2
    
    # Student probabilities (all views)
    student_center = center.unsqueeze(1).expand(student_outputs.size(0), student_outputs.size(1), -1)
    student_logits = (student_outputs - student_center) / tau_s
    student_probs = F.log_softmax(student_logits, dim=-1)  # [batch_size, 8, embedding_dim]
    
    # Teacher probabilities (global views)
    teacher_center = center.unsqueeze(1).expand(teacher_outputs.size(0), teacher_outputs.size(1), -1)
    teacher_logits = (teacher_outputs - teacher_center) / tau_t
    teacher_probs = F.softmax(teacher_logits, dim=-1)  # [batch_size, 2, embedding_dim]
    
    loss = 0
    n_loss_terms = 0
    
    # Compare each student view with each teacher view
    for i in range(student_outputs.shape[1]):  # for each student view
        for j in range(teacher_outputs.shape[1]):  # for each teacher view
            loss += -(teacher_probs[:, j, :] * student_probs[:, i, :]).sum(dim=-1).mean()
            n_loss_terms += 1
            
    return loss / n_loss_terms

def load_and_plot_metrics(metrics_path):
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    
    # Plot training curves
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    # Loss plot
    ax1.plot(metrics['steps'], metrics['loss'], alpha=0.3, label='Per step')
    ax1.plot(range(len(metrics['epoch_avg_losses'])), 
             metrics['epoch_avg_losses'], 
             'r-', label='Epoch average')
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Steps')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    # Norms plot
    ax2.plot(metrics['steps'], metrics['student_norm'], label='Student')
    ax2.plot(metrics['steps'], metrics['teacher_norm'], label='Teacher')
    ax2.plot(metrics['steps'], metrics['center_norm'], label='Center')
    ax2.set_title('Network Norms')
    ax2.set_xlabel('Steps')
    ax2.set_ylabel('L2 Norm')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()
