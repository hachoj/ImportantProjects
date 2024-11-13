import torch.nn.functional as F

def compute_dino_loss(student_outputs, teacher_outputs, center, tau_s, tau_t):
    loss = 0
    total_terms = 0

    # For each student output (from all views)
    for student_output in student_outputs:
        # Apply sharpening to student output
        student_logits = (student_output - center) / tau_s
        student_probs = F.log_softmax(student_logits, dim=-1)

        # For each teacher output (from global views)
        for teacher_output in teacher_outputs:
            # Apply centering and sharpening to teacher output
            teacher_logits = (teacher_output - center) / tau_t
            teacher_probs = F.softmax(teacher_logits, dim=-1)

            # Compute cross-entropy loss
            loss += - (teacher_probs * student_probs).sum(dim=-1).mean()
            total_terms += 1

    loss /= total_terms
    return loss
