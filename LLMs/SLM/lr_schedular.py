import math

def get_lr(it, warmup_steps, max_steps, min_lr, max_lr):
    # linear warmup followed by cosine decay
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    if it > max_steps:
        return min_lr
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)

def get_alpha_cosine_decay(current_step, total_steps, initial_alpha=0.9, final_alpha=0.1):
    """
    Computes the alpha value for the current training step using cosine decay.

    Args:
    - current_step: The current training step.
    - total_steps: The total number of training steps.
    - initial_alpha: The starting value of alpha (typically high, e.g., 0.9 or 1.0).
    - final_alpha: The ending value of alpha (typically lower, e.g., 0.1).

    Returns:
    - alpha: The alpha value for the current step.
    """
    progress = current_step / total_steps
    # Cosine decay from initial_alpha to final_alpha
    alpha = final_alpha + 0.5 * (initial_alpha - final_alpha) * (1 + math.cos(math.pi * progress))
    return alpha