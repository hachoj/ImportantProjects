from dataclasses import dataclass

@dataclass
class config:
    # Training settings
    batch_size: int = 16
    num_workers: int = 4
    base_lr: float = 5e-5
    min_lr: float = 1e-6
    num_epochs: int = 100
    warmup_epochs: int = 3
    weight_decay: float = 0.05

    # Optimizer settings
    betas: tuple = (0.9, 0.999)
    
    # Model architecture (ViT-Sish)
    img_size: int = 224
    patch_size: int = 16
    num_patches: int = (img_size // patch_size) ** 2
    sequence_len: int = num_patches + 1
    n_layer: int = 12
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.1
    
    
    # DINO specific
    student_temp: float = 0.1
    teacher_temp: float = 0.04
    beta: float = 0.9996
    m: float = 0.996

    # Training settings
    print_every: int = 50
    checkpoint_freq: int = 5
