import torch
from config import config
from data_loader import TinyImageNetDataLoader
from vit_model import ViT
from dino_model import DINO
from train import train_dino
import inspect

def main():
    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Initialize models
    student = ViT(config)
    teacher = ViT(config)
    dino_model = DINO(student, teacher, device)
    
    # Setup data
    data_loader = TinyImageNetDataLoader(
        batch_size=config.batch_size, 
        num_workers=config.num_workers
    )
    
    # Setup optimizer
    fused_availabel = "fused" in inspect.signature(torch.optim.AdamW).parameters  # type: ignore
    use_fused = fused_availabel and "cuda" in device
    print(f"using fused AdamW: {use_fused}")

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
        lr=config.base_lr,  # Start with warmup learning rate
        betas=config.betas,
        fused=use_fused,
    )
    
    # Train
    train_dino(
        dino_model,
        data_loader,
        optimizer,
        device,
        config=config,
        save_dir='./checkpoints'
    )

if __name__ == "__main__":
    main()