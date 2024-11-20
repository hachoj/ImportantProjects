import torch
from config import config
from data_loader import TinyImageNetDataLoader
from vit_model import ViT
from dino_model import DINO
from train import train_dino
import inspect
from torch.distributed import init_process_group, destroy_process_group
import os


def main():
    # Run with torchrun --nproc_per_node=NUM_GPUS main.py
    # DDP setup
    ddp = int(os.environ.get("RANK", -1)) != -1
    if ddp:
        init_process_group(backend="nccl")
        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])
        device = f"cuda:{ddp_local_rank}"
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0
    else:
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        master_process = True
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Adjust batch size for DDP
    batch_size = config.batch_size // ddp_world_size if ddp else config.batch_size

    # Setup data with DDP parameters
    data_loader = TinyImageNetDataLoader(
        batch_size=batch_size,
        num_workers=config.num_workers,
        ddp=ddp,
        ddp_rank=ddp_rank,
        ddp_world_size=ddp_world_size,
    )

    # Initialize models
    student = ViT(config)
    teacher = ViT(config)
    dino_model = DINO(student, teacher, device)

    torch.set_float32_matmul_precision("high")

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

    # Train with DDP flag
    train_dino(
        dino_model,
        data_loader,
        optimizer,
        device,
        config=config,
        ddp=ddp,
        ddp_local_rank=ddp_local_rank,
        master_process=master_process,
        save_dir="./checkpoints",
    )

    if ddp:
        destroy_process_group()


if __name__ == "__main__":
    main()
