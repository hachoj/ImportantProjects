
import torch
import torch.nn as nn
from torch.nn import functional as F
import time
import math
import os

from config import GPT_config
from data_loader import DataLoaderLite
from model import GPT
from lr_schedular import get_lr

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ----------------------------------------------------------------------
    # Setting up DDP
    # torchrun --standalone --nproc_per_node=NUMGPUS FILENAME.py

    from torch.distributed import init_process_group, destroy_process_group
    from torch.nn.parallel import DistributedDataParallel as DDP
    import torch.distributed as dist

    ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run
    print(f"using ddp: {ddp}")
    if ddp:
        # use of DDP atm demands CUDA, we set the device appropriately according to the rank
        assert torch.cuda.is_available()
        init_process_group(backend='nccl')
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0
    else:
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        master_process = True
        # attempt  to autodetect device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Running on {device}")

    # ----------------------------------------------------------------------

    torch.manual_seed(1337)
    torch.cuda.manual_seed(1337)

    # calculate the number of gradient accumulation steps
    # for the desired batch size
    total_batch_size = 524288
    B=4
    T=1024
    assert total_batch_size % (B * T * ddp_world_size) == 0
    grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
    if master_process:
        print(f"total desired batch size: {total_batch_size}")
        print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

    train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size)

    torch.set_float32_matmul_precision('high')

    model = GPT(GPT_config)
    model.to(device)
    model = torch.compile(model)
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])
    raw_model = model.module if ddp else model    

    # taken from GPT-3 paper
    max_lr = 6e-4
    min_lr = max_lr * 0.1
    warmup_steps = 200
    max_steps = 2000 #TODO TBD

    optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device=device)

    # training loop
    for step in range(max_steps):
        t0 = time.time()
        optimizer.zero_grad()
        loss_accum = 0.0  # this is just for printing the loss
        for microstep in range(grad_accum_steps):
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                logits, loss = model(x, y)
            # each loss.backward() call accumulates gradients
            loss = loss / grad_accum_steps # scale the loss for the gradient accumulation
            loss_accum += loss.detach()
            if ddp:
                model.require_backward_grad_sync = (microstep == grad_accum_steps - 1)
            loss.backward()
        if ddp:
            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
        # taken from GPT-3 paper
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # vairable learning rate
        lr = get_lr(step, warmup_steps, max_steps, min_lr, max_lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        optimizer.step()
        torch.cuda.synchronize()
        t1 = time.time()
        dt = (t1 - t0) # time difference in seconds
        tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
        tokens_per_sec = tokens_processed / dt
        if master_process:
            print(f"step {step:4d} | loss: {loss_accum.item():.6f} | lr {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")  # type: ignore

    if ddp:
        destroy_process_group()

    torch.save(model.state_dict(), 'models/.pth')