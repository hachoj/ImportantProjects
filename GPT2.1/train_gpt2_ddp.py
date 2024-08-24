import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass
import inspect
import tiktoken
import numpy as np
import time
import math
import os

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        # makes sure that the number of embedding is some multiple of
        # the number of heads
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1 # type: ignore
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd

        # not really a 'bias', more fo a mask, but following the OpenAI/HF naming
        self.register_buffer('bias', torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        qkv = self.c_attn(x)
        q, k, v  = qkv.split(self.n_embd, dim=2)
        # k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        # q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        # v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # (B, nh, T, hs) Flash attention

        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(config.n_embd * 4, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1 # type: ignore

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPT_config:
    block_size:int = 1024
    vocab_size:int = 50257
    n_layer:int = 12
    n_head:int = 12
    n_embd:int = 768

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config 

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd)
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.transformer.wte.weight = self.lm_head.weight # weight tying

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
     
    def forward(self, idx, targets=None):
        # idx is of shape (B, T)
        B, T = idx.size()
        assert T <= self.config.block_size, "Cannot forward, model block size is exhausted."
        # forward the token and position embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
        pos_emb = self.transformer.wpe(pos) # position embedding of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx) # token embedding of shape (B, T, n_embd)
        x = tok_emb + pos_emb 
        # froward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        # forward the final layer norm and the classifier 
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPT_config(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict() # type:ignore 

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model


    def configure_optimizers(self, weight_decay, learning_rate, device):
        # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        # print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        # print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # create a fused adamW optimizer if possible
        fused_availabel = 'fused' in inspect.signature(torch.optim.AdamW).parameters # type: ignore
        use_fused = fused_availabel and 'cuda' in device
        # print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8) # type: ignore
        return optimizer

class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes

        with open('input.txt', 'r') as f:
            text = f.read()
        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        # print(f"loaded {len(self.tokens)} tokens")
        # print(f"1 epoch = {len(self.tokens) // (B*T)} batches")

        # state
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position:self.current_position + B * T + 1]
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)
        self.current_position += B * T * self.num_processes
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_position = self.B * self.T * self.process_rank
        return x, y



if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # fine_tuning = True
    # lr = 1e-5 if fine_tuning else 3e-4

    # ----------------------------------------------------------------------
    # Setting up DDP

    from torch.distributed import init_process_group, destroy_process_group
    from torch.nn.parallel import DistributedDataParallel as DDP
    import torch.distributed as dist

    ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run
    # ddp = False
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
        device = 'cpu'
        if torch.cuda.is_available():
            device = 'cuda'
        print(f"Running on {device}")

    # ----------------------------------------------------------------------

    torch.manual_seed(1337)
    torch.cuda.manual_seed(1337)

    # calculate the number of gradient accumulation steps
    # for the desired batch size
    total_batch_size = 524288
    B=32
    T=1024
    assert total_batch_size % (B * T * ddp_world_size) == 0
    grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
    if master_process:
        print(f"total desired batch size: {total_batch_size}")
        print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

    train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size)

    torch.set_float32_matmul_precision('high')

    # create model
    # if master_process:
    #     load_pretrained = True if input("Load pretrained model? [y/n]: ") == 'y' else False
    load_pretrained = False
    model = GPT.from_pretrained('gpt2') if load_pretrained else GPT(GPT_config)
    model.to(device)
    model = torch.compile(model)
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])
    raw_model = model.module if ddp else model    

    # taken from GPT-3 paper
    max_lr = 6e-4
    min_lr = max_lr * 0.1
    warmup_steps = 10
    max_steps = 50

    def get_lr(it):
        # linear warmup followed by cosine decay
        if it < warmup_steps:
            return max_lr * (it+1) / warmup_steps
        if it > max_steps:
            return min_lr
        decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
        return min_lr + coeff * (max_lr - min_lr)    

    optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device=device)

    t00 = time.time()

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
        lr = get_lr(step)
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

        # if step%(max_steps/20) == 0:
        #     print(f"step {step+1}/{max_steps} | loss: {loss.item():.6f} | norm: {norm:.2f} | tokens/s: {tokens_per_sec:.2f} | ms/batch: {dt:.2f}")

    t11 = time.time()
    print("--------------------------------------------------")
    print(f"Total time: {(t11 - t00):.2f} s")

    torch.save(model.state_dict(), 'models/gpt2_nano_test.pth')

    import sys; sys.exit(0)




# --------------------------------------------------------------------------------------

# model = GPT.from_pretrained('gpt2')
# # model.eval()
# model.to('cuda')

# # prefix tokens
# import tiktoken
# enc = tiktoken.get_encoding('gpt2')
# text = open('shakespeare_long.txt', 'r').read()
# tokens = enc.encode(text)
# # tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
# # x = tokens.to('cuda')

# buf = torch.tensor(tokens[:GPT_config.block_size*GPT_config.batch_size + 1], requires_grad=False)
# x = buf[:-1].view(GPT_config.batch_size, GPT_config.block_size).to(GPT_config.device)
# y = buf[1:].view(GPT_config.batch_size, GPT_config.block_size).to(GPT_config.device)

# # model = GPT(GPT_config).to(GPT_config.device)
# logits, loss = model(x, y)
# print(logits.shape, loss)


# # generate
# num_return_sequences = 5
# max_length = 30
# torch.manual_seed()
# torch.cuda.manual_seed(42)
# tokens = enc.encode('')
# tokens = torch.tensor(tokens, dtype=torch.long, device='cuda')
# tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
# x = tokens.to('cuda')
# while x.size(1) < max_length:
#     with torch.no_grad():
#         logits, loss = model(x)

#         logits = logits[:, -1, :]

#         probs = F.softmax(logits, dim=-1)

#         topk_probs, topk_indices = torch.topk(probs, 10, dim=-1)

#         ix = torch.multinomial(topk_probs, num_samples=1)

#         xcol = torch.gather(topk_indices, -1, ix)

#         x = torch.cat((x, xcol), dim=1)

# # decode
# for i in range(num_return_sequences):
#     tokens = x[i, :max_length].tolist()
#     decoded = enc.decode(tokens)
#     print(">", decoded)


# --------------------------------------------------------------------------------------
# fine-tuning


# if __name__ == '__main__':
#     import tiktoken
#     enc = tiktoken.get_encoding('gpt2')

#     model = GPT.from_pretrained('gpt2')
#     model.to('cuda')

#     # prefix tokens
#     text = open('kendrick.txt', 'r').read()
#     tokens = enc.encode(text)
#     n = int(0.9*len(tokens))
#     train_data = tokens[:n]
#     val_data = tokens[n:]
#     def get_batch(split):
#         data = train_data if split == 'train' else val_data
#         ix = torch.randint(0, len(data) - GPT_config.block_size * GPT_config.batch_size, (1,)).item()
#         buf = torch.tensor(data[ix:ix+GPT_config.block_size*GPT_config.batch_size + 1], requires_grad=False)

#         x = buf[:-1].view(GPT_config.batch_size, GPT_config.block_size).to(GPT_config.device)
#         y = buf[1:].view(GPT_config.batch_size, GPT_config.block_size).to(GPT_config.device)
#         return x, y
#     parameters = 0
#     for parameter in model.parameters():
#         pc = 1
#         for dim in range(parameter.ndim):
#             pc *= parameter.size(dim=dim)
#         parameters += pc
#     print(f'Num parameters: {parameters}')
#     optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
#     model.train()
#     for steps in range(GPT_config.max_iters):
#         break
#         print(F"{steps}/{GPT_config.max_iters}")
#         xb, yb = get_batch('train')

#         logits, loss = model(xb, yb)
#         optimizer.zero_grad(set_to_none=True)
#         loss.backward()
#         optimizer.step()
#     # torch.save(model.state_dict(), 'model/gpt2_funny.pth')