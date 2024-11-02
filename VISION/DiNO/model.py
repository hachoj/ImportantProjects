import torch
import torch.nn as nn
import torch.nn.functional as F

import inspect

from model_block import Block
from PatchEmbedding import PatchEmbedding

class ViT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config 
        self.transformer = nn.ModuleDict(dict(
            embd = PatchEmbedding(config.img_size, config.patch_size, config.embd_dim),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd)
        ))

        # the final layer is a linear projection that maps the output of the transformer to the vocabulary
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        # some weight initialization that works for ViT
        return module
     
    def forward(self, x):
        # get the embeddings from the image
        x = self.transformer.embd(x)

        # froward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        # forward the final layer norm and the classifier 
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        return logits

    def configure_optimizers(self, weight_decay, learning_rate, device):
        # SOME OPTIMIZATION THAT WORKS FOR ViT

        # # start with all of the candidate parameters (that require grad)
        # param_dict = {pn: p for pn, p in self.named_parameters()}
        # param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        # decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        # nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        # optim_groups = [
        #     {'params': decay_params, 'weight_decay': weight_decay},
        #     {'params': nodecay_params, 'weight_decay': 0.0}
        # ]
        # num_decay_params = sum(p.numel() for p in decay_params)
        # num_nodecay_params = sum(p.numel() for p in nodecay_params)
        # print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        # print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # # create a fused adamW optimizer if possible
        # fused_availabel = 'fused' in inspect.signature(torch.optim.AdamW).parameters # type: ignore
        # use_fused = fused_availabel and 'cuda' in device
        # print(f"using fused AdamW: {use_fused}")
        # optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8) # type: ignore
        # return optimizer
        return None
