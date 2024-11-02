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
            embd = PatchEmbedding(config.img_size, config.patch_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
        ))

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Parameter):
            nn.init.normal_(module, std=0.02)
     
    def forward(self, x):
        # get the embeddings from the image
        x = self.transformer.embd(x)

        # froward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        # forward the final layer norm and the classifier 
        return x

    def configure_optimizers(self, weight_decay, learning_rate, betas, device):
        # SOME OPTIMIZATION THAT WORKS FOR ViT

        # Separate parameters into those with and without weight decay
        decay_params = []
        no_decay_params = []
        for name, param in self.named_parameters():
            if param.requires_grad:
                if len(param.shape) > 1:
                    decay_params.append(param)
                else:
                    no_decay_params.append(param)

        # create a fused adamW optimizer if possible
        fused_availabel = 'fused' in inspect.signature(torch.optim.AdamW).parameters # type: ignore
        use_fused = fused_availabel and 'cuda' in device
        print(f"using fused AdamW: {use_fused}")

        optimizer = torch.optim.AdamW([
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ], lr=learning_rate, betas=betas, fused=use_fused)
        return optimizer