import torch
import torch.nn as nn
import torch.nn.functional as F

from attention import CausalSelfAttention
from mlp import MLP

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.rms_1 = nn.RMSNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.rms_2 = nn.RMSNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x, positions=None):
        x = x + self.attn(self.rms_1(x), positions)
        x = x + self.mlp(self.rms_2(x))
        return x