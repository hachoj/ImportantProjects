import torch
import torch.nn as nn
import torch.nn.functional as F

from swiglu import swiglu

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.swiglu = swiglu(4 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd * 4, config.n_embd)
        self.c_proj.REGULARIZE = 1 # type: ignore

    def forward(self, x):
        x = self.c_fc(x)
        x = self.swiglu(x)
        x = self.c_proj(x)
        return x