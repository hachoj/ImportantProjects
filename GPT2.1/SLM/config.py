import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class GPT_config:
    block_size:int = 1024
    vocab_size:int = 50280
    n_layer:int = 12
    n_head:int = 12
    n_embd:int = 768
