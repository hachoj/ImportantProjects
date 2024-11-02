from dataclasses import dataclass

@dataclass
class config:
    img_size:int = 224
    patch_size:int = 16
    num_patches:int = (img_size // patch_size) ** 2
    sequence_len:int = num_patches + 1  # 1 for CLS token
    n_layer:int = 12
    n_head:int = 6
    n_embd:int = 384

    dropout:float = 0.1
    weight_decay:float = 0.1
    betas:tuple = (0.9, 0.999)
