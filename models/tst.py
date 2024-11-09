# Time-series Transformer model
from typing import Tuple
import torch
import torch.nn as nn
from collections import OrderedDict
from dataclasses import dataclass
import json

# for imu data, which is 2 channels, 3x200 data, input.
# or process the data to 1 channels but 6x200 data, input. TODO::
@dataclass
class ModelArgs:
    # Input
    in_channels: int = 2 # 1 or 2 c
    img_height: int = 3 #  6 or 3 with channel
    img_width: int = 200
    
    # Patch Embedding
    patch_size_height: int = 1 # 1 or 3 h
    patch_size_width: int = 20

    # Transformer
    d_model: int = 64 # d
    n_head: int = 8 # nh
    n_layer: int = 3 # nl
    dropout: float = 0.1

    @classmethod
    def from_json(cls, file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        return cls(**data)
    

class Block(nn.Module):
    '''
    ResidualAttentionBlock
    Add dropout to avid overfitting
    '''
    def __init__(self, d_model: int, n_head: int, dropout: float=0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", nn.GELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))

    def attention(self, x):
        return self.attn(x, x, x)[0]

    def forward(self, x):
        # x: (seq_length, batch_size, d_model)
        x = x + self.attention(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class Transformer(nn.Module):
    def __init__(self, n_model: int, n_layer: int, n_head: int, dropout: float=0):
        super().__init__()
        self.width = n_model
        self.layers = n_layer
        self.blocks = nn.Sequential(*[Block(n_model, n_head, dropout) for _ in range(n_layer)])
    
    def forward(self, x: torch.Tensor):
        return self.blocks(x)

class VisionTransformer(nn.Module):
    def __init__(self, in_channels: int, img_size: Tuple[int, int], patch_size: Tuple[int, int], d_model: int, n_layer: int, n_head: int, dropout: float):
        super().__init__()
        self.num_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])

        # Patch Embedding
        self.conv1 = nn.Conv2d(in_channels, d_model, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = d_model ** -0.5
        self.cls_token = nn.Parameter(scale * torch.randn(d_model))
        self.pos_embed = nn.Parameter(
            scale * torch.randn(self.num_patches + 1, d_model)
        )
        self.ln_pre = nn.LayerNorm(d_model)

        self.transformer = Transformer(d_model, n_layer, n_head, dropout)

        self.ln_post = nn.LayerNorm(d_model)
        # self.proj = nn.Parameter(scale * torch.randn(d_model, out_dim))
        self.proj = None

    def forward(self, x):
        x = self.conv1(x) # (batch_size, d_model, h, w)
        x = x.flatten(2).transpose(1, 2) # (batch_size, num_patches, d_model)

        x = torch.cat([self.cls_token.to(x.dtype) + torch.zeros(x.size(0), 1, x.size(-1), device=x.device), x], dim=1) # (batch_size, num_patches + 1, d_model)
        x = x + self.pos_embed.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2) # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2) # LND -> NLD

        x = self.ln_post(x[:, 0, :]) # (batch_size, d_model)
        if self.proj is not None:
            x = x @ self.proj
        return x

class Classifier(nn.Module):
    def __init__(self, d_model, num_classes):
        super().__init__()
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        return self.classifier(x)

class TestModel(nn.Module):
    def __init__(self, params: ModelArgs, num_classes: int):
        super().__init__()
        self.in_channels = params.in_channels
        assert self.in_channels in [1, 2], f"Invalid in_channels: {self.in_channels}"
        self.vit = VisionTransformer(params.in_channels, (params.img_height, params.img_width), (params.patch_size_height, params.patch_size_width), params.d_model, params.n_layer, params.n_head, params.dropout)
        self.classifier = Classifier(params.d_model, num_classes)

    def forward(self, x):
        # x: (batch_size, feature, seq_len)
        if self.in_channels == 1:
            x = x.unsqueeze(1) # (batch_size, 1, feature, seq_len)
        else:
            x = x.view(x.size(0), 2, -1, x.size(2))  # (batch_size, 2, feature//2, seq_len)
        x = self.vit(x)
        x = self.classifier(x)
        return x

if __name__ == '__main__':
    # Parameters
    args = ModelArgs()
    n_class = 7

    # Create a random batch of images
    batch_size = 2
    x = torch.randn(batch_size, args.in_channels, args.img_height, args.img_width)
    print(x.shape)  # Output: torch.Size([2, 2, 3, 200])
    # Instantiate the TestModel
    model = TestModel(args, n_class)

    # Forward pass
    logits = model(x)
    print(logits.shape)  # Output: torch.Size([2, 10])
    print(args.__dict__)