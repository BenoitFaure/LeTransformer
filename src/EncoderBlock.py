import torch
from torch import nn

from .EncoderLayer import EncoderLayer

class EncoderBlock(nn.Module):
    """
        Transformer Encoder Layer
    """

    def __init__(self, n_latent: int, n_heads: int, n_layers: int):
        """
            Transformer Encoder Block - Chains Encoder Layers

            Parameters
            ----------
            n_latent
                Dimension of the output of each head
            n_heads
                Number of head per layer
            n_layers
                Number of layers to stack
        """
        super().__init__()

        self.layer_stack = nn.Sequential(*[
            EncoderLayer(n_latent, n_heads) for _ in range(n_layers)
        ])
    
    def forward(self, values: torch.Tensor):

        return self.layer_stack(values)
