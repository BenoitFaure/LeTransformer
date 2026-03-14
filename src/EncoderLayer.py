import torch
from torch import nn

from .AttentionLayer import AttentionLayer

class EncoderLayer(nn.Module):
    """
        Transformer Encoder Layer
    """

    def __init__(self, n_latent: int, n_heads: int):
        """
            Transformer Encoder Layer

            Parameters
            ----------
            n_latent
                Dimension of the output of each head
            n_heads
                Number of head

            Notes
            -----
            Input/Ouput Dimensions
                The input and output dimensions are to be of
                size n_latent*n_heads
        """
        super().__init__()

        layer_dimension = n_latent * n_heads
        self.multi_head_attention = AttentionLayer(layer_dimension, n_latent, n_heads)
        self.layer_norm_mha = nn.LayerNorm(layer_dimension)

        fcn_latent = layer_dimension * 4
        self.fully_connected = nn.Sequential(
            nn.Linear(layer_dimension, fcn_latent),
            nn.ReLU(),
            nn.Linear(fcn_latent, layer_dimension)
        )
        self.layer_norm_fcn = nn.LayerNorm(layer_dimension)
        
        nn.init.kaiming_uniform_(self.fully_connected[0].weight)
        nn.init.xavier_uniform_(self.fully_connected[2].weight)
    
    def forward(self, values: torch.Tensor):

        context = self.multi_head_attention(values)
        x = self.layer_norm_mha(context + values)

        fcn_out = self.fully_connected(x)
        x = self.layer_norm_fcn(fcn_out + x)

        return x
