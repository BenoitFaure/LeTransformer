import torch
from torch import nn

from .AttentionLayer import AttentionLayer

class DecoderLayer(nn.Module):
    """
        Transformer Decoder Layer
    """

    def __init__(self, n_latent: int, n_heads: int, max_seq_len: int = 512):
        """
            Transformer Decoder Layer

            Parameters
            ----------
            n_latent
                Dimension of the output of each head
            n_heads
                Number of head
            max_seq_len
                Maximum sequence length to expect for
                masked forward attention during training

            Notes
            -----
            Input/Ouput Dimensions
                The input and output dimensions are to be of
                size n_latent*n_heads
        """
        super().__init__()

        layer_dimension = n_latent * n_heads
        self.multi_head_attention = AttentionLayer(
            layer_dimension, n_latent, n_heads,
            mask_forward_attention=True,
            max_seq_len=max_seq_len
            )
        self.layer_norm_mha = nn.LayerNorm(layer_dimension)

        self.multi_head_cross_attention = AttentionLayer(layer_dimension, n_latent, n_heads)
        self.layer_norm_mhca = nn.LayerNorm(layer_dimension)

        fcn_latent = layer_dimension * 4
        self.fully_connected = nn.Sequential(
            nn.Linear(layer_dimension, fcn_latent),
            nn.ReLU(),
            nn.Linear(fcn_latent, layer_dimension)
        )
        self.layer_norm_fcn = nn.LayerNorm(layer_dimension)
        
        nn.init.kaiming_uniform_(self.fully_connected[0].weight)
        nn.init.xavier_uniform_(self.fully_connected[1].weight)
    
    def forward(self, decoder_seq: torch.Tensor, encoder_context: torch.Tensor):

        decoder_context = self.multi_head_attention(decoder_seq)
        decoder_context = self.layer_norm_mha(decoder_context + decoder_seq)

        x = self.multi_head_cross_attention(decoder_context, context=encoder_context)
        x = self.layer_norm_mhca(x + decoder_context)

        fcn_out = self.fully_connected(x)
        x = self.layer_norm_fcn(fcn_out + x)

        return x
