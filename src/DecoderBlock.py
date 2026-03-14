import torch
from torch import nn

from .DecoderLayer import DecoderLayer

class DecoderBlock(nn.Module):
    """
        Transformer Decoder Block
    """

    def __init__(self, n_latent: int, n_heads: int, n_layers: int, max_seq_len: int = 512):
        """
            Transformer Decoder Block

            Parameters
            ----------
            n_latent
                Dimension of the output of each head
            n_heads
                Number of head
            max_seq_len
                Maximum sequence length to expect for
                masked forward attention during training
        """
        super().__init__()

        self.layer_stack = nn.ModuleList([
            DecoderLayer(n_latent, n_heads, max_seq_len=max_seq_len) for _ in range(n_layers)
        ])

    def forward(self, decoder_seq: torch.Tensor, encoder_context: torch.Tensor):

        for decoder_block in self.layer_stack:
            decoder_seq = decoder_block(decoder_seq, encoder_context)
        
        return decoder_seq