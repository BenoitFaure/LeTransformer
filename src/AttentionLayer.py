import math

import torch
from torch import nn

class AttentionLayer(nn.Module):
    """
    Multihead attention layer
    """

    def __init__(self, n_in: int, n_out: int, n_heads: int):
        """
            Multihead attention layer.

            Parameters
            ----------
            n_in: int
                input dimension size
            n_out: int
                output dimension size per head
            n_heads: int
                number of attention heads
        """
        super().__init__()

        self.W_Q = nn.Parameter(torch.empty(n_heads, n_in, n_out))
        self.W_K = nn.Parameter(torch.empty(n_heads, n_in, n_out))
        self.W_V = nn.Parameter(torch.empty(n_heads, n_in, n_out))

        nn.init.xavier_uniform_(self.W_Q)
        nn.init.xavier_uniform_(self.W_K)
        nn.init.xavier_uniform_(self.W_V)

        scaling_param = math.sqrt(n_out)
        scaling_param = torch.as_tensor(scaling_param)
        self.register_buffer("scaling_param", scaling_param)

    def forward(self, values: torch.Tensor):
        batch_size, seq_len, _ = values.size()

        Q = torch.einsum('bsi, hio -> bhso', values, self.W_Q)
        K = torch.einsum('bsi, hio -> bhso', values, self.W_K)
        V = torch.einsum('bsi, hio -> bhso', values, self.W_V)

        attention = Q @ K.transpose(-2, -1)
        attention = attention / self.scaling_param
        attention = torch.softmax(attention, dim=-1)

        V = attention @ V
        V = V.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

        return V
