from typing import Optional

import math

import torch
from torch import nn

class AttentionLayer(nn.Module):
    """
        Multihead attention layer. Incorporates the possibility of masking forward attention.
    """

    def __init__(self, n_in: int, n_out: int, n_heads: int, mask_forward_attention: bool = False, max_seq_len: int = 512):
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
            mask_forward_attention: bool
                whether to mask forward attention
            max_seq_len: int
                maximum_seq_len to expect
        """
        super().__init__()
        
        self.mask_forward_attention = mask_forward_attention
        if self.mask_forward_attention:
            attention_mask = torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1).bool()
            self.register_buffer("attention_mask", attention_mask)

        self.W_Q = nn.Parameter(torch.empty(n_heads, n_in, n_out))
        self.W_K = nn.Parameter(torch.empty(n_heads, n_in, n_out))
        self.W_V = nn.Parameter(torch.empty(n_heads, n_in, n_out))

        nn.init.xavier_uniform_(self.W_Q)
        nn.init.xavier_uniform_(self.W_K)
        nn.init.xavier_uniform_(self.W_V)

        scaling_param = math.sqrt(n_out)
        scaling_param = torch.as_tensor(scaling_param)
        self.register_buffer("scaling_param", scaling_param)

    def forward(self, values: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = values.size()

        if context is None:
            context = values

        Q = torch.einsum('bsi, hio -> bhso', values, self.W_Q)
        K = torch.einsum('bsi, hio -> bhso', context, self.W_K)
        V = torch.einsum('bsi, hio -> bhso', context, self.W_V)

        attention = Q @ K.transpose(-2, -1)
        attention = attention / self.scaling_param

        if self.mask_forward_attention:
            attention = torch.masked_fill(attention, self.attention_mask[:seq_len, :seq_len], float('-inf'))
        attention = torch.softmax(attention, dim=-1)

        V = attention @ V
        V = V.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

        return V
