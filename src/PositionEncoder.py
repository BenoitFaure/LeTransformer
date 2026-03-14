import math

import torch
from torch import nn

class PositionEncoder(nn.Module):
    """
        Position encode a sequence
    """

    def __init__(self, n_model: int, ctx_window: int):
        """
            Add sinusoidal positional encodings

            Parameters
            ----------
            n_model: int
                Dimension of 1st layer of model
            ctx_window: int
                Max length of the sequence
        """
        super().__init__()
        
        embedding_axis = torch.arange(0, n_model, 2).float()
        embedding_axis = torch.exp(embedding_axis * (-math.log(10000.0) / n_model))

        embedding_axis = embedding_axis.unsqueeze(0)
        ctx_axis = torch.arange(ctx_window).float().unsqueeze(1)
        position_matrix = ctx_axis @ embedding_axis

        even_pos_embeddings = torch.sin(position_matrix)
        odd_pos_embeddings = torch.cos(position_matrix)

        pos_embeddings = torch.stack((even_pos_embeddings, odd_pos_embeddings), dim=-1)
        pos_embeddings = pos_embeddings.flatten(start_dim=-2)

        scaling_factor = torch.sqrt(torch.Tensor(n_model))

        self.register_buffer("pos_embeddings", pos_embeddings.unsqueeze(0))
        self.register_buffer("scaling_factor", scaling_factor)
    
    def forward(self, values: torch.Tensor):

        ctx_length = values.size(-2)
        values = values + self.pos_embeddings[:, :ctx_length, :]

        return values
