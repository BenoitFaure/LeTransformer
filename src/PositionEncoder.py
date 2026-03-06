import math

import torch
from torch import nn

class PositionEncoder(nn.Module):
    """
        First encoding of transformer (Encoding + Position tags)
    """

    def __init__(self, n_dict: int, n_model: int, ctx_window: int):
        """
            Layer:
                1] Project from dictionary space to embedding space
                2] Adding sinusoidal positional encodings

            Parameters
            ----------
            n_dict: int
                Dimension of dictionary
            n_model: int
                Dimension of 1st layer of model
        """
        super().__init__()

        # Projection
        self.W_1 = nn.Parameter(torch.empty(n_dict, n_model))
        self.b_1 = nn.Parameter(torch.empty(n_model))

        nn.init.xavier_uniform_(self.W_1)
        nn.init.constant_(self.b_1, 0)
        
        # Positional Encoddings
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

        # Projection
        values = values @ self.W_1 + self.b_1
        values = values * self.scaling_factor

        # Position Embeddings
        ctx_length = values.size(-2)
        values = values + self.pos_embeddings[:, :ctx_length, :]

        return values
