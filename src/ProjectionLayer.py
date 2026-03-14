import math

import torch
from torch import nn

class ProjectionLayer(nn.Module):
    """
        Projection Logic of Transformer
    """

    def __init__(self, n_dict: int, n_model: int):
        """
            Project to and from dictionary space to embedding space

            Parameters
            ----------
            n_dict: int
                Dimension of dictionary
            n_model: int
                Dimension of model
        """
        super().__init__()

        self.embeddings = nn.Embedding(n_dict, n_model)
        self.project_embeddings = nn.Linear(n_model, n_dict, bias=False)
        self.project_embeddings.weight = self.embeddings.weight

        nn.init.xavier_uniform_(self.embeddings.weight)

        scaling_factor = torch.sqrt(torch.tensor(float(n_model)))
        self.register_buffer("scaling_factor", scaling_factor)
    
    def project_to_embeddings(self, values: torch.Tensor):

        return self.embeddings(values) * self.scaling_factor

    def project_to_vocabulary(self, values: torch.Tensor):

        return self.project_embeddings(values)
