from typing import Optional

import torch
from torch import nn

from .ProjectionLayer import ProjectionLayer
from .PositionEncoder import PositionEncoder
from .EncoderBlock import EncoderBlock
from .DecoderBlock import DecoderBlock

class Transformer(nn.Module):
    """
        Transformer Module
    """

    def __init__(self, n_latent: int, n_heads: int, n_layers : int, n_vocab: int, max_seq_len: int):
        """
            Tranformer module

            Parameters
            ----------
            n_latent
                Dimension of the output of each head
            n_heads
                Number of head
            n_layers
                Number of layers in the encoder and decoder
            n_vocab
                Size of the vocabulary
            max_seq_len
                Maximum sequence length to allow
        """
        super().__init__()

        self.max_seq_length = max_seq_len

        n_model = n_latent * n_heads
        self.projection = ProjectionLayer(n_vocab, n_model)
        self.position_encoder = PositionEncoder(n_model, max_seq_len)

        self.encoder = EncoderBlock(n_latent, n_heads, n_layers)
        self.decoder = DecoderBlock(n_latent, n_heads, n_layers, max_seq_len)
    
    def forward(self, tockens: torch.Tensor, teacher_tockens: Optional[torch.Tensor] = None, temperature = 1):

        embeddings = self.projection.project_to_embeddings(tockens)
        embeddings = self.position_encoder(embeddings)

        encoder_context = self.encoder(embeddings)

        if teacher_tockens is not None:
            teacher_embeddings = self.projection.project_to_embeddings(teacher_tockens)
            teacher_embeddings = self.position_encoder(teacher_embeddings)
            decoded_embeddings = self.decoder(teacher_embeddings, encoder_context)
            return self.projection.project_to_vocabulary(decoded_embeddings)
        
        else:
            return self.generate(encoder_context, temperature)
    
    @torch.no_grad()
    def generate(self, encoder_context: torch.Tensor, temperature):
        sos_tocken_id = 0

        device = encoder_context.device
        batch_size = encoder_context.size(0)

        generated_tockens = torch.full((batch_size, 1), sos_tocken_id, dtype=torch.long, device=device)

        for _ in range(self.max_seq_length - 1):
            
            generated_embeddings = self.projection.project_to_embeddings(generated_tockens)
            generated_embeddings = self.position_encoder(generated_embeddings)

            decoder_embeddings = self.decoder(generated_embeddings, encoder_context)
            decoder_predicted_tocken_logits = self.projection.project_to_vocabulary(decoder_embeddings[:, -1, :])

            decoder_predicted_tocken_logits = decoder_predicted_tocken_logits / temperature
            decoder_predicted_tocken_probs = torch.softmax(decoder_predicted_tocken_logits, dim=-1)
            decoder_predicted_tocken = torch.multinomial(decoder_predicted_tocken_probs, num_samples=1)

            generated_tockens = torch.cat([generated_tockens, decoder_predicted_tocken], dim=1)
        
        return generated_tockens