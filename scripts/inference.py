import torch

from src import Transformer

# --- 1. REBUILD TOKENIZER ---
class CharTokenizer:
    def __init__(self, vocab_dict):
        self.vocab = vocab_dict
        self.inv_vocab = {i: ch for ch, i in self.vocab.items()}
        self.n_vocab = len(self.vocab)

    def encode(self, text):
        return [self.vocab[ch] for ch in text if ch in self.vocab]

    def decode(self, ids):
        # We stop decoding if we hit the EOS token (ID 1)
        res = ""
        for i in ids:
            if i == 1: break # Stop at EOS
            res += self.inv_vocab.get(i, "")
        return res

# --- 2. GENERATION SCRIPT ---
def run_generation(prompt_text: str, temperature: float = 0.8):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model on: {device}")

    checkpoint = torch.load("./model/letransformer.pt", map_location=device)

    tokenizer = CharTokenizer(checkpoint['tokenizer_vocab'])
    hp = checkpoint['hyperparams']

    model = Transformer(
        n_latent=hp['n_latent'],
        n_heads=hp['n_heads'],
        n_layers=hp['n_layers'],
        n_vocab=hp['n_vocab'],
        max_seq_len=hp['max_seq_len']
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"--- Generating based on prompt: '{prompt_text}' ---")

    prompt_ids = torch.tensor([tokenizer.encode(prompt_text)], device=device)

    with torch.no_grad():
        prompt_emb = model.projection.project_to_embeddings(prompt_ids)
        prompt_emb = model.position_encoder(prompt_emb)

        encoder_context = model.encoder(prompt_emb)
        generated_ids = model.generate(encoder_context, temperature=temperature)

    # G. Decode the Result
    decoded_output = tokenizer.decode(generated_ids[0].tolist())
    
    print("\n[Generated Script]:\n")
    print(decoded_output)

if __name__ == "__main__":
    # Example usage:
    # Give it a snippet of code as a starting point for the Encoder
    seed = "class DecoderLayer(nn.Module):"
    run_generation(seed, temperature=0.7)