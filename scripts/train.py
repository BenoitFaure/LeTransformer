from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from src import Transformer

# --- 1. SIMPLE TOKENIZER ---
class CharTokenizer:
    def __init__(self, text):
        self.chars = sorted(list(set(text)))
        # Reserve 0 for SOS, 1 for EOS, 2 for PAD
        self.vocab = {ch: i + 3 for i, ch in enumerate(self.chars)}
        self.vocab['<SOS>'] = 0
        self.vocab['<EOS>'] = 1
        self.vocab['<PAD>'] = 2
        self.inv_vocab = {i: ch for ch, i in self.vocab.items()}
        self.n_vocab = len(self.vocab)

    def encode(self, text):
        return [self.vocab[ch] for ch in text if ch in self.vocab]

    def decode(self, ids):
        return "".join([self.inv_vocab.get(i, "") for i in ids])

# --- 2. DATASET PREPARATION ---
class CodeDataset(Dataset):
    def __init__(self, data_ids, seq_len):
        self.data_ids = data_ids
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data_ids) - (self.seq_len * 2)

    def __getitem__(self, idx):
        # Encoder input: A chunk of code
        # Decoder input: The next chunk, shifted for Teacher Forcing
        src = torch.tensor(self.data_ids[idx : idx + self.seq_len])
        tgt = torch.tensor([0] + self.data_ids[idx + self.seq_len : idx + 2 * self.seq_len - 1])
        labels = torch.tensor(self.data_ids[idx + self.seq_len : idx + 2 * self.seq_len])
        return src, tgt, labels

# --- 3. THE MAIN SCRIPT ---
def main():
    # Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # A. Gather Code from src/
    src_dir = Path("./src")
    all_text = ""
    for file_path in src_dir.rglob("*.py"): # Collecting Python files
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                all_text += f.read() + "\n\n"
        except Exception as e:
            print(f"Skipping {file_path}: {e}")

    print(f"Total characters loaded: {len(all_text)}")

    # B. Tokenize
    tokenizer = CharTokenizer(all_text)
    encoded_text = tokenizer.encode(all_text)

    # C. Hyperparameters
    n_latent = 32
    n_heads = 4
    n_layers = 3
    max_seq_len = 256
    batch_size = 128
    epochs = 20

    # D. Initialize Model
    model = Transformer(n_latent, n_heads, n_layers, tokenizer.n_vocab, max_seq_len).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=2) # Ignore padding

    # E. DataLoader
    dataset = CodeDataset(encoded_text, max_seq_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # F. Training Loop
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for i, (src, tgt, labels) in enumerate(loader):
            src, tgt, labels = src.to(device), tgt.to(device), labels.to(device)

            optimizer.zero_grad()
            
            # Forward pass (Teacher Forcing)
            output_logits = model(src, teacher_tockens=tgt)
            
            # Reshape for CrossEntropy
            # output_logits: [batch, seq, vocab] -> [batch * seq, vocab]
            loss = criterion(output_logits.view(-1, tokenizer.n_vocab), labels.view(-1))
            
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if i % 10 == 0:
                print(f"Epoch {epoch}, Batch {i}, Loss: {loss.item():.4f}")

    # G. Save Model
    save_path = "./model/letransformer.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'tokenizer_vocab': tokenizer.vocab,
        'hyperparams': {
            'n_latent': n_latent, 'n_heads': n_heads, 
            'n_layers': n_layers, 'n_vocab': tokenizer.n_vocab, 
            'max_seq_len': max_seq_len
        }
    }, save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    main()