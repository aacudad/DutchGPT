import os
import glob
import random

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader # type: ignore
from model_init import GPTModel, TextDataset, prepare_dataset

def load_text_files(directory: str, shuffle_sentences: bool = False) -> list:
    """
    Loads all .txt files from the given directory and returns a list of text strings.
    If shuffle_sentences is True, splits each file into sentences (using newlines),
    shuffles them, and then joins them back together.
    
    Parameters:
        directory (str): Path to the directory containing text files.
        shuffle_sentences (bool): Whether to shuffle the sentences within each file.
        
    Returns:
        texts (list): A list of text strings.
    """
    file_paths = glob.glob(os.path.join(directory, "*.txt"))
    texts = []
    for file in file_paths:
        with open(file, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if shuffle_sentences:
                # Split by newlines, filter empty lines, shuffle, and join back
                sentences = content.splitlines()
                sentences = [s.strip() for s in sentences if s.strip()]
                random.shuffle(sentences)
                content = " ".join(sentences)
            texts.append(content)
    return texts

# ---------------------------
# Setup device.
# ---------------------------
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# ---------------------------
# Hyperparameters.
# ---------------------------
embedding_dim = 512
n_heads = 8
n_layers = 6
max_length = 1024
batch_size = 1
num_epochs = 100
learning_rate = 1e-4

# ---------------------------
# Prepare dataset.
# ---------------------------
# Load texts from files in the "data" directory.
# Set shuffle_sentences=True to randomly shuffle the sentences within each file.
data_directory = "data"  # Update this path to where your .txt files reside.
texts = load_text_files(data_directory, shuffle_sentences=True)

# If you prefer to use hard-coded data instead, uncomment the next line:
# texts = ["Here comes the text data", "Another sample of text", "More text data..."]

# Tokenize and pad the texts; also get the vocabulary size.
padded_sequences, vocab_size = prepare_dataset(texts, max_length)
data_loader = DataLoader(TextDataset(padded_sequences), batch_size=batch_size, shuffle=True)

# ---------------------------
# Instantiate the model.
# ---------------------------
max_seq_length = padded_sequences.size(1)
model = GPTModel(vocab_size=vocab_size,
                 embedding_dim=embedding_dim,
                 n_heads=n_heads,
                 n_layers=n_layers,
                 max_seq_length=max_seq_length).to(device)

criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# ---------------------------
# Training loop.
# ---------------------------
for epoch in range(num_epochs):
    total_loss = 0.0
    model.train()
    for input_ids, target_ids in data_loader:
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)
        optimizer.zero_grad()
        logits = model(input_ids)
        logits = logits.view(-1, vocab_size)
        target_ids = target_ids.view(-1)
        loss = criterion(logits, target_ids)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(data_loader)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

# ---------------------------
# Save model weights and configuration.
# ---------------------------
torch.save(model.state_dict(), "model_weights.pth")
model_config = {
    "vocab_size": vocab_size,
    "embedding_dim": embedding_dim,
    "n_heads": n_heads,
    "n_layers": n_layers,
    "max_seq_length": max_seq_length
}
torch.save(model_config, "model_config.pth")
