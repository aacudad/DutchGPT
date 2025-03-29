import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken  # For tokenization (e.g., OpenAI token encodings)
import torch.nn.utils.rnn as rnn_utils
from torch.utils.data import DataLoader, Dataset

def tokenize(text_input: str, encoding_type: str = "r50k_base") -> tuple[list, int]:
    """
    Converts input text into tokens using the specified encoding.
    
    Returns:
        tokens (list): List of token IDs.
        vocab_size (int): The vocabulary size for the chosen encoding.
    """
    enc = tiktoken.get_encoding(encoding_type)
    vocab_size = enc.n_vocab
    tokens = enc.encode(text_input)
    return tokens, vocab_size

class GPTModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_heads, n_layers, max_seq_length):
        """
        Initializes a GPT-style language model.
        
        Components:
          - Token Embedding: Maps token indices to dense vectors.
          - Positional Embedding: Adds positional information.
          - Transformer Encoder Layers: Each layer internally performs:
                * Multi-head self-attention (with causal masking).
                * A feedforward network.
          - Final LayerNorm and Linear Head: Normalizes and projects hidden states to vocabulary logits.
          
        Parameters:
            vocab_size (int): Vocabulary size.
            embedding_dim (int): Embedding dimension.
            n_heads (int): Number of attention heads.
            n_layers (int): Number of transformer encoder layers.
            max_seq_length (int): Maximum sequence length.
        """
        super(GPTModel, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(max_seq_length, embedding_dim)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=n_heads, batch_first=True)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(embedding_dim)
        self.head = nn.Linear(embedding_dim, vocab_size, bias=False)
        self.max_seq_length = max_seq_length

    def forward(self, x):
        """
        Forward pass:
          1. Sum token and positional embeddings.
          2. Apply a causal mask so each token only attends to previous tokens.
          3. Pass through transformer encoder layers (self-attention and feedforward).
          4. Normalize and project to logits.
        """
        batch_size, seq_length = x.size()
        positions = torch.arange(0, seq_length, device=x.device).unsqueeze(0).expand(batch_size, seq_length)
        token_emb = self.token_embedding(x)
        pos_emb = self.position_embedding(positions)
        x = token_emb + pos_emb

        # Create a causal (lower triangular) mask.
        mask = torch.tril(torch.ones(seq_length, seq_length, device=x.device)).bool()
        for layer in self.layers:
            x = layer(x, src_mask=mask)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits

    def generate(self, start_tokens, max_new_tokens=20, temperature=1.5, top_k=50, top_p=0.9):
        """
        Autoregressively generate text.
        
        Parameters:
            start_tokens (Tensor): Starting tokens (batch_size, seq_length).
            max_new_tokens (int): Number of tokens to generate.
            temperature (float): Controls randomness.
            top_k (int): Top-k sampling parameter.
            top_p (float): Nucleus sampling parameter.
        
        Returns:
            generated (Tensor): Generated token IDs.
        """
        self.eval()
        generated = start_tokens.clone()
        for _ in range(max_new_tokens):
            logits = self(generated)
            next_logits = logits[:, -1, :] / temperature

            # Top-k filtering.
            if top_k is not None and top_k > 0:
                topk_values, _ = torch.topk(next_logits, k=top_k, dim=-1)
                threshold = topk_values[:, -1].unsqueeze(-1)
                next_logits = torch.where(next_logits < threshold,
                                          torch.full_like(next_logits, float('-inf')),
                                          next_logits)

            # Nucleus (top-p) filtering.
            if top_p is not None and top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True, dim=-1)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_logits[cumulative_probs > top_p] = float('-inf')
                next_logits = torch.zeros_like(next_logits).scatter_(dim=-1, index=sorted_indices, src=sorted_logits)

            next_logits = torch.nan_to_num(next_logits, nan=-1e10, posinf=1e10, neginf=-1e10)
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)
        return generated

class TextDataset(Dataset):
    def __init__(self, sequences):
        """
        Dataset for language modeling.
        
        Each sample consists of:
            - Input: Token IDs excluding the last token.
            - Target: Token IDs excluding the first token.
        """
        self.sequences = sequences

    def __len__(self):
        return self.sequences.size(0)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        if seq.size(0) < 2:
            raise ValueError("Sequence length must be at least 2 for shifting")
        input_ids = seq[:-1]
        target_ids = seq[1:]
        return input_ids, target_ids

def prepare_dataset(texts, max_length=1024, padding_value=0):
    """
    Tokenizes and pads a list of text strings.
    
    Returns:
        padded_sequences (Tensor): Padded token sequences.
        vocab_size (int): Vocabulary size.
    """
    tokenized_texts = []
    vocab_size = None
    for text in texts:
        tokens, vs = tokenize(text)
        tokens = tokens[:max_length]
        tokenized_texts.append(torch.tensor(tokens, dtype=torch.long))
        if vocab_size is None:
            vocab_size = vs
    padded_sequences = rnn_utils.pad_sequence(tokenized_texts, batch_first=True, padding_value=padding_value)
    return padded_sequences, vocab_size
