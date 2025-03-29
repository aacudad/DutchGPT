import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from model_init import GPTModel, TextDataset, prepare_dataset

# Setup device.
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Hyperparameters for continued training.
learning_rate = 1e-4
num_epochs = 60  # Number of epochs for continued training.

# Load saved model configuration and weights.
model_config = torch.load("model_config.pth", map_location=device)
model = GPTModel(**model_config).to(device)
model.load_state_dict(torch.load("model_weights.pth", map_location=device))
model.train()  # Set to training mode.

# Prepare new training data.
new_texts = ["New training data goes here"]  # Replace with your new data.
padded_sequences, _ = prepare_dataset(new_texts, max_length=model_config["max_seq_length"])
data_loader = DataLoader(TextDataset(padded_sequences), batch_size=1, shuffle=True)

criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Continued training loop.
for epoch in range(num_epochs):
    total_loss = 0.0
    model.train()
    for input_ids, target_ids in data_loader:
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)
        optimizer.zero_grad()
        logits = model(input_ids)
        logits = logits.view(-1, model_config["vocab_size"])
        target_ids = target_ids.view(-1)
        loss = criterion(logits, target_ids)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(data_loader)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

# Optionally save the updated weights.
torch.save(model.state_dict(), "model_weights_continued.pth")
