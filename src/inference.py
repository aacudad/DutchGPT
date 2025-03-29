import torch
import tiktoken
from model_init import GPTModel, tokenize

def load_model(model_config_path="model_config.pth", model_weights_path="model_weights.pth", device="cpu"):
    # Load the saved configuration and rebuild the model.
    model_config = torch.load(model_config_path, map_location=torch.device(device))
    model = GPTModel(**model_config).to(device)
    model.load_state_dict(torch.load(model_weights_path, map_location=torch.device(device)))
    model.eval()  # Set the model to evaluation mode.
    return model

def generate_text(model, prompt, max_new_tokens=30, temperature=1.5, top_k=50, top_p=0.9):
    # Tokenize the prompt.
    tokens, _ = tokenize(prompt)
    device = next(model.parameters()).device
    input_ids = torch.tensor([tokens], dtype=torch.long).to(device)
    generated_ids = model.generate(input_ids, max_new_tokens=max_new_tokens,
                                    temperature=temperature, top_k=top_k, top_p=top_p)
    enc = tiktoken.get_encoding("o200k_base")
    return enc.decode(generated_ids[0].tolist())

if __name__ == "__main__":
    device = "cpu"  # Change to "cuda" if using a GPU.
    model = load_model(device=device)
    prompt_text = "College"
    generated_text = generate_text(model, prompt_text)
    print("Generated text:\n", generated_text)
