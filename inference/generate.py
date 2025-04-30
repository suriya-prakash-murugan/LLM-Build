import sys
sys.path.append("./")
import argparse
import torch
from model.transformer import MiniGPT
from utils.dataset import ValyrianDataset
from tokenizer.tokenizer import ValyrianBPETokenizer
from pathlib import Path

@torch.no_grad()
def generate(model, tokenizer, prompt, block_size, max_new_tokens=20, temperature=1.0, top_p=0.9, device='cpu'):
    model.eval()

    # Encode prompt
    input_ids = tokenizer.encode(prompt)
    print(input_ids)
    input_ids = torch.tensor([input_ids], dtype=torch.long).to(device)

    for _ in range(max_new_tokens):
        if input_ids.size(1) > model.block_size:
            input_ids = input_ids[:, -model.block_size:]

        logits = model(input_ids)[0]
        logits = logits[:, -1, :] / temperature
        probs = torch.softmax(logits, dim=-1)

        # Top-p filtering
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        # Remove tokens with cumulative prob > top_p
        mask = cumulative_probs > top_p
        if mask[0, 0]:  # ensure at least one token is available
            mask[0, 0] = False

        sorted_probs[mask] = 0.0
        sorted_probs = sorted_probs / sorted_probs.sum()  # re-normalize

        next_token = sorted_indices[0, torch.multinomial(sorted_probs, 1)]
        input_ids = torch.cat((input_ids, next_token.view(1,1)), dim=1)

    output_tokens = input_ids[0].tolist()
    return tokenizer.decode(output_tokens)


def main():
    # Paths
    tokenizer_path = Path("tokenizer/")
    checkpoint_path = Path("checkpoints/best_model.pt")  

    block_size = 64
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load tokenizer
    tokenizer = ValyrianBPETokenizer()
    tokenizer.load(tokenizer_path)


    # Load model
    vocab_size = tokenizer.vocab_size

    model = MiniGPT(
        vocab_size=vocab_size,
        embed_dim=128,
        block_size=block_size,
        layers=4,
        heads=4,
        dropout=0.0
    ).to(device)

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    # Set up argument parser
    parser = argparse.ArgumentParser(description="Generate text from a prompt")
    parser.add_argument('--prompt', required=True, help="Input prompt for text generation")
    
    # Parse arguments
    args = parser.parse_args()

    # Generate
    prompt = args.prompt
    generated_text = generate(model, tokenizer, prompt, block_size, temperature=1.0, top_p=0.9, device=device)

    print("=== Generated High Valyrian Text ===")
    print(generated_text)

if __name__ == "__main__":
    main()
