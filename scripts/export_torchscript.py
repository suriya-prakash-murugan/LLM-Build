import sys
sys.path.append("./")
import torch
from model.transformer import MiniGPT
from tokenizer.tokenizer import ValyrianBPETokenizer
from pathlib import Path

def export_model():
    # Paths
    tokenizer_path = Path("tokenizer/")
    checkpoint_path = Path("checkpoints/best_model.pt")
    export_path = Path("checkpoints/model_scripted.pt")

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
        dropout=0.0  # during inference dropout should be 0
    ).to(device)

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    # Example input
    dummy_input = torch.randint(0, vocab_size, (1, block_size), dtype=torch.long)

    # TorchScript export
    traced_model = torch.jit.trace_module(model, {"generate_logits": dummy_input})
    traced_model.save(str(export_path))

    print(f"TorchScript model saved to {export_path}")

if __name__ == "__main__":
    export_model()
