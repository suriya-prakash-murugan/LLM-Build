import sys
sys.path.append("./")
import torch
import wandb
from torch.utils.data import DataLoader, random_split
from model.transformer import MiniGPT
from utils.dataset import ValyrianDataset
from tokenizer.tokenizer import ValyrianBPETokenizer
from pathlib import Path

def train():
    # Config
    project_name = "high-valyrian-gpt"
    run_name = "run-" + wandb.util.generate_id()
    wandb.init(project=project_name, name=run_name)

    corpus_path = Path("data/High Valyrian.txt")
    tokenizer_path = Path("tokenizer/")
    save_path = Path("checkpoints/")
    save_path.mkdir(parents=True, exist_ok=True)

    block_size = 64
    batch_size = 32
    embed_dim = 128
    layers = 4
    heads = 4
    dropout = 0.2  
    learning_rate = 3e-4
    weight_decay = 0.01
    n_epochs = 100
    patience = 5

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load tokenizer
    tokenizer = ValyrianBPETokenizer()
    tokenizer.load(tokenizer_path)

    # Dataset and DataLoader
    dataset = ValyrianDataset(corpus_path, tokenizer, block_size=block_size)
    vocab_size = tokenizer.vocab_size
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Model
    model = MiniGPT(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        block_size=block_size,
        layers=layers,
        heads=heads,
        dropout=dropout   
    ).to(device)

    # Optimizer and Loss
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = torch.nn.CrossEntropyLoss()

    # Early stopping
    best_val_loss = float('inf')
    epochs_without_improvement = 0

    # WandB watch
    wandb.watch(model)

    # Training loop
    for epoch in range(n_epochs):
        model.train()
        train_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)

            optimizer.zero_grad()
            logits, _ = model(xb)
            loss = criterion(logits.view(-1, logits.size(-1)), yb.view(-1))
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

    # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits, _ = model(xb)
                loss = criterion(logits.view(-1, logits.size(-1)), yb.view(-1))
                val_loss += loss.item()

        val_loss /= len(val_loader)

    # Log to WandB
        wandb.log({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

    # Early Stopping Check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), "checkpoints/best_model.pt")  # Save best model
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping triggered at epoch {epoch}")
                break

    print("Training Complete!")
    wandb.finish()

if __name__ == "__main__":
    train()
