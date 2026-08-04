import os
import sys
import torch
import torch.nn as nn
import json

# Ensure relative import of src modules
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from src.models.learned_gating import AutonomousWeightedTransformerDecoder
from src.data.synthetic import SyntheticNeedleDataset

def train_synthetic(
    vocab_size: int = 1000,
    d_model: int = 256,
    num_heads: int = 8,
    num_layers: int = 4,
    steps: int = 500,
    lr: float = 1e-3,
    reg_lambda: float = 0.005,
    device: str = None
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 70)
    print(f"   TRAINING AUTONOMOUS TRANSFORMER ON SYNTHETIC DATA ({device.upper()})   ")
    print(f"   Model Config: d_model={d_model}, num_heads={num_heads}, num_layers={num_layers}")
    print("=" * 70)

    model = AutonomousWeightedTransformerDecoder(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=d_model * 4,
        max_seq_len=2048
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    dataset = SyntheticNeedleDataset(vocab_size=vocab_size)

    model.train()
    for step in range(steps):
        curr_seq_len = 64 if step % 2 == 0 else 128
        tokens, _, targets = dataset.generate_batch(batch_size=32, seq_len=curr_seq_len, target_weight=1.0)
        tokens, targets = tokens.to(device), targets.to(device)

        logits, _, predicted_weights = model(tokens, use_autonomous_gating=True)
        
        last_token_logits = logits[:, -1, :]
        task_loss = criterion(last_token_logits, targets)
        reg_loss = reg_lambda * torch.mean((predicted_weights - 1.0) ** 2)
        total_loss = task_loss + reg_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if (step + 1) % 100 == 0:
            inst_w = predicted_weights[:, 0:3].mean().item()
            dist_w = predicted_weights[:, 3:-1].mean().item()
            print(f"Step {step+1:3d}/{steps} | Task Loss: {task_loss.item():.4f} | Instruction Weight: {inst_w:.3f} | Distractor Weight: {dist_w:.3f}")

    # Save model weights to relative path under results/
    results_dir = os.path.join(BASE_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)
    save_path = os.path.join(results_dir, "autonomous_synthetic_model.pt")
    torch.save(model.state_dict(), save_path)
    print(f"\nTrained model weights saved successfully to: {save_path}")

if __name__ == "__main__":
    train_synthetic()
