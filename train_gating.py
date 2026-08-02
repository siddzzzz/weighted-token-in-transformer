import torch
import torch.nn as nn
import numpy as np
import json
from learned_gating import AutonomousWeightedTransformerDecoder
from synthetic_data import SyntheticNeedleDataset

def train_autonomous_gating_transformer(
    vocab_size: int = 1000,
    d_model: int = 128,
    num_heads: int = 4,
    num_layers: int = 2,
    steps: int = 500,
    lr: float = 1e-3,
    reg_lambda: float = 0.005,
    device: str = "cpu"
) -> AutonomousWeightedTransformerDecoder:
    """
    Trains AutonomousWeightedTransformerDecoder end-to-end.
    The model receives NO ground-truth token weight labels.
    It must autonomously learn to assign high weights to instruction tokens via task loss backpropagation!
    """
    print("=" * 70)
    print("   TRAINING AUTONOMOUS LEARNED IMPORTANCE GATING TRANSFORMER   ")
    print("=" * 70)

    model = AutonomousWeightedTransformerDecoder(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=256,
        max_seq_len=2048
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    dataset = SyntheticNeedleDataset(vocab_size=vocab_size)

    model.train()
    for step in range(steps):
        # Vary sequence length during training between 32 and 96
        curr_seq_len = 64 if step % 2 == 0 else 96
        tokens, _, targets = dataset.generate_batch(batch_size=32, seq_len=curr_seq_len, target_weight=1.0)
        tokens, targets = tokens.to(device), targets.to(device)

        logits, _, predicted_weights = model(tokens, use_autonomous_gating=True)
        
        # Task Loss (predict target token after final query)
        last_token_logits = logits[:, -1, :] # [B, vocab_size]
        task_loss = criterion(last_token_logits, targets)

        # Sparsity / Regularization Loss (penalty for deviating from 1.0)
        reg_loss = reg_lambda * torch.mean((predicted_weights - 1.0) ** 2)

        total_loss = task_loss + reg_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if (step + 1) % 100 == 0:
            inst_w = predicted_weights[:, 0:3].mean().item()
            dist_w = predicted_weights[:, 3:-1].mean().item()
            print(f"Step {step+1:3d}/{steps} | Task Loss: {task_loss.item():.4f} | Reg Loss: {reg_loss.item():.4f} | Instruction Weight w: {inst_w:.3f} | Distractor Weight w: {dist_w:.3f}")

    return model


def evaluate_autonomous_model(
    model: AutonomousWeightedTransformerDecoder,
    seq_lengths: list = [64, 128, 256, 512, 1024],
    total_trials: int = 40,
    eval_batch_size: int = 10,
    device: str = "cpu"
) -> dict:
    """
    Evaluates the trained Autonomous Transformer across sequence lengths.
    Compares Autonomous Gating vs Standard Unweighted Baseline.
    """
    model.eval()
    dataset = SyntheticNeedleDataset(vocab_size=model.vocab_size)

    modes = ["autonomous_gating", "unweighted_baseline"]
    results = {mode: {"accuracy": [], "attention_mass": [], "avg_inst_weight": [], "avg_dist_weight": []} for mode in modes}

    print("\n" + "=" * 70)
    print("   EVALUATING AUTONOMOUS GATING VS UNWEIGHTED BASELINE   ")
    print("=" * 70)

    for N in seq_lengths:
        print(f"\nEvaluating Sequence Length N = {N} ...")

        for mode in modes:
            accuracies = []
            attn_masses = []
            inst_weights = []
            dist_weights = []

            num_batches = total_trials // eval_batch_size
            for b in range(num_batches):
                tokens, _, targets = dataset.generate_batch(batch_size=eval_batch_size, seq_len=N, target_weight=1.0)
                tokens, targets = tokens.to(device), targets.to(device)

                use_gating = (mode == "autonomous_gating")
                with torch.no_grad():
                    logits, layer_attn, pred_w = model(tokens, use_autonomous_gating=use_gating)

                    predictions = logits[:, -1, :].argmax(dim=-1)
                    acc = (predictions == targets).float().mean().item() * 100.0

                    last_layer_attn = layer_attn[-1]
                    query_attn = last_layer_attn[:, :, -1, :]
                    inst_attn_mass = query_attn[:, :, 0:3].sum(dim=-1).mean().item() * 100.0

                    accuracies.append(acc)
                    attn_masses.append(inst_attn_mass)
                    inst_weights.append(pred_w[:, 0:3].mean().item())
                    dist_weights.append(pred_w[:, 3:-1].mean().item())

            avg_acc = float(np.mean(accuracies))
            avg_mass = float(np.mean(attn_masses))
            avg_inst_w = float(np.mean(inst_weights))
            avg_dist_w = float(np.mean(dist_weights))

            results[mode]["accuracy"].append(avg_acc)
            results[mode]["attention_mass"].append(avg_mass)
            results[mode]["avg_inst_weight"].append(avg_inst_w)
            results[mode]["avg_dist_weight"].append(avg_dist_w)

            print(f"  Mode [{mode:20s}] | Accuracy: {avg_acc:5.1f}% | Inst Attn Mass: {avg_mass:5.1f}% | Learned Inst w: {avg_inst_w:.3f} | Dist w: {avg_dist_w:.3f}")

    return results


def run_gating_experiment():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vocab_size = 1000

    # 1. Train model with autonomous gating
    model = train_autonomous_gating_transformer(vocab_size=vocab_size, device=device)

    # 2. Evaluate across sequence lengths
    seq_lengths = [64, 128, 256, 512, 1024]
    results = evaluate_autonomous_model(model, seq_lengths=seq_lengths, total_trials=40, eval_batch_size=10, device=device)

    # Save results and trained weights
    output_data = {
        "seq_lengths": seq_lengths,
        "results": results
    }

    with open("gating_benchmark_results.json", "w") as f:
        json.dump(output_data, f, indent=2)

    torch.save(model.state_dict(), "autonomous_transformer.pt")
    print("\nTrack 1 experiment complete! Results saved to gating_benchmark_results.json and autonomous_transformer.pt")

if __name__ == "__main__":
    run_gating_experiment()
