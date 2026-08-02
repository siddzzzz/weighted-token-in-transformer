import torch
import torch.nn as nn
import numpy as np
import json
from weighted_attention import WeightedTransformerDecoder
from synthetic_data import SyntheticNeedleDataset, SyntheticPriorityDataset

def train_base_retrieval_model(
    vocab_size: int = 1000,
    d_model: int = 128,
    num_heads: int = 4,
    num_layers: int = 2,
    steps: int = 400,
    lr: float = 1e-3,
    device: str = "cpu"
) -> WeightedTransformerDecoder:
    """
    Trains a base transformer model to solve key-value retrieval on short context (seq_len=32).
    """
    print("--- Training Base Toy Transformer on Short Sequences (seq_len=32) ---")
    model = WeightedTransformerDecoder(
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
        tokens, weights, targets = dataset.generate_batch(batch_size=32, seq_len=32, target_weight=1.0)
        tokens, targets = tokens.to(device), targets.to(device)

        logits, _ = model(tokens, entrypoint="baseline")
        # Predict target after final query token
        last_token_logits = logits[:, -1, :] # [B, vocab_size]
        loss = criterion(last_token_logits, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (step + 1) % 100 == 0:
            print(f"Step {step+1}/{steps} | Loss: {loss.item():.4f}")

    return model


def evaluate_entrypoints_across_lengths(
    model: WeightedTransformerDecoder,
    seq_lengths: list = [64, 128, 256, 512, 1024],
    target_weight: float = 3.0,
    total_trials: int = 40,
    eval_batch_size: int = 10,
    device: str = "cpu"
) -> dict:
    """
    Evaluates baseline, logit_bias, v_scale, k_scale, and combo across expanding sequence lengths in batches.
    """
    model.eval()
    dataset = SyntheticNeedleDataset(vocab_size=model.vocab_size)
    entrypoints = ["baseline", "logit_bias", "v_scale", "k_scale", "combo"]

    results = {ep: {"accuracy": [], "attention_mass": []} for ep in entrypoints}

    print("\n--- Evaluating Token Weight Entrypoints Across Context Lengths ---")
    for N in seq_lengths:
        print(f"\nEvaluating Sequence Length N = {N} ...")

        for ep in entrypoints:
            try:
                accuracies = []
                attn_masses = []
                
                num_batches = total_trials // eval_batch_size
                for b in range(num_batches):
                    tokens, weights, targets = dataset.generate_batch(batch_size=eval_batch_size, seq_len=N, target_weight=target_weight)
                    tokens, weights, targets = tokens.to(device), weights.to(device), targets.to(device)

                    with torch.no_grad():
                        logits, layer_attn = model(tokens, token_weights=weights if ep != "baseline" else None, entrypoint=ep)
                        
                        predictions = logits[:, -1, :].argmax(dim=-1)
                        acc = (predictions == targets).float().mean().item() * 100.0
                        
                        last_layer_attn = layer_attn[-1] # [B, H, N, N]
                        query_attn = last_layer_attn[:, :, -1, :] # [B, H, N]
                        inst_attn_mass = query_attn[:, :, 0:3].sum(dim=-1).mean().item() * 100.0

                        accuracies.append(acc)
                        attn_masses.append(inst_attn_mass)

                avg_acc = float(np.mean(accuracies))
                avg_mass = float(np.mean(attn_masses))

                results[ep]["accuracy"].append(avg_acc)
                results[ep]["attention_mass"].append(avg_mass)

                print(f"  Entrypoint [{ep:10s}] | Accuracy: {avg_acc:5.1f}% | Instruction Attn Mass: {avg_mass:5.1f}%")
            except Exception as e:
                import traceback
                print(f"  Entrypoint [{ep:10s}] FAILED with error: {e}")
                traceback.print_exc()
                results[ep]["accuracy"].append(0.0)
                results[ep]["attention_mass"].append(0.0)

    return results


def run_benchmark():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vocab_size = 1000
    
    # 1. Train base model on short context
    model = train_base_retrieval_model(vocab_size=vocab_size, device=device)
    
    # 2. Evaluate entrypoints
    seq_lengths = [64, 128, 256, 512, 1024]
    results = evaluate_entrypoints_across_lengths(
        model,
        seq_lengths=seq_lengths,
        target_weight=3.0,
        total_trials=40,
        eval_batch_size=10,
        device=device
    )

    summary_data = {
        "seq_lengths": seq_lengths,
        "results": results
    }

    with open("benchmark_results.json", "w") as f:
        json.dump(summary_data, f, indent=2)

    print("\nBenchmark complete! Results saved to benchmark_results.json")

if __name__ == "__main__":
    run_benchmark()
