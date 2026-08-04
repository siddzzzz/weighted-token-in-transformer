import os
import sys
import argparse
import torch
import numpy as np
import json

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from src.models.weighted_attention import WeightedTransformerDecoder
from src.models.learned_gating import AutonomousWeightedTransformerDecoder
from src.data.synthetic import SyntheticNeedleDataset

def evaluate_synthetic_benchmarks(args):
    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 75)
    print(f"   EVALUATING WEIGHTED TOKEN ENTRYPOINTS ACROSS CONTEXT LENGTHS ({device.upper()})   ")
    print("=" * 75)

    model = WeightedTransformerDecoder(
        vocab_size=1000,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        d_ff=args.d_model * 4,
        max_seq_len=2048
    ).to(device)

    dataset = SyntheticNeedleDataset(vocab_size=1000)
    entrypoints = ["baseline", "logit_bias", "v_scale", "k_scale", "combo"]
    seq_lengths = [64, 128, 256, 512, 1024]

    results = {ep: {"accuracy": [], "attention_mass": []} for ep in entrypoints}

    for N in seq_lengths:
        print(f"\nEvaluating Sequence Length N = {N} ...")

        for ep in entrypoints:
            accuracies = []
            attn_masses = []
            num_batches = args.total_trials // args.batch_size

            for _ in range(num_batches):
                tokens, weights, targets = dataset.generate_batch(batch_size=args.batch_size, seq_len=N, target_weight=args.target_weight)
                tokens, weights, targets = tokens.to(device), weights.to(device), targets.to(device)

                with torch.no_grad():
                    logits, layer_attn = model(tokens, token_weights=weights if ep != "baseline" else None, entrypoint=ep)
                    predictions = logits[:, -1, :].argmax(dim=-1)
                    acc = (predictions == targets).float().mean().item() * 100.0

                    last_layer_attn = layer_attn[-1]
                    query_attn = last_layer_attn[:, :, -1, :]
                    inst_attn_mass = query_attn[:, :, 0:3].sum(dim=-1).mean().item() * 100.0

                    accuracies.append(acc)
                    attn_masses.append(inst_attn_mass)

            avg_acc = float(np.mean(accuracies))
            avg_mass = float(np.mean(attn_masses))

            results[ep]["accuracy"].append(avg_acc)
            results[ep]["attention_mass"].append(avg_mass)

            print(f"  Entrypoint [{ep:10s}] | Accuracy: {avg_acc:5.1f}% | Inst Attn Mass: {avg_mass:5.1f}%")

    results_dir = os.path.join(BASE_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)
    out_file = os.path.join(results_dir, "benchmark_results.json")

    with open(out_file, "w") as f:
        json.dump({"seq_lengths": seq_lengths, "results": results}, f, indent=2)

    print(f"\nBenchmark evaluations completed! Saved to {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Weighted Token Transformers")
    parser.add_argument("--d_model", type=int, default=256, help="Embedding dimension")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--num_layers", type=int, default=4, help="Number of layers")
    parser.add_argument("--target_weight", type=float, default=3.0, help="Weight assigned to target instruction")
    parser.add_argument("--total_trials", type=int, default=40, help="Total evaluation trials per length")
    parser.add_argument("--batch_size", type=int, default=10, help="Evaluation batch size")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda or cpu)")
    args = parser.parse_args()

    evaluate_synthetic_benchmarks(args)
