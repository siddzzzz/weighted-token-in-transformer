import os
import sys
import json

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")

def visualize():
    json_path = os.path.join(RESULTS_DIR, "benchmark_results.json")
    if not os.path.exists(json_path):
        print(f"Results file not found at {json_path}. Please run scripts/evaluate.py first.")
        return

    with open(json_path, "r") as f:
        data = json.load(f)

    seq_lengths = data["seq_lengths"]
    results = data["results"]

    print("=" * 70)
    print("      WEIGHTED TOKEN TRANSFORMER BENCHMARK SUMMARY REPORT      ")
    print("=" * 70)

    header = f"{'Entrypoint':<12} | " + " | ".join([f"N={n:<4}" for n in seq_lengths])
    print("\n--- TASK ACCURACY (%) ---")
    print(header)
    print("-" * len(header))
    for ep, metrics in results.items():
        acc_str = " | ".join([f"{acc:5.1f}%" for acc in metrics["accuracy"]])
        print(f"{ep:<12} | {acc_str}")

    print("\n--- INSTRUCTION ATTENTION MASS (%) ---")
    print(header)
    print("-" * len(header))
    for ep, metrics in results.items():
        mass_str = " | ".join([f"{mass:5.1f}%" for mass in metrics["attention_mass"]])
        print(f"{ep:<12} | {mass_str}")
    print("=" * 70)

    try:
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        styles = {
            "baseline": ("gray", "--", "s"),
            "logit_bias": ("blue", "-", "o"),
            "v_scale": ("green", "-.", "^"),
            "k_scale": ("purple", ":", "d"),
            "combo": ("red", "-", "p")
        }

        for ep, metrics in results.items():
            color, linestyle, marker = styles.get(ep, ("black", "-", "o"))
            ax1.plot(seq_lengths, metrics["accuracy"], label=ep, color=color, linestyle=linestyle, marker=marker, linewidth=2)
            ax2.plot(seq_lengths, metrics["attention_mass"], label=ep, color=color, linestyle=linestyle, marker=marker, linewidth=2)

        ax1.set_title("Accuracy vs Sequence Length", fontsize=12, fontweight="bold")
        ax1.set_xlabel("Sequence Length (Tokens)")
        ax1.set_ylabel("Accuracy (%)")
        ax1.set_ylim(0, 105)
        ax1.grid(True, linestyle=":", alpha=0.6)
        ax1.legend()

        ax2.set_title("Instruction Attention Mass vs Sequence Length", fontsize=12, fontweight="bold")
        ax2.set_xlabel("Sequence Length (Tokens)")
        ax2.set_ylabel("Attention Mass (%)")
        ax2.grid(True, linestyle=":", alpha=0.6)
        ax2.legend()

        plt.tight_layout()
        plot_path = os.path.join(RESULTS_DIR, "weighted_token_benchmark_plot.png")
        plt.savefig(plot_path, dpi=300)
        print(f"\nPlot saved successfully to: {plot_path}")
    except ImportError:
        print("\nNote: matplotlib not installed. Skipping plot generation.")

if __name__ == "__main__":
    visualize()
