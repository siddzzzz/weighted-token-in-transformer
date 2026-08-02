import json
import os

def load_results():
    if not os.path.exists("benchmark_results.json"):
        print("benchmark_results.json not found! Run evaluate.py first.")
        return None
    with open("benchmark_results.json", "r") as f:
        return json.load(f)

def generate_text_summary(data):
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
        attn_str = " | ".join([f"{mass:5.1f}%" for mass in metrics["attention_mass"]])
        print(f"{ep:<12} | {attn_str}")
        
    print("=" * 70)

def generate_matplotlib_plots(data):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nNote: matplotlib not installed. Skipping PNG image generation.")
        return

    seq_lengths = data["seq_lengths"]
    results = data["results"]

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

    ax1.set_title("Task Accuracy vs Context Length (N)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Sequence Length (Tokens)", fontsize=10)
    ax1.set_ylabel("Target Retrieval Accuracy (%)", fontsize=10)
    ax1.set_ylim(0, 105)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend()

    ax2.set_title("Instruction Attention Mass vs Context Length (N)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Sequence Length (Tokens)", fontsize=10)
    ax2.set_ylabel("Attention Score Mass on Instruction (%)", fontsize=10)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend()

    plt.tight_layout()
    plot_path = "weighted_token_benchmark_plot.png"
    plt.savefig(plot_path, dpi=300)
    print(f"\nBenchmark chart saved successfully to {plot_path}")

if __name__ == "__main__":
    data = load_results()
    if data:
        generate_text_summary(data)
        generate_matplotlib_plots(data)
