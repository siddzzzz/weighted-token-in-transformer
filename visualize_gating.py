import json
import os

def load_gating_results():
    if not os.path.exists("gating_benchmark_results.json"):
        print("gating_benchmark_results.json not found! Run train_gating.py first.")
        return None
    with open("gating_benchmark_results.json", "r") as f:
        return json.load(f)

def generate_gating_summary(data):
    seq_lengths = data["seq_lengths"]
    results = data["results"]

    print("=" * 75)
    print("      AUTONOMOUS LEARNED GATING TRANSFORMER BENCHMARK REPORT      ")
    print("=" * 75)
    
    header = f"{'Evaluation Mode':<22} | " + " | ".join([f"N={n:<4}" for n in seq_lengths])
    
    print("\n--- TASK ACCURACY (%) ---")
    print(header)
    print("-" * len(header))
    for mode, metrics in results.items():
        acc_str = " | ".join([f"{acc:5.1f}%" for acc in metrics["accuracy"]])
        print(f"{mode:<22} | {acc_str}")

    print("\n--- INSTRUCTION ATTENTION MASS (%) ---")
    print(header)
    print("-" * len(header))
    for mode, metrics in results.items():
        mass_str = " | ".join([f"{mass:5.1f}%" for mass in metrics["attention_mass"]])
        print(f"{mode:<22} | {mass_str}")

    print("\n--- LEARNED INSTRUCTION WEIGHT (w_inst) ---")
    print(header)
    print("-" * len(header))
    for mode, metrics in results.items():
        w_str = " | ".join([f"{w:5.2f} " for w in metrics["avg_inst_weight"]])
        print(f"{mode:<22} | {w_str}")
        
    print("=" * 75)

def generate_gating_plots(data):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nNote: matplotlib not installed. Skipping PNG plot generation.")
        return

    seq_lengths = data["seq_lengths"]
    results = data["results"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    styles = {
        "autonomous_gating": ("blue", "-", "o"),
        "unweighted_baseline": ("gray", "--", "s")
    }

    for mode, metrics in results.items():
        color, linestyle, marker = styles.get(mode, ("black", "-", "o"))
        ax1.plot(seq_lengths, metrics["accuracy"], label=mode, color=color, linestyle=linestyle, marker=marker, linewidth=2.5)
        ax2.plot(seq_lengths, metrics["attention_mass"], label=mode, color=color, linestyle=linestyle, marker=marker, linewidth=2.5)

    ax1.set_title("Accuracy: Autonomous Gating vs Baseline", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Sequence Length (Tokens)", fontsize=10)
    ax1.set_ylabel("Retrieval Accuracy (%)", fontsize=10)
    ax1.set_ylim(0, 105)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend()

    ax2.set_title("Instruction Attention Mass: Gating vs Baseline", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Sequence Length (Tokens)", fontsize=10)
    ax2.set_ylabel("Attention Mass on Instruction (%)", fontsize=10)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend()

    plt.tight_layout()
    plot_path = "learned_gating_benchmark_plot.png"
    plt.savefig(plot_path, dpi=300)
    print(f"\nPlot saved successfully to {plot_path}")

if __name__ == "__main__":
    data = load_gating_results()
    if data:
        generate_gating_summary(data)
        generate_gating_plots(data)
