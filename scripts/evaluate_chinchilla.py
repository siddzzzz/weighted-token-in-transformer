import os
import sys
import argparse
import torch
import torch.nn as nn
import numpy as np
import json

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from src.models.learned_gating import AutonomousWeightedTransformerDecoder
from src.models.weighted_attention import WeightedTransformerDecoder
from src.data.real_wikitext import WikiTextDatasetLoader

CHINCHILLA_TIERS = {
    "micro": {"d_model": 384, "num_layers": 6, "num_heads": 6, "d_ff": 1536},
    "small": {"d_model": 512, "num_layers": 8, "num_heads": 8, "d_ff": 2048},
    "medium": {"d_model": 768, "num_layers": 12, "num_heads": 12, "d_ff": 3072}
}

def evaluate_chinchilla(args):
    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    
    if args.tier in CHINCHILLA_TIERS:
        t_cfg = CHINCHILLA_TIERS[args.tier]
        d_model, num_layers, num_heads, d_ff = t_cfg["d_model"], t_cfg["num_layers"], t_cfg["num_heads"], t_cfg["d_ff"]
    else:
        d_model, num_layers, num_heads, d_ff = args.d_model, args.num_layers, args.num_heads, args.d_model * 4

    print("=" * 75)
    print(f"   EVALUATING CHINCHILLA MODEL ({args.tier.upper()} TIER) ({device.upper()})   ")
    print(f"   Config: d_model={d_model}, num_heads={num_heads}, num_layers={num_layers}")
    print("=" * 75)

    loader = WikiTextDatasetLoader(heading_weight=args.heading_weight, max_seq_len=args.max_seq_len)
    val_dataset = loader.load_wikitext_dataset(split="validation")
    vocab_size = len(loader.tokenizer)

    if args.model_type == "autonomous":
        model = AutonomousWeightedTransformerDecoder(
            vocab_size=vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            d_ff=d_ff,
            max_seq_len=args.max_seq_len
        ).to(device)
    else:
        model = WeightedTransformerDecoder(
            vocab_size=vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            d_ff=d_ff,
            max_seq_len=args.max_seq_len
        ).to(device)

    # Load trained checkpoint
    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        checkpoint_path = os.path.join(BASE_DIR, "results", f"chinchilla_{args.tier}_{args.model_type}.pt")

    if os.path.exists(checkpoint_path):
        print(f"--> Loading trained Chinchilla checkpoint from: {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)
    else:
        print(f"--> [Warning] Checkpoint not found at {checkpoint_path}! Evaluating initialized model.")

    model.eval()
    criterion = nn.CrossEntropyLoss(ignore_index=loader.tokenizer.pad_token_id)

    num_eval_samples = min(len(val_dataset), args.max_eval_samples)
    batch_size = args.batch_size
    num_batches = num_eval_samples // (batch_size * 2)

    total_loss = 0.0
    heading_attn_masses = []
    learned_weights = []

    print(f"\nEvaluating on {num_eval_samples} WikiText-103 validation text lines ...")

    with torch.no_grad():
        for b in range(num_batches):
            batch_slice = val_dataset[b * batch_size : (b + 1) * batch_size]
            text_lines = [ex["text"] for ex in batch_slice if isinstance(ex, dict) and "text" in ex]
            if not text_lines:
                continue

            input_ids, token_weights, target_ids = loader.prepare_batch(text_lines, device=device)

            if args.model_type == "autonomous":
                logits, layer_attn, pred_weights = model(input_ids, use_autonomous_gating=True, output_attentions=True)
                learned_weights.append(pred_weights[:, 0:10].mean().item())
            else:
                logits, layer_attn = model(input_ids, token_weights=token_weights, entrypoint=args.mode, output_attentions=True)

            loss = criterion(logits.view(-1, vocab_size), target_ids.view(-1))
            total_loss += loss.item()

            if len(layer_attn) > 0:
                last_layer_attn = layer_attn[-1]
                query_attn = last_layer_attn[:, :, -1, :]
                head_mass = query_attn[:, :, 0:10].sum(dim=-1).mean().item() * 100.0
                heading_attn_masses.append(head_mass)

    avg_val_loss = total_loss / max(num_batches, 1)
    perplexity = float(np.exp(avg_val_loss))
    avg_head_mass = float(np.mean(heading_attn_masses)) if len(heading_attn_masses) > 0 else 0.0
    avg_weight = float(np.mean(learned_weights)) if len(learned_weights) > 0 else 1.0

    print("\n" + "=" * 75)
    print(f"      CHINCHILLA {args.tier.upper()} TIER EVALUATION RESULTS      ")
    print("=" * 75)
    print(f"  Model Tier               : {args.tier.upper()}")
    print(f"  Model Type               : {args.model_type}")
    print(f"  Validation Loss          : {avg_val_loss:.4f}")
    print(f"  Validation Perplexity    : {perplexity:.2f}")
    print(f"  Heading Attention Mass   : {avg_head_mass:.2f}%")
    if args.model_type == "autonomous":
        print(f"  Learned Heading Weight   : {avg_weight:.3f}")
    print("=" * 75)

    results_dir = os.path.join(BASE_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)
    out_file = os.path.join(results_dir, f"chinchilla_{args.tier}_eval_results.json")
    
    eval_summary = {
        "tier": args.tier,
        "model_type": args.model_type,
        "validation_loss": avg_val_loss,
        "perplexity": perplexity,
        "heading_attention_mass": avg_head_mass,
        "learned_heading_weight": avg_weight
    }

    with open(out_file, "w") as f:
        json.dump(eval_summary, f, indent=2)

    print(f"\nEvaluation complete! Summary saved to {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Chinchilla Weighted Transformer Model")
    parser.add_argument("--tier", type=str, default="micro", choices=["micro", "small", "medium", "custom"])
    parser.add_argument("--model_type", type=str, default="autonomous", choices=["autonomous", "explicit"])
    parser.add_argument("--mode", type=str, default="logit_bias")
    parser.add_argument("--d_model", type=int, default=384)
    parser.add_argument("--num_heads", type=int, default=6)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--max_seq_len", type=int, default=256)
    parser.add_argument("--heading_weight", type=float, default=3.0)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_eval_samples", type=int, default=500)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)

    args = parser.parse_args()
    evaluate_chinchilla(args)
