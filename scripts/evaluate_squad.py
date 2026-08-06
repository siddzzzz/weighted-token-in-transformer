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
from src.data.real_squad import SQuADDatasetLoader

def evaluate_squad(args):
    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 75)
    print(f"   EVALUATING SQUAD V2.0 QUESTION-ANSWERING MODEL ({device.upper()})   ")
    print("=" * 75)

    loader = SQuADDatasetLoader(question_weight=args.question_weight, max_seq_len=args.max_seq_len)
    val_dataset = loader.load_squad_dataset(split="validation")
    vocab_size = len(loader.tokenizer)

    if args.model_type == "autonomous":
        model = AutonomousWeightedTransformerDecoder(
            vocab_size=vocab_size,
            d_model=args.d_model,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            d_ff=args.d_model * 4,
            max_seq_len=args.max_seq_len
        ).to(device)
    else:
        model = WeightedTransformerDecoder(
            vocab_size=vocab_size,
            d_model=args.d_model,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            d_ff=args.d_model * 4,
            max_seq_len=args.max_seq_len
        ).to(device)

    # Load trained checkpoint
    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        checkpoint_path = os.path.join(BASE_DIR, "results", f"weighted_squad_{args.model_type}_{args.mode}.pt")

    if os.path.exists(checkpoint_path):
        print(f"--> Loading trained SQuAD model checkpoint from: {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)
    else:
        print(f"--> [Warning] Checkpoint not found at {checkpoint_path}! Evaluating initialized model.")

    model.eval()
    criterion = nn.CrossEntropyLoss(ignore_index=loader.tokenizer.pad_token_id)

    num_eval_samples = min(len(val_dataset), args.max_eval_samples)
    batch_size = args.batch_size
    num_batches = num_eval_samples // batch_size

    total_loss = 0.0
    question_attn_masses = []
    learned_q_weights = []

    print(f"\nEvaluating on {num_eval_samples} SQuAD v2.0 validation samples ...")

    with torch.no_grad():
        for b in range(num_batches):
            batch_examples = val_dataset[b * batch_size : (b + 1) * batch_size]
            input_ids, token_weights, target_ids = loader.prepare_batch(batch_examples, device=device)

            if args.model_type == "autonomous":
                logits, layer_attn, pred_weights = model(input_ids, use_autonomous_gating=True, output_attentions=True)
                learned_q_weights.append(pred_weights[:, 0:20].mean().item())
            else:
                logits, layer_attn = model(input_ids, token_weights=token_weights, entrypoint=args.mode, output_attentions=True)

            loss = criterion(logits.view(-1, vocab_size), target_ids.view(-1))
            total_loss += loss.item()

            if len(layer_attn) > 0:
                last_layer_attn = layer_attn[-1] # [B, H, N, N]
                query_attn = last_layer_attn[:, :, -1, :] # [B, H, N]
                # Measure attention mass allocated to first 25 tokens (System Prompt + Question)
                q_mass = query_attn[:, :, 0:25].sum(dim=-1).mean().item() * 100.0
                question_attn_masses.append(q_mass)

    avg_val_loss = total_loss / num_batches
    avg_q_mass = float(np.mean(question_attn_masses)) if len(question_attn_masses) > 0 else 0.0
    avg_q_weight = float(np.mean(learned_q_weights)) if len(learned_q_weights) > 0 else 1.0

    print("\n" + "=" * 75)
    print("      SQUAD V2.0 VALIDATION EVALUATION RESULTS      ")
    print("=" * 75)
    print(f"  Model Type               : {args.model_type}")
    print(f"  Weighting Mode           : {args.mode}")
    print(f"  Validation Loss (Perp)   : {avg_val_loss:.4f} (Perplexity: {np.exp(avg_val_loss):.2f})")
    print(f"  Question Attention Mass  : {avg_q_mass:.2f}%")
    if args.model_type == "autonomous":
        print(f"  Learned Question Weight  : {avg_q_weight:.3f}")
    print("=" * 75)

    results_dir = os.path.join(BASE_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)
    out_file = os.path.join(results_dir, "squad_eval_results.json")
    
    eval_summary = {
        "model_type": args.model_type,
        "mode": args.mode,
        "validation_loss": avg_val_loss,
        "perplexity": float(np.exp(avg_val_loss)),
        "question_attention_mass": avg_q_mass,
        "learned_question_weight": avg_q_weight
    }

    with open(out_file, "w") as f:
        json.dump(eval_summary, f, indent=2)

    print(f"\nSQuAD evaluation complete! Summary saved to {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate SQuAD v2.0 Model")
    parser.add_argument("--model_type", type=str, default="autonomous", choices=["autonomous", "explicit"])
    parser.add_argument("--mode", type=str, default="logit_bias")
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--max_seq_len", type=int, default=256)
    parser.add_argument("--question_weight", type=float, default=3.0)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_eval_samples", type=int, default=500)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)

    args = parser.parse_args()
    evaluate_squad(args)
