import os
import sys
import argparse
import torch
import torch.nn as nn
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from src.models.learned_gating import AutonomousWeightedTransformerDecoder
from src.models.weighted_attention import WeightedTransformerDecoder
from src.data.real_wikitext import WikiTextDatasetLoader

# Chinchilla Optimal Presets (D = 20 * N)
CHINCHILLA_TIERS = {
    "micro": {
        "params": "5.2M",
        "d_model": 384,
        "num_layers": 6,
        "num_heads": 6,
        "d_ff": 1536,
        "target_tokens": "104M"
    },
    "small": {
        "params": "15.6M",
        "d_model": 512,
        "num_layers": 8,
        "num_heads": 8,
        "d_ff": 2048,
        "target_tokens": "312M"
    },
    "medium": {
        "params": "42.0M",
        "d_model": 768,
        "num_layers": 12,
        "num_heads": 12,
        "d_ff": 3072,
        "target_tokens": "840M"
    }
}

def train_chinchilla(args):
    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    
    # Configure tier settings
    if args.tier in CHINCHILLA_TIERS:
        tier_cfg = CHINCHILLA_TIERS[args.tier]
        d_model = tier_cfg["d_model"]
        num_layers = tier_cfg["num_layers"]
        num_heads = tier_cfg["num_heads"]
        d_ff = tier_cfg["d_ff"]
        param_label = tier_cfg["params"]
        token_label = tier_cfg["target_tokens"]
    else:
        d_model, num_layers, num_heads, d_ff = args.d_model, args.num_layers, args.num_heads, args.d_model * 4
        param_label, token_label = "Custom", "Custom"

    print("=" * 75)
    print(f"   CHINCHILLA COMPUTE-OPTIMAL TRAINING ({args.tier.upper()} TIER) ({device.upper()})   ")
    print(f"   Model Size: ~{param_label} Parameters | Chinchilla Target Tokens: {token_label}")
    print(f"   Config: d_model={d_model}, num_heads={num_heads}, num_layers={num_layers}, d_ff={d_ff}")
    print("=" * 75)

    loader = WikiTextDatasetLoader(heading_weight=args.heading_weight, max_seq_len=args.max_seq_len)
    raw_train_dataset = loader.load_wikitext_dataset(split="train")
    vocab_size = len(loader.tokenizer)

    if args.model_type == "autonomous":
        model = AutonomousWeightedTransformerDecoder(
            vocab_size=vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            d_ff=d_ff,
            max_seq_len=args.max_seq_len,
            dropout=args.dropout
        ).to(device)
    else:
        model = WeightedTransformerDecoder(
            vocab_size=vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            d_ff=d_ff,
            max_seq_len=args.max_seq_len,
            dropout=args.dropout
        ).to(device)

    # Print total trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"--> Total Model Parameters: {total_params:,} ({total_params/1e6:.2f} Million)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=loader.tokenizer.pad_token_id)
    
    # AMP Mixed Precision Scaler
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    num_samples = len(raw_train_dataset)
    steps_per_epoch = min(num_samples // (args.batch_size * 4), args.max_steps_per_epoch)

    # Cosine Annealing LR Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs * steps_per_epoch, eta_min=1e-5)

    accum_steps = args.gradient_accumulation_steps
    optimizer.zero_grad()

    model.train()
    for epoch in range(args.epochs):
        print(f"\n--- Epoch {epoch+1}/{args.epochs} ---")
        epoch_loss = 0.0

        for i in range(steps_per_epoch):
            batch_slice = raw_train_dataset[i * args.batch_size : (i + 1) * args.batch_size]
            text_lines = [ex["text"] for ex in batch_slice if isinstance(ex, dict) and "text" in ex]

            if not text_lines:
                continue

            input_ids, token_weights, target_ids = loader.prepare_batch(text_lines, device=device)

            autocast_cm = torch.amp.autocast("cuda", enabled=(device == "cuda")) if hasattr(torch, "amp") else torch.cuda.amp.autocast(enabled=(device == "cuda"))
            with autocast_cm:
                if args.model_type == "autonomous":
                    logits, _, pred_weights = model(input_ids, use_autonomous_gating=True, output_attentions=False)
                else:
                    logits, _ = model(input_ids, token_weights=token_weights, entrypoint=args.mode, output_attentions=False)

                loss = criterion(logits.view(-1, vocab_size), target_ids.view(-1))
                loss = loss / accum_steps

            scaler.scale(loss).backward()

            if (i + 1) % accum_steps == 0 or (i + 1) == steps_per_epoch:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            epoch_loss += loss.item() * accum_steps

            if (i + 1) % args.log_interval == 0 or (i + 1) == steps_per_epoch:
                avg_l = epoch_loss / (i + 1)
                curr_lr = scheduler.get_last_lr()[0]
                print(f"Step {i+1:4d}/{steps_per_epoch} | Loss: {loss.item() * accum_steps:.4f} | Avg Loss: {avg_l:.4f} | Perplexity: {np.exp(avg_l):.2f} | LR: {curr_lr:.6f}")
                if device == "cuda":
                    torch.cuda.empty_cache()

    results_dir = os.path.join(BASE_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)
    save_path = os.path.join(results_dir, f"chinchilla_{args.tier}_{args.model_type}.pt")
    torch.save(model.state_dict(), save_path)
    print(f"\nChinchilla model weights saved successfully to: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Chinchilla Compute-Optimal Weighted Transformer")
    parser.add_argument("--tier", type=str, default="micro", choices=["micro", "small", "medium", "custom"], help="Chinchilla scaling tier")
    parser.add_argument("--model_type", type=str, default="autonomous", choices=["autonomous", "explicit"], help="Model type")
    parser.add_argument("--mode", type=str, default="logit_bias", choices=["baseline", "logit_bias", "k_scale", "v_scale", "combo"])
    parser.add_argument("--d_model", type=int, default=384, help="Embedding dimension (for custom tier)")
    parser.add_argument("--num_heads", type=int, default=6, help="Attention heads (for custom tier)")
    parser.add_argument("--num_layers", type=int, default=6, help="Layers (for custom tier)")
    parser.add_argument("--max_seq_len", type=int, default=256, help="Maximum sequence length")
    parser.add_argument("--heading_weight", type=float, default=3.0, help="Weight assigned to section headings")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Micro batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--max_steps_per_epoch", type=int, default=500, help="Max steps per epoch")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout probability")
    parser.add_argument("--log_interval", type=int, default=50, help="Logging step interval")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda or cpu)")

    args = parser.parse_args()
    train_chinchilla(args)
