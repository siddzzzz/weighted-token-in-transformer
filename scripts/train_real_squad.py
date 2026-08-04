import os
import sys
import argparse
import torch
import torch.nn as nn
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from src.models.learned_gating import AutonomousWeightedTransformerDecoder
from src.models.weighted_attention import WeightedTransformerDecoder
from src.data.real_squad import SQuADDatasetLoader

def train_real_squad(args):
    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    
    print("=" * 75)
    print(f"   TRAINING WEIGHTED TRANSFORMER ON SQUAD V2.0 QA DATASET ({device.upper()})   ")
    print(f"   Model Config: d_model={args.d_model}, num_heads={args.num_heads}, num_layers={args.num_layers}, vocab_size=50257")
    print(f"   Weighting Mode: {args.mode} | Logit Bias Question Weight: {args.question_weight}")
    print("=" * 75)

    loader = SQuADDatasetLoader(question_weight=args.question_weight, max_seq_len=args.max_seq_len)
    raw_train_dataset = loader.load_squad_dataset(split="train")
    
    vocab_size = len(loader.tokenizer)

    if args.model_type == "autonomous":
        model = AutonomousWeightedTransformerDecoder(
            vocab_size=vocab_size,
            d_model=args.d_model,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            d_ff=args.d_model * 4,
            max_seq_len=args.max_seq_len,
            dropout=args.dropout
        ).to(device)
    else:
        model = WeightedTransformerDecoder(
            vocab_size=vocab_size,
            d_model=args.d_model,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            d_ff=args.d_model * 4,
            max_seq_len=args.max_seq_len,
            dropout=args.dropout
        ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=loader.tokenizer.pad_token_id)
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    num_samples = len(raw_train_dataset)
    steps_per_epoch = min(num_samples // args.batch_size, args.max_steps_per_epoch)

    accum_steps = getattr(args, 'gradient_accumulation_steps', 1)
    optimizer.zero_grad()
    
    model.train()
    for epoch in range(args.epochs):
        print(f"\n--- Epoch {epoch+1}/{args.epochs} ---")
        epoch_loss = 0.0

        for i in range(steps_per_epoch):
            batch_examples = raw_train_dataset[i * args.batch_size : (i + 1) * args.batch_size]
            input_ids, token_weights, target_ids = loader.prepare_batch(batch_examples, device=device)

            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
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

            epoch_loss += loss.item() * accum_steps

            if (i + 1) % args.log_interval == 0 or (i + 1) == steps_per_epoch:
                avg_l = epoch_loss / (i + 1)
                print(f"Step {i+1:4d}/{steps_per_epoch} | Batch Loss: {loss.item() * accum_steps:.4f} | Avg Loss: {avg_l:.4f}")
                if device == "cuda":
                    torch.cuda.empty_cache()

    results_dir = os.path.join(BASE_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)
    save_path = os.path.join(results_dir, f"weighted_squad_{args.model_type}_{args.mode}.pt")
    torch.save(model.state_dict(), save_path)
    print(f"\nTrained model weights saved successfully to: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Weighted Token Transformer on SQuAD v2.0")
    parser.add_argument("--model_type", type=str, default="autonomous", choices=["autonomous", "explicit"], help="Model type: autonomous or explicit")
    parser.add_argument("--mode", type=str, default="logit_bias", choices=["baseline", "logit_bias", "k_scale", "v_scale", "combo"], help="Token weight entrypoint")
    parser.add_argument("--d_model", type=int, default=256, help="Embedding dimension")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--num_layers", type=int, default=4, help="Number of transformer layers")
    parser.add_argument("--max_seq_len", type=int, default=256, help="Maximum sequence length")
    parser.add_argument("--question_weight", type=float, default=3.0, help="Weight assigned to question tokens")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Micro batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--max_steps_per_epoch", type=int, default=500, help="Max steps per epoch")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout probability")
    parser.add_argument("--log_interval", type=int, default=50, help="Logging step interval")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda or cpu)")
    
    args = parser.parse_args()
    train_real_squad(args)
