# Weighted Token Transformer

A PyTorch research framework for incorporating dynamic **token importance weights** ($w_j > 0$) directly into Transformer Self-Attention matrices ($\text{Scores} + \log(w_j)$) to prevent instruction drift, attention dilution, and context decay in long sequence windows.

---

## 📁 Repository Structure

```text
weighted-token-in-transformer/
├── src/
│   ├── models/
│   │   ├── weighted_attention.py   # WeightedMultiHeadAttention & TransformerBlocks
│   │   └── learned_gating.py       # ImportanceGatingHead & AutonomousWeightedTransformer
│   └── data/
│       ├── synthetic.py            # Synthetic needle & priority datasets
│       ├── real_squad.py           # SQuAD v2.0 loader & GPT-2 BPE tokenizer wrapper
│       └── real_wikitext.py        # WikiText-103 loader & BPE tokenizer wrapper
├── scripts/
│   ├── train_synthetic.py          # Train & evaluate on synthetic tasks
│   ├── train_real_squad.py         # CLI CUDA trainer for SQuAD v2.0 Question Answering
│   ├── train_chinchilla.py         # Chinchilla Compute-Optimal Scaling CLI Trainer (D = 20 * N)
│   ├── evaluate.py                 # Evaluation benchmark runner (Synthetic)
│   ├── evaluate_squad.py           # SQuAD v2.0 QA model evaluation runner
│   ├── evaluate_chinchilla.py      # Chinchilla model evaluation runner
│   └── visualize.py                # Plotting tool & summary reporter
├── results/                        # Output directory for plots, metrics, & checkpoints
├── requirements.txt                # Dependencies
└── README.md                       # Project documentation
```

---

## 📐 Chinchilla Compute-Optimal Model Scaling ($D = 20 \times N$)

Following Hoffmann et al. (2022, DeepMind), parameters $N$ and training tokens $D$ scale in a **1:1 equal ratio** ($D \approx 20 \times N$):

| Tier | Model Parameters ($N$) | Chinchilla Optimal Tokens ($D$) | Architecture Config |
| :--- | :--- | :--- | :--- |
| **Micro** | **~5.2 Million** | **104 Million tokens** (~1 epoch WikiText-103) | `d_model=384`, `layers=6`, `heads=6`, `d_ff=1536` |
| **Small** | **~15.6 Million** | **312 Million tokens** (~3 epochs WikiText-103) | `d_model=512`, `layers=8`, `heads=8`, `d_ff=2048` |
| **Medium**| **~42.0 Million** | **840 Million tokens** (~1.8 epochs TinyStories) | `d_model=768`, `layers=12`, `heads=12`, `d_ff=3072` |

---

## ⚡ Quick Start: Running in your Miniconda CUDA Environment

### 1. Train Chinchilla-Micro (5.2M Parameters) on WikiText-103
```bash
python scripts/train_chinchilla.py --tier micro --device cuda
```

### 2. Evaluate Trained Chinchilla-Micro Model
```bash
python scripts/evaluate_chinchilla.py --tier micro --device cuda
```

### 3. Train on Real SQuAD v2.0 QA Dataset (CUDA Accelerated)
```bash
python scripts/train_real_squad.py --model_type autonomous --d_model 256 --num_layers 4 --num_heads 8 --epochs 3 --batch_size 2 --gradient_accumulation_steps 8 --max_seq_len 256 --device cuda
```

### 4. Evaluate SQuAD v2.0 QA Model
```bash
python scripts/evaluate_squad.py --device cuda
```

### 5. Evaluate Synthetic Needle Benchmarks
```bash
python scripts/evaluate.py --device cuda
```

### 6. Generate Summary Reports & Plots
```bash
python scripts/visualize.py
```
All outputs, trained `.pt` checkpoints, and plots will be saved to `./results/`.