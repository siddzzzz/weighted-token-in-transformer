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
│       └── real_squad.py           # SQuAD v2.0 loader & GPT-2 BPE tokenizer wrapper
├── scripts/
│   ├── train_synthetic.py          # Train & evaluate on synthetic tasks
│   ├── train_real_squad.py         # CLI trainer for SQuAD v2.0 Question Answering
│   ├── evaluate.py                 # Evaluation benchmark runner
│   └── visualize.py                # Visualizer and plotting tool
├── results/                        # Output directory for plots & metrics
├── requirements.txt                # Dependencies
└── README.md                       # Project documentation
```

---

## ⚡ Quick Start: Running in your Miniconda CUDA Environment

### 1. Install Dependencies
Activate your Miniconda CUDA environment and install the required dependencies:
```bash
pip install -r requirements.txt
```

### 2. Train Autonomous Gating on Synthetic Data
```bash
python scripts/train_synthetic.py
```

### 3. Train on Real SQuAD v2.0 Dataset (with GPU / CUDA Acceleration)
```bash
python scripts/train_real_squad.py --model_type autonomous --d_model 256 --num_layers 4 --num_heads 8 --epochs 3 --batch_size 16 --device cuda
```

Options for `train_real_squad.py`:
- `--model_type`: `autonomous` (learned gating) or `explicit`
- `--mode`: `logit_bias`, `k_scale`, `v_scale`, `combo`, or `baseline`
- `--question_weight`: Weight assigned to question tokens (default: `3.0`)
- `--device`: `cuda` or `cpu`

### 4. Run Evaluation Benchmark
```bash
python scripts/evaluate.py --device cuda
```

### 5. Generate Summary Reports & Plots
```bash
python scripts/visualize.py
```
Outputs and plots will be saved inside the `./results/` folder.

---

## 🧠 Architectural Overview

1. **Logit Bias Entrypoint (`logit_bias`)**:
   $$\text{Scores}_{ij} = \frac{Q_i K_j^T}{\sqrt{d_k}} + \log(w_j)$$
   Adds $\log(w_j)$ to unnormalized attention scores prior to softmax.

2. **Autonomous Learned Importance Gating (`ImportanceGatingHead`)**:
   $$w_j = 1.0 + \text{Softplus}\left(W_{\text{gate}} \cdot h_j + b_{\text{gate}}\right)$$
   Automatically predicts token weights from hidden representations end-to-end without requiring ground-truth human labels.