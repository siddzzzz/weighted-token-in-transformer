import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List

class WeightedMultiHeadAttention(nn.Module):
    """
    Multi-Head Attention supporting multiple token weight entrypoints:
    1. baseline: Standard scaled dot-product attention
    2. logit_bias: Adds log(w_j) to unnormalized attention logits
    3. v_scale: Scales Value vectors V_j by w_j
    4. k_scale: Scales Key vectors K_j by sqrt(w_j)
    5. combo: Combines logit_bias and v_scale
    """
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        token_weights: Optional[torch.Tensor] = None,
        causal: bool = True,
        entrypoint: str = "baseline"
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: [B, N, C]
        token_weights: [B, N] (positive importance scalars, e.g., 1.0 = default, 3.0 = high priority)
        returns: (output [B, N, C], attention_weights [B, num_heads, N, N])
        """
        B, N, C = x.shape

        # Linear projections
        Q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, N, D]
        K = self.k_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, N, D]
        V = self.v_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, N, D]

        # Apply token weight entrypoint modifications
        if token_weights is not None and entrypoint != "baseline":
            w = torch.clamp(token_weights, min=1e-4) # [B, N]
            
            if entrypoint == "k_scale":
                w_k = torch.sqrt(w).view(B, 1, N, 1) # [B, 1, N, 1]
                K = K * w_k

            if entrypoint in ["v_scale", "combo"]:
                w_v = w.view(B, 1, N, 1) # [B, 1, N, 1]
                V = V * w_v

        # Compute raw attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim) # [B, H, N, N]

        # Apply logit bias token weighting
        if token_weights is not None and entrypoint in ["logit_bias", "combo"]:
            w = torch.clamp(token_weights, min=1e-4)
            log_w = torch.log(w).view(B, 1, 1, N) # [B, 1, 1, N]
            scores = scores + log_w

        # Causal mask
        if causal:
            causal_mask = torch.triu(torch.full((N, N), float("-inf"), device=x.device), diagonal=1)
            scores = scores + causal_mask.unsqueeze(0).unsqueeze(0)

        # Softmax normalization
        attn_weights = F.softmax(scores, dim=-1) # [B, H, N, N]
        attn_weights_dropped = self.dropout(attn_weights)

        # Attention output
        out = torch.matmul(attn_weights_dropped, V) # [B, H, N, D]
        out = out.transpose(1, 2).contiguous().view(B, N, C)
        
        return self.out_proj(out), attn_weights


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = WeightedMultiHeadAttention(d_model, num_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        token_weights: Optional[torch.Tensor] = None,
        causal: bool = True,
        entrypoint: str = "baseline"
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        norm_x = self.ln1(x)
        attn_out, weights = self.attn(norm_x, token_weights=token_weights, causal=causal, entrypoint=entrypoint)
        x = x + attn_out
        x = x + self.ffn(self.ln2(x))
        return x, weights


class WeightedTransformerDecoder(nn.Module):
    def __init__(
        self,
        vocab_size: int = 50257,
        d_model: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        d_ff: int = 1024,
        max_seq_len: int = 2048,
        dropout: float = 0.1
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.drop = nn.Dropout(dropout)
        
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(
        self,
        idx: torch.Tensor,
        token_weights: Optional[torch.Tensor] = None,
        entrypoint: str = "baseline"
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        B, N = idx.shape
        device = idx.device

        pos = torch.arange(0, N, dtype=torch.long, device=device).unsqueeze(0)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        x = self.drop(x)

        layer_attn_weights = []
        for layer in self.layers:
            x, weights = layer(x, token_weights=token_weights, causal=True, entrypoint=entrypoint)
            layer_attn_weights.append(weights)

        x = self.ln_f(x)
        logits = self.head(x)
        return logits, layer_attn_weights
