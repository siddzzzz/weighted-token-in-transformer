import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List
from .weighted_attention import TransformerBlock

class ImportanceGatingHead(nn.Module):
    """
    Learned Importance Gating Head.
    Maps token hidden representations h_j [B, N, d_model] -> token weight scalars w_j [B, N].
    w_j = 1.0 + Softplus(Linear(h_j))
    Initial bias is set to -5.0 so that Softplus(-5.0) ~ 0.0067 -> w_j ~ 1.0067 at initialization.
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, 1)
        nn.init.constant_(self.gate_proj.bias, -5.0)
        nn.init.normal_(self.gate_proj.weight, std=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, N, d_model]
        returns: token_weights [B, N] where w_j >= 1.0
        """
        raw_gate = self.gate_proj(x).squeeze(-1) # [B, N]
        token_weights = 1.0 + F.softplus(raw_gate) # [B, N]
        return token_weights


class AutonomousWeightedTransformerDecoder(nn.Module):
    """
    Autonomous Weighted Transformer Decoder.
    Automatically predicts token importance weights using an internal ImportanceGatingHead
    and injects them into multi-head attention logit biases end-to-end.
    """
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
        self.gating_head = ImportanceGatingHead(d_model)
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
        custom_weights: Optional[torch.Tensor] = None,
        use_autonomous_gating: bool = True
    ) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]:
        B, N = idx.shape
        device = idx.device

        pos = torch.arange(0, N, dtype=torch.long, device=device).unsqueeze(0)
        h = self.tok_emb(idx) + self.pos_emb(pos)
        h = self.drop(h)

        if use_autonomous_gating:
            predicted_weights = self.gating_head(h) # [B, N]
            token_weights = predicted_weights
        else:
            token_weights = custom_weights if custom_weights is not None else torch.ones((B, N), device=device)

        x = h
        layer_attn_weights = []
        for layer in self.layers:
            x, weights = layer(x, token_weights=token_weights, causal=True, entrypoint="logit_bias")
            layer_attn_weights.append(weights)

        x = self.ln_f(x)
        logits = self.head(x)
        return logits, layer_attn_weights, token_weights
