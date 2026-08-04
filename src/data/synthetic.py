import torch
import random
from typing import Tuple

INST_PREFIX = 1
DISTRACTOR_START = 500

class SyntheticNeedleDataset:
    def __init__(self, vocab_size: int = 1000):
        self.vocab_size = vocab_size

    def generate_batch(
        self,
        batch_size: int,
        seq_len: int,
        target_weight: float = 3.0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tokens = torch.zeros((batch_size, seq_len), dtype=torch.long)
        weights = torch.ones((batch_size, seq_len), dtype=torch.float32)
        targets = torch.zeros(batch_size, dtype=torch.long)
        
        for b in range(batch_size):
            tokens[b] = torch.randint(DISTRACTOR_START, self.vocab_size - 1, (seq_len,))
            
            key_id = random.randint(10, 50)
            val_id = random.randint(100, 200)
            
            tokens[b, 0] = INST_PREFIX
            tokens[b, 1] = key_id
            tokens[b, 2] = val_id
            
            weights[b, 0:3] = target_weight
            
            tokens[b, seq_len - 1] = key_id
            targets[b] = val_id

        return tokens, weights, targets


class SyntheticPriorityDataset:
    def __init__(self, vocab_size: int = 1000):
        self.vocab_size = vocab_size

    def generate_batch(
        self,
        batch_size: int,
        seq_len: int,
        weight_low: float = 1.0,
        weight_high: float = 4.0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tokens = torch.zeros((batch_size, seq_len), dtype=torch.long)
        weights = torch.ones((batch_size, seq_len), dtype=torch.float32)
        targets = torch.zeros(batch_size, dtype=torch.long)

        for b in range(batch_size):
            tokens[b] = torch.randint(DISTRACTOR_START, self.vocab_size - 1, (seq_len,))
            
            key_id = random.randint(10, 50)
            val1_id = random.randint(100, 150)
            val2_id = random.randint(151, 200)

            pos1 = 2
            tokens[b, pos1] = key_id
            tokens[b, pos1 + 1] = val1_id
            weights[b, pos1:pos1+2] = weight_low

            pos2 = seq_len // 2
            tokens[b, pos2] = key_id
            tokens[b, pos2 + 1] = val2_id
            weights[b, pos2:pos2+2] = weight_high

            tokens[b, seq_len - 1] = key_id
            targets[b] = val2_id

        return tokens, weights, targets
