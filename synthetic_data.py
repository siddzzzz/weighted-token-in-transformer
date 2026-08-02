import torch
import random
from typing import Tuple, List

# Fixed special token IDs
INST_PREFIX = 1
KEY_OFFSET = 10
VAL_OFFSET = 100
DISTRACTOR_START = 500

class SyntheticNeedleDataset:
    """
    Synthetic dataset for testing instruction retention across varying sequence lengths.
    Structure:
    [INST_PREFIX, Key, Value] + [Distractor_1, ..., Distractor_K] + [Query(Key)]
    Target to predict after Query(Key) is Value.
    """
    def __init__(self, vocab_size: int = 1000):
        self.vocab_size = vocab_size

    def generate_batch(
        self,
        batch_size: int,
        seq_len: int,
        target_weight: float = 3.0,
        num_rules: int = 1
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        returns: (tokens [B, seq_len], weights [B, seq_len], target_values [B], inst_indices [B, num_inst_tokens])
        """
        tokens = torch.zeros((batch_size, seq_len), dtype=torch.long)
        weights = torch.ones((batch_size, seq_len), dtype=torch.float32)
        targets = torch.zeros(batch_size, dtype=torch.long)
        
        for b in range(batch_size):
            # Fill with random distractor tokens
            distractors = torch.randint(DISTRACTOR_START, self.vocab_size - 1, (seq_len,))
            tokens[b] = distractors
            
            # Place System Instruction at start
            key_id = random.randint(10, 50)
            val_id = random.randint(100, 200)
            
            tokens[b, 0] = INST_PREFIX
            tokens[b, 1] = key_id
            tokens[b, 2] = val_id
            
            # Set weights for the instruction tokens
            weights[b, 0:3] = target_weight
            
            # Place Query at the end
            tokens[b, seq_len - 1] = key_id
            targets[b] = val_id

        return tokens, weights, targets

class SyntheticPriorityDataset:
    """
    Synthetic dataset testing priority resolution (Rule 1 vs Rule 2).
    Rule 1 (Early, Low Weight): Key -> Val1 (w=1.0)
    Rule 2 (Middle, High Weight): Key -> Val2 (w=4.0)
    Query(Key) at sequence end should yield Val2.
    """
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
            # Distractors
            tokens[b] = torch.randint(DISTRACTOR_START, self.vocab_size - 1, (seq_len,))
            
            key_id = random.randint(10, 50)
            val1_id = random.randint(100, 150)
            val2_id = random.randint(151, 200)

            # Rule 1 (Early)
            pos1 = 2
            tokens[b, pos1] = key_id
            tokens[b, pos1 + 1] = val1_id
            weights[b, pos1:pos1+2] = weight_low

            # Rule 2 (Middle)
            pos2 = seq_len // 2
            tokens[b, pos2] = key_id
            tokens[b, pos2 + 1] = val2_id
            weights[b, pos2:pos2+2] = weight_high

            # Query at end
            tokens[b, seq_len - 1] = key_id
            targets[b] = val2_id

        return tokens, weights, targets
