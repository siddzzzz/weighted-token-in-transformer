import os
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Dict, Any

class SQuADDatasetLoader:
    """
    SQuAD v2.0 Dataset Loader with GPT-2 Tokenizer.
    Generates subword token sequences and assigns token importance weights:
    - System Prompt + Question Tokens: weight = question_weight (e.g. 3.0)
    - Context Passage Tokens: weight = 1.0
    """
    def __init__(self, question_weight: float = 3.0, max_seq_len: int = 1024):
        self.question_weight = question_weight
        self.max_seq_len = max_seq_len
        self.tokenizer = None
        self._init_tokenizer()

    def _init_tokenizer(self):
        try:
            from transformers import GPT2TokenizerFast
            self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        except Exception as e:
            print(f"[Warning] Could not initialize GPT2TokenizerFast: {e}")

    def load_squad_dataset(self, split: str = "train", cache_dir: str = None):
        if cache_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            cache_dir = os.path.join(base_dir, "data", "squad_cache")
        os.makedirs(cache_dir, exist_ok=True)

        from datasets import load_dataset
        dataset = load_dataset("squad_v2", split=split, cache_dir=cache_dir)
        return dataset

    def prepare_batch(
        self,
        examples: list,
        device: str = "cpu"
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Tokenizes SQuAD QA pairs:
        Prompt: System_Instruction + Question + Passage
        Returns: (input_ids [B, N], token_weights [B, N], target_ids [B, N])
        """
        if self.tokenizer is None:
            self._init_tokenizer()

        batch_input_ids = []
        batch_weights = []
        batch_targets = []

        system_prompt = "Answer the question based on the passage.\nQuestion: "

        # Handle Hugging Face Dataset batch slicing (returns dict of lists)
        if isinstance(examples, dict):
            num_items = len(examples["question"])
            item_list = []
            for idx in range(num_items):
                item_list.append({
                    "question": examples["question"][idx],
                    "context": examples["context"][idx],
                    "answers": examples["answers"][idx]
                })
            examples = item_list

        for ex in examples:
            question = ex["question"]
            context = ex["context"]
            answers = ex.get("answers", {})
            if isinstance(answers, dict):
                ans_text_list = answers.get("text", [])
            elif isinstance(answers, list):
                ans_text_list = answers
            else:
                ans_text_list = []
            target_ans = ans_text_list[0] if len(ans_text_list) > 0 else "unanswerable"

            # Tokenize sections
            sys_ids = self.tokenizer.encode(system_prompt)
            q_ids = self.tokenizer.encode(question + "\nPassage: ")
            ctx_ids = self.tokenizer.encode(context + "\nAnswer: ")
            ans_ids = self.tokenizer.encode(target_ans)

            # Combine sequence
            full_ids = sys_ids + q_ids + ctx_ids
            # Truncate if exceeding max length
            if len(full_ids) > self.max_seq_len - len(ans_ids):
                full_ids = full_ids[: self.max_seq_len - len(ans_ids)]

            seq_ids = full_ids + ans_ids
            seq_len = len(seq_ids)

            # Assign weights
            # System prompt + Question tokens get question_weight
            # Passage tokens get 1.0
            q_end_idx = len(sys_ids) + len(q_ids)
            weights = [1.0] * seq_len
            for i in range(min(q_end_idx, seq_len)):
                weights[i] = self.question_weight

            batch_input_ids.append(torch.tensor(seq_ids, dtype=torch.long))
            batch_weights.append(torch.tensor(weights, dtype=torch.float32))
            batch_targets.append(torch.tensor(seq_ids[1:] + [self.tokenizer.eos_token_id], dtype=torch.long))

        # Pad batch
        padded_ids = torch.nn.utils.rnn.pad_sequence(
            batch_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        padded_weights = torch.nn.utils.rnn.pad_sequence(
            batch_weights, batch_first=True, padding_value=1.0
        )
        padded_targets = torch.nn.utils.rnn.pad_sequence(
            batch_targets, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )

        return padded_ids.to(device), padded_weights.to(device), padded_targets.to(device)
