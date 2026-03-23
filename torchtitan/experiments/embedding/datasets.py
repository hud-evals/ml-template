"""Training data formatting and loading for embedding training."""

import json
import random
from collections.abc import Iterator
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

from torchtitan.components.dataloader import BaseDataLoader


@dataclass
class TrainingPair:
    instruction: str
    query: str
    positive: str
    negatives: list[str]


def format_query_input(
    instruction: str,
    query: str,
    eos_token: str = "<|endoftext|>",
) -> str:
    if instruction:
        return f"{instruction} {query}{eos_token}"
    return f"{query}{eos_token}"


def format_document_input(
    document: str,
    eos_token: str = "<|endoftext|>",
) -> str:
    return f"{document}{eos_token}"


def create_training_batch(
    pairs: list[TrainingPair],
    eos_token: str = "<|endoftext|>",
) -> dict:
    query_texts = []
    positive_texts = []
    negative_texts = []

    for pair in pairs:
        query_texts.append(format_query_input(pair.instruction, pair.query, eos_token))
        positive_texts.append(format_document_input(pair.positive, eos_token))
        negative_texts.append(
            [format_document_input(neg, eos_token) for neg in pair.negatives]
        )

    return {
        "query_texts": query_texts,
        "positive_texts": positive_texts,
        "negative_texts": negative_texts,
    }


class EmbeddingDataset(Dataset):
    def __init__(
        self,
        path: str,
        tokenizer,
        max_seq_length: int = 512,
        num_hard_negatives: int = 7,
    ):
        self.pairs: list[TrainingPair] = []
        with open(path) as f:
            for line in f:
                obj = json.loads(line)
                self.pairs.append(
                    TrainingPair(
                        instruction=obj.get("instruction", ""),
                        query=obj["query"],
                        positive=obj["positive"],
                        negatives=obj.get("negatives", []),
                    )
                )
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.num_hard_negatives = num_hard_negatives

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]

        query_text = format_query_input(
            pair.instruction, pair.query, self.tokenizer.eos_token
        )
        pos_text = format_document_input(pair.positive, self.tokenizer.eos_token)

        negs = pair.negatives[: self.num_hard_negatives]
        while len(negs) < self.num_hard_negatives:
            other_idx = random.randint(0, len(self.pairs) - 1)
            if other_idx != idx:
                negs.append(self.pairs[other_idx].positive)
        neg_texts = [format_document_input(n, self.tokenizer.eos_token) for n in negs]

        query_enc = self.tokenizer(
            query_text,
            max_length=self.max_seq_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        pos_enc = self.tokenizer(
            pos_text,
            max_length=self.max_seq_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        neg_encs = self.tokenizer(
            neg_texts,
            max_length=self.max_seq_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "query_input_ids": query_enc["input_ids"].squeeze(0),
            "query_attention_mask": query_enc["attention_mask"].squeeze(0),
            "pos_input_ids": pos_enc["input_ids"].squeeze(0),
            "pos_attention_mask": pos_enc["attention_mask"].squeeze(0),
            "neg_input_ids": neg_encs["input_ids"],
            "neg_attention_mask": neg_encs["attention_mask"],
        }


class EmbeddingDataLoader(BaseDataLoader):
    """Torchtitan-compatible dataloader for contrastive embedding training."""

    @dataclass(kw_only=True, slots=True)
    class Config(BaseDataLoader.Config):
        train_path: str = ""
        eval_path: str | None = None
        num_hard_negatives: int = 7
        num_epochs: int = 3
        model_name: str = "Qwen/Qwen3-0.6B"

    def __init__(self, config: Config, **kwargs):
        self.config = config
        self._seq_len = kwargs.get("seq_len", 512)
        self._batch_size = kwargs.get("local_batch_size", 4)

        from transformers import AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(
            config.model_name, trust_remote_code=True
        )

        self._dataset: EmbeddingDataset | None = None
        self._epoch = 0
        self._position = 0

    def _ensure_dataset(self):
        if self._dataset is None:
            self._dataset = EmbeddingDataset(
                self.config.train_path,
                self._tokenizer,
                self._seq_len,
                self.config.num_hard_negatives,
            )

    def __iter__(self) -> Iterator[tuple[dict[str, torch.Tensor], torch.Tensor]]:
        self._ensure_dataset()
        assert self._dataset is not None

        for epoch in range(self._epoch, self.config.num_epochs):
            self._epoch = epoch
            loader = torch.utils.data.DataLoader(
                self._dataset,
                batch_size=self._batch_size,
                shuffle=True,
                drop_last=True,
            )
            for i, batch in enumerate(loader):
                if i < self._position:
                    continue
                self._position = i + 1
                input_dict = {
                    "input": batch["query_input_ids"],
                    "query_attention_mask": batch["query_attention_mask"],
                    "pos_input_ids": batch["pos_input_ids"],
                    "pos_attention_mask": batch["pos_attention_mask"],
                    "neg_input_ids": batch["neg_input_ids"],
                    "neg_attention_mask": batch["neg_attention_mask"],
                }
                labels = torch.zeros(
                    batch["query_input_ids"].shape[0], dtype=torch.long
                )
                yield input_dict, labels
            self._position = 0

    def state_dict(self):
        return {"epoch": self._epoch, "position": self._position}

    def load_state_dict(self, state_dict):
        self._epoch = state_dict.get("epoch", 0)
        self._position = state_dict.get("position", 0)
