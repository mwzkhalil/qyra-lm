from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd
from datasets import Dataset, IterableDataset, load_dataset
from transformers import PreTrainedTokenizerBase

from .config import Config


def load_corpus_as_dataset(path: Path) -> Dataset:
    texts: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                texts.append(line)
    return Dataset.from_dict({"text": texts})


def load_corpus_as_dataset_hf(
    dataset_name: str,
    split: str,
    text_field: str,
    max_samples: int | None = None,
) -> Dataset:
    """
    Load a text corpus from a Hugging Face dataset, using a single text field.
    Uses streaming mode to avoid loading the full dataset when max_samples is set.
    """
    if max_samples is None:
        raise ValueError("max_samples must be set to avoid loading 50M+ examples. Set max_train_samples in config.")
    
    # ALWAYS use streaming when max_samples is set to prevent full dataset materialization
    # Streaming mode naturally bypasses Arrow cache since it doesn't materialize the dataset
    print(f"Loading dataset with streaming mode, limiting to {max_samples:,} samples...")
    ds_hf = load_dataset(
        dataset_name,
        split=split,
        streaming=True,
    )

    def _extract(example: Dict[str, str]) -> str | None:
        text = str(example.get(text_field, "") or "")
        if not text.strip():
            return None
        return text.strip()

    # For streaming: take first N samples, then convert to regular dataset
    texts = []
    count = 0
    for idx, example in enumerate(ds_hf):
        if count >= max_samples:
            break
        if idx % 10000 == 0 and idx > 0:
            print(f"  Processed {idx:,} examples, collected {count:,} valid samples...")
        result = _extract(example)
        if result is not None:
            texts.append(result)
            count += 1
    
    print(f"Collected {len(texts):,} samples. Creating dataset...")
    return Dataset.from_dict({"text": texts})


def load_sft_dataset(path: Path) -> Dataset:
    df = pd.read_csv(path)
    if not {"input", "output"}.issubset(df.columns):
        raise ValueError("SFT CSV must contain 'input' and 'output' columns.")
    merged = df["input"].astype(str) + "\n\n" + df["output"].astype(str)
    return Dataset.from_dict({"text": merged.tolist()})


def load_sft_dataset_hf(cfg: Config) -> Dataset:
    """
    Load SFT dataset from Hugging Face hub, e.g.:
    ds = load_dataset("mahwizzzz/dumm", split="train")
    """
    ds = load_dataset(cfg.hf_sft_dataset, split=cfg.hf_sft_split)
    in_field = cfg.hf_sft_input_field
    out_field = cfg.hf_sft_output_field

    def _build_text(example: Dict[str, str]) -> Dict[str, str]:
        inp = str(example.get(in_field, "") or "")
        out = str(example.get(out_field, "") or "")
        text = inp + "\n\n" + out if out else inp
        return {"text": text}

    ds = ds.map(
        _build_text,
        remove_columns=list(ds.features),
        load_from_cache_file=False,
        keep_in_memory=False,
        writer_batch_size=1000,
    )
    return ds


def tokenize_causal_lm(
    dataset: Dataset | IterableDataset,
    tokenizer: PreTrainedTokenizerBase,
    cfg: Config,
) -> Dataset:
    def _tokenize(batch: Dict[str, List[str]]) -> Dict[str, List[List[int]]]:
        return tokenizer(
            batch["text"],
            max_length=cfg.max_seq_length,
            truncation=True,
            padding="max_length",
        )

    remove_cols = list(getattr(dataset, "features", {}).keys()) or dataset.column_names
    tokenized = dataset.map(
        _tokenize,
        batched=True,
        remove_columns=remove_cols,
        desc="Tokenizing dataset for causal LM",
        load_from_cache_file=False,
        keep_in_memory=False,
        writer_batch_size=1000,
    )

    def _set_labels(batch: Dict[str, List[List[int]]]) -> Dict[str, List[List[int]]]:
        batch["labels"] = batch["input_ids"].copy()
        return batch

    tokenized = tokenized.map(
        _set_labels,
        batched=True,
        load_from_cache_file=False,
        keep_in_memory=False,
        writer_batch_size=1000,
    )
    return tokenized


__all__ = [
    "load_corpus_as_dataset",
    "load_corpus_as_dataset_hf",
    "load_sft_dataset",
    "load_sft_dataset_hf",
    "tokenize_causal_lm",
]

