from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from .config import load_config


def load_base_tokenizer(model_name: str) -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def read_vocab_file(path: Path) -> List[str]:
    tokens: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            tok = line.split("\t", 1)[0].strip()
            if tok:
                tokens.append(tok)
    return tokens


def add_new_tokens(
    tokenizer: PreTrainedTokenizerBase,
    new_tokens: Iterable[str],
) -> int:
    existing = set(tokenizer.get_vocab().keys())
    # Hard cap on vocabulary growth to avoid excessive expansion.
    MAX_NEW_TOKENS = 10_000
    filtered = [t for t in new_tokens if t not in existing]
    if len(filtered) > MAX_NEW_TOKENS:
        filtered = filtered[:MAX_NEW_TOKENS]
    if not filtered:
        print("No new tokens to add.")
        return 0
    num_added = tokenizer.add_tokens(filtered, special_tokens=False)
    print(f"Added {num_added} new tokens.")
    return num_added


def resize_embeddings_for_new_tokens(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
) -> PreTrainedModel:
    old_embeddings = model.get_input_embeddings().weight.data
    old_vocab_size, hidden_size = old_embeddings.shape
    new_vocab_size = len(tokenizer)

    if new_vocab_size == old_vocab_size:
        print("Tokenizer size unchanged; skipping resize_token_embeddings.")
        return model

    model.resize_token_embeddings(new_vocab_size)

    new_embeddings = model.get_input_embeddings().weight.data
    if new_vocab_size > old_vocab_size:
        with torch.no_grad():
            existing_slice = old_embeddings
            std = existing_slice.std(dim=0, keepdim=True)
            mean = existing_slice.mean(dim=0, keepdim=True)
            init = torch.randn(
                (new_vocab_size - old_vocab_size, hidden_size),
                device=existing_slice.device,
                dtype=existing_slice.dtype,
            )
            init = init * std + mean
            new_embeddings[old_vocab_size:] = init

    print(
        f"Resized embeddings from {old_vocab_size} to {new_vocab_size}; "
        "existing rows left untouched."
    )
    return model


def expand_tokenizer_and_model(config_path: str | Path = "config.yaml") -> None:
    cfg = load_config(config_path)
    vocab_path = cfg.vocab_output_path
    if not vocab_path.is_file():
        raise FileNotFoundError(f"Vocab file not found: {vocab_path}")

    base_model_name = cfg.model_name
    tokenizer = load_base_tokenizer(base_model_name)
    vocab_tokens = read_vocab_file(vocab_path)
    add_new_tokens(tokenizer, vocab_tokens)

    paths = cfg.paths
    tokenizer_out = paths["tokenizer_expanded"]
    tokenizer_out.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(tokenizer_out)
    print(f"Expanded tokenizer saved to {tokenizer_out}")

    model = AutoModelForCausalLM.from_pretrained(base_model_name)
    resize_embeddings_for_new_tokens(model, tokenizer)
    model_out = paths["model_token_expanded"]
    model_out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(model_out)
    tokenizer.save_pretrained(model_out)
    print(f"Model with resized embeddings saved to {model_out}")


if __name__ == "__main__":
    expand_tokenizer_and_model()

