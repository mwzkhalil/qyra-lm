from __future__ import annotations

from pathlib import Path
import os
import shutil

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from .config import load_config
from .data_utils import (
    load_corpus_as_dataset,
    load_corpus_as_dataset_hf,
    tokenize_causal_lm,
)


def _bf16_supported() -> bool:
    return torch.cuda.is_available() and torch.cuda.is_bf16_supported()


def run_pretraining(config_path: str | Path = "config.yaml") -> None:
    # Fail fast if there is not enough free disk space for preprocessing / checkpoints.
    total, used, free = shutil.disk_usage("/")
    if free < 20 * 1024**3:
        raise RuntimeError("Not enough disk space. Require at least 20GB free.")

    cfg = load_config(config_path)
    paths = cfg.paths

    tokenizer_dir = paths["tokenizer_expanded"]
    model_dir = paths["model_token_expanded"]
    if not tokenizer_dir.is_dir() or not model_dir.is_dir():
        raise FileNotFoundError(
            "Expanded tokenizer/model not found; run vocab extraction and tokenizer "
            "expansion first."
        )

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir)

    # Require max_train_samples to be set to prevent accidental full dataset loading
    if not cfg.max_train_samples:
        raise ValueError(
            "max_train_samples must be set in config to prevent loading 50M+ examples. "
            "Set training.pretrain.max_train_samples (e.g., 500000 for testing)."
        )

    # If a local corpus file exists, use it; otherwise fall back to HF dataset
    if cfg.corpus_path.is_file():
        corpus_ds = load_corpus_as_dataset(cfg.corpus_path)
        # Apply limit immediately for local files too
        if len(corpus_ds) > cfg.max_train_samples:
            print(f"Limiting local dataset from {len(corpus_ds):,} to {cfg.max_train_samples:,} samples...")
            corpus_ds = corpus_ds.select(range(cfg.max_train_samples))
    else:
        # Pass max_samples to loader to avoid loading 50M+ examples
        corpus_ds = load_corpus_as_dataset_hf(
            dataset_name=cfg.hf_vocab_dataset,
            split=cfg.hf_vocab_split,
            text_field=cfg.hf_vocab_text_field,
            max_samples=cfg.max_train_samples,
        )

    # Verify dataset size matches expectation
    actual_size = len(corpus_ds)
    if actual_size > cfg.max_train_samples * 1.1:  # Allow 10% tolerance
        raise RuntimeError(
            f"Dataset size ({actual_size:,}) exceeds max_train_samples "
            f"({cfg.max_train_samples:,}). This indicates a bug in the data loading."
        )
    print(f"Dataset loaded: {actual_size:,} samples (max allowed: {cfg.max_train_samples:,})")

    # Keep only the text column to minimize memory / disk usage.
    if "text" in corpus_ds.column_names and len(corpus_ds.column_names) > 1:
        corpus_ds = corpus_ds.remove_columns(
            [col for col in corpus_ds.column_names if col != "text"]
        )
    tokenized = tokenize_causal_lm(corpus_ds, tokenizer, cfg)

    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    pre_cfg = cfg.pretrain_cfg
    use_bf16 = pre_cfg.get("bf16", True) and _bf16_supported()

    training_args = TrainingArguments(
        output_dir=str(paths["model_pretrained"]),
        per_device_train_batch_size=pre_cfg["batch_size"],
        gradient_accumulation_steps=pre_cfg["gradient_accumulation_steps"],
        learning_rate=pre_cfg["learning_rate"],
        weight_decay=pre_cfg["weight_decay"],
        num_train_epochs=pre_cfg["num_epochs"],
        warmup_ratio=pre_cfg["warmup_ratio"],
        max_grad_norm=pre_cfg["max_grad_norm"],
        logging_steps=pre_cfg["logging_steps"],
        save_strategy=pre_cfg["save_strategy"],
        save_total_limit=3,
        bf16=use_bf16,
        fp16=not use_bf16,
        gradient_checkpointing=pre_cfg.get("gradient_checkpointing", True),
        dataloader_pin_memory=False,  # Reduce memory usage
        dataloader_num_workers=0,  # Reduce memory overhead
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=collator,
    )

    trainer.train()

    out_dir = paths["model_pretrained"]
    out_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"Pretrained model saved to {out_dir}")


if __name__ == "__main__":
    run_pretraining()

