from __future__ import annotations

from pathlib import Path

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from .config import load_config
from .data_utils import load_sft_dataset_hf, tokenize_causal_lm


def _bf16_supported() -> bool:
    return torch.cuda.is_available() and torch.cuda.is_bf16_supported()


def _find_latest_checkpoint(checkpoint_dir: Path) -> Path:
    """Find the latest checkpoint directory."""
    checkpoint_dirs = [d for d in checkpoint_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")]
    if not checkpoint_dirs:
        # No checkpoint subdirs, use the directory itself
        return checkpoint_dir
    
    # Sort by checkpoint number (extract number from "checkpoint-31250")
    def get_checkpoint_num(path: Path) -> int:
        try:
            return int(path.name.split("-")[1])
        except (IndexError, ValueError):
            return 0
    
    latest = max(checkpoint_dirs, key=get_checkpoint_num)
    print(f"Using checkpoint: {latest.name}")
    return latest


def run_sft(config_path: str | Path = "config.yaml") -> None:
    cfg = load_config(config_path)
    paths = cfg.paths

    base_dir = paths["model_pretrained"]
    if not base_dir.is_dir():
        raise FileNotFoundError(
            "Pretrained model not found; run continued pretraining first."
        )

    # Find latest checkpoint if checkpoints exist, otherwise use base_dir
    checkpoint_path = _find_latest_checkpoint(base_dir)
    
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    model = AutoModelForCausalLM.from_pretrained(checkpoint_path)

    # Use Hugging Face SFT dataset: ds = load_dataset("mahwizzzz/dumm")
    sft_ds = load_sft_dataset_hf(cfg)
    tokenized = tokenize_causal_lm(sft_ds, tokenizer, cfg)

    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    sft_cfg = cfg.sft_cfg
    use_bf16 = sft_cfg.get("bf16", True) and _bf16_supported()

    training_args = TrainingArguments(
        output_dir=str(paths["model_sft"]),
        per_device_train_batch_size=sft_cfg["batch_size"],
        gradient_accumulation_steps=sft_cfg["gradient_accumulation_steps"],
        learning_rate=sft_cfg["learning_rate"],
        weight_decay=sft_cfg["weight_decay"],
        num_train_epochs=sft_cfg["num_epochs"],
        warmup_ratio=sft_cfg["warmup_ratio"],
        max_grad_norm=sft_cfg["max_grad_norm"],
        logging_steps=sft_cfg["logging_steps"],
        save_strategy=sft_cfg["save_strategy"],
        save_total_limit=3,
        bf16=use_bf16,
        fp16=not use_bf16,
        gradient_checkpointing=True,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=collator,
    )

    trainer.train()

    out_dir = paths["model_sft"]
    out_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"SFT model saved to {out_dir}")


if __name__ == "__main__":
    run_sft()

