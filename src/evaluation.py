from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import load_config


def _prepare_model(path: Path, device: torch.device) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path)
    model.to(device)
    model.eval()
    return model, tokenizer


def _find_latest_checkpoint(checkpoint_dir: Path) -> Path:
    """
    Find the latest checkpoint directory inside `checkpoint_dir`.
    If no checkpoint subdirectories exist, return `checkpoint_dir` itself.
    """
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    checkpoint_dirs = [
        d for d in checkpoint_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")
    ]
    if not checkpoint_dirs:
        return checkpoint_dir

    def get_checkpoint_num(path: Path) -> int:
        try:
            return int(path.name.split("-")[1])
        except (IndexError, ValueError):
            return 0

    latest = max(checkpoint_dirs, key=get_checkpoint_num)
    print(f"Using adapted checkpoint: {latest}")
    return latest


def compute_perplexity(
    model: AutoModelForCausalLM,
    tokenizer,
    texts: Iterable[str],
    device: torch.device,
    max_length: int,
) -> float:
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for text in texts:
            enc = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
            )
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)
            out = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = out.loss
            total_loss += loss.item() * input_ids.numel()
            total_tokens += input_ids.numel()
    if total_tokens == 0:
        return float("inf")
    return math.exp(total_loss / total_tokens)


def fragmentation_ratio(tokenizer, texts: Iterable[str]) -> float:
    total_words = 0
    total_tokens = 0
    for text in texts:
        words = [w for w in text.split() if w.strip()]
        if not words:
            continue
        tokens = tokenizer.encode(text, add_special_tokens=False)
        total_words += len(words)
        total_tokens += len(tokens)
    if total_words == 0:
        return 0.0
    return total_tokens / float(total_words)


def generate_samples(
    model: AutoModelForCausalLM,
    tokenizer,
    prompts: List[str],
    device: torch.device,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> List[str]:
    outputs: List[str] = []
    with torch.no_grad():
        for p in prompts:
            enc = tokenizer(p, return_tensors="pt").to(device)
            gen = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
            )
            text = tokenizer.decode(gen[0], skip_special_tokens=True)
            outputs.append(text)
    return outputs


def main(config_path: str | Path = "config.yaml") -> None:
    cfg = load_config(config_path)
    paths = cfg.paths
    eval_cfg = cfg.eval_cfg

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sample_texts = [
        "اردو ایک خوبصورت زبان ہے۔",
        "مجھے پاکستان کی تاریخ کے بارے میں بتائیں۔",
        "آج موسم کیسا ہے؟",
        "آپ کا پسندیدہ شاعر کون ہے؟",
    ]

    base_model_path = Path(cfg.model_name)
    adapted_root = paths["model_sft"]
    adapted_model_path = _find_latest_checkpoint(adapted_root)

    print("Loading base model...")
    base_model, base_tok = _prepare_model(base_model_path, device)
    print("Loading adapted model...")
    # Load adapted model weights from latest checkpoint; reuse base tokenizer for compatibility.
    adapted_model = AutoModelForCausalLM.from_pretrained(adapted_model_path)
    adapted_model.to(device)
    adapted_model.eval()
    adapted_tok = base_tok

    # Use a safe maximum sequence length to avoid overflow from very large tokenizer.model_max_length
    max_len = int(min(cfg.max_seq_length, getattr(base_tok, "model_max_length", cfg.max_seq_length)))

    print("Computing perplexity...")
    base_ppl = compute_perplexity(base_model, base_tok, sample_texts, device, max_length=max_len)
    adapted_ppl = compute_perplexity(adapted_model, adapted_tok, sample_texts, device, max_length=max_len)

    print("Computing fragmentation...")
    base_frag = fragmentation_ratio(base_tok, sample_texts)
    adapted_frag = fragmentation_ratio(adapted_tok, sample_texts)

    print(f"Base model perplexity: {base_ppl:.3f}")
    print(f"Adapted model perplexity: {adapted_ppl:.3f}")
    print(f"Base model fragmentation ratio: {base_frag:.3f}")
    print(f"Adapted model fragmentation ratio: {adapted_frag:.3f}")

    print("Generating qualitative samples from adapted model...")
    samples = generate_samples(
        adapted_model,
        adapted_tok,
        prompts=sample_texts[: eval_cfg.get("num_eval_prompts", 4)],
        device=device,
        max_new_tokens=eval_cfg["max_new_tokens"],
        temperature=eval_cfg["temperature"],
        top_p=eval_cfg["top_p"],
    )
    for i, (prompt, sample) in enumerate(zip(sample_texts, samples), start=1):
        print(f"\n=== Sample {i} ===")
        print(f"Prompt: {prompt}")
        print(f"Generation: {sample}")


if __name__ == "__main__":
    main()

