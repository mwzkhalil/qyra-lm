from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Iterable, List, Tuple

from tqdm import tqdm

from .config import load_config
from datasets import load_dataset


URDU_CHAR_RANGE = r"\u0600-\u06FF"  # Arabic + Urdu block


def is_urdu_word(token: str) -> bool:
    if not token:
        return False
    return all(re.match(f"[{URDU_CHAR_RANGE}]", ch) for ch in token)


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def iter_urdu_tokens(lines: Iterable[str]) -> Iterable[str]:
    for line in lines:
        line = normalize_whitespace(line)
        if not line:
            continue
        for tok in line.split(" "):
            tok = tok.strip()
            if is_urdu_word(tok):
                yield tok


def extract_top_urdu_words(
    texts: Iterable[str],
    min_frequency: int,
    top_k: int | None,
) -> List[Tuple[str, int]]:
    counter: Counter[str] = Counter()
    for token in tqdm(iter_urdu_tokens(texts), desc="Counting Urdu tokens"):
        counter[token] += 1

    items = [(w, c) for w, c in counter.items() if c >= min_frequency]
    items.sort(key=lambda x: x[1], reverse=True)
    if top_k is not None and top_k > 0:
        items = items[:top_k]
    return items


def save_vocab(words: List[Tuple[str, int]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for w, c in words:
            f.write(f"{w}\t{c}\n")


def main(config_path: str | Path = "config.yaml") -> None:
    cfg = load_config(config_path)
    # Use Hugging Face dataset for vocab extraction:
    # ds = load_dataset("ReySajju742/shaistagi_clean") by default.
    hf_name = cfg.hf_vocab_dataset
    hf_split = cfg.hf_vocab_split
    text_field = cfg.hf_vocab_text_field

    # Some large community datasets have outdated split metadata on the hub.
    # Disable strict verification to avoid NonMatchingSplitsSizesError.
    dataset = load_dataset(
        hf_name,
        split=hf_split,
        verification_mode="no_checks",
    )

    def _iter_texts() -> Iterable[str]:
        for ex in dataset:
            txt = ex.get(text_field, "")
            if isinstance(txt, str) and txt.strip():
                yield txt

    words = extract_top_urdu_words(
        texts=_iter_texts(),
        min_frequency=cfg.vocab_min_frequency,
        top_k=cfg.vocab_top_k,
    )
    save_vocab(words, cfg.vocab_output_path)
    print(f"Saved {len(words)} Urdu vocab entries to {cfg.vocab_output_path}")


if __name__ == "__main__":
    main()

