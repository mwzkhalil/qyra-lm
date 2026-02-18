from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from .config import load_config


def load_inference_pipeline(
    model_dir: Path,
    device_str: str | None = None,
):
    if device_str is None:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device_index = 0 if device_str == "cuda" else -1

    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir)

    gen_pipe = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tok,
        device=device_index,
    )
    return gen_pipe


def generate_urdu(
    prompts: Iterable[str],
    model_dir: Path,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> List[str]:
    pipe = load_inference_pipeline(model_dir)
    outputs: List[str] = []
    for p in prompts:
        res = pipe(
            p,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            return_full_text=True,
        )
        outputs.append(res[0]["generated_text"])
    return outputs


def main(config_path: str | Path = "config.yaml") -> None:
    cfg = load_config(config_path)
    paths = cfg.paths
    inf_cfg = cfg.inference_cfg

    model_dir = paths["model_sft"]
    if not model_dir.is_dir():
        raise FileNotFoundError(
            f"SFT model directory not found at {model_dir}. Run SFT before inference."
        )

    prompts = [
        "اردو میں موسمِ بہار کے بارے میں ایک پیراگراف لکھیں۔",
        "پاکستان کی ثقافت کی اہم خصوصیات بیان کریں۔",
        "علامہ اقبال کے انداز میں مختصر شاعری تحریر کریں۔",
    ]

    generations = generate_urdu(
        prompts,
        model_dir=model_dir,
        max_new_tokens=inf_cfg["max_new_tokens"],
        temperature=inf_cfg["temperature"],
        top_p=inf_cfg["top_p"],
    )

    for p, g in zip(prompts, generations):
        print("\n=== Prompt ===")
        print(p)
        print("=== Generation ===")
        print(g)


if __name__ == "__main__":
    main()

