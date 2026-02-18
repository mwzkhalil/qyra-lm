from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass
class Config:
    raw: Dict[str, Any]

    @property
    def model_name(self) -> str:
        return str(self.raw["model"]["base_model_name"])

    @property
    def max_seq_length(self) -> int:
        return int(self.raw["model"]["max_seq_length"])

    @property
    def max_train_samples(self) -> int | None:
        value = self.raw["training"]["pretrain"].get("max_train_samples")
        if value is None:
            return None
        return int(value)

    @property
    def corpus_path(self) -> Path:
        return Path(self.raw["data"]["corpus_path"])

    @property
    def sft_path(self) -> Path:
        return Path(self.raw["data"]["sft_path"])

    @property
    def hf_vocab_dataset(self) -> str:
        return str(self.raw["data"]["hf_vocab_dataset"])

    @property
    def hf_vocab_split(self) -> str:
        return str(self.raw["data"]["hf_vocab_split"])

    @property
    def hf_vocab_text_field(self) -> str:
        return str(self.raw["data"]["hf_vocab_text_field"])

    @property
    def hf_sft_dataset(self) -> str:
        return str(self.raw["data"]["hf_sft_dataset"])

    @property
    def hf_sft_split(self) -> str:
        return str(self.raw["data"]["hf_sft_split"])

    @property
    def hf_sft_input_field(self) -> str:
        return str(self.raw["data"]["hf_sft_input_field"])

    @property
    def hf_sft_output_field(self) -> str:
        return str(self.raw["data"]["hf_sft_output_field"])

    @property
    def vocab_min_frequency(self) -> int:
        return int(self.raw["vocab"]["min_frequency"])

    @property
    def vocab_top_k(self) -> int:
        return int(self.raw["vocab"]["top_k"])

    @property
    def vocab_output_path(self) -> Path:
        return Path(self.raw["vocab"]["output_vocab_path"])

    @property
    def paths(self) -> Dict[str, Path]:
        p = self.raw["paths"]
        return {
            "tokenizer_expanded": Path(p["tokenizer_expanded_dir"]),
            "model_token_expanded": Path(p["model_token_expanded_dir"]),
            "model_pretrained": Path(p["model_pretrained_dir"]),
            "model_sft": Path(p["model_sft_dir"]),
        }

    @property
    def pretrain_cfg(self) -> Dict[str, Any]:
        return dict(self.raw["training"]["pretrain"])

    @property
    def sft_cfg(self) -> Dict[str, Any]:
        return dict(self.raw["training"]["sft"])

    @property
    def eval_cfg(self) -> Dict[str, Any]:
        return dict(self.raw["evaluation"])

    @property
    def inference_cfg(self) -> Dict[str, Any]:
        return dict(self.raw["inference"])


def load_config(path: str | Path = "config.yaml") -> Config:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data: Dict[str, Any] = yaml.safe_load(f)
    return Config(raw=data)


__all__ = ["Config", "load_config"]

