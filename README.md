# qyra-lm

Adapting `LiquidAI/LFM2-350M` to Urdu via vocabulary expansion, continued pretraining, and supervised fine-tuning.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Configuration

Edit `config.yaml` to set:
- Model paths and hyperparameters
- Dataset paths (corpus, SFT)
- Training parameters (batch size, learning rate, epochs)

## Training Pipeline

```bash
# 1. Extract Urdu vocabulary
bash scripts/01_extract_vocab.sh

# 2. Expand tokenizer and run continued pretraining
bash scripts/02_pretrain.sh

# 3. Supervised fine-tuning
bash scripts/03_sft.sh

# 4. Evaluation
bash scripts/04_evaluate.sh

# 5. Inference
bash scripts/05_inference.sh
```

## Architecture

- **Base model**: `LiquidAI/LFM2-350M` (350M parameters)
- **Tokenizer**: LFM-2 tokenizer + Urdu vocabulary expansion (capped at +10k tokens)
- **Training**: Causal LM with `transformers.Trainer`, bf16/fp16 mixed precision, gradient checkpointing
- **Data pipeline**: Streaming datasets, memory-safe preprocessing (no Arrow cache explosion)

## Repository Structure

```
.
├── config.yaml              # Configuration
├── scripts/                 # Training scripts
│   ├── 01_extract_vocab.sh
│   ├── 02_pretrain.sh
│   ├── 03_sft.sh
│   ├── 04_evaluate.sh
│   └── 05_inference.sh
└── src/                     # Source code
    ├── config.py
    ├── vocab_extraction.py
    ├── tokenizer_utils.py
    ├── data_utils.py
    ├── pretrain.py
    ├── sft.py
    ├── evaluation.py
    └── inference.py
```

## Model Cards

- **Base model**: [`mahwizzzz/qyra-350m`](https://huggingface.co/mahwizzzz/qyra-350m) 
- **SFT model**: [`Cont.`] 

## Requirements

- Python 3.12+
- CUDA-capable GPU (8GB+ VRAM recommended)
- PyTorch 2.1+, Transformers 4.46+, Datasets 2.18+

## License

See base model license: `LiquidAI/LFM2-350M`
