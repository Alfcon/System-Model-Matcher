# LLM Model Finder

A desktop application that detects your system hardware and finds the best GGUF-format LLM models from Hugging Face Hub for your machine.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python src/main.py
```

## App Flow

1. **Hardware Detection** — automatically scans your CPU, RAM, and GPU (VRAM)
2. **Preferences** — select your inference app and optionally enter a search keyword
3. **Results** — top 10 ranked models displayed in a sortable table

## Features

- **Hardware Detection**: Detects CPU model/cores, total/available RAM, and GPU VRAM using `nvidia-ml-py`, `nvidia-smi`, or `lspci` (Linux fallback); falls back to 8 GB effective VRAM on CPU-only systems
- **Smart Model Search**: Queries Hugging Face Hub for GGUF models sorted by downloads and likes (top 30 each), deduplicates, and evaluates up to 60 candidates
- **Quantization-Aware Filtering**: Selects the highest-quality quant that fits within 80% of your VRAM; falls back to the smallest available file if none fit
- **Hardware-Aware Ranking**: Scores each model on four weighted criteria:
  - VRAM fit — 50%
  - Parameter count suitability (prefers 7–13B for chat, 13B+ for other tasks) — 30%
  - Quantization quality — 10%
  - Popularity (downloads + likes) — 10%
- **Results Table**: Displays rank, model name, parameter count, quant type, file size, estimated VRAM usage, and estimated inference speed (tokens/sec)
- **Copy to Clipboard**: Export the results table as tab-separated text

## Supported Quantization Formats

Q2_K, Q3_K_S, Q3_K_M, Q3_K_L, Q4_0, Q4_K_S, Q4_K_M, Q5_K_S, Q5_K_M, Q6_K, Q8_0

## Supported Inference Apps

- Ollama
- LM Studio
- GPT4All
- Custom

## Requirements

- Python 3.8+
- Internet connection (for Hugging Face Hub search)
- NVIDIA GPU recommended; CPU-only systems are supported with a conservative VRAM estimate
