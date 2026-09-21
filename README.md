# LLM Model Finder

A desktop application that detects your system hardware and finds the best GGUF-format LLM models from Hugging Face Hub for your machine.

## Installation

Follow these steps to set up the project using Miniconda.

1.  **Install Miniconda**: If you don't already have it, download and install [Miniconda](https://docs.conda.io/en/latest/miniconda.html).

2.  **Open Anaconda Prompt**: Launch the Anaconda Prompt.

3.  **Clone the Repository**: Navigate to where you want to store the project and run the following commands:
    ```bash
    git clone https://github.com/Alfcon-Industries/System-Model-Matcher.git
    cd System-Model-Matcher
    ```

4.  **Create and Activate Conda Environment**: This creates an isolated environment for the project's dependencies.
    ```bash
    conda create -n llm_finder python=3.9 -y
    conda activate llm_finder
    ```

5.  **Install Requirements**: Install the necessary Python packages from `requirements.txt`.
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  **Open Anaconda Prompt**: Launch the Anaconda Prompt.

2.  **Activate the Conda Environment**:
    ```bash
    conda activate llm_finder
    ```

3.  **Run the Application**:
    ```bash
    python src/main.py
    ```

## App Flow

1. **Hardware Detection** — automatically scans your CPU, RAM, and GPU (VRAM)
2. **Preferences** — select your inference app, what you'll use the model for, and optionally a search keyword
3. **Results** — top 10 models ranked for your hardware, in a sortable table

## Features
- **Hardware Detection**: Detects CPU model, cores and threads, total/available RAM, and GPUs:
  - **NVIDIA** — all cards via `nvidia-ml-py` (falls back to `nvidia-smi`); VRAM is summed across multiple GPUs
  - **AMD** — `rocm-smi` or `lspci` on Linux; WMI plus the registry on Windows (WMI alone caps at 4 GB)
  - **Apple Silicon** — unified memory, so the GPU memory pool is system RAM
  - No GPU or unknown VRAM — models are sized against available system RAM
- **Smart Model Search**: Queries Hugging Face Hub for GGUF models by downloads and likes (plus a use-case keyword pass such as "coder" when no keyword is given), filters out non-chat repos (embeddings, detectors, TTS), and reads each model's exact parameter count, architecture and context length from its GGUF metadata
- **Every Quant Considered**: Lists all GGUF files in each repo (combining split files, skipping draft/MTP heads and vision projectors) and picks the one to recommend:
  1. The fastest execution path any file can use: **GPU** → **MoE offload** → **CPU+GPU** → **CPU**
  2. The highest-quality quant on that path, preferring one that leaves memory headroom
- **Memory Estimate**: file size + KV cache (at an 8k-token context) + 0.5 GB runtime overhead
- **Fit Levels**: by how full the memory pool is — **Perfect** (≤60%, GPU only), **Good** (≤85%), **Marginal** (≤98%); anything tighter is dropped
- **Mixture-of-Experts Aware**: MoE models (e.g. `30B-A3B`) that don't fit in VRAM can keep active experts on the GPU and inactive experts in RAM; speed and quality use the *active* parameter count
- **Speed Estimate**: token generation is memory-bandwidth-bound, so tokens/sec ≈ GPU bandwidth ÷ model size × 0.55 for known NVIDIA, AMD and Apple Silicon GPUs, with per-backend constants for others (and for laptop GPUs, whose memory bus differs from the desktop card)
- **Use-Case-Aware Ranking**: Each model is scored 0–100 on four dimensions, weighted by use case:

  | Use case | Quality | Speed | Fit | Context |
  |---|---|---|---|---|
  | General | 45% | 30% | 15% | 10% |
  | Chat, Roleplay / Creative | 40% | 35% | 15% | 10% |
  | Coding | 50% | 20% | 15% | 15% |
  | Reasoning | 55% | 15% | 15% | 15% |
  | Multimodal | 50% | 20% | 15% | 15% |

  Quality combines parameter count, model family, recency, quantization loss and a per-family task benchmark table (coding / reasoning / chat)
- **One Result per Model**: re-uploads of the same model by different quantizers (bartowski, unsloth, lmstudio-community, ...) are collapsed to the best-scoring one
- **Results Table**: rank, model, parameters, quant, file size, memory needed, fit, run mode, estimated speed, context length and score; click a heading to sort, select a row for the score breakdown and notes, double-click to open the model on Hugging Face
- **Copy to Clipboard**: Export the results table as tab-separated text

The fit, speed and scoring model is a Python port of [llmfit](https://github.com/AlexsJones/llmfit) (MIT License), adapted to use real GGUF file sizes from Hugging Face.

Set `HF_TOKEN` in your environment to use your Hugging Face token and avoid anonymous rate limits.

## Supported Quantization Formats

Any GGUF quant in a repo is recognized, including K-quants (Q2_K … Q6_K, with _S/_M/_L/_XL variants), legacy quants (Q4_0, Q5_1, Q8_0), i-quants (IQ1–IQ4), Unsloth dynamic quants (UD-*), MXFP4, F16 and BF16.

## Supported Inference Apps

- Ollama
- LM Studio
- GPT4All
- Llama.cpp app
- Custom

## Requirements

- Python 3.8+
- Internet connection (for Hugging Face Hub search)
- NVIDIA, AMD or Apple Silicon GPU recommended; CPU-only systems are supported (models are sized against system RAM)
