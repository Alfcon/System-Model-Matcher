# LLM Model Finder

A desktop application to find and download GGUF-format LLM models from HuggingFace based on your system hardware and use case.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python src/main.py
```

## Features

- **Hardware Detection**: Automatically detects your CPU, RAM, and GPU (NVIDIA)
- **Smart Model Search**: Searches HuggingFace for GGUF models matching your task/app/theme
- **Hardware-Aware Ranking**: Ranks models by suitability for your system
- **One-Click Download**: Download models directly from the app
- **App Integration**: Auto-integrates with Ollama, LM Studio, or GPT4All

## Supported Apps

- Ollama
- LM Studio
- GPT4All
- Custom (manual integration)

## Requirements

- Python 3.8+
- 100MB free disk space (for app)
- Internet connection (for downloading models)
