# LLM Model Finder Application Design
**Date:** 2026-04-25  
**Status:** Design Phase

---

## Overview

A desktop GUI application that helps users find and download GGUF-format LLM models from HuggingFace based on their system hardware and use case requirements. The app detects system specs, gathers user preferences, searches HuggingFace for suitable models, ranks them, and provides one-click download/integration with popular inference platforms.

---

## Tech Stack

- **Framework:** Tkinter (Python built-in, cross-platform desktop GUI)
- **Hardware Detection:** `psutil` (CPU/RAM), `GPUtil` or `nvidia-ml-py` (GPU info)
- **Web Scraping:** `requests` + `BeautifulSoup` (HuggingFace Tasks, Apps, Themes)
- **Model Data:** HuggingFace Hub API (`huggingface_hub` library) + scraping
- **HTTP Downloads:** `requests` with streaming for large files
- **Platform Support:** Windows, Linux, macOS

---

## Core Modules

1. **main.py**
   - Tkinter app entry point
   - Window management (geometry, title, icon)
   - Screen navigation (forward/back state management)
   - Session state tracking (user profile, hardware, preferences)

2. **hardware_detector.py**
   - Detect CPU model, core count
   - Detect total/available RAM
   - Detect GPU model and VRAM
   - Fail gracefully if hardware cannot be detected
   - Return standardized hardware dict

3. **hf_scraper.py**
   - Scrape HuggingFace website for: Tasks, Apps, Themes/Use Cases
   - Cache results in-memory for the session
   - Handle scraping errors (return empty list, show warning)

4. **model_finder.py**
   - Search HuggingFace Hub API for GGUF models matching task/app/theme
   - Extract model metadata: parameters, quantization type, file size
   - Calculate estimated VRAM requirement based on parameters + quantization
   - Estimate tokens/sec based on GPU VRAM and quantization
   - Rank models by relevance (algorithm: TBD with user requirements)
   - Return top 10 sorted

5. **ui_screens.py**
   - Modular screen classes: ProfileScreen, HardwareScreen, PreferencesScreen, ResultsScreen
   - Each screen: handles input validation, layout, button callbacks
   - Back button logic: return to previous screen state

6. **app_integrator.py**
   - Detect installed Ollama, LM Studio, GPT4All on system
   - Know model directory paths for each app (OS-specific)
   - Download GGUF file to temp location
   - Copy/integrate into target app directory
   - Provide manual integration instructions if app not found

---

## UI Flow & Screens

### Screen 1: User Profile
**Purpose:** Capture user identity and location (optional for analytics/preferences)

**Elements:**
- Label: "Welcome to LLM Model Finder"
- Input: Text field "Your Name" (required, validate non-empty)
- Input: Text field "Location" (optional)
- Button: "Next" (enabled only if name is filled)
- Button: "Quit"

**Next Screen:** Hardware Detection

---

### Screen 2: Hardware Detection
**Purpose:** Display detected system specs for user awareness and performance estimation

**Elements:**
- Label: "Detecting System Hardware..."
- Display (read-only, labels + values):
  - CPU Model
  - CPU Cores
  - Total RAM (GB)
  - Available RAM (GB)
  - GPU Model (or "No GPU detected")
  - GPU VRAM (GB, or "N/A")
- Status indicator: "✓ Hardware detected" or "⚠ Partial detection"
- Button: "Next" (enabled after detection completes)
- Button: "Back"

**Logic:**
- On screen load, immediately call `hardware_detector.detect_all()` in background thread
- Display results when ready (max 5-second timeout)
- If GPU detection fails, proceed with CPU-only (show "No GPU detected")

**Next Screen:** LLM Preferences

---

### Screen 3: LLM Preferences (Combined)
**Purpose:** Gather user's preferences for model selection

**Elements:**
- Dropdown: "What should the LLM do?" (Task)
  - Populated by scraping HF website
  - Example options: "Text Generation", "Chat", "Code Generation", "Summarization"
- Dropdown: "What app will it be used for?" (App)
  - Populated by scraping HF website
  - Example options: "Ollama", "LM Studio", "GPT4All", "Custom"
- Dropdown: "What theme/use case?" (Theme)
  - Populated by scraping HF website
  - Example options: "Story", "Adult Story", "Coding", "General Chat", "Research"
- Button: "Generate Results"
- Button: "Back"

**Validation:**
- All three dropdowns must have a selection before "Generate Results" is enabled

**Next Screen:** Results Display

---

### Screen 4: Results Display
**Purpose:** Show ranked list of top 10 GGUF models and allow download/integration

**Elements:**
- Label: "Top 10 Models for [Task] / [App] / [Theme]"
- Table with columns:
  - Rank (1-10)
  - Model Name (linked/clickable to HF model page, if feasible)
  - Parameters (e.g., "7B", "13B", "70B")
  - Quantization (e.g., "Q4_K_M", "Q5_K_M", "Q8")
  - File Size (GB)
  - Est. VRAM (GB, based on params + quant + user's GPU)
  - Est. Speed (tokens/sec, based on GPU VRAM)
  - Model Type (e.g., "Mistral", "Llama", "Phi")
- For each row:
  - Button: "Download" — downloads GGUF file, shows progress bar + success/error
  - Button: "Apply to [App]" — if app is Ollama/LM Studio/GPT4All, auto-integrates; if Custom, shows instructions
- Button: "Copy to Clipboard" — copies entire table as formatted text
- Button: "Back" (goes to Preferences)
- Button: "New Search" (returns to Profile screen)

**Download/Integration Logic:**
- Download to user-selected directory (or default temp location)
- On successful download, offer immediate integration if app is detected
- Show status: "Downloaded 2.4 GB / 2.4 GB", then "✓ Ready to integrate"
- If integration fails, show error with manual instructions

---

## Data Sources

### HuggingFace Scraping Targets
1. **Tasks:** Scrape from https://huggingface.co/tasks or hardcode: Text Generation, Chat, Code Generation, Summarization, Translation, etc.
2. **Apps:** Hardcode list of supported inference platforms: Ollama, LM Studio, GPT4All, Custom (these are what we support for auto-integration)
3. **Themes/Use Cases:** Scrape from model library filters or hardcode: Story, Adult Story, Coding, General Chat, Research, Roleplay, etc. (based on common model tags on HF)

### HuggingFace Hub API
- Search endpoint: `/models` with filters (task, library, etc.)
- Model metadata: downloads, likes, last modified, gated status
- Focus on GGUF-format models (filter by filename)

---

## Hardware Detection Details

**CPU Detection:**
- Use `psutil.cpu_count(logical=False)` for physical cores
- Use `platform.processor()` for model name
- Handle case where CPU model is unavailable (show "Unknown")

**RAM Detection:**
- Use `psutil.virtual_memory().total` for total RAM
- Use `psutil.virtual_memory().available` for available RAM

**GPU Detection:**
- **NVIDIA:** Use `nvidia-ml-py` for reliable VRAM and model detection.
- **AMD (Linux):** Use `rocm-smi` command-line tool to get VRAM and model name. Fallback to `lspci` for model name if `rocm-smi` is not available.
- **AMD (Windows):** Use `WMI` to query for video controller details to get VRAM and model name.
- **Fallback:** If no specific GPU is detected, or detection fails, proceed with CPU-only mode (show "No GPU detected", set VRAM to 0 for ranking purposes, which then defaults to an 8GB effective VRAM assumption in `model_finder`).

---

## Model Ranking Algorithm

**Criteria (weighted):**
1. **VRAM Fit** (critical) — Model must fit in user's available VRAM + some buffer (e.g., 20%)
2. **Parameter Count** (high) — Favor models with good balance for task (7B–13B for chat/roleplay, varies by task)
3. **Quantization** (medium) — Prefer higher quality quantization (Q8 > Q5 > Q4) if VRAM allows
4. **Downloads/Likes** (medium) — Community validation; popular models are usually well-tested
5. **File Size** (low) — Smaller is faster to download; secondary consideration

**Output:** Rank 1-10 models by weighted score, exclude those that don't fit VRAM

**Scoring Algorithm:**
```
For each model M:
  1. Check if M.vram_required <= (user_gpu_vram * 0.8)  # 20% safety buffer
     If no: exclude M
  2. Calculate fit_score: 100 if fits, 0 if doesn't
  3. Calculate param_score: based on task (e.g., chat prefers 7B-13B, coding prefers 13B+)
  4. Calculate quant_score: higher quant (Q8) > lower quant (Q4)
  5. Calculate popularity_score: normalize downloads/likes to 0-100
  6. final_score = (fit_score * 0.5) + (param_score * 0.3) + (quant_score * 0.1) + (popularity_score * 0.1)
  
Sort by final_score descending, return top 10
```

Weights are preliminary; adjust based on testing results.

---

## App Integration Paths

### Ollama
- **Detection:** Check for `ollama` command in PATH or `~/.ollama/` directory
- **Model Directory:** `~/.ollama/models/blobs/`
- **Integration:** Copy GGUF file, create model file in `~/.ollama/models/manifests/`
- **Verification:** Run `ollama list` to confirm model appears

### LM Studio
- **Detection:** Check common install paths (Windows: `C:\Users\[user]\AppData\Local\LM Studio`, macOS: `/Applications/LM Studio.app`, Linux: `~/.lm-studio/`)
- **Model Directory:** LM Studio's models folder (varies; usually auto-detected from app settings)
- **Integration:** Copy GGUF file, LM Studio auto-discovers on restart
- **Verification:** Show instructions to restart LM Studio and refresh model list

### GPT4All
- **Detection:** Check common install paths or registry (Windows)
- **Model Directory:** `~/.cache/gpt4all/` or equivalent
- **Integration:** Copy GGUF file; GPT4All auto-discovers
- **Verification:** Show instructions to restart GPT4All

### Fallback (Custom/Manual)
- If app not detected, show: "Manual Integration Required"
- Provide step-by-step instructions for user to move file to correct location
- Option to open file explorer to downloaded file location

---

## Error Handling

1. **Hardware Detection Failure:**
   - If GPU not detected: proceed with CPU-only estimates (show warning)
   - If total detection fails after timeout: allow user to proceed but disable VRAM-based ranking

2. **HuggingFace Scraping Failure:**
   - If dropdowns cannot be populated: show cached/default list or allow manual entry
   - Show warning: "Could not fetch latest options from HuggingFace"

3. **Model Search Failure:**
   - If no models found matching criteria: show message "No models found, try different selections"
   - Offer to broaden search (e.g., ignore theme filter)

4. **Download Failure:**
   - Retry logic: up to 3 retries with exponential backoff
   - Show error with reason (network, storage full, etc.)
   - Allow user to retry or select different model

5. **App Integration Failure:**
   - If app path not found: show manual integration instructions
   - If copy fails: show error and open file explorer to source/destination

---

## Session State Management

**In-Memory State Dict:**
```
{
  "name": str,
  "location": str,
  "hardware": {
    "cpu_model": str,
    "cpu_cores": int,
    "ram_total_gb": float,
    "ram_available_gb": float,
    "gpu_model": str,
    "gpu_vram_gb": float
  },
  "preferences": {
    "task": str,
    "app": str,
    "theme": str
  },
  "results": [list of model dicts],
  "current_screen": int (0-3)
}
```

---

## Future Enhancements (Out of Scope)

- Benchmark/performance comparison UI
- Model testing sandbox within app
- User preferences persistence (JSON config file)
- Update checking for new models on HuggingFace
- Support for non-GGUF formats (safetensors, etc.)

---

## Success Criteria

- [ ] App launches and displays Profile screen
- [ ] Hardware detection works on Windows/Linux/macOS
- [ ] HuggingFace dropdowns populate correctly
- [ ] Model search returns relevant results
- [ ] Top 10 ranking is accurate for user's hardware
- [ ] Download and app integration work for Ollama/LM Studio/GPT4All
- [ ] Back button navigation works correctly
- [ ] No unhandled exceptions; graceful error messages shown to user
