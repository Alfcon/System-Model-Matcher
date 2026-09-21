# src/model_finder.py
"""
Search Hugging Face Hub for GGUF models and rank them for the user's hardware.

Model metadata (exact parameter count, architecture, context length) comes from
the Hub's parsed GGUF header rather than guessing from repo names, and every
quantized file in a repo is evaluated so the best-fitting one is recommended.
Fit, speed and scoring live in fit_engine.py.
"""
import os
import re
from concurrent.futures import ThreadPoolExecutor

import requests

import fit_engine
from fit_engine import (DEFAULT_ESTIMATION_CTX, GENERAL, CODING, REASONING, MULTIMODAL, ROLEPLAY,
                        TOO_TIGHT, detect_quant, estimate_memory_gb, params_from_name)

HF_API = "https://huggingface.co/api"
REQUEST_TIMEOUT = 15
MAX_WORKERS = 8

# Nominal GB per 1B parameters for each quant; used only when a file size is unknown.
QUANT_VRAM_MULTIPLIER = {
    "Q2_K": 0.37,
    "Q3_K_S": 0.44,
    "Q3_K_M": 0.48,
    "Q3_K_L": 0.52,
    "Q4_0": 0.58,
    "Q4_K_S": 0.57,
    "Q4_K_M": 0.58,
    "Q5_K_S": 0.66,
    "Q5_K_M": 0.68,
    "Q6_K": 0.80,
    "Q8_0": 1.05,
}

# Extra search keyword per use case, to widen the candidate pool beyond the
# most-downloaded generalist models when the user leaves the keyword blank.
USE_CASE_SEARCH_HINTS = {
    CODING: "coder",
    REASONING: "R1",
    MULTIMODAL: "VL",
    ROLEPLAY: "roleplay",
}

# Only chat / text-generation repos are recommended (untagged repos are allowed).
_ALLOWED_PIPELINES = {
    None, "text-generation", "text2text-generation", "conversational", "image-text-to-text",
    "image-to-text", "any-to-any",
}
_EXCLUDED_NAME_HINTS = ("embed", "rerank", "bge-", "whisper", "-tts", "stable-diffusion", "flux")
_SPLIT_SUFFIX = re.compile(r"-\d{5}-of-\d{5}(?=\.gguf$)", re.IGNORECASE)
# Speculative-decoding draft / multi-token-prediction heads, not runnable models.
_DRAFT_FILE = re.compile(r"(^|[-_./])(mtp|draft|eagle\d*|dflash)([-_./]|$)", re.IGNORECASE)


def _headers():
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    return {"Authorization": f"Bearer {token}"} if token else {}


def estimate_vram_requirement(params_billions, quant="Q4_K_M", ctx=DEFAULT_ESTIMATION_CTX):
    """
    Estimate memory in GB to run a model when its file size is unknown:
    weights (params * bytes/param) + KV cache + runtime overhead.
    """
    multiplier = QUANT_VRAM_MULTIPLIER.get(quant, 0.58)
    return round(estimate_memory_gb(params_billions * multiplier, params_billions, ctx), 2)


def extract_params_from_name(model_name):
    """Extract parameter count from model name (e.g. 'mistral-7b' -> 7.0). Defaults to 7."""
    params = params_from_name(model_name)
    return params if params else 7


def search_gguf_models(task="text-generation", limit=50, sort="downloads", user_vram_gb=None):
    """
    Search Hugging Face Hub for GGUF repos. `task` is a free-text search keyword
    (blank or 'text-generation' means no keyword filter).
    Returns a list of candidate dicts with repo metadata; files are fetched later.
    `user_vram_gb` is accepted for backwards compatibility and ignored.
    """
    params = {
        "filter": "gguf",
        "sort": sort,
        "direction": -1,
        "limit": limit,
        "expand[]": ["gguf", "downloads", "likes", "pipeline_tag", "createdAt", "lastModified"],
    }
    if task and task != "text-generation":
        params["search"] = task

    try:
        response = requests.get(f"{HF_API}/models", params=params, headers=_headers(), timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        entries = response.json()
    except Exception as e:
        print(f"Warning: Could not search HF Hub: {e}")
        return []

    candidates = []
    for entry in entries:
        candidate = _candidate_from_entry(entry)
        if candidate:
            candidates.append(candidate)
    return candidates


def _candidate_from_entry(entry):
    model_id = entry.get("id") or entry.get("modelId")
    if not model_id:
        return None
    pipeline = entry.get("pipeline_tag")
    lower = model_id.lower()
    if pipeline not in _ALLOWED_PIPELINES or any(h in lower for h in _EXCLUDED_NAME_HINTS):
        return None

    gguf = entry.get("gguf") or {}
    total_params = gguf.get("total")
    name_params = params_from_name(model_id)
    params_b = round(total_params / 1e9, 2) if total_params else name_params
    # The Hub parses one file per repo; if that was a draft/MTP head its count
    # is far off the size in the repo name, so trust the name instead.
    if total_params and name_params and not (name_params / 2 <= params_b <= name_params * 2):
        params_b = name_params

    return {
        "model_name": model_id,
        "downloads": entry.get("downloads") or 0,
        "likes": entry.get("likes") or 0,
        "created_at": entry.get("createdAt"),
        "last_modified": str(entry.get("lastModified") or ""),
        "pipeline_tag": pipeline,
        "architecture": gguf.get("architecture"),
        "context_length": gguf.get("context_length"),
        "params_b": params_b,
    }


def list_gguf_files(model_id):
    """
    List the quantized GGUF files in a repo with their sizes. Split files
    (model-Q8_0-00001-of-00002.gguf) are combined into one entry.
    """
    try:
        response = requests.get(f"{HF_API}/models/{model_id}/tree/main", params={"recursive": "true"},
                                headers=_headers(), timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        tree = response.json()
    except Exception:
        return []

    grouped = {}
    for item in tree:
        path = item.get("path", "")
        if item.get("type") != "file" or not path.lower().endswith(".gguf"):
            continue
        name_lower = os.path.basename(path).lower()
        if "mmproj" in name_lower or "imatrix" in name_lower or _DRAFT_FILE.search(path):
            continue
        quant = detect_quant(path)
        if not quant:
            continue
        key = _SPLIT_SUFFIX.sub("", path)
        size = item.get("size") or (item.get("lfs") or {}).get("size") or 0
        entry = grouped.setdefault(key, {"quant": quant, "file_name": key, "single": 0, "split": 0, "parts": 0,
                                         "first_shard": None})
        if key == path:
            entry["single"] = size
        else:
            entry["split"] += size
            entry["parts"] += 1
            if entry["first_shard"] is None or path < entry["first_shard"]:
                entry["first_shard"] = path

    files = []
    for entry in grouped.values():
        # Some repos ship the same quant both whole and split; count it once.
        single, split = entry.pop("single"), entry.pop("split")
        first_shard = entry.pop("first_shard")
        size_bytes = single or split
        if size_bytes <= 0:
            continue
        if single:
            entry["parts"] = 1
        # Path of the file to load: the whole file, or the first shard of a split set.
        entry["file_path"] = entry["file_name"] if single else first_shard
        entry["size_bytes"] = size_bytes
        entry["size_gb"] = size_bytes / (1024**3)
        files.append(entry)
    return files


def canonical_model_name(model_id):
    """
    Collapse re-uploads of the same model by different quantizers
    ('bartowski/Qwen_Qwen3-8B-GGUF', 'unsloth/Qwen3-8B-GGUF') to one key.
    """
    name = model_id.split("/")[-1].lower()
    name = re.sub(r"[-_.]?gguf$", "", name)
    name = re.sub(r"[-_]i1$", "", name)
    if "_" in name:
        prefix, rest = name.split("_", 1)
        if rest and prefix.isalpha():  # 'qwen_qwen3-8b' -> 'qwen3-8b'
            name = rest
    name = name.replace("_", "-")
    name = re.sub(r"^meta-(?=llama)", "", name)
    return name


def gather_candidates(search_param="", use_case=GENERAL, per_sort_limit=30):
    """Run the Hub searches for a request and return de-duplicated candidates."""
    queries = [(search_param, "downloads", per_sort_limit), (search_param, "likes", per_sort_limit)]
    hint = USE_CASE_SEARCH_HINTS.get(use_case)
    if hint and not search_param:
        queries.append((hint, "downloads", per_sort_limit // 2))
        queries.append((hint, "likes", per_sort_limit // 2))

    seen = set()
    candidates = []
    for query, sort, limit in queries:
        for model in search_gguf_models(task=query, limit=limit, sort=sort):
            if model["model_name"] not in seen:
                seen.add(model["model_name"])
                candidates.append(model)
    return candidates


def evaluate_candidates(candidates, hardware, use_case=GENERAL, progress_callback=None):
    """
    Fetch each repo's GGUF files and pick the best-fitting file for the hardware.
    Returns a list of evaluated model dicts (models that cannot run are dropped).
    """
    evaluated = []
    total = len(candidates)

    def evaluate(model):
        if not model.get("params_b"):
            return None
        files = list_gguf_files(model["model_name"])
        if not files:
            return None
        best = fit_engine.best_file_for_model(model, files, hardware, use_case)
        if best is None:
            return None
        return {**model, **best}

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        for done, result in enumerate(pool.map(evaluate, candidates), 1):
            if progress_callback:
                progress_callback(done, total, result)
            if result is not None:
                evaluated.append(result)
    return evaluated


def rank_models(models, user_vram_gb=None, task="text-generation", app_name="", use_case=GENERAL,
                hardware=None, top_n=10):
    """
    Rank evaluated models by composite score (quality, speed, fit, context) and
    return the top N, keeping only the best-scoring upload of each model.

    Models that have not been through evaluate_candidates() (no 'final_score')
    are scored here from their params/quant using estimated memory.
    """
    if hardware is None:
        hardware = _hardware_from_vram(user_vram_gb)

    scored = []
    for model in models:
        if "final_score" not in model:
            model = _score_without_files(model, hardware, use_case)
            if model is None:
                continue
        if model.get("fit_level") == TOO_TIGHT:
            continue
        scored.append(model)

    scored.sort(key=lambda m: (m["final_score"], fit_engine.FIT_RANK[m["fit_level"]],
                               m.get("downloads", 0) + m.get("likes", 0)), reverse=True)

    unique = []
    seen = set()
    for model in scored:
        key = canonical_model_name(model["model_name"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(model)
    return unique[:top_n]


def _hardware_from_vram(user_vram_gb):
    """Minimal hardware dict for callers that only know VRAM."""
    vram = float(user_vram_gb or 0)
    return {
        "cpu": {"cores": 8, "threads": 8},
        "ram": {"total_gb": max(vram * 2, 16.0), "available_gb": max(vram * 2, 16.0)},
        "gpu": {"model": "", "vram_gb": vram, "backend": "cuda" if vram > 0 else "cpu_x86"},
    }


def _score_without_files(model, hardware, use_case):
    params_b = model.get("params_b") or 7
    quant = model.get("quant", "Q4_K_M")
    size_gb = model.get("file_size_gb") or params_b * QUANT_VRAM_MULTIPLIER.get(quant, 0.58)
    base = {"model_name": model.get("model_name", "unknown"), "params_b": params_b,
            "context_length": model.get("context_length"), "architecture": model.get("architecture"),
            "created_at": model.get("created_at"), "pipeline_tag": model.get("pipeline_tag"),
            "downloads": model.get("downloads"), "likes": model.get("likes")}
    result = fit_engine.analyze_file(base, {"quant": quant, "size_gb": size_gb}, hardware, use_case)
    return {**model, **result}
