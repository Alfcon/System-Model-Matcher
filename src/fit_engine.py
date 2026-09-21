# src/fit_engine.py
"""
Hardware fit, speed and scoring model for GGUF LLMs.

Ported from llmfit (https://github.com/AlexsJones/llmfit, MIT License,
Copyright (c) 2026 Alex Jones), adapted to use the real GGUF file sizes
reported by Hugging Face instead of estimating weights from parameter counts.

Pipeline for a single GGUF file:
  1. memory required  = weights (file size) + KV cache + runtime overhead
  2. run mode         = GPU / MoE offload / CPU+GPU / CPU, whichever pool it fits
  3. fit level        = Perfect / Good / Marginal / Too Tight from pool utilization
  4. est. tokens/sec  = memory-bandwidth roofline, or per-backend constant fallback
  5. composite score  = Quality, Speed, Fit, Context weighted per use case
"""
import json
import math
import os
import re
from datetime import datetime, timezone

# ── Constants (llmfit defaults) ────────────────────────────────────────────

# Most runtimes (llama.cpp, Ollama, LM Studio) default to a much smaller context
# than a model's advertised maximum, so KV cache is estimated at this cap.
DEFAULT_ESTIMATION_CTX = 8192
RUNTIME_OVERHEAD_GB = 0.5  # CUDA/Metal context, scratch buffers
BANDWIDTH_EFFICIENCY = 0.55  # Kernel overhead, KV reads, memory controller effects
DEFAULT_DDR_BANDWIDTH_GBPS = 50.0  # DDR4-3200 dual channel
MOE_DEFAULT_OVERHEAD = 0.60  # Expert count unknown from HF metadata

# Pool utilization bands for the fit verdict.
FIT_PERFECT_MAX_RATIO = 0.60
FIT_GOOD_MAX_RATIO = 0.85
FIT_MARGINAL_MAX_RATIO = 0.98

PERFECT, GOOD, MARGINAL, TOO_TIGHT = "Perfect", "Good", "Marginal", "Too Tight"
FIT_RANK = {PERFECT: 3, GOOD: 2, MARGINAL: 1, TOO_TIGHT: 0}

GPU, MOE_OFFLOAD, CPU_OFFLOAD, CPU_ONLY = "GPU", "MoE offload", "CPU+GPU", "CPU"

RUN_MODE_FACTORS = {GPU: 1.0, MOE_OFFLOAD: 0.8, CPU_OFFLOAD: 0.5, CPU_ONLY: 0.3}

# Per-backend throughput constants used when the GPU's bandwidth is unknown.
BACKEND_SPEED_K = {
    "cuda": 220.0,
    "metal": 160.0,
    "rocm": 180.0,
    "vulkan": 150.0,
    "sycl": 100.0,
    "cpu_arm": 90.0,
    "cpu_x86": 70.0,
}

# ── Use cases ──────────────────────────────────────────────────────────────

GENERAL, CHAT, CODING, REASONING, MULTIMODAL = "General", "Chat", "Coding", "Reasoning", "Multimodal"
ROLEPLAY = "Roleplay / Creative"

USE_CASES = [GENERAL, CHAT, ROLEPLAY, CODING, REASONING, MULTIMODAL]

# (quality, speed, fit, context) weights
SCORING_WEIGHTS = {
    GENERAL: (0.45, 0.30, 0.15, 0.10),
    CODING: (0.50, 0.20, 0.15, 0.15),
    REASONING: (0.55, 0.15, 0.15, 0.15),
    CHAT: (0.40, 0.35, 0.15, 0.10),
    MULTIMODAL: (0.50, 0.20, 0.15, 0.15),
}

SPEED_TARGET_TPS = {GENERAL: 40.0, CODING: 40.0, CHAT: 40.0, MULTIMODAL: 40.0, REASONING: 25.0}
CONTEXT_TARGET = {GENERAL: 4096, CHAT: 4096, MULTIMODAL: 4096, CODING: 8192, REASONING: 8192}


def scoring_category(use_case):
    """Map a UI use case onto one of llmfit's scoring categories."""
    if use_case == ROLEPLAY:
        return CHAT
    return use_case if use_case in SCORING_WEIGHTS else GENERAL


# ── Quantization ───────────────────────────────────────────────────────────

# class -> (nominal bytes/param for the bandwidth roofline, quality penalty, speed multiplier)
_QUANT_CLASSES = {
    "F32": (4.0, 0.0, 0.4),
    "F16": (2.0, 0.0, 0.6),
    "Q8": (1.0, 0.0, 0.8),
    "Q6": (0.75, -1.0, 0.95),
    "Q5": (0.625, -2.0, 1.0),
    "Q4": (0.5, -5.0, 1.15),
    "MXFP4": (0.53, 0.0, 1.15),
    "Q3": (0.375, -8.0, 1.25),
    "Q2": (0.25, -12.0, 1.35),
    "TQ": (0.40, -6.0, 1.3),
    "Q1": (0.20, -18.0, 1.4),
}

# Fine-grained quality adjustments within a class (K_S is a little worse than K_M,
# K_L / K_XL a little better, i-quant XXS/XS a little worse again).
_VARIANT_ADJUST = {
    "_K_XL": 0.5, "_K_L": 0.5, "_K_M": 0.0, "_K_S": -0.5, "_K": 0.0,
    "_XXS": -1.5, "_XS": -1.0, "_S": -0.5, "_M": 0.0, "_NL": 0.0,
    "_0": -0.5, "_1": -0.25,
}

QUANT_PATTERN = re.compile(
    r"(?:^|[-_.\s/])((?:UD-)?(?:IQ\d_[A-Z]+|Q\d_K_[A-Z]+|Q\d_K|Q\d_\d|BF16|F16|FP16|F32|MXFP4|TQ\d_\d))(?=[-_.\s/]|$)",
    re.IGNORECASE,
)


def detect_quant(filename):
    """Return the quant label in a GGUF filename (e.g. 'Q4_K_M', 'UD-Q4_K_XL'), or None."""
    base = os.path.basename(filename)
    matches = QUANT_PATTERN.findall(base) or QUANT_PATTERN.findall(filename)
    if not matches:
        return None
    label = matches[-1].upper()
    return "F16" if label == "FP16" else label


def quant_class(quant):
    """Collapse a quant label to its precision class ('Q4_K_M' -> 'Q4')."""
    if not quant:
        return "Q4"
    q = quant.upper()
    if q.startswith("UD-"):
        q = q[3:]
    if q in ("F32",):
        return "F32"
    if q in ("F16", "BF16", "FP16"):
        return "F16"
    if q == "MXFP4":
        return "MXFP4"
    if q.startswith("TQ"):
        return "TQ"
    m = re.match(r"I?Q(\d)", q)
    if m:
        bits = int(m.group(1))
        if bits >= 8:
            return "Q8"
        if bits <= 1:
            return "Q1"
        return "Q%d" % bits
    return "Q4"


def quant_bytes_per_param(quant):
    return _QUANT_CLASSES[quant_class(quant)][0]


def quant_speed_multiplier(quant):
    return _QUANT_CLASSES[quant_class(quant)][2]


def quant_quality_penalty(quant):
    cls = quant_class(quant)
    penalty = _QUANT_CLASSES[cls][1]
    if cls in ("F32", "F16", "Q8", "MXFP4"):
        return penalty  # effectively lossless; ties resolve to the smaller file
    q = (quant or "").upper()
    if q.startswith("IQ") or q.startswith("UD-IQ"):
        penalty += 0.5  # importance-matrix quants hold up better than plain k-quants
    for suffix, adjust in _VARIANT_ADJUST.items():
        if q.endswith(suffix):
            return penalty + adjust
    return penalty


# ── GPU bandwidth lookup (from llmfit hardware.rs) ─────────────────────────

# Ordered: more specific names must come before their prefixes ("4070 ti" before "4070").
_GPU_BANDWIDTH_GBPS = [
    # NVIDIA RTX 50
    ("5090", 1792), ("5080", 960), ("5070 ti", 896), ("5070", 672), ("5060 ti", 448), ("5060", 256),
    # NVIDIA RTX 40
    ("4090", 1008), ("4080 super", 736), ("4080", 717), ("4070 ti super", 672), ("4070 ti", 504),
    ("4070 super", 504), ("4070", 504), ("4060 ti", 288), ("4060", 272),
    # NVIDIA RTX 30
    ("3090 ti", 1008), ("3090", 936), ("3080 ti", 912), ("3080", 760), ("3070 ti", 608),
    ("3070", 448), ("3060 ti", 448), ("3060", 360),
    # NVIDIA RTX 20 / GTX 16
    ("2080 ti", 616), ("2080 super", 496), ("2080", 448), ("2070 super", 448), ("2070", 448),
    ("2060 super", 448), ("2060", 336), ("1660 ti", 288), ("1660 super", 336), ("1660", 192),
    ("1650 super", 192), ("1650", 128),
    # NVIDIA data center / workstation
    ("h100 sxm", 3350), ("h100", 2039), ("h200", 4800), ("a100 sxm", 2039), ("a100", 1555),
    ("l40s", 864), ("l40", 864), ("l4", 300), ("a10g", 600), ("a10", 600), ("t4", 320),
    ("v100 sxm", 900), ("v100", 897), ("a6000", 768), ("a5000", 768), ("a4000", 448),
    # AMD Strix Halo APUs
    ("8060s", 256), ("8050s", 256), ("strix halo", 256), ("ryzen ai max", 256),
    # AMD RDNA 4 / 3 / 2
    ("9070 xt", 624), ("9070", 488), ("7900 xtx", 960), ("7900 xt", 800), ("7900 gre", 576),
    ("7800 xt", 624), ("7700 xt", 432), ("7600", 288), ("6950 xt", 576), ("6900 xt", 512),
    ("6800 xt", 512), ("6800", 512), ("6700 xt", 384), ("6600 xt", 256), ("6600", 224),
    # AMD CDNA
    ("mi300x", 5300), ("mi300", 5300), ("mi250x", 3277), ("mi250", 3277), ("mi210", 1638), ("mi100", 1229),
    # Apple Silicon
    ("m5 max", 614), ("m5 pro", 307), ("m5", 153.6), ("m4 ultra", 819), ("m4 max", 546),
    ("m4 pro", 273), ("m4", 120), ("m3 ultra", 800), ("m3 max", 400), ("m3 pro", 150), ("m3", 100),
    ("m2 ultra", 800), ("m2 max", 400), ("m2 pro", 200), ("m2", 100), ("m1 ultra", 800),
    ("m1 max", 400), ("m1 pro", 200), ("m1", 68),
]


def gpu_memory_bandwidth_gbps(name):
    """
    GPU memory bandwidth in GB/s from its model name, or None if unknown.

    Laptop parts share model numbers with desktop cards but not their memory bus
    (an RTX 5070 Laptop is 128-bit, the desktop card 192-bit), so they return
    None and the caller falls back to the per-backend constant.
    """
    if not name:
        return None
    lower = name.lower()
    if "laptop" in lower or "mobile" in lower or "max-q" in lower:
        return None
    for key, bandwidth in _GPU_BANDWIDTH_GBPS:
        if key in lower:
            # Apple chip names are short; require a word boundary so "m1" does not match "rm10".
            if key[0] == "m" and key[1].isdigit() and not re.search(r"\b" + re.escape(key) + r"\b", lower):
                continue
            return float(bandwidth)
    return None


# ── Model metadata helpers ────────────────────────────────────────────────

_ACTIVE_PARAMS_PATTERN = re.compile(r"(?:^|[-_.])a(\d+(?:\.\d+)?)b(?=[-_.]|$)", re.IGNORECASE)
_MIXTRAL_PATTERN = re.compile(r"(\d+)x(\d+(?:\.\d+)?)b", re.IGNORECASE)
_SIZE_PATTERN = re.compile(r"(?:^|[-_.\s/])(\d+(?:\.\d+)?)\s*([bm])(?=[-_.\s]|$)", re.IGNORECASE)


def params_from_name(model_name):
    """Parameter count in billions parsed from a repo name ('Mistral-7B' -> 7.0), or None."""
    name = model_name.split("/")[-1]
    moe = _MIXTRAL_PATTERN.search(name)
    if moe:
        # Mixtral-8x7B is ~46.7B total: experts share attention weights.
        return round(int(moe.group(1)) * float(moe.group(2)) * 0.83, 1)
    for value, unit in _SIZE_PATTERN.findall(name):
        # Skip the active-parameter marker in names like "30B-A3B".
        if re.search(r"(?:^|[-_.])a" + re.escape(value) + unit + r"(?=[-_.]|$)", name, re.IGNORECASE):
            continue
        number = float(value)
        return number / 1000.0 if unit.lower() == "m" else number
    return None


def moe_info(model_name, architecture=None, params_b=None):
    """
    Detect Mixture-of-Experts models and their active parameter count.
    Returns (is_moe, active_params_b or None).
    """
    name = model_name.split("/")[-1]
    arch = (architecture or "").lower()
    active = _ACTIVE_PARAMS_PATTERN.search(name)
    if active:
        return True, float(active.group(1))
    mixtral = _MIXTRAL_PATTERN.search(name)
    if mixtral:
        # Two of N experts active per token, plus shared attention.
        per_expert = float(mixtral.group(2))
        return True, round(per_expert * 2 * 0.92, 1)
    if "moe" in arch or arch in ("mixtral", "gpt_oss", "gpt-oss", "deepseek2", "qwen3next", "llama4"):
        if "gpt-oss-20b" in name.lower():
            return True, 3.6
        if "gpt-oss-120b" in name.lower():
            return True, 5.1
        return True, None
    return False, None


# ── Memory & fit ───────────────────────────────────────────────────────────

def kv_cache_gb(params_b, ctx):
    """Coarse fp16 KV cache estimate (llmfit fallback formula)."""
    return 0.000008 * params_b * ctx


def estimate_memory_gb(weights_gb, params_b, ctx=DEFAULT_ESTIMATION_CTX):
    """Memory needed to run a model: weights + KV cache + runtime overhead."""
    return weights_gb + kv_cache_gb(params_b, ctx) + RUNTIME_OVERHEAD_GB


def fit_level(mem_required, mem_available, run_mode=GPU):
    """
    Verdict from how full the run mode's memory pool is. Only fully GPU-resident
    runs can be Perfect; offload and CPU paths cap at Good.
    """
    if mem_available <= 0:
        return TOO_TIGHT
    ratio = mem_required / mem_available
    if not math.isfinite(ratio) or ratio > FIT_MARGINAL_MAX_RATIO:
        level = TOO_TIGHT
    elif ratio <= FIT_PERFECT_MAX_RATIO:
        level = PERFECT
    elif ratio <= FIT_GOOD_MAX_RATIO:
        level = GOOD
    else:
        level = MARGINAL
    if level == PERFECT and run_mode != GPU:
        level = GOOD
    return level


def _pools(hardware):
    gpu = hardware.get("gpu", {}) or {}
    ram = hardware.get("ram", {}) or {}
    vram = float(gpu.get("vram_gb") or 0.0)
    ram_available = float(ram.get("available_gb") or 0.0)
    unified = bool(gpu.get("unified_memory"))
    return vram, ram_available, unified


def choose_run_mode(mem_required, weights_gb, params_b, active_params_b, is_moe, hardware,
                    ctx=DEFAULT_ESTIMATION_CTX):
    """
    Pick the best execution path for a model and return
    (run_mode, memory_required_gb, memory_available_gb, notes).
    """
    vram, ram_available, unified = _pools(hardware)
    notes = []

    if vram > 0 and unified:
        notes.append("Unified memory: GPU and CPU share the same pool")
        return GPU, mem_required, vram, notes

    if vram > 0:
        if mem_required <= vram * FIT_MARGINAL_MAX_RATIO:
            notes.append("Fits entirely in VRAM")
            return GPU, mem_required, vram, notes

        if is_moe and active_params_b and params_b > active_params_b:
            bytes_per_param = weights_gb / params_b if params_b > 0 else 0.6
            active_vram = max(active_params_b * bytes_per_param * 1.1, 0.5)
            active_vram += kv_cache_gb(active_params_b, ctx) + RUNTIME_OVERHEAD_GB
            offloaded = (params_b - active_params_b) * bytes_per_param
            if active_vram <= vram * FIT_MARGINAL_MAX_RATIO and offloaded <= ram_available * FIT_MARGINAL_MAX_RATIO:
                notes.append("MoE: active experts in VRAM (%.1f GB), inactive experts in RAM (%.1f GB)"
                             % (active_vram, offloaded))
                return MOE_OFFLOAD, active_vram, vram, notes

        notes.append("Insufficient VRAM: spills to system RAM, performance significantly reduced")
        return CPU_OFFLOAD, mem_required, ram_available, notes

    notes.append("No GPU: runs from system RAM, inference will be slow")
    return CPU_ONLY, mem_required, ram_available, notes


# ── Speed ──────────────────────────────────────────────────────────────────

def _backend(hardware):
    gpu = hardware.get("gpu", {}) or {}
    backend = gpu.get("backend")
    if backend:
        return backend
    return "cpu_arm" if "arm" in (hardware.get("cpu", {}).get("arch") or "").lower() else "cpu_x86"


def estimate_tps(params_b, quant, run_mode, hardware, is_moe=False, active_params_b=None,
                 weights_gb=None):
    """
    Estimated decode tokens/sec. Decode is memory-bandwidth-bound: every token
    reads the (active) weights once, so tps ~= bandwidth / model_size * efficiency.
    """
    speed_params = max(active_params_b if (is_moe and active_params_b) else params_b, 0.1)
    gpu = hardware.get("gpu", {}) or {}
    cpu_threads = (hardware.get("cpu", {}) or {}).get("threads") or (hardware.get("cpu", {}) or {}).get("cores") or 0
    bytes_pp = quant_bytes_per_param(quant)
    mode_factor = RUN_MODE_FACTORS.get(run_mode, 1.0)
    bandwidth = gpu_memory_bandwidth_gbps(gpu.get("model")) if run_mode != CPU_ONLY else None

    if bandwidth:
        active_gb = speed_params * bytes_pp
        if is_moe and run_mode == MOE_OFFLOAD:
            expert_read = active_gb / DEFAULT_DDR_BANDWIDTH_GBPS
            gpu_compute = active_gb / (bandwidth * BANDWIDTH_EFFICIENCY)
            return max(1.0 / (expert_read + gpu_compute) * mode_factor, 0.1)
        if is_moe and run_mode == GPU:
            return max(bandwidth / active_gb * BANDWIDTH_EFFICIENCY * MOE_DEFAULT_OVERHEAD * mode_factor, 0.1)
        return max(bandwidth / active_gb * BANDWIDTH_EFFICIENCY * mode_factor, 0.1)

    # Fallback: per-backend constant.
    backend = _backend(hardware)
    k = BACKEND_SPEED_K.get(backend, 70.0)
    if run_mode == CPU_ONLY:
        k = 90.0 if backend == "cpu_arm" or "arm" in (hardware.get("cpu", {}).get("arch") or "").lower() else 70.0
    thread_bonus = 1.1 if cpu_threads >= 8 else 1.0

    if run_mode == MOE_OFFLOAD:
        est_gpu_bw = k * bytes_pp / BANDWIDTH_EFFICIENCY
        active_gb = speed_params * bytes_pp
        expert_read = active_gb / DEFAULT_DDR_BANDWIDTH_GBPS
        gpu_compute = active_gb / (est_gpu_bw * BANDWIDTH_EFFICIENCY)
        return max(1.0 / (expert_read + gpu_compute), 0.1) * thread_bonus

    tps = k / speed_params * quant_speed_multiplier(quant) * thread_bonus
    return max(tps * mode_factor, 0.1)


# ── Scores ─────────────────────────────────────────────────────────────────

_BENCH_TABLE = None


def _bench_table():
    global _BENCH_TABLE
    if _BENCH_TABLE is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "use_case_benchmarks.json")
        try:
            with open(path, "r", encoding="utf-8") as f:
                _BENCH_TABLE = json.load(f).get("families", [])
        except (OSError, ValueError):
            _BENCH_TABLE = []
    return _BENCH_TABLE


def task_benchmark_score(name_lower, task):
    """Curated per-family task score (0-100); the longest matching pattern wins."""
    best = None
    for entry in _bench_table():
        score = entry.get("scores", {}).get(task)
        if score is None:
            continue
        for pattern in entry.get("match", []):
            if pattern in name_lower and (best is None or len(pattern) > best[0]):
                best = (len(pattern), float(score))
    return best[1] if best else None


def _months_since(iso_date, now=None):
    if not iso_date:
        return None
    try:
        created = datetime.strptime(str(iso_date)[:10], "%Y-%m-%d")
    except ValueError:
        return None
    now = now or datetime.now(timezone.utc).replace(tzinfo=None)
    return max((now.year - created.year) * 12 + (now.month - created.month), 0)


_ROLEPLAY_HINTS = ("roleplay", "-rp", "_rp", "story", "creative", "writer", "mythomax", "stheno",
                   "lumimaid", "magnum", "nemomix", "cydonia", "rocinante")
_VISION_HINTS = ("vision", "-vl", "_vl", "llava", "pixtral", "minicpm-v", "moondream")


def quality_score(model_name, params_b, quant, use_case, active_params_b=None, created_at=None,
                  pipeline_tag=None, now=None):
    """Base quality from (active) parameter count + family + recency + quant + task alignment."""
    name_lower = model_name.lower()
    quality_params = active_params_b or params_b

    if quality_params < 1:
        base = 30.0
    elif quality_params < 3:
        base = 45.0
    elif quality_params < 7:
        base = 60.0
    elif quality_params < 10:
        base = 75.0
    elif quality_params < 20:
        base = 82.0
    elif quality_params < 40:
        base = 89.0
    else:
        base = 95.0

    if "deepseek" in name_lower:
        family = 3.0
    elif "qwen" in name_lower or "llama" in name_lower:
        family = 2.0
    elif any(f in name_lower for f in ("mistral", "mixtral", "gemma", "starcoder")):
        family = 1.0
    else:
        family = 0.0

    # Same-size models improve over time; the repo creation date is a proxy for release.
    months = _months_since(created_at, now)
    recency = 0.0 if months is None else (3.0 if months < 3 else 1.5 if months < 9 else 0.0)

    category = scoring_category(use_case)
    task = {CODING: "coding", REASONING: "reasoning", CHAT: "chat"}.get(category)
    bench = task_benchmark_score(name_lower, task) if task else None
    if bench is not None:
        task_bump = max(-8.0, min(9.0, (bench - 72.0) * 0.4))
    elif category == CODING:
        task_bump = 6.0 if any(h in name_lower for h in ("code", "coder", "starcoder", "wizard")) else 0.0
    elif category == REASONING:
        task_bump = 5.0 if params_b >= 13 or any(h in name_lower for h in ("-r1", "qwq", "reason", "think")) else 0.0
    elif category == MULTIMODAL:
        is_vision = pipeline_tag == "image-text-to-text" or any(h in name_lower for h in _VISION_HINTS)
        task_bump = 6.0 if is_vision else 0.0
    else:
        task_bump = 0.0
    if use_case == ROLEPLAY and any(h in name_lower for h in _ROLEPLAY_HINTS):
        task_bump += 4.0

    total = base + family + recency + quant_quality_penalty(quant) + task_bump
    return max(0.0, min(100.0, total))


def speed_score(tps, use_case):
    target = SPEED_TARGET_TPS[scoring_category(use_case)]
    return max(0.0, min(100.0, tps / target * 100.0))


def fit_score(required, available):
    """Flat 100 up to 70% utilization, then a one-sided Gaussian falloff."""
    if available <= 0 or required > available:
        return 0.0
    z = max((required / available - 0.70) / 0.20, 0.0)
    return max(0.0, min(100.0, 100.0 * math.exp(-0.5 * z * z)))


def context_score(context_length, use_case):
    target = CONTEXT_TARGET[scoring_category(use_case)]
    context_length = context_length or 4096
    if context_length >= target:
        return 100.0
    if context_length >= target // 2:
        return 70.0
    return 30.0


def weighted_score(quality, speed, fit, context, use_case):
    wq, ws, wf, wc = SCORING_WEIGHTS[scoring_category(use_case)]
    return round(quality * wq + speed * ws + fit * wf + context * wc, 1)


# ── Whole-file analysis ────────────────────────────────────────────────────

def analyze_file(model, gguf_file, hardware, use_case=GENERAL):
    """
    Evaluate one GGUF file of a model against the hardware.

    model:     dict with model_name, params_b, context_length, architecture,
               created_at, pipeline_tag
    gguf_file: dict with quant and size_gb
    Returns a dict with memory, run mode, fit level, speed and scores.
    """
    params_b = model["params_b"]
    quant = gguf_file["quant"]
    weights_gb = gguf_file["size_gb"]
    context_length = model.get("context_length") or 4096
    ctx = min(context_length, DEFAULT_ESTIMATION_CTX)
    is_moe, active_params_b = moe_info(model["model_name"], model.get("architecture"), params_b)

    mem_required = estimate_memory_gb(weights_gb, params_b, ctx)
    run_mode, mem_used, mem_available, notes = choose_run_mode(
        mem_required, weights_gb, params_b, active_params_b, is_moe, hardware, ctx)
    level = fit_level(mem_used, mem_available, run_mode)
    tps = estimate_tps(params_b, quant, run_mode, hardware, is_moe, active_params_b, weights_gb)

    quality = quality_score(model["model_name"], params_b, quant, use_case, active_params_b,
                            model.get("created_at"), model.get("pipeline_tag"))
    speed = speed_score(tps, use_case)
    fit = fit_score(mem_used, mem_available)
    context = context_score(context_length, use_case)
    score = weighted_score(quality, speed, fit, context, use_case) if level != TOO_TIGHT else 0.0

    if ctx < context_length:
        notes.append("KV cache estimated at %d tokens (model supports %d)" % (ctx, context_length))

    return {
        "quant": quant,
        "file_name": gguf_file.get("file_name"),
        "file_size_gb": round(weights_gb, 2),
        "vram_needed": round(mem_used, 2),
        "memory_required_gb": round(mem_required, 2),
        "memory_available_gb": round(mem_available, 2),
        "utilization_pct": round(mem_used / mem_available * 100, 1) if mem_available > 0 else None,
        "run_mode": run_mode,
        "fit_level": level,
        "est_tokens_per_sec": round(tps, 1),
        "is_moe": is_moe,
        "active_params_b": active_params_b,
        "quality_score": round(quality, 1),
        "speed_score": round(speed, 1),
        "fit_score": round(fit, 1),
        "context_score": round(context, 1),
        "final_score": score,
        "notes": notes,
    }


def is_plausible_file(params_b, gguf_file):
    """
    Reject files far smaller than their quant implies for the model's size:
    speculative-decoding draft / MTP heads and mislabelled uploads.
    """
    if not params_b or params_b <= 0:
        return True
    expected_gb = params_b * quant_bytes_per_param(gguf_file["quant"]) * 1e9 / (1024**3)
    return gguf_file["size_gb"] >= expected_gb * 0.5


RUN_MODE_PREFERENCE = [GPU, MOE_OFFLOAD, CPU_OFFLOAD, CPU_ONLY]


def best_file_for_model(model, gguf_files, hardware, use_case=GENERAL):
    """
    Pick the GGUF file to recommend, llmfit-style: take the fastest execution
    path any file can use (GPU > MoE offload > CPU+GPU > CPU), then the
    highest-quality quant on that path, preferring one that leaves headroom
    (Good or better) over a Marginal squeeze.
    Returns the analysis dict, or None if nothing fits.
    """
    results = [analyze_file(model, f, hardware, use_case)
               for f in gguf_files if is_plausible_file(model.get("params_b"), f)]
    runnable = [r for r in results if r["fit_level"] != TOO_TIGHT]
    if not runnable:
        return None

    for mode in RUN_MODE_PREFERENCE:
        on_path = [r for r in runnable if r["run_mode"] == mode]
        if not on_path:
            continue
        roomy = [r for r in on_path if FIT_RANK[r["fit_level"]] >= FIT_RANK[GOOD]]
        pool = roomy or on_path
        return max(pool, key=lambda r: (quant_quality_penalty(r["quant"]), -r["file_size_gb"]))
    return None
