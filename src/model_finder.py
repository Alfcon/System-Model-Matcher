from huggingface_hub import HfApi

api = HfApi()

QUANT_VRAM_MULTIPLIER = {
    "Q2_K": 0.35,
    "Q3_K_S": 0.40,
    "Q3_K_M": 0.45,
    "Q3_K_L": 0.50,
    "Q4_0": 0.50,
    "Q4_K_S": 0.52,
    "Q4_K_M": 0.56,
    "Q5_K_S": 0.65,
    "Q5_K_M": 0.72,
    "Q6_K": 0.80,
    "Q8_0": 1.0,
}


def estimate_vram_requirement(params_billions, quant="Q4_K_M"):
    """
    Estimate VRAM needed in GB for a model.
    Formula: params_billions * quant_multiplier
    The multiplier represents GB needed per 1B parameters for that quantization.
    """
    multiplier = QUANT_VRAM_MULTIPLIER.get(quant, 0.56)  # Default Q4_K_M
    return round(params_billions * multiplier, 2)


def search_gguf_models(task="text-generation", limit=50):
    """
    Search HuggingFace Hub for GGUF models matching a task.
    Returns list of model dicts with metadata.
    """
    try:
        # Search with filters: GGUF format only (remove "featured" filter as it's too restrictive)
        models = api.list_models(
            search=task,
            filter=["gguf"],
            sort="downloads",
            direction=-1,
            limit=limit
        )

        result = []
        for model in models:
            # Extract model details
            try:
                model_info = {
                    "model_name": model.id,
                    "downloads": model.downloads,
                    "likes": model.likes,
                    "last_modified": str(model.last_modified),
                    "params_b": extract_params_from_name(model.id),
                    "quant": "Q4_K_M",  # Default, would need to scrape actual quant from model repo
                    "file_size_gb": 5.0,  # Placeholder, would fetch from model files
                }
                result.append(model_info)
            except Exception:
                continue

        return result[:limit]
    except Exception as e:
        print(f"Warning: Could not search HF Hub: {e}")
        return []


def extract_params_from_name(model_name):
    """Extract parameter count from model name (e.g., 'mistral-7b' -> 7)."""
    name_lower = model_name.lower()
    for size in ["70", "65", "34", "32", "30", "13", "12", "11", "10", "9", "8", "7", "6", "5", "4", "3", "2", "1"]:
        if f"{size}b" in name_lower:
            return int(size)
    return 7  # Default to 7B


def rank_models(models, user_vram_gb, task="text-generation"):
    """
    Rank models by suitability for user's hardware and preferences.
    Scoring: VRAM fit (50%) + param suitability (30%) + quant quality (10%) + popularity (10%)
    """
    scored = []

    for model in models:
        params_b = model.get("params_b", 7)
        quant = model.get("quant", "Q4_K_M")
        downloads = model.get("downloads", 0)
        likes = model.get("likes", 0)

        # Estimate VRAM requirement
        vram_needed = estimate_vram_requirement(params_b, quant)

        # 1. VRAM fit score (0 if doesn't fit, 100 if fits)
        safety_buffer = user_vram_gb * 0.8  # 20% safety margin
        if vram_needed > safety_buffer:
            vram_score = 0
        else:
            vram_score = 100

        # 2. Param suitability (prefer 7-13B for chat, 13B+ for coding)
        if task in ["conversational", "text-generation"]:
            if 7 <= params_b <= 13:
                param_score = 100
            elif 5 <= params_b <= 20:
                param_score = 80
            elif params_b < 5:
                param_score = 50
            else:
                param_score = 70
        else:
            if 13 <= params_b <= 70:
                param_score = 100
            elif 7 <= params_b < 13:
                param_score = 80
            else:
                param_score = 60

        # 3. Quant quality score
        quant_quality = {"Q2_K": 40, "Q3_K_M": 60, "Q4_K_M": 80, "Q5_K_M": 90, "Q6_K": 95, "Q8_0": 100}
        quant_score = quant_quality.get(quant, 75)

        # 4. Popularity score (normalized 0-100)
        popularity_combined = downloads + likes
        popularity_score = min(100, (popularity_combined / 50000) * 100) if popularity_combined > 0 else 50

        # Weighted final score
        final_score = (vram_score * 0.5) + (param_score * 0.3) + (quant_score * 0.1) + (popularity_score * 0.1)

        scored.append({
            **model,
            "vram_needed": vram_needed,
            "vram_score": vram_score,
            "param_score": param_score,
            "quant_score": quant_score,
            "popularity_score": popularity_score,
            "final_score": round(final_score, 2)
        })

    # Sort by final score, exclude models that don't fit (vram_score == 0)
    scored = [m for m in scored if m["vram_score"] > 0]
    scored.sort(key=lambda x: x["final_score"], reverse=True)

    return scored[:10]  # Return top 10
