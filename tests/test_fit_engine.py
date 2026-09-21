import unittest
from datetime import datetime

from src import fit_engine as fe


def _hardware(vram=8.0, ram_available=24.0, gpu_model="NVIDIA GeForce RTX 3060", backend="cuda",
              unified=False, threads=16):
    return {
        "cpu": {"cores": threads // 2, "threads": threads, "arch": "x86_64"},
        "ram": {"total_gb": ram_available + 8, "available_gb": ram_available},
        "gpu": {"model": gpu_model if vram else "No compatible GPU detected", "vram_gb": vram,
                "backend": backend if vram else "cpu_x86", "unified_memory": unified},
    }


def _model(name="owner/Llama-3.1-8B-Instruct-GGUF", params_b=8.0, ctx=131072):
    return {"model_name": name, "params_b": params_b, "context_length": ctx}


class TestQuantization(unittest.TestCase):
    def test_detect_quant(self):
        assert fe.detect_quant("Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf") == "Q4_K_M"
        assert fe.detect_quant("model.IQ4_XS.gguf") == "IQ4_XS"
        assert fe.detect_quant("Qwen3-30B-UD-Q4_K_XL.gguf") == "UD-Q4_K_XL"
        assert fe.detect_quant("BF16/model-BF16-00001-of-00002.gguf") == "BF16"
        assert fe.detect_quant("qwen2.5-coder-7b-instruct-q5_k_m.gguf") == "Q5_K_M"
        assert fe.detect_quant("model-fp16.gguf") == "F16"
        assert fe.detect_quant("README.gguf") is None

    def test_quality_penalty_ordering(self):
        order = ["Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q3_K_M", "Q2_K"]
        penalties = [fe.quant_quality_penalty(q) for q in order]
        assert penalties == sorted(penalties, reverse=True)
        assert fe.quant_quality_penalty("Q4_K_S") < fe.quant_quality_penalty("Q4_K_M")
        assert fe.quant_quality_penalty("BF16") == fe.quant_quality_penalty("Q8_0")


class TestMemoryAndFit(unittest.TestCase):
    def test_memory_includes_kv_cache_and_overhead(self):
        mem = fe.estimate_memory_gb(weights_gb=4.6, params_b=8, ctx=8192)
        assert abs(mem - (4.6 + 0.000008 * 8 * 8192 + 0.5)) < 1e-9

    def test_fit_level_bands(self):
        assert fe.fit_level(5.0, 10.0) == fe.PERFECT
        assert fe.fit_level(8.0, 10.0) == fe.GOOD
        assert fe.fit_level(9.5, 10.0) == fe.MARGINAL
        assert fe.fit_level(9.9, 10.0) == fe.TOO_TIGHT
        assert fe.fit_level(1.0, 0.0) == fe.TOO_TIGHT

    def test_offload_paths_cap_at_good(self):
        assert fe.fit_level(2.0, 10.0, fe.CPU_ONLY) == fe.GOOD
        assert fe.fit_level(2.0, 10.0, fe.CPU_OFFLOAD) == fe.GOOD

    def test_run_mode_selection(self):
        hw = _hardware(vram=8.0, ram_available=24.0)
        assert fe.choose_run_mode(6.0, 5.0, 8, None, False, hw)[0] == fe.GPU
        assert fe.choose_run_mode(12.0, 11.0, 14, None, False, hw)[0] == fe.CPU_OFFLOAD
        assert fe.choose_run_mode(6.0, 5.0, 8, None, False, _hardware(vram=0))[0] == fe.CPU_ONLY
        mac = _hardware(vram=32.0, gpu_model="Apple M2 Pro", backend="metal", unified=True)
        assert fe.choose_run_mode(20.0, 19.0, 30, None, False, mac)[0] == fe.GPU

    def test_moe_offload_when_full_model_exceeds_vram(self):
        hw = _hardware(vram=8.0, ram_available=24.0)
        mode, used, available, notes = fe.choose_run_mode(19.0, 17.3, 30.5, 3.0, True, hw)
        assert mode == fe.MOE_OFFLOAD
        assert used < 8.0 and available == 8.0

    def test_fit_score_smooth(self):
        assert fe.fit_score(5.0, 10.0) == 100.0
        assert 0 < fe.fit_score(9.0, 10.0) < fe.fit_score(8.0, 10.0) < 100.0
        assert fe.fit_score(11.0, 10.0) == 0.0


class TestSpeed(unittest.TestCase):
    def test_bandwidth_lookup(self):
        assert fe.gpu_memory_bandwidth_gbps("NVIDIA GeForce RTX 4090") == 1008.0
        assert fe.gpu_memory_bandwidth_gbps("NVIDIA GeForce RTX 4070 Ti SUPER") == 672.0
        assert fe.gpu_memory_bandwidth_gbps("AMD Radeon RX 7900 XTX") == 960.0
        assert fe.gpu_memory_bandwidth_gbps("Apple M1 Max") == 400.0
        # Laptop parts do not share the desktop memory bus
        assert fe.gpu_memory_bandwidth_gbps("NVIDIA GeForce RTX 4060 Laptop GPU") is None
        assert fe.gpu_memory_bandwidth_gbps("Mystery GPU") is None

    def test_faster_gpu_is_faster(self):
        slow = fe.estimate_tps(8, "Q4_K_M", fe.GPU, _hardware(gpu_model="NVIDIA GeForce RTX 3060"))
        fast = fe.estimate_tps(8, "Q4_K_M", fe.GPU, _hardware(gpu_model="NVIDIA GeForce RTX 4090"))
        assert fast > slow * 2
        # RTX 4090, 8B Q4: 1008 / 4 * 0.55 ~= 139 tok/s
        assert 120 < fast < 160

    def test_run_mode_penalties(self):
        hw = _hardware(gpu_model="Unknown GPU")
        gpu = fe.estimate_tps(8, "Q4_K_M", fe.GPU, hw)
        offload = fe.estimate_tps(8, "Q4_K_M", fe.CPU_OFFLOAD, hw)
        cpu = fe.estimate_tps(8, "Q4_K_M", fe.CPU_ONLY, hw)
        assert gpu > offload > cpu > 0

    def test_moe_uses_active_parameters(self):
        hw = _hardware(gpu_model="NVIDIA GeForce RTX 4090", vram=24.0)
        dense = fe.estimate_tps(30, "Q4_K_M", fe.GPU, hw)
        moe = fe.estimate_tps(30, "Q4_K_M", fe.GPU, hw, is_moe=True, active_params_b=3)
        assert moe > dense * 3


class TestModelMetadata(unittest.TestCase):
    def test_params_from_name(self):
        assert fe.params_from_name("Qwen3-Coder-30B-A3B-Instruct-GGUF") == 30
        assert fe.params_from_name("Llama-3.1-8B-Instruct") == 8
        assert fe.params_from_name("SmolLM2-360M-Instruct") == 0.36
        assert 45 < fe.params_from_name("Mixtral-8x7B-Instruct-v0.1") < 48
        assert fe.params_from_name("gemma-4-E4B-it") is None

    def test_moe_detection(self):
        assert fe.moe_info("unsloth/Qwen3-30B-A3B-GGUF") == (True, 3.0)
        assert fe.moe_info("TheBloke/Mixtral-8x7B-v0.1-GGUF")[0] is True
        assert fe.moe_info("owner/something-GGUF", architecture="qwen3moe") == (True, None)
        assert fe.moe_info("bartowski/Llama-3.1-8B-GGUF", architecture="llama") == (False, None)


class TestScoring(unittest.TestCase):
    def test_quality_prefers_bigger_and_less_quantized(self):
        small = fe.quality_score("x/foo-3b", 3, "Q4_K_M", fe.GENERAL)
        big = fe.quality_score("x/foo-13b", 13, "Q4_K_M", fe.GENERAL)
        q2 = fe.quality_score("x/foo-13b", 13, "Q2_K", fe.GENERAL)
        assert big > small and big > q2

    def test_coding_use_case_favours_coder_models(self):
        coder = fe.quality_score("Qwen/Qwen2.5-Coder-7B-Instruct-GGUF", 7.6, "Q4_K_M", fe.CODING)
        general = fe.quality_score("owner/Generic-7B-GGUF", 7.6, "Q4_K_M", fe.CODING)
        assert coder > general

    def test_recency_bonus(self):
        now = datetime(2026, 9, 1)
        fresh = fe.quality_score("x/foo-7b", 7, "Q4_K_M", fe.GENERAL, created_at="2026-08-01", now=now)
        old = fe.quality_score("x/foo-7b", 7, "Q4_K_M", fe.GENERAL, created_at="2024-01-01", now=now)
        assert fresh - old == 3.0

    def test_weighted_score_uses_use_case_weights(self):
        # Reasoning weights quality higher than Chat, Chat weights speed higher
        assert fe.weighted_score(100, 0, 0, 0, 0, fe.REASONING) > fe.weighted_score(100, 0, 0, 0, 0, fe.CHAT)
        assert fe.weighted_score(0, 100, 0, 0, 0, fe.CHAT) > fe.weighted_score(0, 100, 0, 0, 0, fe.REASONING)
        assert fe.weighted_score(0, 100, 0, 0, 0, fe.ROLEPLAY) == fe.weighted_score(0, 100, 0, 0, 0, fe.CHAT)

    def test_popularity_is_ten_percent_of_score(self):
        assert fe.weighted_score(100, 100, 100, 100, 100, fe.GENERAL) == 100.0
        assert fe.weighted_score(100, 100, 100, 100, 0, fe.GENERAL) == 90.0
        assert fe.weighted_score(0, 0, 0, 0, 100, fe.CODING) == 10.0

    def test_popularity_score_is_log_scaled(self):
        assert fe.popularity_score(0, 0) == 0.0
        assert fe.popularity_score(10_000_000, 3162) > 99.9
        small = fe.popularity_score(1_000, 10)
        medium = fe.popularity_score(100_000, 100)
        assert 0 < small < medium < 100
        # 100x the downloads is a fixed step, not 100x the score
        assert abs((medium - small) - (fe.popularity_score(10_000_000, 1000) - medium)) < 1e-9

    def test_popular_upload_outranks_identical_obscure_one(self):
        files = [{"quant": "Q4_K_M", "size_gb": 4.6}]
        popular = fe.best_file_for_model({**_model(), "downloads": 5_000_000, "likes": 2000}, files, _hardware())
        obscure = fe.best_file_for_model({**_model(), "downloads": 50, "likes": 0}, files, _hardware())
        assert popular["final_score"] > obscure["final_score"]


class TestBestFile(unittest.TestCase):
    def _files(self):
        return [
            {"quant": "Q2_K", "size_gb": 3.0},
            {"quant": "Q4_K_M", "size_gb": 4.6},
            {"quant": "Q5_K_M", "size_gb": 5.4},
            {"quant": "Q8_0", "size_gb": 8.0},
            {"quant": "BF16", "size_gb": 15.0},
        ]

    def test_picks_highest_quality_that_fits_gpu_with_headroom(self):
        best = fe.best_file_for_model(_model(), self._files(), _hardware(vram=8.0))
        # Q5_K_M: 5.4 + 0.52 KV + 0.5 = 6.4 GB -> 80% of 8 GB (Good)
        assert best["quant"] == "Q5_K_M"
        assert best["run_mode"] == fe.GPU

    def test_big_gpu_takes_q8_not_bf16(self):
        best = fe.best_file_for_model(_model(), self._files(), _hardware(vram=24.0))
        assert best["quant"] == "Q8_0"

    def test_cpu_only_system_uses_ram(self):
        best = fe.best_file_for_model(_model(), self._files(), _hardware(vram=0, ram_available=12.0))
        assert best["run_mode"] == fe.CPU_ONLY
        assert best["fit_level"] in (fe.GOOD, fe.MARGINAL)

    def test_nothing_fits(self):
        assert fe.best_file_for_model(_model(params_b=70), [{"quant": "Q4_K_M", "size_gb": 40.0}],
                                      _hardware(vram=8.0, ram_available=16.0)) is None

    def test_ignores_implausibly_small_files(self):
        files = [{"quant": "Q8_0", "size_gb": 0.4}, {"quant": "Q4_K_M", "size_gb": 4.6}]
        assert fe.best_file_for_model(_model(), files, _hardware(vram=8.0))["quant"] == "Q4_K_M"


if __name__ == "__main__":
    unittest.main()
