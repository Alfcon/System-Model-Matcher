import unittest
from src.model_finder import search_gguf_models, rank_models, estimate_vram_requirement


class TestModelFinder(unittest.TestCase):
    def test_search_gguf_models_returns_list(self):
        result = search_gguf_models(task="text-generation", limit=5)
        assert isinstance(result, list)
        # Result may be empty if API unreachable, but should be a list

    def test_rank_models_with_hardware_constraints(self):
        sample_models = [
            {"model_name": "model-a", "params_b": 7, "quant": "Q4_K_M", "downloads": 1000, "likes": 100},
            {"model_name": "model-b", "params_b": 13, "quant": "Q5_K_M", "downloads": 5000, "likes": 500},
            {"model_name": "model-c", "params_b": 70, "quant": "Q2_K", "downloads": 2000, "likes": 200},
        ]
        user_vram_gb = 8
        ranked = rank_models(sample_models, user_vram_gb)

        assert isinstance(ranked, list)
        assert all(model["vram_needed"] <= user_vram_gb * 0.8 for model in ranked)
        assert all(model["vram_score"] > 0 for model in ranked)

    def test_rank_models_filters_oversized_models(self):
        sample_models = [
            {"model_name": "too-big", "params_b": 70, "quant": "Q8_0", "downloads": 1000, "likes": 100},
            {"model_name": "fits", "params_b": 7, "quant": "Q4_K_M", "downloads": 2000, "likes": 200},
        ]
        ranked = rank_models(sample_models, 8)

        assert len(ranked) == 1
        assert ranked[0]["model_name"] == "fits"
        assert ranked[0]["vram_needed"] <= 6.4

    def test_estimate_vram_requirement(self):
        # 7B model with Q4 quantization should need ~4GB
        vram_7b_q4 = estimate_vram_requirement(params_billions=7, quant="Q4_K_M")
        assert 3 <= vram_7b_q4 <= 5

        # 13B model with Q5 quantization should need ~8GB
        vram_13b_q5 = estimate_vram_requirement(params_billions=13, quant="Q5_K_M")
        assert 7 <= vram_13b_q5 <= 10


if __name__ == "__main__":
    unittest.main()
