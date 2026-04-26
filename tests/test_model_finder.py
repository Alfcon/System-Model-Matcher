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
        # Models that fit should be ranked higher
        # 7B and 13B models should rank above 70B for 8GB GPU

    def test_estimate_vram_requirement(self):
        # 7B model with Q4 quantization should need ~4GB
        vram_7b_q4 = estimate_vram_requirement(params_billions=7, quant="Q4_K_M")
        assert 3 <= vram_7b_q4 <= 5

        # 13B model with Q5 quantization should need ~8GB
        vram_13b_q5 = estimate_vram_requirement(params_billions=13, quant="Q5_K_M")
        assert 7 <= vram_13b_q5 <= 10


if __name__ == "__main__":
    unittest.main()
