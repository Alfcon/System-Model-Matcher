import unittest
from unittest import mock

from src.model_finder import (search_gguf_models, rank_models, estimate_vram_requirement,
                              extract_params_from_name, list_gguf_files, canonical_model_name,
                              _candidate_from_entry)


def _fake_response(payload):
    response = mock.Mock()
    response.json.return_value = payload
    response.raise_for_status.return_value = None
    return response


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
        assert all(model["fit_level"] != "Too Tight" for model in ranked)
        # Anything placed on the GPU must fit in VRAM
        assert all(model["vram_needed"] <= user_vram_gb for model in ranked if model["run_mode"] == "GPU")
        assert [m["final_score"] for m in ranked] == sorted((m["final_score"] for m in ranked), reverse=True)

    def test_rank_models_filters_oversized_models(self):
        sample_models = [
            {"model_name": "too-big", "params_b": 70, "quant": "Q8_0", "downloads": 1000, "likes": 100},
            {"model_name": "fits", "params_b": 7, "quant": "Q4_K_M", "downloads": 2000, "likes": 200},
        ]
        ranked = rank_models(sample_models, 8)

        assert len(ranked) == 1
        assert ranked[0]["model_name"] == "fits"
        assert ranked[0]["run_mode"] == "GPU"
        assert ranked[0]["vram_needed"] <= 6.4

    def test_rank_models_keeps_one_upload_per_model(self):
        sample_models = [
            {"model_name": "bartowski/Qwen_Qwen3-8B-GGUF", "params_b": 8, "quant": "Q4_K_M", "downloads": 10},
            {"model_name": "unsloth/Qwen3-8B-GGUF", "params_b": 8, "quant": "Q4_K_M", "downloads": 99},
        ]
        ranked = rank_models(sample_models, 8)
        assert [m["model_name"] for m in ranked] == ["unsloth/Qwen3-8B-GGUF"]

    def test_estimate_vram_requirement(self):
        # Weights + 8k-token KV cache + runtime overhead
        vram_7b_q4 = estimate_vram_requirement(params_billions=7, quant="Q4_K_M")
        assert 4.5 <= vram_7b_q4 <= 6

        vram_13b_q5 = estimate_vram_requirement(params_billions=13, quant="Q5_K_M")
        assert 9 <= vram_13b_q5 <= 11

    def test_extract_params_from_name(self):
        assert extract_params_from_name("TheBloke/Mistral-7B-Instruct-v0.2-GGUF") == 7
        assert extract_params_from_name("bartowski/Meta-Llama-3.1-8B-Instruct-GGUF") == 8
        assert extract_params_from_name("unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF") == 30
        assert extract_params_from_name("Qwen/Qwen2.5-0.5B-Instruct-GGUF") == 0.5
        assert extract_params_from_name("some/unsized-model-GGUF") == 7

    def test_canonical_model_name_merges_reuploads(self):
        assert canonical_model_name("bartowski/Qwen_Qwen3-8B-GGUF") == canonical_model_name("unsloth/Qwen3-8B-GGUF")
        assert canonical_model_name("lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF") == \
            canonical_model_name("bartowski/Llama-3.1-8B-Instruct-GGUF")
        assert canonical_model_name("mradermacher/Foo-7B-i1-GGUF") == canonical_model_name("mradermacher/Foo-7B-GGUF")

    def test_list_gguf_files_handles_split_and_draft_files(self):
        gb = 1024**3
        tree = [
            {"type": "directory", "path": "BF16", "size": 0},
            {"type": "file", "path": "m-Q4_K_M.gguf", "size": 5 * gb},
            # Same quant shipped both whole and split: must be counted once
            {"type": "file", "path": "m-Q4_K_M-00001-of-00002.gguf", "size": 3 * gb},
            {"type": "file", "path": "m-Q4_K_M-00002-of-00002.gguf", "size": 2 * gb},
            {"type": "file", "path": "BF16/m-BF16-00001-of-00002.gguf", "size": 10 * gb},
            {"type": "file", "path": "BF16/m-BF16-00002-of-00002.gguf", "size": 6 * gb},
            {"type": "file", "path": "mmproj-m-F16.gguf", "size": 1 * gb},
            {"type": "file", "path": "mtp-m-Q8_0.gguf", "size": gb // 10},
            {"type": "file", "path": "m-draft-Q4_0.gguf", "size": gb // 5},
            {"type": "file", "path": "m-noMTP-Q8_0.gguf", "size": 8 * gb},
            {"type": "file", "path": "README.md", "size": 100},
        ]
        with mock.patch("src.model_finder.requests.get", return_value=_fake_response(tree)):
            files = {f["quant"]: f for f in list_gguf_files("owner/m-GGUF")}

        assert set(files) == {"Q4_K_M", "BF16", "Q8_0"}
        assert files["Q4_K_M"]["size_gb"] == 5
        assert files["BF16"]["size_gb"] == 16 and files["BF16"]["parts"] == 2
        assert files["Q8_0"]["file_name"] == "m-noMTP-Q8_0.gguf"

    def test_candidate_uses_gguf_metadata(self):
        entry = {"id": "owner/Foo-7B-GGUF", "downloads": 5, "likes": 1, "pipeline_tag": "text-generation",
                 "gguf": {"total": 7_241_732_096, "architecture": "llama", "context_length": 32768}}
        candidate = _candidate_from_entry(entry)
        assert candidate["params_b"] == 7.24
        assert candidate["context_length"] == 32768
        assert candidate["architecture"] == "llama"

    def test_candidate_distrusts_metadata_far_from_name(self):
        # Hub parsed a draft head (1.86B) for a 27B repo: trust the name
        entry = {"id": "owner/Big-27B-MTP-GGUF", "gguf": {"total": 1_860_000_000}}
        assert _candidate_from_entry(entry)["params_b"] == 27

    def test_candidate_filters_non_llm_repos(self):
        assert _candidate_from_entry({"id": "x/detector-GGUF", "pipeline_tag": "object-detection"}) is None
        assert _candidate_from_entry({"id": "x/nomic-embed-text-GGUF"}) is None


if __name__ == "__main__":
    unittest.main()
