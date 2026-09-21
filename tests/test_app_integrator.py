import unittest
import os
from src.app_integrator import (detect_app_path, get_model_directory, ollama_run_command,
                                llama_cpp_run_command)

class TestAppIntegrator(unittest.TestCase):
    def test_detect_app_path_returns_string_or_none(self):
        result = detect_app_path("ollama")
        # Result may be None if app not installed, but type should be str or None
        assert result is None or isinstance(result, str)

    def test_get_model_directory_returns_valid_path(self):
        # Test that model dirs are returned even if app not installed (returns expected path)
        result = get_model_directory("ollama")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_get_model_directory_different_apps(self):
        ollama_dir = get_model_directory("ollama")
        lm_studio_dir = get_model_directory("lm_studio")
        gpt4all_dir = get_model_directory("gpt4all")

        # Paths should be different
        assert ollama_dir != lm_studio_dir
        assert lm_studio_dir != gpt4all_dir


class TestRunCommands(unittest.TestCase):
    MODEL = {"model_name": "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF", "quant": "Q5_K_M",
             "file_path": "qwen2.5-coder-7b-instruct-q5_k_m.gguf", "file_parts": 1, "run_mode": "GPU"}

    def test_ollama_pulls_quant_from_hugging_face(self):
        command, note = ollama_run_command(self.MODEL)
        assert command == "ollama run hf.co/Qwen/Qwen2.5-Coder-7B-Instruct-GGUF:Q5_K_M"
        assert note is None

    def test_ollama_warns_about_split_files(self):
        command, note = ollama_run_command({**self.MODEL, "file_parts": 3})
        assert "llama.cpp" in note

    def test_llama_cpp_uses_exact_file_and_full_gpu_offload(self):
        command, _ = llama_cpp_run_command(self.MODEL)
        assert command == ("llama-cli --hf-repo Qwen/Qwen2.5-Coder-7B-Instruct-GGUF "
                           "--hf-file qwen2.5-coder-7b-instruct-q5_k_m.gguf -c 8192 -ngl 99")

    def test_llama_cpp_split_file_uses_quant_tag(self):
        command, _ = llama_cpp_run_command({**self.MODEL, "file_parts": 2, "file_path": "Q8/x-00001-of-00002.gguf"})
        assert "-hf Qwen/Qwen2.5-Coder-7B-Instruct-GGUF:Q5_K_M" in command
        assert "--hf-file" not in command

    def test_llama_cpp_flags_follow_run_mode(self):
        assert llama_cpp_run_command({**self.MODEL, "run_mode": "CPU"})[0].endswith("-ngl 0")
        assert llama_cpp_run_command({**self.MODEL, "run_mode": "MoE offload"})[0].endswith("-ngl 99 --cpu-moe")
        command, note = llama_cpp_run_command({**self.MODEL, "run_mode": "CPU+GPU"})
        assert "-ngl 20" in command and "VRAM" in note


if __name__ == "__main__":
    unittest.main()
