import unittest
import os
from src.app_integrator import detect_app_path, get_model_directory

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

if __name__ == "__main__":
    unittest.main()
