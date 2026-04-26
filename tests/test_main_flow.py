import unittest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from hardware_detector import detect_all
from hf_scraper import get_tasks, get_apps, get_themes
from model_finder import search_gguf_models, rank_models


class TestFullFlow(unittest.TestCase):
    def test_hardware_detection_returns_data(self):
        hw = detect_all()
        assert hw["cpu"]["cores"] > 0
        assert hw["ram"]["total_gb"] > 0

    def test_dropdowns_populated(self):
        tasks = get_tasks()
        apps = get_apps()
        themes = get_themes()

        assert len(tasks) > 0
        assert len(apps) > 0
        assert len(themes) > 0

    def test_model_search_returns_list(self):
        models = search_gguf_models(task="text-generation", limit=5)
        assert isinstance(models, list)


if __name__ == "__main__":
    unittest.main()
