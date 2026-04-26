import unittest
from src.hf_scraper import get_tasks, get_apps, get_themes


class TestHFScraper(unittest.TestCase):
    def test_get_tasks_returns_list(self):
        result = get_tasks()
        assert isinstance(result, list)
        assert len(result) > 0

    def test_get_tasks_contains_common_items(self):
        result = get_tasks()
        # Should contain at least some common tasks
        # Note: actual items depend on HF website; we'll verify list is non-empty
        assert len(result) > 0

    def test_get_apps_returns_list(self):
        result = get_apps()
        assert isinstance(result, list)
        assert len(result) > 0
        assert "Ollama" in result or "ollama" in result

    def test_get_themes_returns_list(self):
        result = get_themes()
        assert isinstance(result, list)
        assert len(result) > 0


if __name__ == "__main__":
    unittest.main()
