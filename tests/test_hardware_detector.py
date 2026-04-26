import unittest
from src.hardware_detector import detect_cpu, detect_ram, detect_gpu, detect_all


class TestHardwareDetector(unittest.TestCase):
    def test_detect_cpu_returns_dict_with_model_and_cores(self):
        result = detect_cpu()
        assert isinstance(result, dict)
        assert "model" in result
        assert "cores" in result
        assert isinstance(result["cores"], int)
        assert result["cores"] > 0

    def test_detect_ram_returns_dict_with_total_and_available(self):
        result = detect_ram()
        assert isinstance(result, dict)
        assert "total_gb" in result
        assert "available_gb" in result
        assert isinstance(result["total_gb"], float)
        assert isinstance(result["available_gb"], float)
        assert result["total_gb"] > 0

    def test_detect_gpu_returns_dict_with_model_and_vram(self):
        result = detect_gpu()
        assert isinstance(result, dict)
        assert "model" in result
        assert "vram_gb" in result
        assert isinstance(result["vram_gb"], float)

    def test_detect_all_combines_all_hardware(self):
        result = detect_all()
        assert isinstance(result, dict)
        assert "cpu" in result
        assert "ram" in result
        assert "gpu" in result
        assert result["cpu"]["cores"] > 0


if __name__ == "__main__":
    unittest.main()
