# src/hardware_detector.py
import platform
import psutil
import subprocess
import os
import re

SYSTEM = platform.system()

def detect_cpu():
    """Detect CPU model and core count."""
    try:
        model = platform.processor()
        cores = psutil.cpu_count(logical=False)
        return {"model": model if model else "Unknown", "cores": cores if cores else 0}
    except Exception:
        return {"model": "Unknown", "cores": 0}

def detect_ram():
    """Detect total and available system RAM."""
    try:
        mem = psutil.virtual_memory()
        return {
            "total_gb": round(mem.total / (1024**3), 2),
            "available_gb": round(mem.available / (1024**3), 2)
        }
    except Exception:
        return {"total_gb": 0, "available_gb": 0}

def _detect_nvidia_gpu():
    """Detect NVIDIA GPU using nvidia-ml-py library."""
    try:
        from pynvml import nvmlInit, nvmlDeviceGetCount, nvmlDeviceGetHandleByIndex, nvmlDeviceGetName, nvmlDeviceGetMemoryInfo, NVMLError
        nvmlInit()
        if nvmlDeviceGetCount() > 0:
            handle = nvmlDeviceGetHandleByIndex(0)
            info = nvmlDeviceGetMemoryInfo(handle)
            name = nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode('utf-8')
            return {
                "model": name,
                "vram_gb": round(info.total / (1024**3), 2),
                "vram_free_gb": round(info.free / (1024**3), 2)
            }
    except (ImportError, NVMLError):
        pass # Library not installed or NVIDIA driver issue
    raise Exception("No NVIDIA GPU found")

def _detect_amd_gpu_windows():
    """Detect AMD GPU on Windows using WMI."""
    try:
        import wmi
        c = wmi.WMI()
        # Look for video controllers from AMD
        gpus = c.Win32_VideoController(AdapterCompatibility="Advanced Micro Devices, Inc.")
        if gpus:
            gpu = gpus[0]
            vram_bytes = getattr(gpu, 'AdapterRAM', 0)
            # WMI may return a signed 32-bit integer for VRAM > 2GB, so handle it.
            if vram_bytes < 0:
                vram_bytes += 2**32
            return {
                "model": getattr(gpu, 'Name', 'AMD GPU'),
                "vram_gb": round(vram_bytes / (1024**3), 2),
                "vram_free_gb": 0.0  # WMI doesn't provide free VRAM easily
            }
    except ImportError:
        pass # WMI not installed
    raise Exception("No AMD GPU found on Windows")

def _detect_amd_gpu_linux():
    """Detect AMD GPU on Linux using rocm-smi or lspci."""
    try:
        # Prefer rocm-smi for detailed info
        result = subprocess.run(['rocm-smi', '--showproductname', '--showmeminfo', 'vram'], capture_output=True, text=True, check=True, timeout=5)
        output = result.stdout
        
        name_match = re.search(r'Card series:\s*(.+)', output)
        name = name_match.group(1).strip() if name_match else "AMD GPU"

        vram_total_match = re.search(r'VRAM Total Memory \(B\):\s*(\d+)', output)
        vram_used_match = re.search(r'VRAM Used Memory \(B\):\s*(\d+)', output)

        if vram_total_match and vram_used_match:
            total_vram_bytes = int(vram_total_match.group(1))
            used_vram_bytes = int(vram_used_match.group(1))
            free_vram_bytes = total_vram_bytes - used_vram_bytes
            return {
                "model": name,
                "vram_gb": round(total_vram_bytes / (1024**3), 2),
                "vram_free_gb": round(free_vram_bytes / (1024**3), 2)
            }
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        # Fallback to lspci if rocm-smi fails or is not installed
        try:
            result = subprocess.run(['lspci'], capture_output=True, text=True, check=True, timeout=5)
            for line in result.stdout.splitlines():
                if "VGA compatible controller" in line and "Advanced Micro Devices" in line:
                    model_match = re.search(r'\[(.*?)\]', line)
                    model = f"AMD {model_match.group(1)}" if model_match else "AMD GPU (Unknown VRAM)"
                    # lspci does not provide VRAM info, so we return 0 and let the app use a fallback.
                    return {"model": model, "vram_gb": 0.0, "vram_free_gb": 0.0}
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            pass # lspci not found or failed

    raise Exception("No AMD GPU found on Linux")

def detect_gpu():
    """Detect GPU model and VRAM, trying NVIDIA first, then AMD."""
    try:
        return _detect_nvidia_gpu()
    except Exception:
        try:
            if SYSTEM == "Windows":
                return _detect_amd_gpu_windows()
            elif SYSTEM == "Linux":
                return _detect_amd_gpu_linux()
        except Exception:
            pass  # Fall through to default
    
    return {"model": "No compatible GPU detected", "vram_gb": 0.0, "vram_free_gb": 0.0}

def detect_all():
    """Detect all hardware components and return a dictionary."""
    return {
        "cpu": detect_cpu(),
        "ram": detect_ram(),
        "gpu": detect_gpu()
    }