# src/hardware_detector.py
import platform
import psutil
import subprocess
import re

SYSTEM = platform.system()
MACHINE = platform.machine().lower()
IS_ARM = MACHINE in ("arm64", "aarch64") or MACHINE.startswith("arm")


def _run(cmd, timeout=5):
    """Run a command and return stdout, or None on any failure."""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=timeout)
        return result.stdout
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None


def _cpu_model_name():
    """Human-readable CPU name. platform.processor() is just 'x86_64' on Linux."""
    if SYSTEM == "Linux":
        try:
            with open("/proc/cpuinfo", "r") as f:
                for line in f:
                    if line.lower().startswith(("model name", "hardware", "cpu model")):
                        return line.split(":", 1)[1].strip()
        except OSError:
            pass
    elif SYSTEM == "Darwin":
        name = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
        if name and name.strip():
            return name.strip()
    return platform.processor() or "Unknown"


def detect_cpu():
    """Detect CPU model, physical core count and logical thread count."""
    try:
        cores = psutil.cpu_count(logical=False) or 0
        threads = psutil.cpu_count(logical=True) or cores
        return {"model": _cpu_model_name(), "cores": cores, "threads": threads, "arch": MACHINE}
    except Exception:
        return {"model": "Unknown", "cores": 0, "threads": 0, "arch": MACHINE}


def detect_ram():
    """Detect total and available system RAM."""
    try:
        mem = psutil.virtual_memory()
        return {
            "total_gb": round(mem.total / (1024**3), 2),
            "available_gb": round(mem.available / (1024**3), 2)
        }
    except Exception:
        return {"total_gb": 0.0, "available_gb": 0.0}


def _gpu_result(names, total_bytes, free_bytes, backend, unified_memory=False):
    """Build the standard GPU dict, aggregating VRAM across identical cards."""
    count = len(names)
    model = names[0] if count else "No compatible GPU detected"
    if count > 1:
        model = f"{count}x {model}"
    return {
        "model": model,
        "vram_gb": round(total_bytes / (1024**3), 2),
        "vram_free_gb": round(free_bytes / (1024**3), 2),
        "count": count,
        "backend": backend,
        "unified_memory": unified_memory,
    }


def _detect_nvidia_gpu():
    """
    Detect NVIDIA GPUs via nvidia-ml-py, falling back to nvidia-smi.
    VRAM is summed across all cards, since llama.cpp and friends can split a
    model across multiple GPUs.
    """
    try:
        from pynvml import (nvmlInit, nvmlShutdown, nvmlDeviceGetCount, nvmlDeviceGetHandleByIndex,
                            nvmlDeviceGetName, nvmlDeviceGetMemoryInfo)
        nvmlInit()
        try:
            names, total, free = [], 0, 0
            for i in range(nvmlDeviceGetCount()):
                handle = nvmlDeviceGetHandleByIndex(i)
                info = nvmlDeviceGetMemoryInfo(handle)
                name = nvmlDeviceGetName(handle)
                if isinstance(name, bytes):
                    name = name.decode('utf-8')
                names.append(name)
                total += info.total
                free += info.free
        finally:
            nvmlShutdown()
        if names:
            return _gpu_result(names, total, free, "cuda")
    except Exception:
        pass  # Library not installed or NVIDIA driver issue

    output = _run(["nvidia-smi", "--query-gpu=name,memory.total,memory.free",
                   "--format=csv,noheader,nounits"])
    if output:
        names, total, free = [], 0, 0
        for line in output.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 3:
                continue
            try:
                total += int(float(parts[1])) * 1024**2
                free += int(float(parts[2])) * 1024**2
            except ValueError:
                continue
            names.append(parts[0])
        if names:
            return _gpu_result(names, total, free, "cuda")

    raise Exception("No NVIDIA GPU found")


def _windows_registry_vram_bytes(adapter_name):
    """
    Read the adapter's 64-bit memory size from the registry. WMI's AdapterRAM is
    a 32-bit field and caps at 4 GB, which under-reports every modern card.
    """
    try:
        import winreg
        key_path = r"SYSTEM\ControlSet001\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}"
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path) as class_key:
            for i in range(64):
                try:
                    sub = winreg.EnumKey(class_key, i)
                except OSError:
                    break
                try:
                    with winreg.OpenKey(class_key, sub) as adapter_key:
                        desc, _ = winreg.QueryValueEx(adapter_key, "DriverDesc")
                        if adapter_name and adapter_name.lower() not in str(desc).lower():
                            continue
                        size, _ = winreg.QueryValueEx(adapter_key, "HardwareInformation.qwMemorySize")
                        if isinstance(size, bytes):
                            size = int.from_bytes(size, "little")
                        return int(size)
                except OSError:
                    continue
    except ImportError:
        pass
    return None


def _detect_amd_gpu_windows():
    """Detect AMD GPU on Windows using WMI plus the registry for >4 GB VRAM."""
    try:
        import wmi
        c = wmi.WMI()
        gpus = c.Win32_VideoController(AdapterCompatibility="Advanced Micro Devices, Inc.")
        if gpus:
            gpu = gpus[0]
            name = getattr(gpu, 'Name', 'AMD GPU')
            vram_bytes = getattr(gpu, 'AdapterRAM', 0) or 0
            if vram_bytes < 0:
                vram_bytes += 2**32
            registry_bytes = _windows_registry_vram_bytes(name)
            if registry_bytes and registry_bytes > vram_bytes:
                vram_bytes = registry_bytes
            # WMI doesn't expose free VRAM
            return _gpu_result([name], vram_bytes, 0, "vulkan")
    except ImportError:
        pass  # WMI not installed
    raise Exception("No AMD GPU found on Windows")


def _detect_amd_gpu_linux():
    """Detect AMD GPU on Linux using rocm-smi, falling back to lspci."""
    output = _run(['rocm-smi', '--showproductname', '--showmeminfo', 'vram'])
    if output:
        names = [n.strip() for n in re.findall(r'Card series:\s*(.+)', output)]
        totals = [int(v) for v in re.findall(r'VRAM Total Memory \(B\):\s*(\d+)', output)]
        used = [int(v) for v in re.findall(r'VRAM Total Used Memory \(B\):\s*(\d+)', output)] or \
               [int(v) for v in re.findall(r'VRAM Used Memory \(B\):\s*(\d+)', output)]
        if totals:
            if len(names) < len(totals):
                names += ["AMD GPU"] * (len(totals) - len(names))
            total = sum(totals)
            return _gpu_result(names[:len(totals)], total, total - sum(used), "rocm")

    output = _run(['lspci'])
    if output:
        for line in output.splitlines():
            if ("VGA compatible controller" in line or "Display controller" in line) \
                    and "Advanced Micro Devices" in line:
                model_match = re.search(r'\[(Radeon[^\]]*|[^\]]*RX[^\]]*)\]', line) or re.search(r'\[(.*?)\]', line)
                model = f"AMD {model_match.group(1)}" if model_match else "AMD GPU"
                # lspci does not report VRAM; 0 lets the app fall back to system RAM.
                return _gpu_result([model], 0, 0, "vulkan")

    raise Exception("No AMD GPU found on Linux")


def _detect_apple_silicon(ram):
    """
    Apple Silicon shares one memory pool between CPU and GPU, so the GPU's
    usable memory is system RAM.
    """
    if SYSTEM != "Darwin" or not IS_ARM:
        raise Exception("Not Apple Silicon")
    name = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
    name = name.strip() if name and name.strip() else "Apple Silicon"
    total = int(ram.get("total_gb", 0) * 1024**3)
    available = int(ram.get("available_gb", 0) * 1024**3)
    return _gpu_result([name], total, available, "metal", unified_memory=True)


def detect_gpu(ram=None):
    """Detect GPU model and VRAM: Apple Silicon, then NVIDIA, then AMD."""
    ram = ram if ram is not None else detect_ram()
    detectors = [lambda: _detect_apple_silicon(ram), _detect_nvidia_gpu]
    if SYSTEM == "Windows":
        detectors.append(_detect_amd_gpu_windows)
    elif SYSTEM == "Linux":
        detectors.append(_detect_amd_gpu_linux)

    for detector in detectors:
        try:
            return detector()
        except Exception:
            continue

    return {
        "model": "No compatible GPU detected",
        "vram_gb": 0.0,
        "vram_free_gb": 0.0,
        "count": 0,
        "backend": "cpu_arm" if IS_ARM else "cpu_x86",
        "unified_memory": False,
    }


def detect_all():
    """Detect all hardware components and return a dictionary."""
    ram = detect_ram()
    return {
        "cpu": detect_cpu(),
        "ram": ram,
        "gpu": detect_gpu(ram)
    }
