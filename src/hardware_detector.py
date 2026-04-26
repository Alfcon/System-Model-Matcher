import psutil
import platform
import os
import subprocess


def detect_cpu():
    """Detect CPU model and core count."""
    try:
        cores = psutil.cpu_count(logical=False) or psutil.cpu_count(logical=True) or 1
        model = _get_cpu_model()

        # Format: "Model × Cores"
        formatted_model = f"{model} × {cores}"
        return {"model": formatted_model, "cores": cores}
    except Exception as e:
        return {"model": f"Unknown × {cores}", "cores": cores}


def _get_cpu_model():
    """Get detailed CPU model name based on OS."""
    system = platform.system()

    try:
        if system == "Linux":
            return _get_cpu_model_linux()
        elif system == "Windows":
            return _get_cpu_model_windows()
        elif system == "Darwin":
            return _get_cpu_model_macos()
        else:
            return platform.processor() or "Unknown"
    except Exception:
        return platform.processor() or "Unknown"


def _get_cpu_model_linux():
    """Extract CPU model from /proc/cpuinfo on Linux."""
    try:
        with open('/proc/cpuinfo', 'r') as f:
            for line in f:
                if line.startswith('model name'):
                    return line.split(':', 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or "Unknown"


def _get_cpu_model_windows():
    """Extract CPU model from Windows registry or WMI."""
    try:
        # Try using wmi module if available
        try:
            import wmi
            c = wmi.WMI()
            for processor in c.Win32_Processor():
                return processor.Name.strip()
        except ImportError:
            pass

        # Fallback: try registry
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"Hardware\Description\System\CentralProcessor\0") as key:
                return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        except Exception:
            pass
    except Exception:
        pass

    return platform.processor() or "Unknown"


def _get_cpu_model_macos():
    """Extract CPU model from macOS using sysctl."""
    try:
        result = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'],
                              capture_output=True, text=True, timeout=2)
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass

    return platform.processor() or "Unknown"


def detect_ram():
    """Detect total and available RAM in GB."""
    try:
        vm = psutil.virtual_memory()
        total_gb = vm.total / (1024 ** 3)
        available_gb = vm.available / (1024 ** 3)
        return {"total_gb": round(total_gb, 2), "available_gb": round(available_gb, 2)}
    except Exception as e:
        return {"total_gb": 0.0, "available_gb": 0.0, "error": str(e)}


def detect_gpu():
    """Detect GPU model and VRAM in GB. Returns 'No GPU' if not found."""
    # Try nvidia-ml-py first
    gpu_info = _detect_gpu_nvidia_smi_library()
    if gpu_info and gpu_info["model"] != "No GPU detected":
        return gpu_info

    # Fallback: try nvidia-smi command
    gpu_info = _detect_gpu_nvidia_smi_command()
    if gpu_info and gpu_info["model"] != "No GPU detected":
        return gpu_info

    # Fallback: try lspci on Linux
    system = platform.system()
    if system == "Linux":
        gpu_info = _detect_gpu_lspci()
        if gpu_info and gpu_info["model"] != "No GPU detected":
            return gpu_info

    # No GPU found
    return {"model": "No GPU detected", "vram_gb": 0.0, "vram_free_gb": 0.0}


def _detect_gpu_nvidia_smi_library():
    """Detect GPU using nvidia-ml-py library."""
    try:
        import nvidia_ml_py as nvidia_smi
        nvidia_smi.nvmlInit()
        device_count = nvidia_smi.nvmlDeviceGetCount()
        if device_count == 0:
            return {"model": "No GPU detected", "vram_gb": 0.0, "vram_free_gb": 0.0}

        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
        gpu_name = nvidia_smi.nvmlDeviceGetName(handle).decode('utf-8')
        mem_info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        vram_gb = mem_info.total / (1024 ** 3)
        vram_free_gb = mem_info.free / (1024 ** 3)
        nvidia_smi.nvmlShutdown()
        return {"model": gpu_name, "vram_gb": round(vram_gb, 2), "vram_free_gb": round(vram_free_gb, 2)}
    except Exception:
        return None


def _detect_gpu_nvidia_smi_command():
    """Detect GPU by running nvidia-smi command."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,memory.total,memory.free', '--format=csv,noheader'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            if lines:
                parts = lines[0].split(',')
                gpu_name = parts[0].strip()
                vram_str = parts[1].strip() if len(parts) > 1 else "0"
                vram_free_str = parts[2].strip() if len(parts) > 2 else "0"
                
                # Parse VRAM (e.g., "12297 MiB" -> 12.29 GB)
                vram_mib = float(vram_str.split()[0])
                vram_gb = vram_mib / 1024
                
                vram_free_mib = float(vram_free_str.split()[0])
                vram_free_gb = vram_free_mib / 1024

                # Try to enhance with lspci details on Linux
                system = platform.system()
                if system == "Linux":
                    lspci_details = _get_lspci_gpu_details()
                    if lspci_details:
                        gpu_name = lspci_details

                return {"model": gpu_name, "vram_gb": round(vram_gb, 2), "vram_free_gb": round(vram_free_gb, 2)}
    except Exception:
        pass
    return None


def _get_lspci_gpu_details():
    """Get detailed GPU info from lspci."""
    try:
        result = subprocess.run(['lspci'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                # Look for NVIDIA GPU entries
                if 'NVIDIA' in line and ('VGA' in line or '3D' in line):
                    # Format: "01:00.0 VGA compatible controller: NVIDIA Corporation AD107M [GeForce RTX 4060 Max-Q / Mobile]"
                    if 'NVIDIA Corporation' in line:
                        # Extract everything after "NVIDIA Corporation"
                        parts = line.split('NVIDIA Corporation', 1)
                        if len(parts) > 1:
                            return ('NVIDIA ' + parts[1].strip()).replace('[', '').replace(']', '')
    except Exception:
        pass
    return None


def _detect_gpu_lspci():
    """Detect GPU using lspci on Linux."""
    try:
        result = subprocess.run(['lspci'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if 'NVIDIA' in line or 'nvidia' in line or 'GeForce' in line or 'RTX' in line:
                    # Extract GPU name from lspci output
                    # Format: "01:00.0 VGA compatible controller: NVIDIA Corporation AD107M [GeForce RTX 4060 Max-Q / Mobile]"
                    if ':' in line:
                        gpu_info = line.split(':', 2)[-1].strip()
                        return {"model": gpu_info, "vram_gb": 0.0}  # Can't get VRAM from lspci
    except Exception:
        pass
    return None


def detect_all():
    """Detect all hardware and return combined dict."""
    return {
        "cpu": detect_cpu(),
        "ram": detect_ram(),
        "gpu": detect_gpu()
    }
