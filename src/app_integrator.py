import os
import platform
import shutil
from pathlib import Path

SYSTEM = platform.system()

def detect_app_path(app_name):
    """
    Detect if app is installed and return its path.
    Returns: path string or None if not found.
    """
    app_name_lower = app_name.lower()

    if "ollama" in app_name_lower:
        if SYSTEM == "Windows":
            path = os.path.expanduser("~\\AppData\\Local\\Programs\\Ollama\\ollama.exe")
        elif SYSTEM == "Darwin":  # macOS
            path = "/Applications/Ollama.app/Contents/MacOS/ollama"
        else:  # Linux
            path = shutil.which("ollama")
        return path if path and os.path.exists(path) else None

    elif "lm studio" in app_name_lower or "lm_studio" in app_name_lower:
        if SYSTEM == "Windows":
            path = os.path.expanduser("~\\AppData\\Local\\LM Studio\\app")
        elif SYSTEM == "Darwin":
            path = "/Applications/LM Studio.app"
        else:  # Linux
            path = os.path.expanduser("~/.lm-studio")
        return path if path and os.path.exists(path) else None

    elif "gpt4all" in app_name_lower:
        if SYSTEM == "Windows":
            path = os.path.expanduser("~\\AppData\\Local\\GPT4All")
        elif SYSTEM == "Darwin":
            path = os.path.expanduser("~/Library/Application Support/GPT4All")
        else:  # Linux
            path = os.path.expanduser("~/.cache/gpt4all")
        return path if path and os.path.exists(path) else None

    return None

def get_model_directory(app_name):
    """
    Get the expected model directory path for an app, regardless of whether it's installed.
    Returns: Path string to where models should be placed.
    """
    app_name_lower = app_name.lower()

    if "ollama" in app_name_lower:
        return os.path.expanduser("~/.ollama/models/blobs")
    elif "lm studio" in app_name_lower or "lm_studio" in app_name_lower:
        if SYSTEM == "Windows":
            return os.path.expanduser("~\\AppData\\Local\\LM Studio\\models")
        elif SYSTEM == "Darwin":
            return os.path.expanduser("~/Library/Application Support/LM Studio/models")
        else:
            return os.path.expanduser("~/.lm-studio/models")
    elif "gpt4all" in app_name_lower:
        return os.path.expanduser("~/.cache/gpt4all")
    else:
        return os.path.expanduser("~/Downloads")  # Fallback

def download_model(model_url, destination_path, progress_callback=None):
    """
    Download a GGUF model file from HuggingFace.
    Args:
        model_url: URL to the GGUF file
        destination_path: Where to save the file
        progress_callback: Optional function(current_bytes, total_bytes) for progress
    Returns: True if successful, False otherwise
    """
    import requests

    try:
        response = requests.get(model_url, stream=True, timeout=30)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0

        os.makedirs(os.path.dirname(destination_path), exist_ok=True)

        with open(destination_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024*1024):  # 1MB chunks
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if progress_callback:
                        progress_callback(downloaded, total_size)

        return True
    except Exception as e:
        print(f"Error downloading model: {e}")
        return False

def integrate_model_with_app(model_path, app_name):
    """
    Copy model to app's model directory.
    Returns: (success: bool, message: str)
    """
    app_name_lower = app_name.lower()
    dest_dir = get_model_directory(app_name)

    try:
        os.makedirs(dest_dir, exist_ok=True)
        model_filename = os.path.basename(model_path)
        dest_path = os.path.join(dest_dir, model_filename)

        shutil.copy2(model_path, dest_path)

        if "ollama" in app_name_lower:
            return True, f"Model copied to {dest_path}. Restart Ollama to see it in the model list."
        elif "lm studio" in app_name_lower:
            return True, f"Model copied to {dest_path}. Restart LM Studio and refresh the model list."
        elif "gpt4all" in app_name_lower:
            return True, f"Model copied to {dest_path}. Restart GPT4All to see it in the model library."
        else:
            return True, f"Model saved to {dest_path}"
    except Exception as e:
        manual_msg = f"Manual integration required: Copy {model_path} to {dest_dir}"
        return False, f"Integration failed: {str(e)}. {manual_msg}"

def get_integration_instructions(app_name, model_filename):
    """
    Get manual integration instructions for an app.
    Returns: Instruction text string.
    """
    dest_dir = get_model_directory(app_name)

    instructions = f"""
Manual Integration Instructions for {app_name}:

1. Download the model file: {model_filename}
2. Place it in: {dest_dir}
3. Restart {app_name}
4. The model should appear in your model list

If you need help:
- {app_name} Documentation: https://docs.ollama.ai (or equivalent)
"""
    return instructions
