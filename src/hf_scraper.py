import requests
from bs4 import BeautifulSoup

# Hardcoded fallback lists
DEFAULT_TASKS = [
    "text-generation",
    "text2text-generation",
    "summarization",
    "translation",
    "question-answering",
    "conversational",
    "fill-mask",
    "token-classification",
    "zero-shot-classification"
]

DEFAULT_APPS = [
    "Ollama",
    "LM Studio",
    "GPT4All",
    "Llama.cpp app",
    "Custom"
]

DEFAULT_THEMES = [
    "Story",
    "Adult Story",
    "Coding",
    "General Chat",
    "Research",
    "Roleplay",
    "Scientific",
    "Creative Writing"
]


def get_tasks():
    """Get list of HF tasks. Returns hardcoded list as fallback."""
    try:
        # Attempt to scrape from HF tasks page
        url = "https://huggingface.co/tasks"
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')

        # Extract task names from task cards
        tasks = []
        task_elements = soup.find_all('a', class_='block-link')
        for elem in task_elements[:20]:  # Limit to first 20
            task_name = elem.get_text(strip=True)
            if task_name:
                tasks.append(task_name)

        return tasks if tasks else DEFAULT_TASKS
    except Exception as e:
        print(f"Warning: Could not scrape tasks from HF: {e}")
        return DEFAULT_TASKS


def get_apps():
    """Get list of supported inference apps. Hardcoded since HF doesn't have a dedicated 'apps' page."""
    return DEFAULT_APPS


def get_themes():
    """Get list of use case themes. Returns hardcoded list of common themes."""
    return DEFAULT_THEMES
