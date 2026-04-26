# src/main.py
import tkinter as tk
from tkinter import messagebox
import threading
import sys
import os

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hardware_detector import detect_all
from hf_scraper import get_tasks, get_apps, get_themes
from model_finder import search_gguf_models, rank_models
from ui_screens import ProfileScreen, HardwareScreen, PreferencesScreen, ResultsScreen

class LLMModelFinderApp:
    def __init__(self, root):
        self.root = root
        self.root.title("LLM Model Finder")
        self.root.geometry("700x600")

        # Session state
        self.state = {
            "name": None,
            "location": None,
            "hardware": None,
            "preferences": None,
            "results": None,
            "current_screen": 0
        }

        # Load dropdown data
        self.dropdowns_data = {
            'tasks': self._load_tasks(),
            'apps': self._load_apps(),
            'themes': self._load_themes()
        }

        # Create screens
        self.screens = []
        self._create_screens()

        # Start with first screen
        self._show_screen(0)

    def _load_tasks(self):
        """Load tasks from scraper in background."""
        try:
            return get_tasks()
        except:
            return ["text-generation", "summarization", "conversational"]

    def _load_apps(self):
        """Load apps from scraper."""
        try:
            return get_apps()
        except:
            return ["Ollama", "LM Studio", "GPT4All", "Custom"]

    def _load_themes(self):
        """Load themes from scraper."""
        try:
            return get_themes()
        except:
            return ["Story", "Coding", "General Chat", "Research"]

    def _create_screens(self):
        """Create all screen instances."""
        # Screen 0: Profile
        profile_screen = ProfileScreen(
            self.root, self.state,
            on_next=lambda: self._show_screen(1)
        )
        self.screens.append(profile_screen)

        # Screen 1: Hardware
        hw_screen = HardwareScreen(
            self.root, self.state,
            on_next=lambda: self._show_screen(2),
            on_back=lambda: self._show_screen(0),
            hardware_detector=sys.modules[__name__]
        )
        self.screens.append(hw_screen)

        # Screen 2: Preferences
        prefs_screen = PreferencesScreen(
            self.root, self.state,
            on_next=self._on_preferences_complete,
            on_back=lambda: self._show_screen(1),
            dropdowns_data=self.dropdowns_data
        )
        self.screens.append(prefs_screen)

        # Screen 3: Results (placeholder, will be created after search)
        self.results_screen = None

    def _show_screen(self, screen_num):
        """Show a specific screen and hide others."""
        # Hide all screens
        for screen in self.screens:
            screen.hide()

        # Show requested screen
        if 0 <= screen_num < len(self.screens):
            self.screens[screen_num].show()
            self.state["current_screen"] = screen_num

    def _reset_and_new_search(self):
        """Reset preferences screen for new search."""
        self.screens[2].generate_btn.config(state=tk.DISABLED)
        self.screens[2]._hide_progress_bar()
        self.screens[2]._hide_models_list()
        self.screens[2].clear_models_list()
        self.screens[2].app_var.set('')
        self.screens[2].search_var.set('')
        self._show_screen(2)  # Return to preferences

    def _on_preferences_complete(self):
        """Called when preferences are selected; search for models."""
        # Run search in background thread
        thread = threading.Thread(target=self._search_and_rank_models)
        thread.daemon = True
        thread.start()

    def _search_and_rank_models(self):
        """Search HuggingFace for GGUF models based on user parameter or best general models."""
        try:
            vram_gb = self.state["hardware"]["gpu"]["vram_gb"]
            search_param = self.state["preferences"].get("search_param", "")

            # Get the preferences screen to update model list
            prefs_screen = self.screens[2]

            # Determine search query
            if search_param:
                # User provided search parameter - use it
                search_query = search_param
                task = "text-generation"  # General LLM task
            else:
                # No search parameter - search for best general LLM models
                search_query = "text-generation"
                task = "text-generation"

            # Search for GGUF models
            all_models = []

            # Primary search with user query
            models = search_gguf_models(task=search_query, limit=30)
            all_models.extend(models)

            # If no search param and few results, also search for popular models
            if not search_param and len(all_models) < 10:
                popular_searches = ["mistral", "llama", "neural", "dolphin"]
                for search_term in popular_searches:
                    models = search_gguf_models(task=search_term, limit=15)
                    all_models.extend(models)

            # Remove duplicates by model name
            seen = set()
            unique_models = []
            for model in all_models:
                name = model.get('model_name', '')
                if name not in seen:
                    seen.add(name)
                    unique_models.append(model)

            # Display found models
            for model in unique_models[:40]:
                model_name = model.get('model_name', 'Unknown')[:50]
                prefs_screen.add_model_to_list(model_name)

            if not unique_models:
                messagebox.showwarning("No Results", "No models found. Try a different search parameter.")
                prefs_screen.generate_btn.config(state=tk.NORMAL)
                prefs_screen._hide_progress_bar()
                prefs_screen._hide_models_list()
                return

            # Rank models by suitability for user's hardware
            ranked = rank_models(unique_models, vram_gb, task=task)

            # Add estimated speed
            for model in ranked:
                vram = model.get('vram_needed', 6)
                params = model.get('params_b', 7)
                model['est_tokens_per_sec'] = round((vram * 10) / params, 1)

            self.state["results"] = ranked

            # Hide progress and models list
            prefs_screen._hide_progress_bar()
            prefs_screen._hide_models_list()

            # Create and show results screen
            results_screen = ResultsScreen(
                self.root, self.state,
                on_new_search=lambda: self._reset_and_new_search(),
                on_back=lambda: self._show_screen(2),
                model_data=ranked
            )
            self.screens.append(results_screen)
            self._show_screen(3)

        except Exception as e:
            messagebox.showerror("Error", f"Search failed: {str(e)}")
            prefs_screen = self.screens[2]
            prefs_screen.generate_btn.config(state=tk.NORMAL)
            prefs_screen._hide_progress_bar()
            prefs_screen._hide_models_list()

def main():
    root = tk.Tk()
    app = LLMModelFinderApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
