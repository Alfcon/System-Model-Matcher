# src/main.py
import tkinter as tk
from tkinter import messagebox
import threading
import sys
import os

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hf_scraper import get_tasks, get_apps, get_themes
from model_finder import gather_candidates, evaluate_candidates, rank_models
from fit_engine import GENERAL
import hardware_detector
from ui_screens import HardwareScreen, PreferencesScreen, ResultsScreen

class LLMModelFinderApp:
    def __init__(self, root):
        self.root = root
        self.root.title("LLM Model Finder")
        self.root.geometry("1200x720")

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
        # Screen 0: Hardware
        hw_screen = HardwareScreen(
            self.root, self.state,
            on_next=lambda: self._show_screen(1),
            on_back=lambda: self.root.quit(),
            hardware_detector=hardware_detector
        )
        self.screens.append(hw_screen)

        # Screen 1: Preferences
        prefs_screen = PreferencesScreen(
            self.root, self.state,
            on_next=self._on_preferences_complete,
            on_back=lambda: self._show_screen(0),
            dropdowns_data=self.dropdowns_data
        )
        self.screens.append(prefs_screen)

        # Screen 2: Results (placeholder, will be created after search)
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
        self.screens[1].generate_btn.config(state=tk.DISABLED)
        self.screens[1]._hide_progress_bar()
        self.screens[1]._hide_models_list()
        self.screens[1].clear_models_list()
        self.screens[1].app_var.set('')
        self.screens[1].search_var.set('')
        self.screens[1].use_case_var.set(GENERAL)
        self._show_screen(1)  # Return to preferences

    def _on_preferences_complete(self):
        """Called when preferences are selected; search for models."""
        # Run search in background thread
        thread = threading.Thread(target=self._search_and_rank_models)
        thread.daemon = True
        thread.start()

    def _ui(self, func, *args):
        """Run func on the Tk main loop; Tkinter widgets must not be touched from worker threads."""
        self.root.after(0, func, *args)

    def _search_and_rank_models(self):
        """Search HuggingFace for GGUF models and rank them for this hardware (worker thread)."""
        try:
            hardware = self.state["hardware"]
            preferences = self.state["preferences"]
            search_param = preferences.get("search_param", "")
            use_case = preferences.get("use_case", GENERAL)
            prefs_screen = self.screens[1]

            # Top downloaded + most liked GGUF repos (plus a use-case keyword pass
            # when no search keyword was given), de-duplicated.
            candidates = gather_candidates(search_param, use_case)
            if not candidates:
                self._ui(self._search_failed, "No Results", "No models found. Try a different search parameter.")
                return

            self._ui(prefs_screen.set_progress_text, f"Checking {len(candidates)} models against your hardware...")

            def on_progress(done, total, result):
                self._ui(prefs_screen.set_progress_text, f"Checked {done}/{total} models...")
                if result is not None:
                    line = f"{result['model_name']}  ({result['quant']}, {result['fit_level']}, {result['run_mode']})"
                    self._ui(prefs_screen.add_model_to_list, line)

            evaluated = evaluate_candidates(candidates, hardware, use_case, progress_callback=on_progress)
            ranked = rank_models(evaluated, use_case=use_case, hardware=hardware)
            if not ranked:
                self._ui(self._search_failed, "No Results",
                         "None of the models found fit your hardware. Try a different search parameter.")
                return

            self.state["results"] = ranked
            self._ui(self._show_results, ranked)

        except Exception as e:
            self._ui(self._search_failed, "Error", f"Search failed: {str(e)}", True)

    def _search_failed(self, title, message, is_error=False):
        prefs_screen = self.screens[1]
        prefs_screen.generate_btn.config(state=tk.NORMAL)
        prefs_screen._hide_progress_bar()
        prefs_screen._hide_models_list()
        (messagebox.showerror if is_error else messagebox.showwarning)(title, message)

    def _show_results(self, ranked):
        prefs_screen = self.screens[1]
        prefs_screen._hide_progress_bar()
        prefs_screen._hide_models_list()

        results_screen = ResultsScreen(
            self.root, self.state,
            on_new_search=lambda: self._reset_and_new_search(),
            on_back=self._back_to_preferences,
            model_data=ranked
        )

        if len(self.screens) > 2:
            self.screens[2].frame.destroy()
            self.screens[2] = results_screen
        else:
            self.screens.append(results_screen)

        self._show_screen(2)

    def _back_to_preferences(self):
        self.screens[1].generate_btn.config(state=tk.NORMAL)
        self._show_screen(1)

def main():
    root = tk.Tk()
    app = LLMModelFinderApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
