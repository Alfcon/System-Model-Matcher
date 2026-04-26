# src/ui_screens.py
import tkinter as tk
from tkinter import ttk, messagebox
import threading

class Screen:
    """Base screen class."""
    def __init__(self, parent, session_state):
        self.parent = parent
        self.session_state = session_state
        self.frame = tk.Frame(parent)

    def show(self):
        self.frame.pack(fill=tk.BOTH, expand=True)

    def hide(self):
        self.frame.pack_forget()

class ProfileScreen(Screen):
    """Screen 1: User profile (name, location)."""
    def __init__(self, parent, session_state, on_next):
        super().__init__(parent, session_state)
        self.on_next = on_next
        self._build_ui()

    def _build_ui(self):
        title = tk.Label(self.frame, text="Welcome to LLM Model Finder", font=("Arial", 16, "bold"))
        title.pack(pady=20)

        # Name
        name_label = tk.Label(self.frame, text="Your Name (required):", font=("Arial", 10))
        name_label.pack(pady=5)
        self.name_entry = tk.Entry(self.frame, width=30)
        self.name_entry.pack(pady=5)

        # Location
        location_label = tk.Label(self.frame, text="Location (optional):", font=("Arial", 10))
        location_label.pack(pady=5)
        self.location_entry = tk.Entry(self.frame, width=30)
        self.location_entry.pack(pady=5)

        # Buttons
        button_frame = tk.Frame(self.frame)
        button_frame.pack(pady=20)

        next_btn = tk.Button(button_frame, text="Next", command=self._on_next)
        next_btn.pack(side=tk.LEFT, padx=10)

        quit_btn = tk.Button(button_frame, text="Quit", command=self.parent.quit)
        quit_btn.pack(side=tk.LEFT, padx=10)

    def _on_next(self):
        name = self.name_entry.get().strip()
        location = self.location_entry.get().strip()

        if not name:
            messagebox.showwarning("Validation", "Please enter your name.")
            return

        self.session_state["name"] = name
        self.session_state["location"] = location
        self.on_next()

class HardwareScreen(Screen):
    """Screen 2: Display detected hardware."""
    def __init__(self, parent, session_state, on_next, on_back, hardware_detector):
        super().__init__(parent, session_state)
        self.on_next = on_next
        self.on_back = on_back
        self.hardware_detector = hardware_detector
        self._build_ui()
        self.detection_complete = False

    def _build_ui(self):
        title = tk.Label(self.frame, text="System Hardware Detection", font=("Arial", 14, "bold"))
        title.pack(pady=20)

        self.status_label = tk.Label(self.frame, text="Detecting hardware...", font=("Arial", 10))
        self.status_label.pack(pady=10)

        # Hardware display (will be populated)
        self.info_frame = tk.Frame(self.frame)
        self.info_frame.pack(pady=10, fill=tk.BOTH, expand=True)

        # Buttons
        button_frame = tk.Frame(self.frame)
        button_frame.pack(pady=20)

        self.next_btn = tk.Button(button_frame, text="Next", command=self.on_next, state=tk.DISABLED)
        self.next_btn.pack(side=tk.LEFT, padx=10)

        back_btn = tk.Button(button_frame, text="Back", command=self.on_back)
        back_btn.pack(side=tk.LEFT, padx=10)

    def show(self):
        super().show()
        if not self.detection_complete:
            # Run detection in background thread
            thread = threading.Thread(target=self._detect_hardware)
            thread.daemon = True
            thread.start()

    def _detect_hardware(self):
        hw = self.hardware_detector.detect_all()
        self.session_state["hardware"] = hw
        self._display_hardware(hw)
        self.detection_complete = True
        self.next_btn.config(state=tk.NORMAL)

    def _display_hardware(self, hw):
        # Clear previous info
        for widget in self.info_frame.winfo_children():
            widget.destroy()

        # CPU
        cpu_frame = tk.LabelFrame(self.info_frame, text="Processor", padx=10, pady=10)
        cpu_frame.pack(fill=tk.X, padx=10, pady=5)
        tk.Label(cpu_frame, text=f"Model: {hw['cpu']['model']}").pack(anchor=tk.W)
        tk.Label(cpu_frame, text=f"Cores: {hw['cpu']['cores']}").pack(anchor=tk.W)

        # RAM
        ram_frame = tk.LabelFrame(self.info_frame, text="Memory", padx=10, pady=10)
        ram_frame.pack(fill=tk.X, padx=10, pady=5)
        tk.Label(ram_frame, text=f"Total: {hw['ram']['total_gb']} GB").pack(anchor=tk.W)
        tk.Label(ram_frame, text=f"Available: {hw['ram']['available_gb']} GB").pack(anchor=tk.W)

        # GPU
        gpu_frame = tk.LabelFrame(self.info_frame, text="Video Card", padx=10, pady=10)
        gpu_frame.pack(fill=tk.X, padx=10, pady=5)
        tk.Label(gpu_frame, text=f"Model: {hw['gpu']['model']}").pack(anchor=tk.W)
        tk.Label(gpu_frame, text=f"VRAM: {hw['gpu']['vram_gb']} GB").pack(anchor=tk.W)
        if 'vram_free_gb' in hw['gpu'] and hw['gpu']['vram_free_gb'] > 0:
            tk.Label(gpu_frame, text=f"Free VRAM: {hw['gpu']['vram_free_gb']} GB").pack(anchor=tk.W)

        self.status_label.config(text="✓ Hardware detection complete")

class PreferencesScreen(Screen):
    """Screen 3: Select task, app, theme."""
    def __init__(self, parent, session_state, on_next, on_back, dropdowns_data):
        super().__init__(parent, session_state)
        self.on_next = on_next
        self.on_back = on_back
        self.dropdowns_data = dropdowns_data  # dict with 'tasks', 'apps', 'themes'
        self._build_ui()

    def _build_ui(self):
        title = tk.Label(self.frame, text="LLM Model Finder", font=("Arial", 14, "bold"))
        title.pack(pady=20)

        # App dropdown
        app_label = tk.Label(self.frame, text="Select your inference app:", font=("Arial", 10))
        app_label.pack(pady=5)
        self.app_var = tk.StringVar()
        self.app_combo = ttk.Combobox(self.frame, textvariable=self.app_var,
                                       values=self.dropdowns_data['apps'], state='readonly', width=40)
        self.app_combo.pack(pady=5)

        # Search parameter field
        search_label = tk.Label(self.frame, text="Search parameter (optional):", font=("Arial", 10))
        search_label.pack(pady=5)
        info_text = tk.Label(self.frame, text="Leave blank for best general models, or enter: mistral, llama, neural, uncensored, etc.",
                            font=("Arial", 8, "italic"), fg="gray")
        info_text.pack(pady=2)

        self.search_var = tk.StringVar()
        self.search_entry = tk.Entry(self.frame, textvariable=self.search_var, width=40)
        self.search_entry.pack(pady=5)
        self.search_entry.bind('<Return>', lambda e: self._on_generate() if self.app_var.get() else None)

        # Buttons
        button_frame = tk.Frame(self.frame)
        button_frame.pack(pady=20)

        self.generate_btn = tk.Button(button_frame, text="Find Top 10 Models", command=self._on_generate, state=tk.DISABLED)
        self.generate_btn.pack(side=tk.LEFT, padx=10)

        back_btn = tk.Button(button_frame, text="Back", command=self.on_back)
        back_btn.pack(side=tk.LEFT, padx=10)

        # Enable generate button when app is selected (search param is optional)
        self.app_combo.bind('<<ComboboxSelected>>', self._check_selections)

        # Progress bar (initially hidden)
        self.progress_frame = tk.Frame(self.frame)
        self.progress_frame.pack(pady=10)

        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            self.progress_frame,
            variable=self.progress_var,
            maximum=100,
            mode='indeterminate',
            length=400
        )
        self.progress_bar.pack(pady=5)

        self.progress_label = tk.Label(self.progress_frame, text="Searching for models...", font=("Arial", 9))
        self.progress_label.pack(pady=2)

        self._hide_progress_bar()

        # Models list display (scrollable)
        self.models_frame = tk.Frame(self.frame)
        self.models_frame.pack(pady=10, fill=tk.BOTH, expand=True, padx=10)

        models_label = tk.Label(self.models_frame, text="Models found:", font=("Arial", 9, "bold"))
        models_label.pack(anchor=tk.W)

        # Scrollbar
        scrollbar = tk.Scrollbar(self.models_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Text widget for displaying models
        self.models_text = tk.Text(
            self.models_frame,
            height=8,
            width=60,
            yscrollcommand=scrollbar.set,
            font=("Courier", 8),
            bg="white",
            fg="black"
        )
        self.models_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.models_text.config(state=tk.DISABLED)  # Read-only
        scrollbar.config(command=self.models_text.yview)

        self._hide_models_list()

    def _check_selections(self, event=None):
        if self.app_var.get():
            self.generate_btn.config(state=tk.NORMAL)
        else:
            self.generate_btn.config(state=tk.DISABLED)

    def _show_progress_bar(self):
        """Show and start the progress bar animation."""
        self.progress_frame.pack(pady=10)
        self.progress_bar.start()

    def _hide_progress_bar(self):
        """Hide and stop the progress bar animation."""
        self.progress_bar.stop()
        self.progress_frame.pack_forget()

    def _show_models_list(self):
        """Show the models list display."""
        self.models_frame.pack(pady=10, fill=tk.BOTH, expand=True, padx=10)
        self.models_text.config(state=tk.NORMAL)
        self.models_text.delete('1.0', tk.END)
        self.models_text.config(state=tk.DISABLED)

    def _hide_models_list(self):
        """Hide the models list display."""
        self.models_frame.pack_forget()

    def add_model_to_list(self, model_name):
        """Add a model name to the models list display."""
        self.models_text.config(state=tk.NORMAL)
        self.models_text.insert(tk.END, f"• {model_name}\n")
        self.models_text.see(tk.END)  # Auto-scroll to bottom
        self.models_text.update()  # Refresh display
        self.models_text.config(state=tk.DISABLED)

    def clear_models_list(self):
        """Clear the models list display."""
        self.models_text.config(state=tk.NORMAL)
        self.models_text.delete('1.0', tk.END)
        self.models_text.config(state=tk.DISABLED)

    def _on_generate(self):
        search_param = self.search_var.get().strip()
        self.session_state["preferences"] = {
            "app": self.app_var.get(),
            "search_param": search_param if search_param else ""
        }
        self.generate_btn.config(state=tk.DISABLED)
        self._show_progress_bar()
        self._show_models_list()
        self.clear_models_list()
        self.on_next()

class ResultsScreen(Screen):
    """Screen 4: Display top 10 models with download/integrate options."""
    def __init__(self, parent, session_state, on_new_search, on_back, model_data):
        super().__init__(parent, session_state)
        self.on_new_search = on_new_search
        self.on_back = on_back
        self.model_data = model_data  # List of top 10 models
        self._build_ui()

    def _build_ui(self):
        title = tk.Label(self.frame, text="Top 10 Models", font=("Arial", 14, "bold"))
        title.pack(pady=10)

        # Create treeview for models
        columns = ("Rank", "Model Name", "Parameters / Billion", "Quant", "File Size", "Est. VRAM", "Est. Speed")
        self.tree = ttk.Treeview(self.frame, columns=columns, height=12, show='headings')

        for col in columns:
            self.tree.column(col, width=100)
            self.tree.heading(col, text=col)

        self.tree.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Populate with data
        for idx, model in enumerate(self.model_data[:10], 1):
            self.tree.insert('', 'end', values=(
                idx,
                model.get('model_name', 'Unknown'),  # Do not truncate names
                model.get('params_b', '?'),
                model.get('quant', '?'),
                f"{model.get('file_size_gb', '?')} GB",
                f"{model.get('vram_needed', '?')} GB",
                f"{model.get('est_tokens_per_sec', '?')} t/s"
            ))

        # Buttons
        button_frame = tk.Frame(self.frame)
        button_frame.pack(pady=10)

        copy_btn = tk.Button(button_frame, text="Copy to Clipboard", command=self._copy_to_clipboard)
        copy_btn.pack(side=tk.LEFT, padx=5)

        back_btn = tk.Button(button_frame, text="Back", command=self.on_back)
        back_btn.pack(side=tk.LEFT, padx=5)

        new_search_btn = tk.Button(button_frame, text="New Search", command=self.on_new_search)
        new_search_btn.pack(side=tk.LEFT, padx=5)

    def _copy_to_clipboard(self):
        # Extract data from tree and copy
        text = "Rank\tModel Name\tParameters / Billion\tQuant\tFile Size\tEst. VRAM\tEst. Speed\n"
        for item in self.tree.get_children():
            values = self.tree.item(item)['values']
            text += "\t".join(str(v) for v in values) + "\n"

        self.frame.clipboard_clear()
        self.frame.clipboard_append(text)
        messagebox.showinfo("Success", "Table copied to clipboard!")
