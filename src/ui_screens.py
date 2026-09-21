# src/ui_screens.py
import tkinter as tk
from tkinter import ttk, messagebox
import threading
import webbrowser

from fit_engine import USE_CASES, GENERAL
from app_integrator import ollama_run_command, llama_cpp_run_command

BACKEND_LABELS = {
    "cuda": "CUDA (NVIDIA)", "rocm": "ROCm (AMD)", "vulkan": "Vulkan", "metal": "Metal (Apple)",
    "sycl": "SYCL (Intel)", "cpu_x86": "CPU (x86)", "cpu_arm": "CPU (ARM)",
}

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
        title = tk.Label(self.frame, text="Welcome to LLM Model Finder", font=("Arial", 20, "bold"))
        title.pack(pady=20)

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
        # Tkinter is not thread-safe: hand the result back to the main loop.
        self.frame.after(0, self._on_detection_complete, hw)

    def _on_detection_complete(self, hw):
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
        threads = hw['cpu'].get('threads')
        cores_text = f"Cores: {hw['cpu']['cores']}" + (f" ({threads} threads)" if threads else "")
        tk.Label(cpu_frame, text=cores_text).pack(anchor=tk.W)

        # RAM
        ram_frame = tk.LabelFrame(self.info_frame, text="Memory", padx=10, pady=10)
        ram_frame.pack(fill=tk.X, padx=10, pady=5)
        tk.Label(ram_frame, text=f"Total: {hw['ram']['total_gb']} GB").pack(anchor=tk.W)
        tk.Label(ram_frame, text=f"Available: {hw['ram']['available_gb']} GB").pack(anchor=tk.W)

        # GPU
        gpu_frame = tk.LabelFrame(self.info_frame, text="Video Card", padx=10, pady=10)
        gpu_frame.pack(fill=tk.X, padx=10, pady=5)
        tk.Label(gpu_frame, text=f"Model: {hw['gpu']['model']}").pack(anchor=tk.W)
        if hw['gpu'].get('unified_memory'):
            tk.Label(gpu_frame, text=f"Unified memory: {hw['gpu']['vram_gb']} GB (shared with CPU)").pack(anchor=tk.W)
        else:
            tk.Label(gpu_frame, text=f"VRAM: {hw['gpu']['vram_gb']} GB").pack(anchor=tk.W)
            if hw['gpu'].get('vram_free_gb', 0) > 0:
                tk.Label(gpu_frame, text=f"Free VRAM: {hw['gpu']['vram_free_gb']} GB").pack(anchor=tk.W)
        backend = hw['gpu'].get('backend')
        if backend:
            tk.Label(gpu_frame, text=f"Backend: {BACKEND_LABELS.get(backend, backend)}").pack(anchor=tk.W)
        if not hw['gpu'].get('vram_gb'):
            tk.Label(gpu_frame, text="No usable VRAM detected: models will be sized against system RAM.",
                     fg="gray").pack(anchor=tk.W)

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

        # Use case dropdown (drives scoring weights and extra search keywords)
        use_case_label = tk.Label(self.frame, text="What will you use it for?", font=("Arial", 10))
        use_case_label.pack(pady=5)
        self.use_case_var = tk.StringVar(value=GENERAL)
        self.use_case_combo = ttk.Combobox(self.frame, textvariable=self.use_case_var,
                                           values=USE_CASES, state='readonly', width=40)
        self.use_case_combo.pack(pady=5)

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

    def set_progress_text(self, text):
        """Update the status line under the progress bar."""
        self.progress_label.config(text=text)

    def add_model_to_list(self, model_name):
        """Add a model name to the models list display."""
        self.models_text.config(state=tk.NORMAL)
        self.models_text.insert(tk.END, f"- {model_name}\n")
        self.models_text.see(tk.END)  # Auto-scroll to bottom
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
            "use_case": self.use_case_var.get() or GENERAL,
            "search_param": search_param if search_param else ""
        }
        self.generate_btn.config(state=tk.DISABLED)
        self.set_progress_text("Searching Hugging Face for GGUF models...")
        self._show_progress_bar()
        self._show_models_list()
        self.clear_models_list()
        self.on_next()

def _format_context(tokens):
    if not tokens:
        return "?"
    return f"{tokens // 1024}k" if tokens >= 1024 else str(tokens)


class ResultsScreen(Screen):
    """Screen 4: Display top 10 models ranked for this hardware."""
    COLUMNS = ("Rank", "Model Name", "Params (B)", "Quant", "File Size", "Memory", "Fit",
               "Run Mode", "Est. Speed", "Context", "Score")
    COLUMN_WIDTHS = {"Rank": 45, "Model Name": 330, "Params (B)": 75, "Quant": 90, "File Size": 75,
                     "Memory": 75, "Fit": 70, "Run Mode": 90, "Est. Speed": 80, "Context": 60, "Score": 55}
    # Numeric sort keys for columns whose display text is not directly sortable.
    SORT_KEYS = {
        "Rank": lambda m: m["_rank"],
        "Params (B)": lambda m: m.get("params_b") or 0,
        "File Size": lambda m: m.get("file_size_gb") or 0,
        "Memory": lambda m: m.get("vram_needed") or 0,
        "Fit": lambda m: {"Perfect": 3, "Good": 2, "Marginal": 1}.get(m.get("fit_level"), 0),
        "Est. Speed": lambda m: m.get("est_tokens_per_sec") or 0,
        "Context": lambda m: m.get("context_length") or 0,
        "Score": lambda m: m.get("final_score") or 0,
    }

    def __init__(self, parent, session_state, on_new_search, on_back, model_data):
        super().__init__(parent, session_state)
        self.on_new_search = on_new_search
        self.on_back = on_back
        self.model_data = model_data[:10]  # List of top 10 models
        for idx, model in enumerate(self.model_data, 1):
            model["_rank"] = idx
        self._sort_state = {}
        self._build_ui()

    def _build_ui(self):
        title = tk.Label(self.frame, text="Top 10 Models", font=("Arial", 14, "bold"))
        title.pack(pady=(10, 2))

        summary = self._hardware_summary()
        if summary:
            tk.Label(self.frame, text=summary, font=("Arial", 9), fg="gray").pack()

        table_frame = tk.Frame(self.frame)
        table_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.tree = ttk.Treeview(table_frame, columns=self.COLUMNS, height=12, show='headings')
        for col in self.COLUMNS:
            anchor = tk.W if col == "Model Name" else tk.CENTER
            self.tree.column(col, width=self.COLUMN_WIDTHS[col], anchor=anchor, stretch=(col == "Model Name"))
            self.tree.heading(col, text=col, command=lambda c=col: self._sort_by(c))
        xscroll = ttk.Scrollbar(table_frame, orient=tk.HORIZONTAL, command=self.tree.xview)
        self.tree.configure(xscrollcommand=xscroll.set)
        self.tree.pack(fill=tk.BOTH, expand=True)
        xscroll.pack(fill=tk.X)

        self.tree.tag_configure("Perfect", foreground="#1b7f3a")
        self.tree.tag_configure("Marginal", foreground="#a15c00")
        self._populate(self.model_data)
        self.tree.bind("<<TreeviewSelect>>", self._on_select)
        self.tree.bind("<Double-1>", self._open_model_page)

        self.details_var = tk.StringVar(value="Select a model for details. Double-click to open it on Hugging Face.")
        tk.Label(self.frame, textvariable=self.details_var, font=("Arial", 9), justify=tk.LEFT,
                 anchor=tk.W, wraplength=1000).pack(fill=tk.X, padx=12)

        # Buttons
        button_frame = tk.Frame(self.frame)
        button_frame.pack(pady=10)

        copy_btn = tk.Button(button_frame, text="Copy to Clipboard", command=self._copy_to_clipboard)
        copy_btn.pack(side=tk.LEFT, padx=5)

        self.ollama_btn = tk.Button(button_frame, text="Copy Ollama Command", state=tk.DISABLED,
                                    command=lambda: self._copy_run_command("Ollama", ollama_run_command))
        self.ollama_btn.pack(side=tk.LEFT, padx=5)

        self.llama_cpp_btn = tk.Button(button_frame, text="Copy llama.cpp Command", state=tk.DISABLED,
                                       command=lambda: self._copy_run_command("llama.cpp", llama_cpp_run_command))
        self.llama_cpp_btn.pack(side=tk.LEFT, padx=5)

        open_btn = tk.Button(button_frame, text="Open on Hugging Face", command=self._open_model_page)
        open_btn.pack(side=tk.LEFT, padx=5)

        back_btn = tk.Button(button_frame, text="Back", command=self.on_back)
        back_btn.pack(side=tk.LEFT, padx=5)

        new_search_btn = tk.Button(button_frame, text="New Search", command=self.on_new_search)
        new_search_btn.pack(side=tk.LEFT, padx=5)

    def _hardware_summary(self):
        hw = self.session_state.get("hardware") or {}
        prefs = self.session_state.get("preferences") or {}
        gpu, ram = hw.get("gpu", {}), hw.get("ram", {})
        parts = []
        if gpu.get("vram_gb"):
            parts.append(f"{gpu.get('model')} ({gpu['vram_gb']} GB)")
        else:
            parts.append("CPU only")
        if ram:
            parts.append(f"{ram.get('available_gb')} of {ram.get('total_gb')} GB RAM free")
        if prefs.get("use_case"):
            parts.append(f"Use case: {prefs['use_case']}")
        return "  |  ".join(parts)

    @staticmethod
    def _row_values(model):
        return (
            model["_rank"],
            model.get('model_name', 'Unknown'),  # Do not truncate names
            model.get('params_b', '?'),
            model.get('quant', '?'),
            f"{model.get('file_size_gb', '?')} GB",
            f"{model.get('vram_needed', '?')} GB",
            model.get('fit_level', '?'),
            model.get('run_mode', '?'),
            f"{model.get('est_tokens_per_sec', '?')} t/s",
            _format_context(model.get('context_length')),
            model.get('final_score', '?'),
        )

    def _populate(self, models):
        self.tree.delete(*self.tree.get_children())
        self._row_models = {}
        # Rebuilding the table clears the selection, so there is nothing to copy yet.
        for button in (getattr(self, "ollama_btn", None), getattr(self, "llama_cpp_btn", None)):
            if button is not None:
                button.config(state=tk.DISABLED)
        for model in models:
            item = self.tree.insert('', 'end', values=self._row_values(model), tags=(model.get("fit_level", ""),))
            self._row_models[item] = model

    def _sort_by(self, column):
        if column in self._sort_state:
            descending = not self._sort_state[column]
        else:
            # Numbers read best high-to-low; rank and text columns start ascending.
            descending = column not in ("Rank", "Model Name", "Quant", "Run Mode")
        self._sort_state = {column: descending}
        if column in self.SORT_KEYS:
            key = self.SORT_KEYS[column]
        else:
            index = self.COLUMNS.index(column)
            key = lambda m: str(self._row_values(m)[index]).lower()
        self._populate(sorted(self.model_data, key=key, reverse=descending))

    def _selected_model(self):
        selection = self.tree.selection()
        return self._row_models.get(selection[0]) if selection else None

    def _on_select(self, event=None):
        model = self._selected_model()
        if not model:
            return
        self.ollama_btn.config(state=tk.NORMAL)
        self.llama_cpp_btn.config(state=tk.NORMAL)
        details = [
            f"{model.get('model_name')}  /  {model.get('file_name', model.get('quant'))}",
            f"Scores: quality {model.get('quality_score')}, speed {model.get('speed_score')}, "
            f"fit {model.get('fit_score')}, context {model.get('context_score')}",
        ]
        if model.get("utilization_pct") is not None:
            details.append(f"Uses {model['utilization_pct']}% of {model.get('memory_available_gb')} GB "
                           f"({model.get('run_mode')} pool)")
        if model.get("is_moe"):
            active = model.get("active_params_b")
            details.append("Mixture-of-Experts" + (f", ~{active}B active per token" if active else ""))
        details.extend(model.get("notes", []))
        self.details_var.set("\n".join(details))

    def _copy_run_command(self, app_label, build_command):
        """Copy the command that downloads and runs the selected model in an inference app."""
        model = self._selected_model()
        if not model:
            return
        command, note = build_command(model)
        self.frame.clipboard_clear()
        self.frame.clipboard_append(command)
        message = f"Copied to clipboard. Paste it into a terminal:\n\n{command}"
        if note:
            message += f"\n\nNote: {note}"
        messagebox.showinfo(f"{app_label} Command", message)

    def _open_model_page(self, event=None):
        model = self._selected_model()
        if model:
            webbrowser.open(f"https://huggingface.co/{model['model_name']}")

    def _copy_to_clipboard(self):
        # Extract data from tree and copy
        text = "\t".join(self.COLUMNS) + "\n"
        for item in self.tree.get_children():
            values = self.tree.item(item)['values']
            text += "\t".join(str(v) for v in values) + "\n"

        self.frame.clipboard_clear()
        self.frame.clipboard_append(text)
        messagebox.showinfo("Success", "Table copied to clipboard!")
