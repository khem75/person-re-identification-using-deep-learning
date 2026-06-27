import os
import sys
import time
import csv
import threading
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import cv2
from PIL import Image, ImageTk

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from reid_core import (
    ReIDExtractor,
    build_gallery_index,
    save_feature_cache,
    load_feature_cache,
    search_topk,
    compute_cosine_similarity
)

# ----------------------------- Config ---------------------------------

GALLERY_DIR = "gallery"
QUERY_DIR = "query"
OUTPUT_DIR = "outputs"
CACHE_PATH = os.path.join(OUTPUT_DIR, "gallery_features.pkl")
EXPORT_CSV_PATH = os.path.join(OUTPUT_DIR, "search_history.csv")

AVAILABLE_MODELS = ["osnet_x0_25", "osnet_x0_5", "osnet_x1_0", "resnet50"]
DEFAULT_MODEL = "osnet_x0_25"
DEFAULT_THRESHOLD = 0.40
DEFAULT_TOP_K = 4

PANEL_W, PANEL_H = 380, 290
THUMB_W, THUMB_H = 140, 90

# --------------------------- Utilities --------------------------------

def ensure_dirs():
    os.makedirs(GALLERY_DIR, exist_ok=True)
    os.makedirs(QUERY_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

def read_image(path: str) -> Optional[np.ndarray]:
    return cv2.imread(path)

def to_pil_bgr(img_bgr: Optional[np.ndarray], max_w: int, max_h: int) -> Image.Image:
    if img_bgr is None:
        return Image.new("RGB", (max_w, max_h), (240, 242, 245))
    h, w = img_bgr.shape[:2]
    scale = min(max_w / w, max_h / h)
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(img_rgb).resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    canvas = Image.new("RGB", (max_w, max_h), (240, 242, 245))
    x = (max_w - new_w) // 2
    y = (max_h - new_h) // 2
    canvas.paste(pil, (x, y))
    return canvas

# ----------------------------- Tk App ---------------------------------

class ReIDApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Person Re-Identification Suite — Modern AI Studio")
        self.geometry("1240x800")
        self.resizable(False, False)
        self.configure(bg="#f4f6f9")

        ensure_dirs()
        self.extractor: Optional[ReIDExtractor] = None
        self.current_model_name = DEFAULT_MODEL
        self.gallery_db: Dict[str, np.ndarray] = {}
        
        self.query_files = [
            os.path.join(QUERY_DIR, f) for f in sorted(os.listdir(QUERY_DIR))
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
        ]
        self.query_idx = 0
        self.last_results: List[Tuple[str, float]] = []
        self._results_tk_imgs = []

        self._init_styles()
        self._build_ui()
        self._lazy_load_model(self.current_model_name)
        self._show_current_query()

    def _init_styles(self):
        self.style = ttk.Style(self)
        try:
            self.style.theme_use('clam')
        except Exception:
            pass
        
        # Customize colors
        self.style.configure(".", background="#f4f6f9", font=("Segoe UI", 10))
        self.style.configure("TLabelframe", background="#ffffff", relief="solid", borderwidth=1)
        self.style.configure("TLabelframe.Label", font=("Segoe UI", 11, "bold"), foreground="#1e293b", background="#ffffff")
        self.style.configure("TButton", font=("Segoe UI", 10, "bold"), padding=6)
        self.style.configure("Accent.TButton", font=("Segoe UI", 10, "bold"), foreground="#ffffff", background="#2563eb")
        self.style.map("Accent.TButton", background=[("active", "#1d4ed8")])

    def _build_ui(self):
        # Header Frame
        header = tk.Frame(self, bg="#1e293b", height=60)
        header.pack(fill="x", side="top")
        
        title_lbl = tk.Label(
            header, text="👤 PERSON RE-IDENTIFICATION STUDIO",
            font=("Segoe UI", 16, "bold"), fg="#ffffff", bg="#1e293b"
        )
        title_lbl.pack(side="left", padx=20, pady=12)

        # Model Selector in Header
        model_frame = tk.Frame(header, bg="#1e293b")
        model_frame.pack(side="right", padx=20)
        tk.Label(model_frame, text="Model:", font=("Segoe UI", 10, "bold"), fg="#94a3b8", bg="#1e293b").pack(side="left", padx=5)
        
        self.model_var = tk.StringVar(value=DEFAULT_MODEL)
        self.model_combo = ttk.Combobox(model_frame, textvariable=self.model_var, values=AVAILABLE_MODELS, state="readonly", width=14)
        self.model_combo.pack(side="left", padx=5)
        self.model_combo.bind("<<ComboboxSelected>>", self._on_model_change)

        # Control Toolbar (Threshold & Top-K)
        toolbar = tk.Frame(self, bg="#ffffff", height=45, highlightthickness=1, highlightbackground="#e2e8f0")
        toolbar.pack(fill="x", side="top", padx=15, pady=(10, 5))

        tk.Label(toolbar, text="Similarity Threshold:", font=("Segoe UI", 10, "bold"), bg="#ffffff", fg="#334155").pack(side="left", padx=(15, 5), pady=8)
        self.threshold_var = tk.DoubleVar(value=DEFAULT_THRESHOLD)
        self.threshold_slider = tk.Scale(
            toolbar, from_=0.0, to=1.0, resolution=0.05, orient="horizontal",
            variable=self.threshold_var, bg="#ffffff", highlightthickness=0, length=150, command=lambda v: self._on_slider_change()
        )
        self.threshold_slider.pack(side="left", padx=5)

        tk.Label(toolbar, text="Top Matches (K):", font=("Segoe UI", 10, "bold"), bg="#ffffff", fg="#334155").pack(side="left", padx=(25, 5))
        self.topk_var = tk.IntVar(value=DEFAULT_TOP_K)
        self.topk_spin = ttk.Spinbox(toolbar, from_=1, to=8, textvariable=self.topk_var, width=5, command=self._on_slider_change)
        self.topk_spin.pack(side="left", padx=5)

        ttk.Button(toolbar, text="💾 Export Match Log", command=self._export_log).pack(side="right", padx=15)

        # Main Workspace Container
        container = tk.Frame(self, bg="#f4f6f9")
        container.pack(fill="both", expand=True, padx=15, pady=5)

        # Left Panel: Gallery
        lf = ttk.LabelFrame(container, text=" GALLERY INDEX ")
        lf.place(x=0, y=0, width=390, height=610)

        self.gallery_canvas = tk.Canvas(lf, width=PANEL_W, height=PANEL_H, bg="#f1f5f9", highlightthickness=0)
        self.gallery_canvas.place(x=10, y=10)
        self.gallery_canvas.create_text(PANEL_W//2, PANEL_H//2, text="(Build index to preview gallery)", fill="#64748b", font=("Segoe UI", 10))

        self.thumb_frame = tk.Frame(lf, bg="#ffffff")
        self.thumb_frame.place(x=10, y=310, width=368, height=260)

        # Middle Panel: Query
        mf = ttk.LabelFrame(container, text=" QUERY PERSON ")
        mf.place(x=405, y=0, width=400, height=610)

        self.query_canvas = tk.Canvas(mf, width=PANEL_W, height=PANEL_H, bg="#ffffff", highlightthickness=2, highlightbackground="#3b82f6")
        self.query_canvas.place(x=10, y=10)

        self.query_info = tk.StringVar(value="ID: —   STATUS: Ready")
        info_lbl = tk.Label(mf, textvariable=self.query_info, font=("Segoe UI", 9), anchor="w", bg="#f8fafc", fg="#334155", relief="solid", bd=1, padx=6, pady=6)
        info_lbl.place(x=10, y=310, width=376)

        btn_frame = tk.Frame(mf, bg="#ffffff")
        btn_frame.place(x=10, y=360, width=376, height=200)

        ttk.Button(btn_frame, text="📂 Load Custom Query...", command=self._load_query_from_file).place(x=0, y=0, width=180)
        search_btn = ttk.Button(btn_frame, text="🔍 Run ReID Search", style="Accent.TButton", command=self._on_search)
        search_btn.place(x=190, y=0, width=186)

        ttk.Button(btn_frame, text="◀ Previous Query", command=self._prev_query).place(x=0, y=50, width=180)
        ttk.Button(btn_frame, text="Next Query ▶", command=self._next_query).place(x=190, y=50, width=186)

        # Right Panel: Search Results
        rf = ttk.LabelFrame(container, text=" TOP MATCHES ")
        rf.place(x=820, y=0, width=390, height=610)

        self.results_frame = tk.Frame(rf, bg="#ffffff")
        self.results_frame.place(x=5, y=5, width=378, height=570)
        
        self.results_placeholder = tk.Label(self.results_frame, text="Click 'Run ReID Search' to view matches", font=("Segoe UI", 10, "italic"), fg="#94a3b8", bg="#ffffff")
        self.results_placeholder.pack(expand=True)

        # Status Bar at Bottom
        bottom = tk.Frame(self, bg="#cbd5e1", height=35)
        bottom.pack(side="bottom", fill="x")

        ttk.Button(bottom, text="⚡ Build/Refresh Gallery Index", command=self._build_index_threaded).pack(side="left", padx=10, pady=3)
        ttk.Button(bottom, text="📁 Open Gallery", command=lambda: self._open_folder(GALLERY_DIR)).pack(side="left", padx=5)
        ttk.Button(bottom, text="📁 Open Queries", command=lambda: self._open_folder(QUERY_DIR)).pack(side="left", padx=5)

        self.status_var = tk.StringVar(value="System ready.")
        status_lbl = tk.Label(bottom, textvariable=self.status_var, font=("Segoe UI", 9, "bold"), bg="#cbd5e1", fg="#1e293b")
        status_lbl.pack(side="right", padx=15)

    # ---------------------- Model Logic ---------------------------

    def _lazy_load_model(self, model_name: str):
        self.status_var.set(f"Loading model ({model_name})...")
        self.update_idletasks()
        def load():
            try:
                self.extractor = ReIDExtractor(model_name=model_name, use_gpu=True)
                device_type = "GPU (CUDA)" if self.extractor.device.type == "cuda" else "CPU"
                self.after(0, lambda: self.status_var.set(f"Model {model_name} loaded successfully on {device_type}."))
            except Exception as e:
                self.after(0, lambda: messagebox.showerror("Model Error", f"Failed to load model {model_name}: {e}"))
                self.after(0, lambda: self.status_var.set("Model load failed."))
        threading.Thread(target=load, daemon=True).start()

    def _on_model_change(self, event=None):
        new_model = self.model_var.get()
        if new_model != self.current_model_name:
            self.current_model_name = new_model
            self._lazy_load_model(new_model)

    # ---------------------- Gallery Indexing --------------------------

    def _build_index_threaded(self):
        t = threading.Thread(target=self._build_index, daemon=True)
        t.start()

    def _build_index(self):
        if self.extractor is None:
            messagebox.showwarning("Model Warning", "Model is still loading. Please wait.")
            return
        self.status_var.set("Building gallery feature database...")
        start = time.time()
        try:
            db = build_gallery_index(self.extractor, GALLERY_DIR)
            save_feature_cache(db, CACHE_PATH)
            self.gallery_db = db
            elapsed = time.time() - start
            self.after(0, lambda: self.status_var.set(f"Gallery indexed: {len(db)} items in {elapsed:.2f}s."))
            self.after(0, self._render_gallery_thumbs)
        except Exception as e:
            self.after(0, lambda: self.status_var.set("Index build failed."))
            self.after(0, lambda: messagebox.showerror("Index Error", str(e)))

    def _render_gallery_thumbs(self):
        for w in self.thumb_frame.winfo_children():
            w.destroy()
        files = list(self.gallery_db.keys())
        if not files:
            self.gallery_canvas.delete("all")
            self.gallery_canvas.create_text(PANEL_W//2, PANEL_H//2, text="(No gallery images found)", fill="#64748b")
            return

        first = read_image(files[0])
        self.gallery_canvas.delete("all")
        self._gallery_tk = ImageTk.PhotoImage(to_pil_bgr(first, PANEL_W, PANEL_H))
        self.gallery_canvas.create_image(PANEL_W//2, PANEL_H//2, image=self._gallery_tk)

        cols = 2
        for i, p in enumerate(files[:6]):
            img = read_image(p)
            pil = to_pil_bgr(img, THUMB_W, THUMB_H)
            tkimg = ImageTk.PhotoImage(pil)
            lbl = tk.Label(self.thumb_frame, image=tkimg, bg="#ffffff", bd=1, relief="solid")
            lbl.image = tkimg
            r, c = divmod(i, cols)
            lbl.grid(row=r, column=c, padx=5, pady=5)

    # ------------------------- Query Nav -------------------------------

    def _prev_query(self):
        if not self.query_files:
            messagebox.showinfo("Query", "No images found in query folder.")
            return
        self.query_idx = (self.query_idx - 1) % len(self.query_files)
        self._show_current_query()

    def _next_query(self):
        if not self.query_files:
            messagebox.showinfo("Query", "No images found in query folder.")
            return
        self.query_idx = (self.query_idx + 1) % len(self.query_files)
        self._show_current_query()

    def _show_current_query(self):
        if not self.query_files:
            self.query_canvas.delete("all")
            self._query_tk = ImageTk.PhotoImage(to_pil_bgr(None, PANEL_W, PANEL_H))
            self.query_canvas.create_image(PANEL_W//2, PANEL_H//2, image=self._query_tk)
            self.query_info.set("ID: —   STATUS: No Query Image")
            return
        path = self.query_files[self.query_idx]
        img = read_image(path)
        self.query_canvas.delete("all")
        pil = to_pil_bgr(img, PANEL_W, PANEL_H)
        self._query_tk = ImageTk.PhotoImage(pil)
        self.query_canvas.create_image(PANEL_W//2, PANEL_H//2, image=self._query_tk)
        self.query_info.set(f"ID: {Path(path).name}   |   Status: Ready")

    def _load_query_from_file(self):
        fp = filedialog.askopenfilename(title="Select query image", filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp")])
        if not fp:
            return
        if fp not in self.query_files:
            self.query_files.append(fp)
            self.query_idx = len(self.query_files) - 1
        else:
            self.query_idx = self.query_files.index(fp)
        self._show_current_query()

    # ------------------------- Search ---------------------------------

    def _on_search(self):
        if self.extractor is None:
            messagebox.showwarning("Model Warning", "Model is not ready yet.")
            return
        if not os.path.exists(CACHE_PATH):
            messagebox.showinfo("Index Required", "Please click 'Build/Refresh Gallery Index' first.")
            return
        if not self.query_files:
            messagebox.showinfo("Query Warning", "No query image selected.")
            return

        try:
            self.gallery_db = load_feature_cache(CACHE_PATH)
        except Exception as e:
            messagebox.showerror("Cache Error", f"Failed to load cache: {e}")
            return

        qpath = self.query_files[self.query_idx]
        img = read_image(qpath)
        if img is None:
            messagebox.showerror("Image Error", "Failed to read query image.")
            return

        self.status_var.set("Extracting query feature & searching...")
        
        def work():
            try:
                qfeat = self.extractor.extract_feature(img)
                thresh = self.threshold_var.get()
                top_k = self.topk_var.get()
                results = search_topk(qfeat, self.gallery_db, top_k=top_k, threshold=thresh)
                self.last_results = results
                
                self.after(0, lambda: self._show_search_results(results))
                self.after(0, lambda: self.status_var.set(f"Search completed. Found {len(results)} match(es)."))
            except Exception as e:
                self.after(0, lambda: messagebox.showerror("Search Error", str(e)))
                self.after(0, lambda: self.status_var.set("Search failed."))

        threading.Thread(target=work, daemon=True).start()

    def _on_slider_change(self):
        if self.last_results and self.query_files:
            self._on_search()

    def _show_search_results(self, results: List[Tuple[str, float]]):
        for w in self.results_frame.winfo_children():
            w.destroy()
        self._results_tk_imgs = []

        if not results:
            lbl = tk.Label(
                self.results_frame, text="❌ No matches found above threshold",
                font=("Segoe UI", 10, "bold"), fg="#ef4444", bg="#ffffff"
            )
            lbl.pack(expand=True)
            return

        cols = 2
        thumb_w, thumb_h = 165, 105
        
        for i, (path, score) in enumerate(results):
            img = read_image(path)
            pil = to_pil_bgr(img, thumb_w, thumb_h)
            tkimg = ImageTk.PhotoImage(pil)
            self._results_tk_imgs.append(tkimg)

            item_frame = tk.Frame(self.results_frame, bg="#ffffff", bd=1, relief="solid")
            
            img_label = tk.Label(item_frame, image=tkimg, bg="#ffffff")
            img_label.pack(padx=3, pady=(3, 1))
            
            conf_pct = score * 100.0
            # Color badge based on score
            if conf_pct >= 70.0:
                badge_bg, badge_fg = "#10b981", "#ffffff"  # Green
            elif conf_pct >= 50.0:
                badge_bg, badge_fg = "#f59e0b", "#ffffff"  # Yellow/Orange
            else:
                badge_bg, badge_fg = "#64748b", "#ffffff"  # Gray
            
            badge = tk.Label(
                item_frame, text=f"Rank {i+1}: {conf_pct:.1f}%",
                font=("Segoe UI", 9, "bold"), bg=badge_bg, fg=badge_fg, padx=4, pady=2
            )
            badge.pack(fill="x", pady=(0, 2))
            
            r, c = divmod(i, cols)
            item_frame.grid(row=r, column=c, padx=8, pady=8)

    # ------------------------- Export & Helpers --------------------------------

    def _export_log(self):
        if not self.last_results:
            messagebox.showinfo("Export Log", "No recent search results to export.")
            return
        
        qpath = self.query_files[self.query_idx]
        file_exists = os.path.exists(EXPORT_CSV_PATH)
        
        try:
            with open(EXPORT_CSV_PATH, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(["Timestamp", "Model", "Query_Image", "Rank", "Matched_Gallery_Image", "Similarity_Score"])
                
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                for rank, (gpath, score) in enumerate(self.last_results, start=1):
                    writer.writerow([timestamp, self.current_model_name, Path(qpath).name, rank, Path(gpath).name, f"{score:.4f}"])
            
            messagebox.showinfo("Export Success", f"Results successfully exported to:\n{EXPORT_CSV_PATH}")
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to write log: {e}")

    def _open_folder(self, path: str):
        path = os.path.abspath(path)
        if sys.platform.startswith("win"):
            os.startfile(path)
        elif sys.platform == "darwin":
            os.system(f'open "{path}"')
        else:
            os.system(f'xdg-open "{path}"')

if __name__ == "__main__":
    app = ReIDApp()
    app.mainloop()