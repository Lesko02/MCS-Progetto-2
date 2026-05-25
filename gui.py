import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
from PIL import Image, ImageTk

from compressione import compress_image, load_bmp_grayscale


MAX_DISPLAY_SIDE = 480


def fit_to_thumbnail(arr, max_side=MAX_DISPLAY_SIDE):
    """Ridimensiona l'array (uint8 2D) a uno PIL Image che entra in
    max_side x max_side, mantenendo le proporzioni. Solo per la visualizzazione:
    l'algoritmo lavora sempre sui dati originali."""
    img = Image.fromarray(arr, mode='L')
    img.thumbnail((max_side, max_side), Image.LANCZOS)
    return img


class DCTApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Compressione DCT2 — Progetto MCS 2025-2026")
        self.geometry("1100x720")
        self.minsize(900, 600)

        self.img_array = None
        self.img_path = None
        self.img_compressed = None  

        self.F_var = tk.IntVar(value=8)
        self.d_var = tk.IntVar(value=8)
        self.path_var = tk.StringVar(value="(nessuna immagine caricata)")
        self.info_var = tk.StringVar(value="")

        self._build_ui()

    def _build_ui(self):
        pad = {'padx': 8, 'pady': 6}

        top = ttk.Frame(self)
        top.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(top, text="Scegli immagine .bmp...",
                   command=self.on_choose_file).pack(side=tk.LEFT, **pad)
        ttk.Label(top, textvariable=self.path_var, foreground='#444').pack(
            side=tk.LEFT, **pad)

        params = ttk.LabelFrame(self, text="Parametri di compressione")
        params.pack(side=tk.TOP, fill=tk.X, padx=8, pady=4)

        ttk.Label(params, text="F  (lato del macro-blocco):").grid(
            row=0, column=0, sticky='w', **pad)
        self.F_spin = ttk.Spinbox(params, from_=1, to=512, width=6,
                                  textvariable=self.F_var,
                                  command=self._on_F_change)
        self.F_spin.grid(row=0, column=1, sticky='w', **pad)

        ttk.Label(params, text="d  (soglia di taglio, 0 ≤ d ≤ 2F-2):").grid(
            row=0, column=2, sticky='w', **pad)
        self.d_spin = ttk.Spinbox(params, from_=0, to=14, width=6,
                                  textvariable=self.d_var)
        self.d_spin.grid(row=0, column=3, sticky='w', **pad)

        ttk.Button(params, text="Esegui compressione",
                   command=self.on_run).grid(row=0, column=4, **pad)
        ttk.Button(params, text="Salva immagine compressa...",
                   command=self.on_save).grid(row=0, column=5, **pad)

        ttk.Label(self, textvariable=self.info_var,
                  foreground='#0a4', font=('TkDefaultFont', 10, 'italic')).pack(
            side=tk.TOP, fill=tk.X, padx=12)

        body = ttk.Frame(self)
        body.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=8, pady=8)

        left = ttk.LabelFrame(body, text="Originale")
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4)
        self.canvas_orig = tk.Label(left, background='#222')
        self.canvas_orig.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        right = ttk.LabelFrame(body, text="Compressa")
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4)
        self.canvas_comp = tk.Label(right, background='#222')
        self.canvas_comp.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        self._update_d_bounds()

    def _update_d_bounds(self):
        F = self.F_var.get()
        dmax = max(0, 2 * F - 2)
        self.d_spin.configure(from_=0, to=dmax)
        if self.d_var.get() > dmax:
            self.d_var.set(dmax)

    def _on_F_change(self):
        try:
            self._update_d_bounds()
        except Exception:
            pass

    def _show_array(self, arr, target):
        pil = fit_to_thumbnail(arr)
        ph = ImageTk.PhotoImage(pil)
        target.configure(image=ph)
        target.image = ph  

    def on_choose_file(self):
        path = filedialog.askopenfilename(
            title="Scegli un'immagine .bmp in toni di grigio",
            filetypes=[("BMP", "*.bmp"), ("Tutti i file", "*.*")])
        if not path:
            return
        try:
            arr = load_bmp_grayscale(path)
        except Exception as e:
            messagebox.showerror("Errore", f"Impossibile caricare l'immagine:\n{e}")
            return
        self.img_array = arr
        self.img_path = path
        self.img_compressed = None
        h, w = arr.shape
        self.path_var.set(f"{os.path.basename(path)}  —  {w} × {h} pixel")
        self.info_var.set("")
        self._show_array(arr, self.canvas_orig)
        self.canvas_comp.configure(image='')
        self.canvas_comp.image = None

    def on_run(self):
        if self.img_array is None:
            messagebox.showwarning("Attenzione", "Carica prima un'immagine .bmp.")
            return
        try:
            F = int(self.F_var.get())
            d = int(self.d_var.get())
        except Exception:
            messagebox.showerror("Errore", "F e d devono essere interi.")
            return
        if F <= 0:
            messagebox.showerror("Errore", "F deve essere positivo.")
            return
        if not (0 <= d <= 2 * F - 2):
            messagebox.showerror(
                "Errore",
                f"d deve essere in [0, 2F − 2] = [0, {2*F - 2}].")
            return

        H, W = self.img_array.shape
        if F > min(H, W):
            messagebox.showerror(
                "Errore",
                f"F = {F} e' piu' grande dell'immagine ({W}x{H}).")
            return

        import time
        t0 = time.perf_counter()
        out = compress_image(self.img_array, F, d)
        t = time.perf_counter() - t0

        self.img_compressed = out
        Hc, Wc = out.shape

        orig_crop = self.img_array[:Hc, :Wc].astype(np.float64)
        diff = orig_crop - out.astype(np.float64)
        mse = float(np.mean(diff * diff))
        psnr = (10.0 * np.log10(255.0 ** 2 / mse)) if mse > 0 else float('inf')

        kept = int(np.sum((np.arange(F).reshape(F, 1) +
                           np.arange(F).reshape(1, F)) < d))
        total = F * F
        self.info_var.set(
            f"F={F}, d={d}  →  blocchi {Hc//F}×{Wc//F},  "
            f"coefficienti tenuti per blocco: {kept}/{total} "
            f"({100*kept/total:.1f}%),  "
            f"MSE={mse:.2f},  PSNR={psnr:.2f} dB,  "
            f"tempo={t*1000:.1f} ms"
        )
        self._show_array(out, self.canvas_comp)

    def on_save(self):
        if self.img_compressed is None:
            messagebox.showwarning("Attenzione", "Esegui prima la compressione.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".bmp",
            filetypes=[("BMP", "*.bmp"), ("PNG", "*.png")],
            initialfile="compressa.bmp")
        if not path:
            return
        Image.fromarray(self.img_compressed, mode='L').save(path)
        messagebox.showinfo("Salvataggio", f"Immagine salvata in:\n{path}")


def main():
    app = DCTApp()
    app.mainloop()


if __name__ == "__main__":
    main()