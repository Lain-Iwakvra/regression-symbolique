#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Interface graphique de régression symbolique - Version refondue."""

from __future__ import annotations

import argparse
import ast
import base64
import io
import queue
import re
import threading
import time
import csv
from tkinter import filedialog

import numpy as np

import regression_symbolique as rs

# --------------------------------------------------------------------------
# Analyse de la fonction saisie
# --------------------------------------------------------------------------
FONCTIONS = {
    "sin": np.sin, "cos": np.cos, "tan": np.tan,
    "arcsin": np.arcsin, "arccos": np.arccos, "arctan": np.arctan,
    "sinh": np.sinh, "cosh": np.cosh, "tanh": np.tanh,
    "exp": np.exp, "log": np.log, "ln": np.log, "log10": np.log10,
    "sqrt": np.sqrt, "abs": np.abs, "floor": np.floor,
}
NOMS_SIMPLES = {"x", "pi", "e", "tau"}

_MOTIF_JETON = re.compile(
    r"(\d+(?:\.\d*)?(?:[eE][+-]?\d+)?|\.\d+)"
    r"|([A-Za-z_][A-Za-z_0-9]*)"
    r"|(\*\*|[+\-*/^(),])"
)

def _jetons(texte: str):
    jetons = []
    pos = 0
    while pos < len(texte):
        if texte[pos].isspace():
            pos += 1
            continue
        m = _MOTIF_JETON.match(texte, pos)
        if m is None:
            raise ValueError(f"caractère inattendu : {texte[pos]!r}")
        if m.group(1) is not None:
            jetons.append(("nb", m.group(1)))
        elif m.group(2) is not None:
            jetons.append(("nom", m.group(2)))
        else:
            val = m.group(3)
            jetons.append(("op", "**" if val == "^" else val))
        pos = m.end()
    return jetons

def _besoin_fois(prec, courant) -> bool:
    (gp, vp), (gc, vc) = prec, courant
    if gp == "nb":
        return gc in ("nb", "nom") or (gc == "op" and vc == "(")
    if gp == "nom":
        if gc in ("nb", "nom"):
            return True
        if gc == "op" and vc == "(":
            return vp not in FONCTIONS
        return False
    if gp == "op" and vp == ")":
        return gc in ("nb", "nom") or (gc == "op" and vc == "(")
    return False

def reecrire(source: str) -> str:
    texte = (source.replace("−", "-").replace("×", "*").replace("÷", "/")
                   .replace("π", "pi").replace("²", "^2").replace("³", "^3")
                   .replace("√", "sqrt"))
    jetons = _jetons(texte)
    if not jetons:
        raise ValueError("la fonction est vide")
    sortie, prec = [], None
    for jeton in jetons:
        if prec is not None and _besoin_fois(prec, jeton):
            sortie.append("*")
        sortie.append(jeton[1])
        prec = jeton
    return " ".join(sortie)

def _verifier_ast(arbre):
    for noeud in ast.walk(arbre):
        if isinstance(noeud, ast.Expression):
            continue
        if isinstance(noeud, ast.Constant):
            if isinstance(noeud.value, bool) or not isinstance(noeud.value, (int, float)):
                raise ValueError("seuls les nombres sont autorisés")
            continue
        if isinstance(noeud, ast.Name):
            if noeud.id not in NOMS_SIMPLES and noeud.id not in FONCTIONS:
                raise ValueError(f"nom inconnu : « {noeud.id} »")
            continue
        if isinstance(noeud, ast.BinOp):
            if not isinstance(noeud.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)):
                raise ValueError("opérateur non autorisé")
            continue
        if isinstance(noeud, ast.UnaryOp):
            if not isinstance(noeud.op, (ast.UAdd, ast.USub)):
                raise ValueError("opérateur non autorisé")
            continue
        if isinstance(noeud, ast.Call):
            if not isinstance(noeud.func, ast.Name) or noeud.func.id not in FONCTIONS:
                raise ValueError("fonction non autorisée")
            if noeud.keywords or len(noeud.args) != 1:
                raise ValueError("chaque fonction prend un seul argument")
            continue
        if isinstance(noeud, (ast.Load, ast.operator, ast.unaryop)):
            continue
        raise ValueError(f"élément non autorisé : {type(noeud).__name__}")

def compiler_fonction(source: str):
    texte = reecrire(source)
    try:
        arbre = ast.parse(texte, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"expression invalide : {exc.msg}") from None
    _verifier_ast(arbre)
    code = compile(arbre, "<formule>", "eval")
    environnement = dict(FONCTIONS)
    def f(x):
        env = dict(environnement)
        env.update({"x": np.asarray(x, dtype=np.float64),
                    "pi": np.pi, "e": np.e, "tau": 2.0 * np.pi})
        with np.errstate(all="ignore"):
            y = eval(code, {"__builtins__": {}}, env)
        return np.broadcast_to(np.asarray(y, dtype=np.float64),
                               np.shape(x)).copy()
    return f

# --------------------------------------------------------------------------
class Options:
    def __init__(self, hparams, candidats=48, affinage=8):
        self.n_points = int(hparams["n_points"])
        self.max_len = int(hparams["max_len"])
        self.candidates = int(candidats)
        self.temperature = 1.0
        self.const_noise = 0.15
        self.refine = int(affinage)
        self.refine_steps = 300
        self.penalite = 0.002
        self.seed = 0
        self.x_range = 4.0

def _rmse(yp, y):
    fini = np.isfinite(yp) & np.isfinite(y)
    if not fini.any():
        return float("inf")
    return float(np.sqrt(np.mean((yp[fini] - y[fini]) ** 2)))

def limites_verticales(y):
    fini = np.asarray(y, dtype=np.float64)
    fini = fini[np.isfinite(fini)]
    if fini.size == 0:
        return -1.0, 1.0
    lo, hi = np.percentile(fini, [2, 98])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-9:
        centre = float(np.mean(fini))
        lo, hi = centre - 1.0, centre + 1.0
    marge = 0.2 * (hi - lo)
    return float(lo - marge), float(hi + marge)

# --------------------------------------------------------------------------
COULEUR_FOND = "#f8f9fa"
COULEUR_PANEL = "#ffffff"
COULEUR_TEXTE = "#1f2937"
COULEUR_REELLE = "#2563eb"
COULEUR_PREDITE = "#dc2626"
COULEUR_POINTS = "#111827"
COULEUR_MANUEL = "#16a34a"

EXEMPLES = [
    "2.5*x - 1", "2*x^2 - 3", "3*sin(2*x)", "sin(x) + 2*x",
    "exp(0.5*x)", "tanh(3*x) + 1", "exp(-x^2/2)", "x*sin(x)",
    "log(x^2 + 1)", "sqrt(x + 4) + x",
]

# --------------------------------------------------------------------------
class App:
    def __init__(self, root, model, opt, device, ckpt_name):
        import tkinter as tk
        from tkinter import ttk, messagebox
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

        self.tk = tk
        self.ttk = ttk
        self.messagebox = messagebox
        self.root = root
        self.model = model
        self.opt = opt
        self.device = device
        self.ckpt_name = ckpt_name
        self.queue = queue.Queue()
        self.running = False
        self.mode = "fonction"  # fonction | manuel | csv

        self.points_x = []
        self.points_y = []
        self.scatter_manuel = None

        root.title("Régression Symbolique")
        root.geometry("1400x900")
        root.minsize(1100, 750)
        root.configure(bg=COULEUR_FOND)

        style = ttk.Style()
        try:
            style.theme_use("clam")
        except:
            pass
        style.configure("TFrame", background=COULEUR_FOND)
        style.configure("Card.TFrame", background=COULEUR_PANEL)
        style.configure("TLabel", background=COULEUR_FOND, foreground=COULEUR_TEXTE)
        style.configure("Card.TLabel", background=COULEUR_PANEL, foreground=COULEUR_TEXTE)
        style.configure("Title.TLabel", font=("Segoe UI", 16, "bold"), background=COULEUR_FOND)
        style.configure("Subtitle.TLabel", font=("Segoe UI", 9), foreground="#6b7280", background=COULEUR_FOND)
        style.configure("Accent.TButton", font=("Segoe UI", 9, "bold"))

        self._build()

    def _build(self):
        tk, ttk = self.tk, self.ttk
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

        # Header
        header = ttk.Frame(self.root)
        header.pack(fill="x", padx=20, pady=(15, 10))
        ttk.Label(header, text="Régression Symbolique", style="Title.TLabel").pack(anchor="w")
        ttk.Label(header, text=f"Modèle : {self.ckpt_name}", style="Subtitle.TLabel").pack(anchor="w")

        # Main container
        main = ttk.Frame(self.root)
        main.pack(fill="both", expand=True, padx=20, pady=(0, 15))

        # Left panel
        left = ttk.Frame(main, width=380)
        left.pack(side="left", fill="y")
        left.pack_propagate(False)

        # Mode selector
        mode_card = ttk.Frame(left, style="Card.TFrame", padding=15)
        mode_card.pack(fill="x", pady=(0, 10))
        ttk.Label(mode_card, text="Mode de saisie", style="Card.TLabel", font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(0, 10))
        self.mode_var = tk.StringVar(value="fonction")
        modes = [("Depuis une fonction", "fonction"), ("Tracer des points", "manuel"), ("Charger CSV", "csv")]
        for text, val in modes:
            ttk.Radiobutton(mode_card, text=text, value=val, variable=self.mode_var, 
                           command=self._change_mode, style="Card.TRadiobutton").pack(anchor="w", pady=2)

        # Function panel
        self.func_card = ttk.Frame(left, style="Card.TFrame", padding=15)
        self.func_card.pack(fill="x", pady=(0, 10))
        ttk.Label(self.func_card, text="Fonction f(x)", style="Card.TLabel", font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(0, 8))
        self.func_var = tk.StringVar(value="2*x^2 - 3")
        self.func_entry = ttk.Entry(self.func_card, textvariable=self.func_var)
        self.func_entry.pack(fill="x", pady=(0, 8))
        self.func_entry.bind("<Return>", lambda e: self.predict())
        ttk.Label(self.func_card, text="Exemples", style="Card.TLabel").pack(anchor="w")
        self.ex_var = tk.StringVar()
        ex_combo = ttk.Combobox(self.func_card, textvariable=self.ex_var, values=EXEMPLES, state="readonly")
        ex_combo.pack(fill="x", pady=(4, 0))
        ex_combo.bind("<<ComboboxSelected>>", lambda e: self.func_var.set(self.ex_var.get()))

        # Manual panel
        self.manual_card = ttk.Frame(left, style="Card.TFrame", padding=15)
        ttk.Label(self.manual_card, text="Points manuels", style="Card.TLabel", font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(0, 8))
        ttk.Label(self.manual_card, text="Cliquez sur le graphique pour ajouter des points", style="Card.TLabel").pack(anchor="w")
        btn_frame = ttk.Frame(self.manual_card, style="Card.TFrame")
        btn_frame.pack(fill="x", pady=(10, 0))
        ttk.Button(btn_frame, text="Effacer", command=self.clear_points).pack(side="left")
        ttk.Button(btn_frame, text="Importer CSV", command=self.load_csv).pack(side="left", padx=(5, 0))

        # Parameters
        param_card = ttk.Frame(left, style="Card.TFrame", padding=15)
        param_card.pack(fill="x", pady=(0, 10))
        ttk.Label(param_card, text="Paramètres", style="Card.TLabel", font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(0, 10))
        
        grid = ttk.Frame(param_card, style="Card.TFrame")
        grid.pack(fill="x")
        ttk.Label(grid, text="Candidats", style="Card.TLabel").grid(row=0, column=0, sticky="w")
        self.cand_var = tk.IntVar(value=self.opt.candidates)
        ttk.Spinbox(grid, from_=8, to=128, increment=8, textvariable=self.cand_var, width=8).grid(row=0, column=1, sticky="e", pady=2)
        ttk.Label(grid, text="Affinage", style="Card.TLabel").grid(row=1, column=0, sticky="w")
        self.ref_var = tk.IntVar(value=self.opt.refine)
        ttk.Spinbox(grid, from_=0, to=12, textvariable=self.ref_var, width=8).grid(row=1, column=1, sticky="e", pady=2)
        ttk.Label(grid, text="Points", style="Card.TLabel").grid(row=2, column=0, sticky="w")
        self.npt_var = tk.IntVar(value=self.opt.n_points)
        ttk.Spinbox(grid, from_=4, to=256, increment=4, textvariable=self.npt_var, width=8).grid(row=2, column=1, sticky="e", pady=2)
        ttk.Label(grid, text="Domaine ±", style="Card.TLabel").grid(row=3, column=0, sticky="w")
        self.dom_var = tk.DoubleVar(value=4.0)
        ttk.Spinbox(grid, from_=0.5, to=12.0, increment=0.5, textvariable=self.dom_var, width=8).grid(row=3, column=1, sticky="e", pady=2)
        grid.columnconfigure(1, weight=1)
        
        self.simp_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(param_card, text="Simplifier la formule", variable=self.simp_var, style="Card.TCheckbutton").pack(anchor="w", pady=(10, 0))

        # Predict button
        self.predict_btn = ttk.Button(left, text="PRÉDIRE LA FORMULE", style="Accent.TButton", command=self.predict)
        self.predict_btn.pack(fill="x", ipady=6)

        self.progress = ttk.Progressbar(left, mode="indeterminate")
        self.progress.pack(fill="x", pady=(8, 0))

        # Right panel
        right = ttk.Frame(main)
        right.pack(side="left", fill="both", expand=True, padx=(15, 0))

        plot_card = ttk.Frame(right, style="Card.TFrame")
        plot_card.pack(fill="both", expand=True)

        self.fig = Figure(figsize=(8, 6), dpi=100, facecolor=COULEUR_PANEL)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor("#ffffff")
        self.ax.text(0.5, 0.5, "Prêt", ha="center", va="center", transform=self.ax.transAxes, color="#9ca3af", fontsize=12)
        self.ax.grid(True, alpha=0.3)
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_card)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        toolbar = NavigationToolbar2Tk(self.canvas, plot_card, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(fill="x")
        self.canvas.mpl_connect('button_press_event', self.on_click)

        # Result panel
        res_card = ttk.Frame(right, style="Card.TFrame", padding=12)
        res_card.pack(fill="x", pady=(10, 0))
        ttk.Label(res_card, text="Formule trouvée", style="Card.TLabel", font=("Segoe UI", 10, "bold")).pack(anchor="w")
        self.formula_var = tk.StringVar(value="—")
        self.formula_lbl = ttk.Label(res_card, textvariable=self.formula_var, style="Card.TLabel", font=("Consolas", 12))
        self.formula_lbl.pack(anchor="w", pady=(5, 0))
        self.metrics_var = tk.StringVar(value="")
        ttk.Label(res_card, textvariable=self.metrics_var, style="Card.TLabel").pack(anchor="w", pady=(3, 0))
        self.status_var = tk.StringVar(value="Prêt")
        ttk.Label(right, textvariable=self.status_var, style="Subtitle.TLabel").pack(anchor="w", pady=(8, 0))

        self._change_mode()
        self.root.after(100, self._check_queue)

    def _change_mode(self):
        self.mode = self.mode_var.get()
        if self.mode == "fonction":
            self.func_card.pack(fill="x", pady=(0, 10))
            self.manual_card.pack_forget()
        elif self.mode == "manuel":
            self.func_card.pack_forget()
            self.manual_card.pack(fill="x", pady=(0, 10))
        else:
            self.func_card.pack_forget()
            self.manual_card.pack(fill="x", pady=(0, 10))
            self.load_csv()

    def on_click(self, event):
        if self.mode != "manuel" or event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return
        self.points_x.append(float(event.xdata))
        self.points_y.append(float(event.ydata))
        self._update_plot()
        self.status_var.set(f"{len(self.points_x)} point(s) ajouté(s)")

    def clear_points(self):
        self.points_x = []
        self.points_y = []
        self._update_plot()
        self.status_var.set("Points effacés")

    def load_csv(self):
        path = filedialog.askopenfilename(filetypes=[("CSV", "*.csv"), ("TXT", "*.txt"), ("Tous", "*.*")])
        if not path:
            return
        try:
            xs, ys = [], []
            with open(path) as f:
                reader = csv.reader(f, delimiter=';')
                for row in reader:
                    if not row or row[0].strip().startswith('#'): continue
                    # try comma or space
                    if ',' in row[0] and len(row) == 1:
                        row = row[0].split(',')
                    if len(row) >= 2:
                        try:
                            xs.append(float(row[0].replace(',', '.')))
                            ys.append(float(row[1].replace(',', '.')))
                            continue
                        except:
                            pass
                    # space separated
                    for r in row:
                        parts = r.replace(';', ' ').replace(',', ' ').split()
                        if len(parts) == 2:
                            xs.append(float(parts[0]))
                            ys.append(float(parts[1]))
                            break
            self.points_x, self.points_y = xs, ys
            self.mode_var.set("manuel")
            self._change_mode()
            self._update_plot()
            self.status_var.set(f"{len(xs)} points chargés depuis CSV")
        except Exception as e:
            self.messagebox.showerror("Erreur", str(e))

    def _update_plot(self):
        self.ax.clear()
        self.ax.set_facecolor("#ffffff")
        if self.points_x:
            self.ax.scatter(self.points_x, self.points_y, s=50, color=COULEUR_MANUEL, edgecolor='white', linewidth=1, zorder=5)
        self.ax.grid(True, alpha=0.3)
        self.ax.axhline(0, color='#d1d5db', lw=0.8)
        self.ax.axvline(0, color='#d1d5db', lw=0.8)
        self.fig.tight_layout()
        self.canvas.draw_idle()

    def predict(self):
        if self.running: return
        try:
            self.opt.candidates = self.cand_var.get()
            self.opt.refine = self.ref_var.get()
            if self.mode == "fonction":
                f = compiler_fonction(self.func_var.get())
                npt = self.npt_var.get()
                dom = self.dom_var.get()
                bruit = 0.0
                threading.Thread(target=self._calc_func, args=(f,npt,dom,bruit), daemon=True).start()
            elif self.mode == "manuel":
                if len(self.points_x) < 4:
                    raise ValueError("Ajoutez au moins 4 points")
                x = np.array(self.points_x); y = np.array(self.points_y)
                idx = np.argsort(x); x,y = x[idx], y[idx]
                threading.Thread(target=self._calc_manual, args=(x,y), daemon=True).start()
            else:
                return
            self.running = True
            self.predict_btn.configure(state='disabled')
            self.progress.start(10)
            self.status_var.set("Calcul en cours...")
        except Exception as e:
            self.messagebox.showerror("Erreur", str(e))

    def _calc_func(self, f, npt, dom, bruit):
        try:
            rng = np.random.default_rng(42)
            x = np.sort(rng.uniform(-dom, dom, npt))
            y = np.asarray(f(x), dtype=float)
            fini = np.isfinite(y)
            x,y = x[fini], y[fini]
            sample = {"x":x,"y":y,"tree":None,"depth":0}
            t0=time.time()
            ids,vals,tree = rs.predict_tree(self.model,sample,self.opt,self.device)
            brut = tree
            if tree is not None and self.simp_var.get():
                tree = rs.simplify_formula(tree,x,y)
            dt=time.time()-t0
            xd = np.linspace(x.min(),x.max(),600)
            ytrue = np.asarray(f(xd), dtype=float)
            ypred = rs.eval_tree(tree,xd)
            rmse = _rmse(rs.eval_tree(tree,x),y)
            rrmse = rmse/max(float(np.std(y)),1e-9)
            self.queue.put(("ok",{"x":x,"y":y,"xd":xd,"ytrue":ytrue,"ypred":ypred,"tree":tree,"brut":brut,"rmse":rmse,"rrmse":rrmse,"dt":dt,"n":len(x)}))
        except Exception as e:
            self.queue.put(("err",e))

    def _calc_manual(self, x,y):
        try:
            sample = {"x":x,"y":y,"tree":None,"depth":0}
            t0=time.time()
            ids,vals,tree = rs.predict_tree(self.model,sample,self.opt,self.device)
            brut = tree
            if tree is not None and self.simp_var.get():
                tree = rs.simplify_formula(tree,x,y)
            dt=time.time()-t0
            xd = np.linspace(x.min(),x.max(),600)
            ypred = rs.eval_tree(tree,xd)
            rmse = _rmse(rs.eval_tree(tree,x),y)
            rrmse = rmse/max(float(np.std(y)),1e-9)
            self.queue.put(("ok",{"x":x,"y":y,"xd":xd,"ytrue":None,"ypred":ypred,"tree":tree,"brut":brut,"rmse":rmse,"rrmse":rrmse,"dt":dt,"n":len(x)}))
        except Exception as e:
            self.queue.put(("err",e))

    def _check_queue(self):
        try:
            typ, data = self.queue.get_nowait()
            self.running = False
            self.predict_btn.configure(state='normal')
            self.progress.stop()
            if typ == "err":
                self.messagebox.showerror("Erreur", str(data))
                self.status_var.set("Erreur")
            else:
                self._show(data)
        except:
            pass
        self.root.after(100, self._check_queue)

    def _show(self, res):
        self.ax.clear()
        self.ax.set_facecolor("#ffffff")
        if res["ytrue"] is not None:
            self.ax.plot(res["xd"], res["ytrue"], color=COULEUR_REELLE, lw=2.2, label="Fonction d'origine")
        self.ax.plot(res["xd"], res["ypred"], color=COULEUR_PREDITE, lw=2.0, ls="--", label="Formule IA")
        self.ax.scatter(res["x"], res["y"], s=40, color=COULEUR_POINTS, edgecolor='white', linewidth=0.8, zorder=4)
        if self.points_x and self.mode == "manuel":
            self.ax.scatter(self.points_x, self.points_y, s=55, color=COULEUR_MANUEL, edgecolor='white', linewidth=1, zorder=5)
        lo,hi = limites_verticales(res["ypred"])
        lo2,hi2 = limites_verticales(res["y"])
        lo=min(lo,lo2); hi=max(hi,hi2)
        if res["ytrue"] is not None:
            lo3,hi3 = limites_verticales(res["ytrue"]); lo=min(lo,lo3); hi=max(hi,hi3)
        self.ax.set_ylim(lo,hi)
        self.ax.set_xlim(res["xd"][0],res["xd"][-1])
        self.ax.grid(True, alpha=0.3)
        self.ax.axhline(0,color='#d1d5db',lw=0.8)
        self.ax.legend(frameon=True, fancybox=True)
        self.fig.tight_layout()
        self.canvas.draw_idle()
        
        tree = res["tree"]
        frm = rs.tree_to_str(tree)
        self.formula_var.set(f"y = {frm}")
        self.metrics_var.set(f"RMSE: {res['rmse']:.4g} ({res['rrmse']*100:.1f}%) | {res['n']} points | {res['dt']:.1f}s")
        self.status_var.set("Terminé")

def charger_modele(chemin, device):
    import torch
    ckpt = torch.load(chemin, map_location=device, weights_only=False)
    model = rs.SymbolicRegressor(**ckpt["hparams"]).to(device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()
    return model, ckpt["hparams"], int(ckpt.get("step",0))

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="modele_regression.pt")
    p.add_argument("--verifier", metavar="FONCTION")
    p.add_argument("--points", type=int, default=64)
    args = p.parse_args()
    import torch
    torch.set_num_threads(min(4, torch.get_num_threads()))
    try:
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            try:
                torch.set_float32_matmul_precision('high')
            except Exception:
                pass
    except Exception:
        pass
    if args.verifier:
        device = torch.device("cpu")
        model,hp,et = charger_modele(args.ckpt,device)
        opt = Options(hp)
        f = compiler_fonction(args.verifier)
        rng = np.random.default_rng(0)
        x = np.sort(rng.uniform(-4,4,args.points))
        y = np.asarray(f(x),float)
        sample = {"x":x,"y":y,"tree":None,"depth":0}
        ids,vals,tree = rs.predict_tree(model,sample,opt,device)
        if tree is not None: tree = rs.simplify_formula(tree,x,y)
        print(f"y = {rs.tree_to_str(tree)}")
        print(f"LaTeX: y = {rs.tree_to_latex(tree)}")
        return
    import tkinter as tk
    device = torch.device("cpu")
    try:
        model,hp,et = charger_modele(args.ckpt,device)
    except Exception as e:
        root = tk.Tk(); root.withdraw()
        from tkinter import messagebox
        messagebox.showerror("Erreur", str(e))
        return
    opt = Options(hp)
    root = tk.Tk()
    App(root,model,opt,device,f"{args.ckpt} (étape {et})")
    root.mainloop()

if __name__ == "__main__":
    main()
