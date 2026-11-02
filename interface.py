#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Interface graphique de régression symbolique.

L'utilisateur saisit une fonction f(x) et choisit le nombre de points du
nuage ; le modèle entraîné devine la formule, on superpose la fonction
d'origine et la formule retrouvée, et on affiche cette dernière en LaTeX.

Lancement :
    .venv/bin/python interface.py
    .venv/bin/python interface.py --ckpt modele_regression.pt
"""

from __future__ import annotations

import argparse
import ast
import base64
import io
import queue
import re
import threading
import time

import numpy as np

import regression_symbolique as rs

# --------------------------------------------------------------------------
# Analyse de la fonction saisie par l'utilisateur
# --------------------------------------------------------------------------

# fonctions acceptées dans la saisie (nom -> fonction numpy)
FONCTIONS = {
    "sin": np.sin, "cos": np.cos, "tan": np.tan,
    "arcsin": np.arcsin, "arccos": np.arccos, "arctan": np.arctan,
    "sinh": np.sinh, "cosh": np.cosh, "tanh": np.tanh,
    "exp": np.exp, "log": np.log, "ln": np.log, "log10": np.log10,
    "sqrt": np.sqrt, "abs": np.abs, "floor": np.floor,
}
NOMS_SIMPLES = {"x", "pi", "e", "tau"}

_MOTIF_JETON = re.compile(
    r"(\d+(?:\.\d*)?(?:[eE][+-]?\d+)?|\.\d+)"      # 1 : nombre
    r"|([A-Za-z_][A-Za-z_0-9]*)"                    # 2 : nom
    r"|(\*\*|[+\-*/^(),])"                          # 3 : opérateur
)


def _jetons(texte: str):
    """Découpe l'expression en jetons (nombre, nom, opérateur)."""
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
    """Faut-il insérer un « * » entre deux jetons (2x, 3sin(x), (x+1)x…) ?"""
    (gp, vp), (gc, vc) = prec, courant
    if gp == "nb":
        return gc in ("nb", "nom") or (gc == "op" and vc == "(")
    if gp == "nom":
        if gc in ("nb", "nom"):
            return True
        if gc == "op" and vc == "(":
            return vp not in FONCTIONS      # sin( ... ) reste un appel
        return False
    if gp == "op" and vp == ")":
        return gc in ("nb", "nom") or (gc == "op" and vc == "(")
    return False


def reecrire(source: str) -> str:
    """Normalise la saisie et ajoute les multiplications implicites."""
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
    """Refuse tout ce qui n'est pas une expression mathématique simple."""
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
    """Texte -> fonction numpy vectorisée f(x)."""
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
            y = eval(code, {"__builtins__": {}}, env)      # noqa: S307 (AST vérifié)
        return np.broadcast_to(np.asarray(y, dtype=np.float64),
                               np.shape(x)).copy()

    return f


# --------------------------------------------------------------------------
# Cœur du calcul (sans Tkinter : testable en ligne de commande)
# --------------------------------------------------------------------------

class Options:
    """Petit conteneur d'options attendu par `predict_tree`."""

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


def predire(model, opt, device, f, n_points, x_range, bruit=0.0, graine=0,
            simplifier=True):
    """Nuage -> dict décrivant la prédiction (formule, courbes, erreurs)."""
    rng = np.random.default_rng(graine)
    x = np.sort(rng.uniform(-x_range, x_range, n_points))
    y = np.asarray(f(x), dtype=np.float64)
    if y.shape != x.shape:
        raise ValueError("la fonction doit renvoyer une valeur par point")
    fini = np.isfinite(y)
    if fini.sum() < 3:
        raise ValueError("la fonction n'est pas définie sur assez de points "
                         "du domaine : essayez un autre domaine")
    x, y = x[fini], y[fini]
    if float(np.std(y)) < 1e-9:
        raise ValueError("la fonction est (quasi) constante : aucune formule "
                         "intéressante à retrouver")
    if bruit > 0:
        y = y + rng.normal(0.0, bruit * float(np.std(y)), x.shape)

    # le modèle attend exactement `opt.n_points` abscisses : si l'utilisateur
    # en fournit autant, on les garde telles quelles (plus fidèle) ; sinon on
    # rééchantillonne en suivant la densité des points mesurés.
    if x.size == opt.n_points:
        xg, yg = x, y
    else:
        q = (np.arange(opt.n_points) + 0.5) / opt.n_points
        xg = np.quantile(x, q)
        yg = np.interp(xg, x, y)
    sample = {"x": xg, "y": yg, "tree": None, "depth": 0}

    depart = time.time()
    ids, values, tree = rs.predict_tree(model, sample, opt, device)
    brut = tree
    if tree is not None and simplifier:
        tree = rs.simplify_formula(tree, x, y)
    duree = time.time() - depart
    if tree is None:
        raise ValueError("aucune formule valide générée ; réessayez avec "
                         "plus de candidats")

    xd = np.linspace(x.min(), x.max(), 600)
    return {
        "x": x, "y": y,
        "xd": xd,
        "y_true": np.asarray(f(xd), dtype=np.float64),
        "y_pred": rs.eval_tree(tree, xd),
        "tree": tree,
        "brut": rs.tree_to_str(brut) if brut is not None else "",
        "simplifiee": tree is not brut and rs.tree_to_str(tree) != rs.tree_to_str(brut),
        "formule": rs.tree_to_str(tree),
        "latex": rs.tree_to_latex(tree),
        "rmse": _rmse(rs.eval_tree(tree, x), y),
        "rmse_rel": _rmse(rs.eval_tree(tree, x), y) / max(float(np.std(y)), 1e-9),
        "duree": duree,
    }


def _rmse(yp, y):
    fini = np.isfinite(yp) & np.isfinite(y)
    if not fini.any():
        return float("inf")
    return float(np.sqrt(np.mean((yp[fini] - y[fini]) ** 2)))


def limites_verticales(y):
    """Bornes d'affichage robustes (ignore les valeurs extrêmes/∞)."""
    fini = np.asarray(y, dtype=np.float64)
    fini = fini[np.isfinite(fini)]
    if fini.size == 0:
        return -1.0, 1.0
    lo, hi = np.percentile(fini, [1, 99])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-9:
        centre = float(np.mean(fini))
        lo, hi = centre - 1.0, centre + 1.0
    marge = 0.15 * (hi - lo)
    return float(lo - marge), float(hi + marge)


# --------------------------------------------------------------------------
# Tracé (partagé entre l'interface et le test en ligne de commande)
# --------------------------------------------------------------------------

COULEUR_FOND = "#f6f6f8"
COULEUR_TEXTE = "#1f2937"
COULEUR_REELLE = "#2563eb"
COULEUR_PREDITE = "#dc2626"
COULEUR_POINTS = "#111827"


def tracer(axes, res, n_points):
    """Superpose fonction réelle, points du nuage et formule prédite."""
    axes.clear()
    axes.set_facecolor("#ffffff")
    axes.plot(res["xd"], res["y_true"], color=COULEUR_REELLE, lw=2.2,
              label="votre fonction")
    axes.plot(res["xd"], res["y_pred"], color=COULEUR_PREDITE, lw=2.0,
              ls="--", label="formule de l'IA")
    axes.scatter(res["x"], res["y"], s=34, color=COULEUR_POINTS,
                 edgecolor="white", linewidth=0.7, zorder=3,
                 label=f"{n_points} points mesurés")
    lo, hi = limites_verticales(res["y_true"])
    axes.set_ylim(lo, hi)
    axes.set_xlim(res["xd"][0], res["xd"][-1])
    axes.grid(True, color="#e5e7eb", lw=0.8)
    axes.axhline(0.0, color="#9ca3af", lw=0.8)
    axes.set_xlabel("x")
    axes.set_ylabel("y")
    axes.legend(loc="best", frameon=True, framealpha=0.9, fontsize=9)
    for cote in ("top", "right"):
        axes.spines[cote].set_visible(False)


def rendre_latex_png(latex: str, largeur_px=780, hauteur_px=110,
                     taille=16):
    """Rend « y = <latex> » en PNG (mathtext), sans LaTeX externe."""
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    fig = Figure(figsize=(largeur_px / 100.0, hauteur_px / 100.0), dpi=100)
    fig.patch.set_facecolor(COULEUR_FOND)
    fig.text(0.5, 0.5, f"$y = {latex}$", ha="center", va="center",
             fontsize=taille, color=COULEUR_TEXTE)
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=COULEUR_FOND)
    return buf.getvalue()


# --------------------------------------------------------------------------
# Interface Tkinter
# --------------------------------------------------------------------------

EXEMPLES = [
    "2.5*x - 1",
    "2*x^2 - 3",
    "3*sin(2*x)",
    "sin(x) + 2*x",
    "exp(0.5*x)",
    "tanh(3*x) + 1",
    "exp(-x^2/2)",
    "x*sin(x)",
    "log(x^2 + 1)",
    "sqrt(x + 4) + x",
]


class Application:
    def __init__(self, racine, model, opt, device, nom_checkpoint):
        import tkinter as tk
        from tkinter import ttk, messagebox

        self.tk = tk
        self.messagebox = messagebox
        self.ttk = ttk
        self.racine = racine
        self.model = model
        self.opt = opt
        self.device = device
        self.file_resultats = queue.Queue()
        self.en_cours = False
        self.image_latex = None

        racine.title("Régression symbolique — de nuage de points à formule")
        racine.geometry("1080x820")
        racine.minsize(900, 700)
        racine.configure(bg=COULEUR_FOND)

        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("TFrame", background=COULEUR_FOND)
        style.configure("TLabel", background=COULEUR_FOND,
                        foreground=COULEUR_TEXTE)
        style.configure("Titre.TLabel", font=("DejaVu Sans", 15, "bold"))
        style.configure("Sous.TLabel", foreground="#6b7280")
        style.configure("Action.TButton", font=("DejaVu Sans", 10, "bold"))

        self._construire(racine, nom_checkpoint)
        racine.after(120, self._verifier_file)

    # ---------------------------------------------------------------- widgets
    def _construire(self, racine, nom_checkpoint):
        tk, ttk = self.tk, self.ttk

        entete = ttk.Frame(racine, padding=(14, 12, 14, 6))
        entete.pack(fill="x")

        ttk.Label(entete, text="Retrouver la formule d'une fonction",
                  style="Titre.TLabel").grid(row=0, column=0, columnspan=6,
                                             sticky="w")
        ttk.Label(entete,
                  text=f"modèle : {nom_checkpoint}   ·   "
                       "entrez f(x) ou tracez des points à la souris, "
                       "puis lancez la prédiction",
                  style="Sous.TLabel").grid(row=1, column=0, columnspan=6,
                                            sticky="w", pady=(0, 10))

        ttk.Label(entete, text="f(x) =").grid(row=2, column=0, sticky="w")
        self.var_fonction = tk.StringVar(value="2*x^2 - 3")
        self.entree = ttk.Entry(entete, textvariable=self.var_fonction,
                                font=("DejaVu Sans Mono", 12))
        self.entree.grid(row=2, column=1, columnspan=3, sticky="ew", padx=(6, 6))
        self.entree.bind("<Return>", lambda _e: self.lancer())

        self.var_exemple = tk.StringVar()
        choix = ttk.Combobox(entete, textvariable=self.var_exemple,
                             values=EXEMPLES, state="readonly", width=20)
        choix.set("exemples…")
        choix.grid(row=2, column=4, sticky="w", padx=(0, 6))
        choix.bind("<<ComboboxSelected>>", self._choisir_exemple)

        ttk.Label(entete, text="points").grid(row=2, column=5, sticky="e")
        self.var_points = tk.IntVar(value=self.opt.n_points)
        ttk.Spinbox(entete, from_=4, to=256, increment=4, width=5,
                    textvariable=self.var_points).grid(row=2, column=6,
                                                       sticky="w", padx=(6, 0))

        reglages = ttk.Frame(entete)
        reglages.grid(row=3, column=0, columnspan=7, sticky="ew", pady=(10, 0))

        ttk.Label(reglages, text="domaine  x ∈ ±").pack(side="left")
        self.var_domaine = tk.DoubleVar(value=4.0)
        ttk.Spinbox(reglages, from_=0.5, to=12.0, increment=0.5, width=5,
                    textvariable=self.var_domaine).pack(side="left", padx=(4, 14))

        ttk.Label(reglages, text="bruit").pack(side="left")
        self.var_bruit = tk.DoubleVar(value=0.0)
        ttk.Spinbox(reglages, from_=0.0, to=0.20, increment=0.01, width=5,
                    textvariable=self.var_bruit,
                    format="%.2f").pack(side="left", padx=(4, 14))

        ttk.Label(reglages, text="candidats").pack(side="left")
        self.var_candidats = tk.IntVar(value=self.opt.candidates)
        ttk.Spinbox(reglages, from_=8, to=128, increment=8, width=5,
                    textvariable=self.var_candidats).pack(side="left",
                                                          padx=(4, 14))

        ttk.Label(reglages, text="affinage").pack(side="left")
        self.var_affinage = tk.IntVar(value=self.opt.refine)
        ttk.Spinbox(reglages, from_=0, to=10, increment=1, width=4,
                    textvariable=self.var_affinage).pack(side="left",
                                                         padx=(4, 14))

        self.var_simplifier = tk.BooleanVar(value=True)
        ttk.Checkbutton(reglages, text="simplifier",
                        variable=self.var_simplifier).pack(side="left",
                                                           padx=(0, 12))

        self.var_mode_trace = tk.BooleanVar(value=False)
        ttk.Checkbutton(reglages, text="tracer à la souris",
                        variable=self.var_mode_trace,
                        command=self._toggle_mode_trace).pack(side="left",
                                                              padx=(0, 12))

        self.bouton_effacer = ttk.Button(reglages, text="Effacer points",
                                         command=self._effacer_points)
        self.bouton_effacer.pack(side="left", padx=(0, 12))

        self.bouton = ttk.Button(reglages, text="Prédire la fonction",
                                 style="Action.TButton", command=self.lancer)
        self.bouton.pack(side="right")
        self.barre = ttk.Progressbar(reglages, mode="indeterminate", length=120)
        self.barre.pack(side="right", padx=(0, 10))

        # ---- tracé
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_tkagg import (
            FigureCanvasTkAgg, NavigationToolbar2Tk)

        cadre = ttk.Frame(racine, padding=(14, 4, 14, 4))
        cadre.pack(fill="both", expand=True)
        self.figure = Figure(figsize=(7.5, 4.6), dpi=100, facecolor=COULEUR_FOND)
        self.axes = self.figure.add_subplot(111)
        self.axes.set_facecolor("#ffffff")
        self.axes.text(0.5, 0.5, "La courbe apparaîtra ici\n\nou activez « tracer à la souris » pour placer des points",
                       ha="center", va="center", color="#9ca3af",
                       fontsize=11, transform=self.axes.transAxes)
        self.axes.set_xlabel("x")
        self.axes.set_ylabel("y")
        self.axes.grid(True, color="#e5e7eb", lw=0.8)
        self.axes.axhline(0.0, color="#9ca3af", lw=0.8)
        self.figure.tight_layout()
        self.canvas = FigureCanvasTkAgg(self.figure, master=cadre)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.canvas.mpl_connect('button_press_event', self._ajouter_point_souris)
        barre_outils = NavigationToolbar2Tk(self.canvas, cadre, pack_toolbar=False)
        barre_outils.update()
        barre_outils.pack(side="left", fill="x")

        # Points tracés à la souris
        self.points_manuel_x = []
        self.points_manuel_y = []
        self.scatter_manuel = None
        self.mode_trace = False

        # ---- formule
        bas = ttk.LabelFrame(racine, text=" Formule trouvée par l'IA ",
                             padding=(12, 8))
        bas.pack(fill="x", padx=14, pady=(6, 10))

        self.label_latex = tk.Label(bas, text="—", bg=COULEUR_FOND,
                                    fg=COULEUR_TEXTE)
        self.label_latex.pack(fill="x")

        ligne = ttk.Frame(bas)
        ligne.pack(fill="x", pady=(4, 0))
        ttk.Label(ligne, text="LaTeX :", style="Sous.TLabel").pack(side="left")
        self.var_latex = tk.StringVar(value="")
        self.entree_latex = tk.Entry(ligne, textvariable=self.var_latex,
                                     state="readonly",
                                     font=("DejaVu Sans Mono", 9),
                                     readonlybackground="#eef1f5")
        self.entree_latex.pack(side="left", fill="x", expand=True, padx=(6, 6))
        ttk.Button(ligne, text="Copier", width=8,
                   command=self._copier_latex).pack(side="left")

        self.var_lisible = tk.StringVar(value="")
        ttk.Label(bas, textvariable=self.var_lisible,
                  style="Sous.TLabel").pack(fill="x", pady=(4, 0))
        self.var_brut = tk.StringVar(value="")
        ttk.Label(bas, textvariable=self.var_brut,
                  style="Sous.TLabel").pack(fill="x")

        # ---- barre d'état
        self.var_etat = tk.StringVar(
            value="Prêt. Le modèle a été entraîné sur des fonctions de "
                  "profondeur ≤ 3.")
        ttk.Label(racine, textvariable=self.var_etat, anchor="w",
                  style="Sous.TLabel", padding=(16, 0, 16, 8)).pack(fill="x")

    # ------------------------------------------------------------- actions
    def _toggle_mode_trace(self):
        self.mode_trace = bool(self.var_mode_trace.get())
        if self.mode_trace:
            self.var_etat.set("Mode tracé : cliquez sur le graphique pour ajouter des points. Utilisez 'Effacer points' pour recommencer.")
            self.entree.configure(state="disabled")
            if self.points_manuel_x:
                self.var_points.set(len(self.points_manuel_x))
        else:
            self.var_etat.set("Mode formule : entrez f(x) ou choisissez un exemple.")
            self.entree.configure(state="normal")

    def _effacer_points(self):
        self.points_manuel_x = []
        self.points_manuel_y = []
        self._maj_affichage_points_manuel()
        self.var_etat.set("Points effacés. Cliquez pour en ajouter de nouveaux.")

    def _ajouter_point_souris(self, event):
        if not self.mode_trace or event.inaxes != self.axes or event.xdata is None or event.ydata is None:
            return
        self.points_manuel_x.append(float(event.xdata))
        self.points_manuel_y.append(float(event.ydata))
        self._maj_affichage_points_manuel()
        self.var_points.set(len(self.points_manuel_x))
        self.var_etat.set(f"{len(self.points_manuel_x)} point(s) tracé(s). Lancez la prédiction pour deviner la formule.")

    def _maj_affichage_points_manuel(self):
        try:
            if self.scatter_manuel is not None:
                self.scatter_manuel.remove()
        except Exception:
            pass
        if self.points_manuel_x:
            self.scatter_manuel = self.axes.scatter(self.points_manuel_x, self.points_manuel_y,
                                                    s=60, color="#16a34a", edgecolor="white",
                                                    linewidth=1.0, zorder=4, label="points tracés")
            try:
                self.axes.legend(loc="best", frameon=True, framealpha=0.9, fontsize=9)
            except Exception:
                pass
        else:
            self.scatter_manuel = None
        self.canvas.draw_idle()

    def _choisir_exemple(self, _event=None):
        self.var_fonction.set(self.var_exemple.get())
        self.entree.focus_set()
        self.entree.icursor("end")

    def _copier_latex(self):
        texte = self.var_latex.get()
        if texte:
            self.racine.clipboard_clear()
            self.racine.clipboard_append(texte)
            self.var_etat.set("LaTeX copié dans le presse-papiers.")

    def lancer(self):
        if self.en_cours:
            return
        try:
            if self.mode_trace and self.points_manuel_x:
                # Utiliser les points tracés à la souris
                x_m = np.array(self.points_manuel_x)
                y_m = np.array(self.points_manuel_y)
                n_points = len(x_m)
                x_range = float(max(abs(x_m.min()), abs(x_m.max()), 1.0))
                if x_range < 0.5:
                    x_range = 4.0
                bruit = float(self.var_bruit.get())
                candidats = int(self.var_candidats.get())
                affinage = int(self.var_affinage.get())
                f = None  # pas de fonction connue
            else:
                f = compiler_fonction(self.var_fonction.get())
                n_points = int(self.var_points.get())
                x_range = float(self.var_domaine.get())
                bruit = float(self.var_bruit.get())
                candidats = int(self.var_candidats.get())
                affinage = int(self.var_affinage.get())
        except (ValueError, TypeError) as exc:
            self.messagebox.showerror("Saisie invalide", str(exc),
                                         parent=self.racine)
            return
        if not 4 <= n_points <= 256:
            self.messagebox.showerror(
                "Saisie invalide", "le nombre de points doit être entre 4 et 256",
                parent=self.racine)
            return
        if x_range > 4.0:
            self.var_etat.set(
                f"⚠ domaine ±{x_range:g} : le modèle a appris sur ±4, "
                "les résultats hors de cette plage sont incertains.")

        self.opt.candidates = candidats
        self.opt.refine = affinage
        self.en_cours = True
        self.bouton.configure(state="disabled")
        self.barre.start(12)
        if self.mode_trace and self.points_manuel_x:
            self.var_etat.set(f"Le modèle cherche une formule à partir de {n_points} points tracés "
                              f"({candidats} candidats, affinage de {affinage})…")
        else:
            self.var_etat.set(f"Le modèle cherche une formule "
                              f"({candidats} candidats, affinage de {affinage})…")

        if self.mode_trace and self.points_manuel_x:
            fil = threading.Thread(
                target=self._calculer_points_manuel,
                args=(np.array(self.points_manuel_x), np.array(self.points_manuel_y),
                      x_range, bruit, bool(self.var_simplifier.get())), daemon=True)
        else:
            fil = threading.Thread(
                target=self._calculer,
                args=(f, n_points, x_range, bruit,
                      bool(self.var_simplifier.get())), daemon=True)
        fil.start()

    def _calculer(self, f, n_points, x_range, bruit, simplifier):
        try:
            res = predire(self.model, self.opt, self.device, f,
                          n_points, x_range, bruit, simplifier=simplifier)
            self.file_resultats.put(("ok", (res, n_points)))
        except Exception as exc:                       # noqa: BLE001
            self.file_resultats.put(("erreur", exc))

    def _calculer_points_manuel(self, x_m, y_m, x_range, bruit, simplifier):
        try:
            # Trier les points
            idx = np.argsort(x_m)
            x_m = x_m[idx]
            y_m = y_m[idx]
            if bruit > 0 and len(y_m) > 1:
                rng = np.random.default_rng(12345)
                y_m = y_m + rng.normal(0.0, bruit * float(np.std(y_m)), y_m.shape)
            sample = {"x": x_m, "y": y_m, "tree": None, "depth": 0}
            depart = time.time()
            ids, values, tree = rs.predict_tree(self.model, sample, self.opt, self.device)
            brut = tree
            if tree is not None and simplifier:
                tree = rs.simplify_formula(tree, x_m, y_m)
            duree = time.time() - depart
            if tree is None:
                raise ValueError("aucune formule valide générée ; réessayez avec plus de candidats")
            xd = np.linspace(x_m.min(), x_m.max(), 600)
            res = {
                "x": x_m, "y": y_m,
                "xd": xd,
                "y_true": rs.eval_tree(tree, xd),
                "y_pred": rs.eval_tree(tree, xd),
                "tree": tree,
                "brut": rs.tree_to_str(brut) if brut is not None else "",
                "simplifiee": tree is not brut and rs.tree_to_str(tree) != rs.tree_to_str(brut),
                "formule": rs.tree_to_str(tree),
                "latex": rs.tree_to_latex(tree),
                "rmse": _rmse(rs.eval_tree(tree, x_m), y_m),
                "rmse_rel": _rmse(rs.eval_tree(tree, x_m), y_m) / max(float(np.std(y_m)), 1e-9),
                "duree": duree,
            }
            self.file_resultats.put(("ok", (res, len(x_m))))
        except Exception as exc:                       # noqa: BLE001
            self.file_resultats.put(("erreur", exc))

    def _verifier_file(self):
        try:
            genre, charge = self.file_resultats.get_nowait()
        except queue.Empty:
            self.racine.after(120, self._verifier_file)
            return
        self.en_cours = False
        self.bouton.configure(state="normal")
        self.barre.stop()
        if genre == "erreur":
            self.var_etat.set("Échec : " + str(charge))
            self.messagebox.showerror("Prédiction impossible", str(charge),
                                         parent=self.racine)
        else:
            self._afficher(*charge)
        self.racine.after(120, self._verifier_file)

    def _afficher(self, res, n_points):
        # Remettre à zéro le graphique et tracer
        self.axes.clear()
        self.axes.set_facecolor("#ffffff")
        self.axes.plot(res["xd"], res["y_pred"], color=COULEUR_PREDITE, lw=2.0,
                       ls="--", label="formule de l'IA")
        self.axes.scatter(res["x"], res["y"], s=34, color=COULEUR_POINTS,
                          edgecolor="white", linewidth=0.7, zorder=3,
                          label=f"{n_points} points")
        # Si on est en mode tracé manuel, garder les points verts aussi
        if self.mode_trace and self.points_manuel_x:
            try:
                if self.scatter_manuel is not None:
                    self.scatter_manuel.remove()
            except Exception:
                pass
            self.scatter_manuel = self.axes.scatter(self.points_manuel_x, self.points_manuel_y,
                                                    s=60, color="#16a34a", edgecolor="white",
                                                    linewidth=1.0, zorder=4, label="points tracés")
        lo, hi = limites_verticales(res["y_pred"])
        lo2, hi2 = limites_verticales(res["y"])
        lo = min(lo, lo2)
        hi = max(hi, hi2)
        self.axes.set_ylim(lo, hi)
        self.axes.set_xlim(res["xd"][0], res["xd"][-1])
        self.axes.grid(True, color="#e5e7eb", lw=0.8)
        self.axes.axhline(0.0, color="#9ca3af", lw=0.8)
        self.axes.set_xlabel("x")
        self.axes.set_ylabel("y")
        self.axes.legend(loc="best", frameon=True, framealpha=0.9, fontsize=9)
        for cote in ("top", "right"):
            self.axes.spines[cote].set_visible(False)
        self.figure.tight_layout()
        self.canvas.draw_idle()

        self.var_latex.set(f"y = {res['latex']}")
        self.var_lisible.set("Forme lisible : y = " + res["formule"])
        self.var_brut.set("Formule brute du réseau : y = " + res["brut"]
                          if res["simplifiee"] else "")
        try:
            png = rendre_latex_png(res["latex"])
            self.image_latex = self.tk.PhotoImage(
                data=base64.b64encode(png).decode("ascii"))
            self.label_latex.configure(image=self.image_latex, text="")
        except Exception:                              # noqa: BLE001
            self.image_latex = None
            self.label_latex.configure(image="", text="y = " + res["formule"])

        self.var_etat.set(
            f"{n_points} points · RMSE = {res['rmse']:.4g} "
            f"({100 * res['rmse_rel']:.1f} % de l'écart-type) · "
            f"calculé en {res['duree']:.1f} s"
            + ("   ⚠ la courbe prédite sort du cadre : formule éloignée"
               if res["rmse_rel"] > 0.5 else ""))


# --------------------------------------------------------------------------
# Lancement
# --------------------------------------------------------------------------

def charger_modele(chemin, device):
    import torch

    ckpt = torch.load(chemin, map_location=device, weights_only=False)
    model = rs.SymbolicRegressor(**ckpt["hparams"]).to(device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()
    return model, ckpt["hparams"], int(ckpt.get("step", 0))


def verifier(chemin, fonction, n_points):
    """Test sans interface : prédit une fonction et écrit les aperçus PNG."""
    import torch

    device = torch.device("cpu")
    model, hparams, etape = charger_modele(chemin, device)
    opt = Options(hparams)
    f = compiler_fonction(fonction)
    res = predire(model, opt, device, f, n_points, 4.0, 0.0)
    print(f"modèle   : {chemin} (étape {etape})")
    print(f"fonction : {fonction}")
    print(f"prédite  : y = {res['formule']}")
    print(f"LaTeX    : y = {res['latex']}")
    print(f"RMSE     : {res['rmse']:.4g} ({100 * res['rmse_rel']:.1f} %) "
          f"en {res['duree']:.1f} s")

    from matplotlib.figure import Figure
    fig = Figure(figsize=(8, 4.8), dpi=100, facecolor=COULEUR_FOND)
    axes = fig.add_subplot(111)
    tracer(axes, res, n_points)
    fig.tight_layout()
    fig.savefig("/tmp/opencode/apercu_courbe.png", facecolor=COULEUR_FOND)
    png = rendre_latex_png(res["latex"])
    with open("/tmp/opencode/apercu_latex.png", "wb") as fh:
        fh.write(png)
    print("aperçus : /tmp/opencode/apercu_courbe.png, "
          "/tmp/opencode/apercu_latex.png")


def main():
    p = argparse.ArgumentParser(
        description="Interface graphique de régression symbolique.")
    p.add_argument("--ckpt", default="modele_regression.pt",
                   help="checkpoint du modèle entraîné")
    p.add_argument("--verifier", metavar="FONCTION",
                   help="test sans interface : prédit cette fonction et "
                        "écrit les aperçus PNG")
    p.add_argument("--points", type=int, default=64,
                   help="nombre de points pour le mode --verifier")
    args = p.parse_args()

    import torch
    torch.set_num_threads(min(4, torch.get_num_threads()))

    if args.verifier:
        verifier(args.ckpt, args.verifier, args.points)
        return

    import tkinter as tk
    from tkinter import messagebox

    device = torch.device("cpu")
    try:
        model, hparams, etape = charger_modele(args.ckpt, device)
    except Exception as exc:                           # noqa: BLE001
        racine = tk.Tk()
        racine.withdraw()
        messagebox.showerror(
            "Chargement impossible",
            f"Impossible de charger « {args.ckpt} » :\n\n{exc}")
        return

    opt = Options(hparams)
    racine = tk.Tk()
    Application(racine, model, opt, device,
                f"{args.ckpt} (étape {etape})")
    racine.mainloop()


if __name__ == "__main__":
    main()
