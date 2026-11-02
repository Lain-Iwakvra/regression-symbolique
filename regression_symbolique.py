#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=========================================================================
 regression_symbolique.py — Régression symbolique par réseau de neurones
=========================================================================

Un Transformer qui **retrouve la formule** d'une fonction à partir d'un
simple nuage de points (et non seulement une interpolation y = f(x)).

PRINCIPE
--------
1. GRAMMAIRE : on génère aléatoirement une expression (arbre) dans :
       opérateurs binaires :  +  -  *  /
       opérateurs unaires  :  sin cos exp log sqrt tanh neg
       terminaux           :  x  et  une constante réelle
2. NUAGE      : on échantillonne (x_i , f(x_i)) avec x_i ∈ [-R, R]
                (les fonctions pathologiques : nan/inf/asymptotes sont rejetées)
3. MODÈLE     : encodeur Transformer sur les points triés,
                décodeur autoregressif produisant la formule en notation
                préfixe + les valeurs des constantes.
4. CONTRAINTE : le décodage est masqué selon la grammaire → à chaque étape
                seuls les jetons syntactiquement possibles sont autorisés.
                Toute prédiction est donc une expression analytique valide.
5. ENTRAÎNE.  : cross-entropie sur les symboles + perte sur les constantes,
                sur un flux de fonctions illimité généré à la volée
                (généralité : aucune fonction n'est prédéfinie). L'option
                `--canonique` réécrit les cibles sous une forme unique.

6. INFÉRENCE : plusieurs formules sont générées, dédoublonnées, leurs
                constantes affinées par moindres carrés (scipy), puis
                classées par RMSE relative + pénalité de complexité.
                Le tirage est reproductible pour un nuage donné.

UTILISATION
-----------
    # entraîner (évaluation automatique en fin d'entraînement)
    .venv/bin/python regression_symbolique.py --steps 3000

    # reprendre un entraînement
    .venv/bin/python regression_symbolique.py --steps 3000 --resume

    # évaluer sur de nouvelles fonctions
    .venv/bin/python regression_symbolique.py --eval-only --examples 10

    # benchmark figé (comparer deux versions à fonctions identiques)
    .venv/bin/python regression_symbolique.py --creer-benchmark benchmark_valide.json
    .venv/bin/python regression_symbolique.py --eval-only --benchmark benchmark_valide.json

    # deviner une formule à partir de SES propres points
    .venv/bin/python regression_symbolique.py --predict "-2,4;-1,1;0,0;1,1;2,4"
    .venv/bin/python regression_symbolique.py --predict-file points.csv

FORMAT --predict-file : deux colonnes x,y (séparateur ';' ',' ou tabulation),
une ligne par point, '#' pour les commentaires.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
import copy
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:  # pragma: no cover
    sys.exit(
        "PyTorch introuvable.\n"
        "Activez l'environnement virtuel du projet :\n"
        "    source .venv/bin/activate\n"
        "puis relancez, ou utilisez directement :\n"
        "    .venv/bin/python regression_symbolique.py"
    )

# Optimisations CPU mesurées : 6 threads + torch.compile(reduce-overhead)
try:
    torch.set_num_threads(6)
    torch.set_num_interop_threads(2)
    os.environ.setdefault("OMP_NUM_THREADS", "6")
    os.environ.setdefault("MKL_NUM_THREADS", "6")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "6")
except Exception:
    pass


# =========================================================================
# 1. GRAMMAIRE DES EXPRESSIONS
# =========================================================================

PAD, BOS, EOS = "<pad>", "<bos>", "<eos>"
BINARY_OPS = ["add", "sub", "mul", "div"]
UNARY_OPS = ["sin", "cos", "exp", "log", "sqrt", "tanh", "neg"]
LEAVES = ["x", "const"]

VOCAB = [PAD, BOS, EOS] + LEAVES + BINARY_OPS + UNARY_OPS
TOK2ID = {t: i for i, t in enumerate(VOCAB)}
ID2TOK = {i: t for t, i in TOK2ID.items()}
PAD_ID = TOK2ID[PAD]
BOS_ID = TOK2ID[BOS]
EOS_ID = TOK2ID[EOS]
CONST_ID = TOK2ID["const"]
V = len(VOCAB)

ARITY = {PAD: 0, BOS: 0, EOS: 0, "x": 0, "const": 0}
ARITY.update({op: 1 for op in UNARY_OPS})
ARITY.update({op: 2 for op in BINARY_OPS})
ARITY_VEC = torch.tensor([ARITY[ID2TOK[i]] for i in range(V)], dtype=torch.long)

# opérateurs binaires pour lesquels un enfant de même priorité à droite
# doit être parenthésé
_PREC = {"add": 1, "sub": 1, "mul": 2, "div": 2}
_SYM = {"add": " + ", "sub": " - ", "mul": "*", "div": "/"}


# --------------------------------------------------------------------------
# Génération aléatoire d'arbres d'expressions
# --------------------------------------------------------------------------

def random_leaf(rng: random.Random):
    """Terminal : 'x' ou une constante (arrondie à 2 décimales)."""
    if rng.random() < 0.45:
        if rng.random() < 0.3:
            v = float(rng.choice([-3, -2, -1, -0.5, 0.5, 1, 2, 3]))
        else:
            v = round(rng.uniform(-3.0, 3.0), 2)
        return ("const", v)
    return ("x",)


def random_tree(rng: random.Random, depth: int):
    """Arbre d'expressions aléatoire de profondeur max `depth`."""
    if depth <= 0:
        return random_leaf(rng)
    r = rng.random()
    if r < 0.25:
        return random_leaf(rng)
    if r < 0.55:
        return (rng.choice(UNARY_OPS), random_tree(rng, depth - 1))
    return (
        rng.choice(BINARY_OPS),
        random_tree(rng, depth - 1),
        random_tree(rng, depth - 1),
    )


def tree_depth(t) -> int:
    kids = [c for c in t[1:] if isinstance(c, tuple)]
    return 1 + (max(tree_depth(c) for c in kids) if kids else 0)


def tree_size(t) -> int:
    return 1 + sum(tree_size(c) for c in t[1:] if isinstance(c, tuple))


# --------------------------------------------------------------------------
# Évaluation numérique (vectorisée, erreurs ignorées → filtrées ensuite)
# --------------------------------------------------------------------------

def _eval(tree, x):
    tag = tree[0]
    if tag == "x":
        return x
    if tag == "const":
        return np.full_like(x, tree[1])
    if tag in UNARY_OPS:
        a = _eval(tree[1], x)
        if tag == "sin":
            return np.sin(a)
        if tag == "cos":
            return np.cos(a)
        if tag == "exp":
            return np.exp(a)
        if tag == "log":
            return np.log(a)
        if tag == "sqrt":
            return np.sqrt(a)
        if tag == "tanh":
            return np.tanh(a)
        return -a
    a = _eval(tree[1], x)
    b = _eval(tree[2], x)
    if tag == "add":
        return a + b
    if tag == "sub":
        return a - b
    if tag == "mul":
        return a * b
    return np.divide(a, b)


def eval_tree(tree, x) -> np.ndarray:
    """Évalue l'expression sur un tableau numpy (nan/inf possibles)."""
    with np.errstate(all="ignore"):
        return _eval(tree, x)


# --------------------------------------------------------------------------
# Sérialisation : arbre  <->  liste de jetons (+ constantes alignées)
# --------------------------------------------------------------------------

def tree_to_tokens(tree):
    """Préfixe : renvoie (ids, values) où values[i] est la constante du
    jeton i (0.0 sinon)."""
    ids: list[int] = []
    values: list[float] = []

    def walk(node):
        tag = node[0]
        ids.append(TOK2ID[tag])
        if tag == "const":
            values.append(float(node[1]))
            return
        values.append(0.0)
        if tag == "x":
            return
        if len(node) == 2:
            walk(node[1])
        else:
            walk(node[1])
            walk(node[2])

    walk(tree)
    return ids, values


def ids_to_tree(ids, values, pos=0):
    """Reconstruit l'arbre. Renvoie (arbre, position suivante)."""
    if pos >= len(ids):
        raise ValueError("expression incomplète")
    tag = ID2TOK[ids[pos]]
    if tag == "x":
        return ("x",), pos + 1
    if tag == "const":
        return ("const", float(values[pos])), pos + 1
    if tag in UNARY_OPS:
        child, nxt = ids_to_tree(ids, values, pos + 1)
        return (tag, child), nxt
    if tag in BINARY_OPS:
        left, p = ids_to_tree(ids, values, pos + 1)
        right, p = ids_to_tree(ids, values, p)
        return (tag, left, right), p
    raise ValueError(f"jeton inattendu : {tag}")


def parse_ids(ids, values):
    """ids -> arbre, ou None si la séquence n'est pas une expression valide."""
    if not ids:
        return None
    try:
        tree, nxt = ids_to_tree(ids, values, 0)
        return tree if nxt == len(ids) else None
    except (ValueError, KeyError, IndexError):
        return None


def tree_to_str(t) -> str:
    """Affichage lisible avec priorités opératoires (x*x s'écrit x^2)."""
    tag = t[0]
    if tag == "x":
        return "x"
    if tag == "const":
        return f"{t[1]:g}"
    if tag == "neg":
        inner = t[1]
        s = tree_to_str(inner)
        return f"-({s})" if inner[0] in _PREC else f"-{s}"
    if tag in UNARY_OPS:
        return f"{tag}({tree_to_str(t[1])})"
    if tag == "mul":
        parties = []
        for i, (f, e) in enumerate(_groupes_mul(t)):
            s = tree_to_str(f)
            if _PREC.get(f[0], 3) < _PREC["mul"] or (
                    _est_negatif(f) and (i > 0 or e > 1)):
                s = f"({s})"
            parties.append(f"{s}^{e}" if e > 1 else s)
        return "*".join(parties)
    p_ = _PREC[tag]
    ls, rs = tree_to_str(t[1]), tree_to_str(t[2])
    if _PREC.get(t[1][0], 3) < p_:
        ls = f"({ls})"
    if _PREC.get(t[2][0], 3) <= p_:
        rs = f"({rs})"
    # constante négative à droite :  a - -3  ->  a - (-3)   /   2 * -3 -> 2 * (-3)
    if t[2][0] == "const" and t[2][1] < 0:
        rs = f"({rs})"
    return ls + _SYM[tag] + rs


def _latex_number(v: float) -> str:
    """Nombre formaté pour mathtext/LaTeX (gère les très petites valeurs)."""
    if v == 0.0:
        return "0"
    a = abs(v)
    if a >= 1e5 or a < 1e-4:
        mant, exp = f"{v:.3e}".split("e")
        return f"{float(mant):g} \\times 10^{{{int(exp)}}}"
    return f"{v:.6g}"


def tree_to_latex(t) -> str:
    """Arbre -> formule en LaTeX (compatible mathtext de matplotlib)."""
    tag = t[0]
    if tag == "x":
        return "x"
    if tag == "const":
        return _latex_number(t[1])
    if tag == "neg":
        inner = t[1]
        s = tree_to_latex(inner)
        return (f"-\\left({s}\\right)" if inner[0] in ("add", "sub", "neg")
                else f"-{s}")
    if tag == "sqrt":
        return f"\\sqrt{{{tree_to_latex(t[1])}}}"
    if tag in UNARY_OPS:
        fn = {"sin": r"\sin", "cos": r"\cos", "exp": r"\exp",
              "log": r"\ln", "tanh": r"\tanh"}.get(tag, "\\operatorname{" + tag + "}")
        return f"{fn}\\left({tree_to_latex(t[1])}\\right)"
    if tag == "mul":
        parties = []
        for i, (f, e) in enumerate(_groupes_mul(t)):
            s = tree_to_latex(f)
            if f[0] in ("add", "sub") or (_est_negatif(f) and (i > 0 or e > 1)):
                s = f"\\left({s}\\right)"
            parties.append(f"{s}^{{{e}}}" if e > 1 else s)
        return " \\cdot ".join(parties)

    ls, rs = tree_to_latex(t[1]), tree_to_latex(t[2])
    if tag == "div":
        return f"\\frac{{{ls}}}{{{rs}}}"
    # add / sub
    p_ = _PREC[tag]
    if _PREC.get(t[1][0], 3) < p_:
        ls = f"\\left({ls}\\right)"
    if _PREC.get(t[2][0], 3) < p_ or (tag == "sub" and t[2][0] in ("add", "sub")):
        rs = f"\\left({rs}\\right)"
    if t[2][0] == "const" and t[2][1] < 0:
        rs = f"\\left({rs}\\right)"
    return f"{ls} {'+' if tag == 'add' else '-'} {rs}"


# --------------------------------------------------------------------------
# Simplification symbolique de la formule retrouvée
# --------------------------------------------------------------------------

class _NonConvertible(Exception):
    """Expression SymPy qui ne peut pas s'écrire avec notre grammaire."""


def _est_zero(t):
    return t[0] == "const" and abs(t[1]) < 1e-12


def _est_un(t):
    return t[0] == "const" and abs(t[1] - 1.0) < 1e-12


def _est_negatif(t):
    return t[0] == "neg" or (t[0] == "const" and t[1] < 0.0)


def _facteurs_mul(t):
    """Aplatit une chaîne de multiplications en liste de facteurs."""
    if t[0] == "mul":
        return _facteurs_mul(t[1]) + _facteurs_mul(t[2])
    return [t]


def _groupes_mul(t):
    """Regroupe les facteurs identiques : [facteur, exposant]."""
    groupes, index = [], {}
    for f in _facteurs_mul(t):
        cle = repr(f)
        if cle in index:
            groupes[index[cle]][1] += 1
        else:
            index[cle] = len(groupes)
            groupes.append([f, 1])
    return groupes


def _eval_fonction(tag, v):
    """Valeur numérique d'une fonction unaire, ou None si indéfinie."""
    try:
        if tag == "sin":
            r = math.sin(v)
        elif tag == "cos":
            r = math.cos(v)
        elif tag == "exp":
            r = math.exp(v)
        elif tag == "log":
            r = math.log(v) if v > 0 else None
        elif tag == "sqrt":
            r = math.sqrt(v) if v >= 0 else None
        elif tag == "tanh":
            r = math.tanh(v)
        elif tag == "neg":
            r = -v
        else:
            return None
    except (ValueError, OverflowError):
        return None
    if r is None or not math.isfinite(r):
        return None
    return r


def simplify_struct(tree):
    """Simplifications exactes au niveau de l'arbre : calcul des constantes,
    éléments neutres (0 + a, 1·a, a/a, a − a…), doubles négations."""
    if tree is None:
        return None
    tag = tree[0]
    if tag in ("x", "const"):
        return tree
    if tag == "neg":
        a = simplify_struct(tree[1])
        if a[0] == "const":
            return ("const", -a[1])
        if a[0] == "neg":
            return a[1]
        return ("neg", a)
    if tag in UNARY_OPS:
        a = simplify_struct(tree[1])
        if a[0] == "const":
            v = _eval_fonction(tag, a[1])
            if v is not None:
                return ("const", v)
        # log(exp(a)) = a  (valable partout)
        if tag == "log" and a[0] == "exp":
            return a[1]
        return (tag, a)

    a, b = simplify_struct(tree[1]), simplify_struct(tree[2])
    if tag == "add":
        if _est_zero(a):
            return b
        if _est_zero(b):
            return a
        if a[0] == b[0] == "const":
            return ("const", a[1] + b[1])
        return ("add", a, b)
    if tag == "sub":
        if _est_zero(b):
            return a
        if a == b:
            return ("const", 0.0)
        if a[0] == b[0] == "const":
            return ("const", a[1] - b[1])
        if b[0] == "neg":
            return simplifier_add(a, b[1])
        return ("sub", a, b)
    if tag == "mul":
        if _est_zero(a) or _est_zero(b):
            return ("const", 0.0)
        if _est_un(a):
            return b
        if _est_un(b):
            return a
        if a[0] == b[0] == "const":
            return ("const", a[1] * b[1])
        if b[0] == "const":
            a, b = b, a
        if a[0] == "const" and abs(a[1] + 1.0) < 1e-12:
            return ("neg", b) if b[0] != "const" else ("const", -b[1])
        return ("mul", a, b)
    # div
    if _est_zero(a):
        return ("const", 0.0)
    if _est_un(b):
        return a
    if a == b:
        return ("const", 1.0)
    if a[0] == b[0] == "const" and b[1] != 0.0:
        return ("const", a[1] / b[1])
    if b[0] == "const" and b[1] != 0.0:
        # x / 0.59 -> (1/0.59) * x : plus lisible et même coût
        return ("mul", ("const", 1.0 / b[1]), a)
    return ("div", a, b)


def simplifier_add(a, b):
    """Additionne deux arbres en gardant la forme la plus lisible
    (a + (-3) devient a - 3)."""
    if b[0] == "const" and b[1] < 0:
        return ("sub", a, ("const", -b[1]))
    if b[0] == "neg":
        return ("sub", a, b[1])
    if b[0] == "mul" and b[1][0] == "const" and b[1][1] < 0:
        reste = ("mul", ("const", -b[1][1]), b[2])
        return ("sub", a, simplify_struct(reste))
    return ("add", a, b)


# ---- pont avec SymPy (facultatif) ----------------------------------------

def _sympy():
    try:
        import sympy
        return sympy
    except ImportError:
        return None


def tree_to_sympy(t):
    sp = _sympy()
    if sp is None:
        raise ImportError("sympy n'est pas installé")
    tag = t[0]
    if tag == "x":
        return sp.Symbol("x")
    if tag == "const":
        return sp.Float(float(t[1]))
    if tag == "neg":
        return -tree_to_sympy(t[1])
    if tag == "sqrt":
        return sp.sqrt(tree_to_sympy(t[1]))
    if tag in UNARY_OPS:
        f = {"sin": sp.sin, "cos": sp.cos, "exp": sp.exp,
             "log": sp.log, "tanh": sp.tanh}[tag]
        return f(tree_to_sympy(t[1]))
    a, b = tree_to_sympy(t[1]), tree_to_sympy(t[2])
    if tag == "add":
        return a + b
    if tag == "sub":
        return a - b
    if tag == "mul":
        return a * b
    if tag == "div":
        return a / b
    raise _NonConvertible(tag)


def _puissance(b, n):
    acc = b
    for _ in range(n - 1):
        acc = ("mul", acc, b)
    return acc


def _somme(termes):
    acc = termes[0]
    for t in termes[1:]:
        acc = simplifier_add(acc, t)
    return acc


def _produit(facteurs):
    if not facteurs:
        return ("const", 1.0)
    # les coefficients numériques d'abord : 2,5·x plutôt que x·2,5
    facteurs = sorted(facteurs, key=lambda f: 0 if f[0] == "const" else 1)
    if len(facteurs) > 1 and facteurs[0][0] == "const" and facteurs[0][1] < 0:
        reste = _produit(facteurs[1:])
        if abs(facteurs[0][1] + 1.0) < 1e-12:
            return ("neg", reste) if reste[0] != "const" else ("const", -reste[1])
    acc = facteurs[0]
    for f in facteurs[1:]:
        acc = ("mul", acc, f)
    return acc


def tree_from_sympy(e, profondeur=0):
    """Expression SymPy -> arbre de notre grammaire."""
    sp = _sympy()
    if profondeur > 32:
        raise _NonConvertible("trop profond")
    e = sp.sympify(e)
    if e.free_symbols - {sp.Symbol("x")}:
        raise _NonConvertible("plusieurs variables")
    if e == sp.Symbol("x"):
        return ("x",)
    if e.is_Number:
        try:
            v = float(e)
        except (TypeError, ValueError):
            raise _NonConvertible("nombre non convertible") from None
        if not math.isfinite(v):
            raise _NonConvertible("nombre non fini")
        return ("const", v)

    if e.is_Add:
        termes = [tree_from_sympy(a, profondeur + 1) for a in e.args]
        # constantes numériques en dernier : « 2,5·x − 1 » plutôt que « −1 + 2,5·x »
        termes.sort(key=lambda t: t[0] == "const")
        return _somme(termes)
    if e.is_Mul:
        num, den = [], []
        for a in e.args:
            if a.is_Pow and a.exp.is_negative:
                den.append(tree_from_sympy(sp.Pow(a.base, -a.exp), profondeur + 1))
            else:
                num.append(tree_from_sympy(a, profondeur + 1))
        n = _produit(num)
        return ("div", n, _produit(den)) if den else n
    if e.is_Pow:
        base, exp = e.base, e.exp
        if base == sp.E:
            return ("exp", tree_from_sympy(exp, profondeur + 1))
        if exp == sp.Rational(1, 2):
            return ("sqrt", tree_from_sympy(base, profondeur + 1))
        if exp.is_Integer:
            n = int(exp)
            if abs(n) > 12:
                raise _NonConvertible("exposant trop grand")
            b = tree_from_sympy(base, profondeur + 1)
            if n == 0:
                return ("const", 1.0)
            if n < 0:
                return ("div", ("const", 1.0), _puissance(b, -n))
            return _puissance(b, n)
        raise _NonConvertible("exposant non entier")

    fonction = {sp.sin: "sin", sp.cos: "cos", sp.exp: "exp",
                sp.log: "log", sp.tanh: "tanh", sp.sqrt: "sqrt"}.get(e.func)
    if fonction is not None:
        return (fonction, tree_from_sympy(e.args[0], profondeur + 1))
    raise _NonConvertible(f"fonction non gérée : {e.func}")


def _variantes_sympy(tree):
    """Formes simplifiées produites par SymPy (les échecs sont ignorés)."""
    sp = _sympy()
    if sp is None or tree_size(tree) > 28:
        return []
    try:
        base = tree_to_sympy(tree)
    except Exception:                                  # noqa: BLE001
        return []
    vues, formes = set(), []
    for operation in (lambda z: z, sp.expand, sp.simplify, sp.trigsimp,
                      sp.cancel, sp.factor):
        try:
            r = sp.sympify(operation(base))
        except Exception:                              # noqa: BLE001
            continue
        cle = sp.srepr(r)
        if cle not in vues:
            vues.add(cle)
            formes.append(r)
    return formes


def _valeur_simple(v):
    """Ramène 2.99997 à 3, 0.499998 à 0.5… sans trahir la valeur."""
    if v == 0.0 or abs(v) > 1e4:
        return v
    candidats = {float(round(v)), round(v * 2) / 2.0, round(v * 4) / 4.0,
                 round(v * 10) / 10.0}
    for q in range(2, 9):
        p = round(v * q)
        if p and abs(p) <= 12:              # fractions simples uniquement
            candidats.add(p / q)
    meilleur = min(candidats, key=lambda c: abs(c - v))
    return meilleur if abs(meilleur - v) <= 1e-3 * max(1.0, abs(v)) else v


def _arrondir_constantes(tree):
    """Remplace les constantes quasi entières par leur valeur simple."""
    tag = tree[0]
    if tag == "const":
        return ("const", _valeur_simple(tree[1]))
    if tag in ("x",):
        return tree
    return (tag,) + tuple(_arrondir_constantes(c) for c in tree[1:])


def _compter(tree, tag):
    return (1 if tree[0] == tag else 0) + sum(
        _compter(c, tag) for c in tree[1:] if isinstance(c, tuple))


def _complexite(tree):
    """Mesure de lisibilité : taille, puis nombre de divisions."""
    return (tree_size(tree), _compter(tree, "div"))


def simplify_formula(tree, x, y):
    """Renvoie la version la plus simple de `tree` qui conserve la même
    fonction sur le nuage (x, y).

    On arrondit d'abord les constantes quasi simples (2,99997 → 3), puis on
    demande à SymPy de regrouper les termes (x + x → 2·x, x·x/x → x…). La
    formule d'origine n'est gardée que si aucune version plus simple ne
    reste fidèle au nuage."""
    if tree is None:
        return None
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    base = reconstruction_error(tree, x, y)
    if not np.isfinite(base):
        return tree
    ecart = float(np.std(y)) if y.size else 1.0
    tolerance_stricte = base * 1.001 + 1e-9
    tolerance_arrondi = base + 2e-3 * max(ecart, 1e-9)

    structurel = simplify_struct(tree)
    arrondi = _arrondir_constantes(structurel)

    candidats = []                     # (arbre, tolérance, préférence)

    def ajouter_formes_sympy(source, tolerance, preference):
        """Ajoute les formes SymPy, puis leur version aux constantes
        arrondies (préférée à qualité de fit égale)."""
        for forme in _variantes_sympy(source):
            try:
                arbre = tree_from_sympy(forme)
            except Exception:                          # noqa: BLE001
                continue
            lisse = _arrondir_constantes(arbre)
            if repr(lisse) != repr(arbre):
                candidats.append((lisse, tolerance_arrondi, preference - 0.5))
            candidats.append((arbre, tolerance, preference))

    # 1. formes canoniques de l'arbre aux constantes arrondies
    ajouter_formes_sympy(arrondi, tolerance_arrondi, 1.0)
    candidats.append((arrondi, tolerance_arrondi, 1.5))
    # 2. mêmes formes sans arrondi (si l'arrondi a été rejeté plus bas)
    ajouter_formes_sympy(structurel, tolerance_stricte, 2.0)
    candidats.append((structurel, tolerance_stricte, 3.0))
    candidats.append((tree, tolerance_stricte, 4.0))

    valides = []
    vus = set()
    for arbre, tolerance, preference in candidats:
        if arbre is None:
            continue
        cle = repr(arbre)
        if cle in vus:                 # on garde la première = la préférée
            continue
        vus.add(cle)
        err = reconstruction_error(arbre, x, y)
        if np.isfinite(err) and err <= tolerance:
            valides.append((_complexite(arbre), preference, err, arbre))
    if not valides:
        return tree
    # moins de nœuds d'abord, puis la forme la plus canonique, puis l'erreur
    valides.sort(key=lambda t: (t[0], t[1], t[2]))
    return valides[0][3]


# =========================================================================
# 2. NUAGE DE POINTS
# =========================================================================

def canonical_tree(tree):
    """Forme canonique d'une expression (sans changer sa valeur).

    - opérandes commutatifs (`+`, `*`) triés de façon déterministe ;
    - constantes repliées (`2 + 3` → `5`, `2 * 3` → `6`) ;
    - éléments neutres supprimés (`0 + a`, `1 · a`, `a − a`, `a / a`) ;
    - doubles négations annulées.

    Utilisée comme **cible d'entraînement** : une seule écriture par fonction,
    donc moins d'entropie à apprendre pour le décodeur."""
    tag = tree[0]
    if tag in ("x", "const"):
        return tree
    if tag == "neg":
        a = canonical_tree(tree[1])
        if a[0] == "const":
            return ("const", -a[1])
        if a[0] == "neg":
            return a[1]
        return ("neg", a)
    if tag in UNARY_OPS:
        return (tag, canonical_tree(tree[1]))

    a, b = canonical_tree(tree[1]), canonical_tree(tree[2])
    if tag == "add":
        if _est_zero(a):
            return b
        if _est_zero(b):
            return a
        if a[0] == b[0] == "const":
            return ("const", a[1] + b[1])
        if repr(a) > repr(b):
            a, b = b, a
        return ("add", a, b)
    if tag == "sub":
        if _est_zero(b):
            return a
        if a == b:
            return ("const", 0.0)
        if a[0] == b[0] == "const":
            return ("const", a[1] - b[1])
        return ("sub", a, b)
    if tag == "mul":
        if _est_zero(a) or _est_zero(b):
            return ("const", 0.0)
        if _est_un(a):
            return b
        if _est_un(b):
            return a
        if a[0] == b[0] == "const":
            return ("const", a[1] * b[1])
        if a[0] != "const" and b[0] == "const":
            a, b = b, a
        elif a[0] != "const" and b[0] != "const" and repr(a) > repr(b):
            a, b = b, a
        return ("mul", a, b)
    # div
    if _est_zero(a):
        return ("const", 0.0)
    if _est_un(b):
        return a
    if a == b:
        return ("const", 1.0)
    if a[0] == b[0] == "const" and b[1] != 0.0:
        return ("const", a[1] / b[1])
    return ("div", a, b)


def generate_sample(rng, n_points, max_depth, noise=0.0, x_range=4.0,
                    max_len=64, max_abs_y=50.0, canonique=False,
                    n_points_var=True, domaine_var=True):
    """Tire une fonction au hasard + son nuage de points.

    Rejette les fonctions non finies, quasi constantes, trop explosées ou
    trop longues en jetons. Renvoie un dict {x, y, tree, depth}.

    `canonique` réécrit l'arbre sous une forme unique (opérandes commutatifs
    triés, constantes repliées, éléments neutres supprimés) : le modèle
    n'apprend plus plusieurs écritures de la même fonction.

    Si `n_points_var` est activé, le nombre de points est tiré aléatoirement
    entre 16 et 128. Si `domaine_var` est activé, le domaine [x0, x0+span]
    est variable (span 2.5–7.0)."""
    for _ in range(200):
        tree = random_tree(rng, max_depth)
        if canonique:
            tree = canonical_tree(tree)
        ids, _ = tree_to_tokens(tree)
        if len(ids) > max_len - 2:            # doit tenir dans le décodeur
            continue
        if n_points_var:
            npts = int(rng.randint(16, 129))
        else:
            npts = int(n_points)
        if domaine_var:
            span = rng.uniform(2.5, 7.0)
            x0 = rng.uniform(-2.5, 2.5)
            x = np.sort(np.array([rng.uniform(x0, x0 + span) for _ in range(npts)], dtype=np.float64))
        else:
            x = np.sort(np.array(
                [rng.uniform(-x_range, x_range) for _ in range(npts)],
                dtype=np.float64))
        y = eval_tree(tree, x)
        if not np.all(np.isfinite(y)):
            continue
        m = float(np.max(np.abs(y)))
        if m > max_abs_y or m < 1e-12:      # asymptotes / constantes
            continue
        s = float(y.std())
        if s < 1e-2:
            continue
        if noise > 0.0:
            y = y + np.array([rng.gauss(0.0, noise * s) for _ in range(npts)])
        return {"x": x, "y": y, "tree": tree, "depth": tree_depth(tree)}
    raise RuntimeError("impossible de générer une fonction valide "
                       "(essayez d'augmenter --max-len ou de baisser --max-depth)")


def sample_features(sample) -> np.ndarray:
    """(n_points, 4) : x normalisé dans [-1,1], y centré-réduit, et les 2
    statistiques de y (moyenne, écart-type) pour que le modèle connaisse
    l'échelle absolue des constantes à prédire."""
    x, y = sample["x"], sample["y"]
    m = float(y.mean())
    s = max(float(y.std()), 1e-8)
    yn = (y - m) / s
    xmin = float(x.min())
    xmax = float(x.max())
    if xmax - xmin < 1e-12:
        xn = np.zeros_like(x)
    else:
        xn = 2.0 * (x - xmin) / (xmax - xmin) - 1.0
    feat = np.stack([xn, yn,
                     np.full_like(x, m),
                     np.full_like(x, math.log(s))], axis=-1)
    return feat.astype(np.float32)


def build_batch(samples, device, max_len):
    """Construit les tenseurs d'un lot d'entraînement."""
    B = len(samples)
    M = samples[0]["x"].shape[0]

    src = torch.from_numpy(np.stack([sample_features(s) for s in samples]))
    seqs = [tree_to_tokens(s["tree"]) for s in samples]
    L = min(max_len, max(len(i) for i, _ in seqs) + 2)

    tgt_in = np.full((B, L), PAD_ID, dtype=np.int64)
    tgt_out = np.full((B, L), PAD_ID, dtype=np.int64)
    const_tgt = np.zeros((B, L), dtype=np.float32)

    for b, (ids, vals) in enumerate(seqs):
        k = len(ids)
        tgt_in[b, 0] = BOS_ID
        tgt_in[b, 1:k + 1] = ids
        tgt_out[b, :k] = ids
        tgt_out[b, k] = EOS_ID
        const_tgt[b, :k] = vals

    return {
        "src": src.to(device),
        "tgt_in": torch.from_numpy(tgt_in).to(device),
        "tgt_out": torch.from_numpy(tgt_out).to(device),
        "const_tgt": torch.from_numpy(const_tgt).to(device),
        "size": M,
    }


# =========================================================================
# 3. MODÈLE
# =========================================================================

def _sinusoidal_pos(d_model, length, device):
    pe = torch.zeros(length, d_model, device=device)
    position = torch.arange(0, length, dtype=torch.float, device=device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float, device=device)
                         * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0)


class SymbolicRegressor(nn.Module):
    """Encodeur Transformer sur le nuage de points + décodeur autoregressif
    produisant la formule (jetons + constantes). Polyvalent : accepte un
    nombre variable de points et un domaine variable."""

    def __init__(self, d_model=256, nhead=8, enc_layers=4, dec_layers=4,
                 ff=1024, dropout=0.1, n_points=64, max_len=64):
        super().__init__()
        self.hparams = dict(d_model=d_model, nhead=nhead,
                            enc_layers=enc_layers, dec_layers=dec_layers,
                            ff=ff, dropout=dropout, n_points=n_points,
                            max_len=max_len)
        self.register_buffer("arity", ARITY_VEC.clone(), persistent=False)

        # --- encodeur (nuage de points triés) ---
        self.in_proj = nn.Sequential(
            nn.Linear(4, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=ff,
            dropout=dropout, activation="gelu", batch_first=True,
            norm_first=True)
        self.encoder = nn.TransformerEncoder(
            enc_layer, enc_layers, norm=nn.LayerNorm(d_model),
            enable_nested_tensor=False)

        # --- décodeur (formule) ---
        self.tok_emb = nn.Embedding(len(VOCAB), d_model)
        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=ff,
            dropout=dropout, activation="gelu", batch_first=True,
            norm_first=True)
        self.decoder = nn.TransformerDecoder(
            dec_layer, dec_layers, norm=nn.LayerNorm(d_model))

        self.lm_head = nn.Linear(d_model, len(VOCAB))
        self.const_head = nn.Linear(d_model, 1)
        self._init_weights()

    # ------------------------------------------------------------------
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    # ------------------------------------------------------------------
    def encode(self, src):
        B, M, _ = src.shape
        h = self.in_proj(src) + _sinusoidal_pos(self.hparams["d_model"], M, src.device)
        return self.encoder(h)

    def decode(self, tgt_ids, memory):
        L = tgt_ids.size(1)
        h = self.tok_emb(tgt_ids) * math.sqrt(self.hparams["d_model"])
        h = h + _sinusoidal_pos(self.hparams["d_model"], L, tgt_ids.device)
        causal = torch.zeros(L, L, device=tgt_ids.device)
        causal.masked_fill_(
            torch.triu(torch.ones(L, L, dtype=torch.bool, device=tgt_ids.device), 1),
            float("-inf"))
        return self.decoder(h, memory, tgt_mask=causal)

    def forward(self, src, tgt_ids):
        memory = self.encode(src)
        h = self.decode(tgt_ids, memory)
        return self.lm_head(h), self.const_head(h).squeeze(-1)

    def step(self, ids, memory):
        """Jetons suivants (logits, constante) pour une mémoire déjà encodée."""
        h = self.decode(ids, memory)[:, -1]
        return self.lm_head(h), self.const_head(h).squeeze(-1)


# =========================================================================
# 4. MASQUAGE GRAMMATICAL
# =========================================================================

def allowed_rows(slots, device):
    """(B,) 'sous-expressions restantes à produire' -> (B, V) bool.

    slots > 0 : tous les jetons sauf pad/bos/eos.
    slots = 0 : seulement eos/pad (l'expression est complète)."""
    base = torch.ones(V, dtype=torch.bool, device=device)
    base[PAD_ID] = False
    base[BOS_ID] = False
    base[EOS_ID] = False
    rows = base.expand(slots.size(0), -1).clone()
    zero = slots == 0
    if bool(zero.any()):
        rows[zero] = False
        rows[zero, EOS_ID] = True
        rows[zero, PAD_ID] = True
    return rows


def target_valid_mask(tgt_out):
    """(B, L, V) : ensembles de jetons autorisés à chaque étape, en suivant
    (en professeur) la structure de la cible. Sert à masquer les logits
    avant la cross-entropie."""
    B, L = tgt_out.shape
    device = tgt_out.device
    out = torch.zeros(B, L, V, dtype=torch.bool, device=device)
    slots = torch.ones(B, dtype=torch.long, device=device)
    for t in range(L):
        out[:, t] = allowed_rows(slots, device)
        open_ = slots > 0
        slots = torch.where(open_,
                            slots - 1 + ARITY_VEC.to(device)[tgt_out[:, t]],
                            slots)
    return out


def update_slots(slots, tok):
    return torch.where(slots > 0, slots - 1 + ARITY_VEC.to(slots.device)[tok],
                       slots)


# =========================================================================
# 5. DÉCODAGE GLOUTON (contraint par la grammaire)
# =========================================================================

@torch.no_grad()
def _decode_from_memory(model, memory, max_len, temperature, device,
                        const_noise=0.0, rng=None, gen=None):
    """Une passe de décodage autoregressif, contraint par la grammaire.

    Si `temperature` > 0, les constantes sont perturbées (bruit gaussien) :
    comme elles sortent d'une tête de régression déterministe, c'est le seul
    moyen de faire varier leurs valeurs d'un candidat à l'autre — la
    sélection par RMSE affine alors les coefficients.

    `rng` (numpy) et `gen` (torch.Generator) rendent le tirage reproductible
    pour un nuage donné."""
    B = memory.size(0)
    ids = torch.full((B, 1), BOS_ID, dtype=torch.long, device=device)
    slots = torch.ones(B, dtype=torch.long, device=device)
    done = torch.zeros(B, dtype=torch.bool, device=device)
    out_ids: list[list[int]] = [[] for _ in range(B)]
    out_vals: list[list[float]] = [[] for _ in range(B)]

    for _ in range(max_len):
        logits, cval = model.step(ids, memory)
        logits = logits.masked_fill(~allowed_rows(slots, device), -1e4)
        if temperature > 0.0:
            probas = F.softmax(logits, dim=-1)
            if gen is not None:
                nxt = torch.multinomial(probas, 1, generator=gen).squeeze(1)
            else:
                nxt = torch.multinomial(probas, 1).squeeze(1)
        else:
            nxt = logits.argmax(dim=-1)
        nxt = torch.where(done, torch.full_like(nxt, PAD_ID), nxt)
        for b in range(B):
            if bool(done[b]):
                continue
            tok = int(nxt[b])
            out_ids[b].append(tok)
            if tok == CONST_ID:
                v = float(cval[b])
                if temperature > 0.0 and const_noise > 0.0:
                    if rng is not None:
                        v += float(rng.normal(0.0, const_noise))
                    else:
                        v += float(np.random.normal(0.0, const_noise))
                out_vals[b].append(v)
            else:
                out_vals[b].append(0.0)
        ids = torch.cat([ids, nxt.unsqueeze(1)], dim=1)
        slots = update_slots(slots, nxt)
        done = done | (slots == 0)
        if bool(done.all()):
            break
    return list(zip(out_ids, out_vals))


@torch.no_grad()
def _decode_batched(model, memory, max_len, temps, const_noises, rng, gen):
    """Décodage en batch : tous les candidats partagent le même encodage."""
    B = memory.size(0)
    device = memory.device
    ids = torch.full((B, 1), BOS_ID, dtype=torch.long, device=device)
    slots = torch.ones(B, dtype=torch.long, device=device)
    done = torch.zeros(B, dtype=torch.bool, device=device)
    out_ids = [[] for _ in range(B)]
    out_vals = [[] for _ in range(B)]
    temps_t = torch.tensor(temps, dtype=torch.float32, device=device)
    cn_t = torch.tensor(const_noises, dtype=torch.float32, device=device)
    if len(temps) != B:
        temps = list(temps) + [0.0] * (B - len(temps))
        temps_t = torch.tensor(temps[:B], dtype=torch.float32, device=device)
    if len(const_noises) != B:
        cn_t = torch.tensor(list(const_noises) + [0.0] * (B - len(const_noises)), dtype=torch.float32, device=device)[:B]

    for _ in range(max_len):
        logits, cval = model.step(ids, memory)
        logits = logits.masked_fill(~allowed_rows(slots, device), -1e4)
        t = temps_t.clone()
        t = torch.where(done, torch.ones_like(t), t)
        probs = F.softmax(logits / t.unsqueeze(1).clamp(min=1e-4), dim=-1)
        if gen is not None:
            nxt = torch.multinomial(probs, 1, generator=gen).squeeze(1)
        elif rng is not None and any(temps[b] > 0 for b in range(B)):
            nxt_list = []
            for b in range(B):
                if bool(done[b]):
                    nxt_list.append(PAD_ID)
                elif temps[b] <= 0:
                    nxt_list.append(int(logits[b].argmax().item()))
                else:
                    p = probs[b].detach().cpu().numpy()
                    p = p / (p.sum() + 1e-18)
                    nxt_list.append(int(rng.choice(len(p), p=p)))
            nxt = torch.tensor(nxt_list, dtype=torch.long, device=device)
        else:
            nxt = logits.argmax(dim=-1)
        nxt = torch.where(done, torch.full_like(nxt, PAD_ID), nxt)
        for b in range(B):
            if bool(done[b]):
                continue
            tok = int(nxt[b])
            out_ids[b].append(tok)
            if tok == CONST_ID:
                v = float(cval[b].item())
                if temps[b] > 0.0 and cn_t[b] > 0.0:
                    if rng is not None:
                        v += float(rng.normal(0.0, float(cn_t[b].item())))
                    else:
                        v += float(np.random.normal(0.0, float(cn_t[b].item())))
                out_vals[b].append(v)
            else:
                out_vals[b].append(0.0)
        ids = torch.cat([ids, nxt.unsqueeze(1)], dim=1)
        slots = update_slots(slots, nxt)
        done = done | (slots == 0)
        if bool(done.all()):
            break
    return list(zip(out_ids, out_vals))


@torch.no_grad()
def decode_sequences(model, src, max_len, n=1, temperature=1.0, const_noise=0.0,
                     rng=None, gen=None):
    """Génère n candidats : le premier glouton, les suivants échantillonnés
    (tous contraints par la grammaire), avec des températures légèrement
    variées pour diversifier les structures. Doublons éliminés."""
    model.eval()
    memory = model.encode(src)          # encodé une seule fois pour les n candidats
    seen: set[tuple] = set()
    out = []
    for cand in range(n):
        if cand == 0:
            temp = 0.0
        else:
            # alterne 0,85× / 1,0× / 1,15× la température demandée
            temp = temperature * (1.0 + 0.15 * ((cand % 3) - 1))
        for ids, values in _decode_from_memory(
                model, memory, max_len, temp, src.device,
                const_noise if cand else 0.0, rng, gen):
            key = tuple(ids)
            if key in seen:
                continue
            seen.add(key)
            out.append((ids, values))
    return out


@torch.no_grad()
def decode_beam(model, src, max_len, beam_width=6):
    """Beam search grammatical : trouve des formules à haute probabilité."""
    model.eval()
    memory = model.encode(src)
    B = 1
    device = src.device
    ids = torch.full((B, 1), BOS_ID, dtype=torch.long, device=device)
    slots = torch.ones(B, dtype=torch.long, device=device)
    beams = [(ids.clone(), slots.clone(), 0.0, [[]], [[0.0]])]
    finished = []

    for _ in range(max_len):
        if not beams:
            break
        all_cands = []
        for b_ids, b_slots, score, out_ids, out_vals in beams:
            if bool((b_slots == 0).all()):
                finished.append((b_ids, score, out_ids, out_vals))
                continue
            logits, cval = model.step(b_ids, memory)
            logits = logits.masked_fill(~allowed_rows(b_slots, device), -1e4)
            logp = F.log_softmax(logits, dim=-1)
            topk = torch.topk(logp, min(beam_width * 2, V))
            for k in range(topk.indices.size(1)):
                tok = int(topk.indices[0, k])
                if tok == PAD_ID or tok == BOS_ID:
                    continue
                lp = float(topk.values[0, k])
                new_ids = torch.cat([b_ids, torch.tensor([[tok]], device=device)], dim=1)
                new_slots = update_slots(b_slots.clone(), torch.tensor([tok], device=device))
                new_out_ids = [list(x) + [tok] for x in out_ids]
                if tok == CONST_ID:
                    new_out_vals = [list(x) + [float(cval[0])] for x in out_vals]
                else:
                    new_out_vals = [list(x) + [0.0] for x in out_vals]
                all_cands.append((new_ids, new_slots, score + lp, new_out_ids, new_out_vals))
        if not all_cands:
            break
        all_cands.sort(key=lambda x: x[2], reverse=True)
        beams = all_cands[:beam_width]
    for b_ids, score, out_ids, out_vals in beams:
        finished.append((b_ids, score, out_ids, out_vals))
    out = []
    seen = set()
    for _, score, out_ids, out_vals in finished:
        for ids, vals in zip(out_ids, out_vals):
            if not ids:
                continue
            key = tuple(ids)
            if key in seen:
                continue
            seen.add(key)
            out.append((ids, vals))
    return out


# =========================================================================
# 6. MÉTRIQUES
# =========================================================================

def relative_rmse(true_tree, pred_tree, x_range, n=512):
    """RMSE entre la fonction vraie et la fonction prédite, normalisée par
    l'écart-type de la vraie fonction. Les valeurs non finies de la
    prédiction sont pénalisées."""
    xg = np.linspace(-x_range, x_range, n)
    yt = eval_tree(true_tree, xg)
    valid = np.isfinite(yt)
    if not valid.any():
        return float("inf")
    scale = float(yt[valid].std())
    if scale < 1e-9:
        scale = 1.0
    if pred_tree is None:
        return 10.0
    yp = eval_tree(pred_tree, xg)
    err2 = np.zeros_like(yt)
    good = valid & np.isfinite(yp)
    err2[good] = (yp[good] - yt[good]) ** 2
    err2[valid & ~np.isfinite(yp)] = (10.0 * scale) ** 2
    return float(np.sqrt(err2.mean())) / scale


def _torch_eval(node, x):
    """Évalue l'arbre avec des tenseurs ; les constantes sont des paramètres
    entraînables, ce qui permet de les affiner."""
    tag = node[0]
    if tag == "x":
        return x
    if tag == "const":
        return node[1] * torch.ones_like(x)
    if tag in UNARY_OPS:
        a = _torch_eval(node[1], x)
        if tag == "sin":
            return torch.sin(a)
        if tag == "cos":
            return torch.cos(a)
        if tag == "exp":
            return torch.exp(a.clamp(-20, 20))
        if tag == "log":
            return torch.log(a.clamp(min=1e-6))
        if tag == "sqrt":
            return torch.sqrt(a.clamp(min=1e-6))
        if tag == "tanh":
            return torch.tanh(a)
        return -a
    a = _torch_eval(node[1], x)
    b = _torch_eval(node[2], x)
    if tag == "add":
        return a + b
    if tag == "sub":
        return a - b
    if tag == "mul":
        return a * b
    return a / (b + torch.sign(b) * 1e-6 + 1e-6)


def refine_constants(tree, x, y, steps=300, lr=0.05):
    """Affine les valeurs des constantes de `tree` par descente de gradient
    pour minimiser l'erreur sur le nuage observé. Renvoie un nouvel arbre.

    Le modèle fournit la *structure* et une estimation initiale : ce réglage
    numérique lui donne la précision qu'un petit réseau ne peut pas atteindre."""
    if tree is None:
        return None
    # collecte des constantes
    consts = []

    def build(node):
        tag = node[0]
        if tag == "const":
            p = torch.nn.Parameter(torch.tensor(float(node[1])))
            consts.append(p)
            return ("const", p)
        if tag == "x":
            return ("x",)
        if len(node) == 2:
            return (tag, build(node[1]))
        return (tag, build(node[1]), build(node[2]))

    root = build(tree)
    if not consts:
        return tree

    xt = torch.from_numpy(np.asarray(x, dtype=np.float64)).to(torch.float32)
    yt = torch.from_numpy(np.asarray(y, dtype=np.float64)).to(torch.float32)
    opt = torch.optim.Adam(consts, lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        pred = _torch_eval(root, xt)
        loss = torch.mean((pred - yt) ** 2)
        if not torch.isfinite(loss):
            return tree                      # divergence : on garde l'original
        loss.backward()
        opt.step()
    # reconstruit l'arbre avec les valeurs affinées
    def to_plain(node):
        tag = node[0]
        if tag == "const":
            return ("const", float(node[1].detach()))
        if tag == "x":
            return ("x",)
        if len(node) == 2:
            return (tag, to_plain(node[1]))
        return (tag, to_plain(node[1]), to_plain(node[2]))

    refined = to_plain(root)
    if reconstruction_error(refined, x, y) < reconstruction_error(tree, x, y):
        return refined
    return tree


def reconstruction_error(tree, x, y):
    """RMSE d'une formule candidate sur le nuage observé : critère de
    sélection entre les candidats (génération + vérification)."""
    if tree is None:
        return float("inf")
    yp = eval_tree(tree, x)
    finite = np.isfinite(yp) & np.isfinite(y)
    if not finite.any():
        return float("inf")
    d = np.clip(yp[finite] - y[finite], -1e6, 1e6)
    rmse = float(np.sqrt(np.mean(d * d)))
    # léger malus si la formule s'annule/expire en dehors du nuage,
    # proportionnel à l'échelle de y (robuste aux très petites/grandes valeurs)
    echelle = max(float(np.std(y[finite])), 1e-12)
    return rmse + 10.0 * echelle * float((~finite).sum()) / max(yp.size, 1)


def _cle_canonique(tree, arrondi=1e-3):
    """Clé de dédoublonnage : deux écritures équivalentes de la même formule
    donnent la même clé (opérandes commutatifs triés, constantes arrondies)."""
    tag = tree[0]
    if tag == "x":
        return "x"
    if tag == "const":
        v = float(tree[1])
        # arrondi pour retrouver les constantes identiques
        if abs(v) < 1e-9:
            v = 0.0
        else:
            v = round(v / arrondi) * arrondi
        # normalisation
        if v == -0.0:
            v = 0.0
        return f"c{v:g}"
    enfants = [_cle_canonique(c, arrondi) for c in tree[1:]]
    if tag in ("add", "mul"):
        enfants.sort()
    return f"{tag}({','.join(enfants)})"


def _eval_numerique(tree, x):
    try:
        yp = eval_tree(tree, x)
        if np.all(np.isfinite(yp)):
            return yp
    except Exception:
        pass
    return None


def _score_semantique(trees, x):
    """Score sémantique : regroupe les formules équivalentes sur une grille
    et garde la plus simple par groupe."""
    if not trees:
        return []
    xg = np.linspace(float(np.min(x)), float(np.max(x)), 256)
    groupes = []
    for tree in trees:
        yp = _eval_numerique(tree, xg)
        if yp is None:
            continue
        trouve = False
        for g in groupes:
            if g["yp"] is None:
                continue
            if np.max(np.abs(yp - g["yp"])) < 1e-4 * (np.std(g["yp"]) + 1e-12):
                if tree_size(tree) < tree_size(g["tree"]):
                    g["tree"] = tree
                trouve = True
                break
        if not trouve:
            groupes.append({"yp": yp, "tree": tree})
    return [g["tree"] for g in groupes]


def score_candidat(tree, x, y, penalite=0.002):
    """Score de sélection : RMSE relative au nuage, plus une pénalité de
    complexité. Sans cette pénalité, un gros arbre profite toujours d'un
    paramètre supplémentaire pour coller aux points (sur-ajustement)."""
    if tree is None:
        return float("inf")
    echelle = max(float(np.std(y)), 1e-9)
    return reconstruction_error(tree, x, y) / echelle + penalite * tree_size(tree)


# ---- affinage des constantes par Levenberg–Marquardt (scipy) --------------

def _liste_constantes(tree):
    """Valeurs des constantes, dans l'ordre de parcours de l'arbre."""
    out = []

    def walk(node):
        if node[0] == "const":
            out.append(float(node[1]))
        for c in node[1:]:
            if isinstance(c, tuple):
                walk(c)

    walk(tree)
    return out


def _remplacer_constantes(tree, vals, compteur):
    tag = tree[0]
    if tag == "const":
        v = vals[compteur[0]]
        compteur[0] += 1
        return ("const", float(v))
    if tag == "x":
        return tree
    return (tag,) + tuple(
        _remplacer_constantes(c, vals, compteur) if isinstance(c, tuple) else c
        for c in tree[1:])


def _eval_params(tree, x, vals, compteur):
    """Évalue l'arbre en remplaçant la i-ème constante par vals[i]."""
    tag = tree[0]
    if tag == "x":
        return x
    if tag == "const":
        v = vals[compteur[0]]
        compteur[0] += 1
        return np.full_like(x, v)
    if tag in UNARY_OPS:
        a = _eval_params(tree[1], x, vals, compteur)
        if tag == "sin":
            return np.sin(a)
        if tag == "cos":
            return np.cos(a)
        if tag == "exp":
            return np.exp(np.clip(a, -20, 20))
        if tag == "log":
            return np.log(np.maximum(a, 1e-6))
        if tag == "sqrt":
            return np.sqrt(np.maximum(a, 0.0))
        if tag == "tanh":
            return np.tanh(a)
        return -a
    a = _eval_params(tree[1], x, vals, compteur)
    b = _eval_params(tree[2], x, vals, compteur)
    if tag == "add":
        return a + b
    if tag == "sub":
        return a - b
    if tag == "mul":
        return a * b
    return a / np.where(np.abs(b) < 1e-9, 1e-9, b)


def _departs_constantes(base, max_departs=3):
    """Points de départ pour l'optimisation : valeur prédite, puis variantes
    (le problème n'est pas convexe, un seul départ peut rester bloqué)."""
    base = np.clip(base, -999.0, 999.0)   # reste strictement dans les bornes
    departs = [base]
    if base.size and max_departs > 1:
        departs.append(np.clip(base * 1.25 + 0.05, -999.0, 999.0))
    if base.size >= 2 and max_departs > 2:
        departs.append(np.clip(base * 0.8 - 0.05, -999.0, 999.0))
    return departs[:max_departs]


def refine_constants_lm(tree, x, y, max_nfev=120, max_departs=3):
    """Affine les constantes par moindres carrés (Levenberg–Marquardt) :
    bien plus précis et bien plus rapide que 300 pas d'Adam. Repli sur la
    descente de gradient si scipy est absent."""
    if tree is None:
        return None
    base = _liste_constantes(tree)
    if not base:
        return tree
    try:
        from scipy.optimize import least_squares
    except ImportError:                                # pragma: no cover
        return refine_constants(tree, x, y)

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    echelle = max(float(np.std(y)), 1e-6)
    depart0 = np.asarray(base, dtype=np.float64)

    def residus(v):
        compteur = [0]
        with np.errstate(all="ignore"):
            yp = _eval_params(tree, x, v, compteur)
        r = yp - y
        return np.where(np.isfinite(r), r, 1e3 * echelle)

    meilleur, meilleure_err = None, reconstruction_error(tree, x, y)
    for depart in _departs_constantes(depart0, max_departs):
        try:
            res = least_squares(residus, depart, method="trf",
                                bounds=(-1e3, 1e3), max_nfev=max_nfev)
        except Exception:                              # noqa: BLE001
            continue
        if not np.all(np.isfinite(res.x)):
            continue
        compteur = [0]
        arbre = _remplacer_constantes(tree, res.x, compteur)
        err = reconstruction_error(arbre, x, y)
        if err < meilleure_err - 1e-12:
            meilleur, meilleure_err = arbre, err
    return meilleur if meilleur is not None else tree


def _graine_donnees(x, y, base=0):
    """Graine déterministe dérivée du nuage : un même jeu de points redonne
    toujours la même formule (indispensable pour comparer et pour l'interface)."""
    h = hashlib.blake2b(digest_size=8)
    h.update(np.ascontiguousarray(x, dtype=np.float64).tobytes())
    h.update(np.ascontiguousarray(y, dtype=np.float64).tobytes())
    h.update(int(base).to_bytes(8, "little", signed=True))
    return int.from_bytes(h.digest(), "little") % (2**63 - 1)


def _score_candidat_cv(tree, x, y, penalite=0.002):
    """Score avec validation croisée sur les points (robuste contre le sur-ajustement)."""
    if tree is None:
        return float("inf")
    n = len(x)
    if n < 6:
        return score_candidat(tree, x, y, penalite)
    idx = np.arange(n)
    err = 0.0
    for k in range(3):
        np.random.RandomState(k).shuffle(idx)
        m = n // 2
        tr_idx, te_idx = idx[:m], idx[m:]
        xt, yt, xe, ye = x[tr_idx], y[tr_idx], x[te_idx], y[te_idx]
        arbre = refine_constants_lm(tree, xt, yt, max_nfev=80, max_departs=1)
        if arbre is None:
            arbre = tree
        e = reconstruction_error(arbre, xe, ye)
        err += e
    base = score_candidat(tree, x, y, penalite)
    return 0.5 * base + 0.5 * (err / 3.0)


def predict_tree(model, sample, args, device):
    """Nuage -> (ids, values, arbre_prédit).

    On génère `--candidates` formules, on dédoublonne par forme canonique,
    on affiche les constantes par moindres carrés, puis on garde le meilleur
    score (RMSE relative + pénalité de complexité). Le tirage est déterministe
    pour un nuage donné (sauf `--aleatoire`)."""
    x, y = sample["x"], sample["y"]
    src = torch.from_numpy(sample_features(sample)[None]).to(device)

    penalite = getattr(args, "penalite", 0.002)
    if getattr(args, "aleatoire", False):
        rng, gen = None, None
    else:
        graine = _graine_donnees(x, y, getattr(args, "seed", 0))
        rng = np.random.default_rng(graine)
        try:
            gen = torch.Generator(device=src.device.type)
            gen.manual_seed(graine)
        except Exception:                              # noqa: BLE001
            gen = None

    model.eval()
    # 1. générer les candidats (échantillonnage + beam)
    n_cand = int(getattr(args, "candidates", 48))
    candidats_bruts = []
    vus = set()
    for ids, values in decode_sequences(model, src, args.max_len,
                                        n_cand, args.temperature,
                                        args.const_noise, rng, gen):
        key = tuple(ids)
        if key in vus:
            continue
        vus.add(key)
        tree = parse_ids(ids, values)
        if tree is None:
            continue
        cle = _cle_canonique(tree)
        if cle in vus:
            continue
        vus.add(cle)
        candidats_bruts.append(tree)

    # beam search si peu de diversité
    try:
        beam_w = int(getattr(args, "beam", 6))
        if beam_w > 0:
            for ids, values in decode_beam(model, src, args.max_len, beam_width=beam_w):
                key = tuple(ids)
                if key in vus:
                    continue
                vus.add(key)
                tree = parse_ids(ids, values)
                if tree is None:
                    continue
                cle = _cle_canonique(tree)
                if cle in vus:
                    continue
                vus.add(cle)
                candidats_bruts.append(tree)
    except Exception:
        pass

    # déduplication sémantique
    candidats_bruts = _score_semantique(candidats_bruts, x)
    # budget adaptatif
    try:
        echelle = max(float(np.std(y)), 1e-9)
        # estimation rapide sur le meilleur vu
        best_rapide = float("inf")
        for tree in candidats_bruts[:10]:
            best_rapide = min(best_rapide, reconstruction_error(tree, x, y) / echelle)
        if best_rapide > 1e-3 and n_cand < 96:
            for ids, values in decode_sequences(model, src, args.max_len,
                                                48, args.temperature,
                                                args.const_noise, rng, gen):
                key = tuple(ids)
                if key in vus:
                    continue
                vus.add(key)
                tree = parse_ids(ids, values)
                if tree is None:
                    continue
                cle = _cle_canonique(tree)
                if cle in vus:
                    continue
                vus.add(cle)
                candidats_bruts.append(tree)
            candidats_bruts = _score_semantique(candidats_bruts, x)
    except Exception:
        pass

    scored = []
    for tree in candidats_bruts:
        scored.append((score_candidat(tree, x, y, penalite), tree_to_tokens(tree)[0],
                       tree_to_tokens(tree)[1], tree))
    if not scored:
        return [], [], None

    # 2. affinage rapide de TOUS les candidats (LM ≈ milliseconde), puis tri
    affines = []
    for _, ids, values, tree in scored:
        arbre = refine_constants_lm(tree, x, y, max_nfev=80, max_departs=1)
        if arbre is not None:
            # arrondi subtil
            try:
                arbre = _arrondir_constantes(arbre)
            except Exception:
                pass
            ids, values = tree_to_tokens(arbre)
            tree = arbre
        affines.append((_score_candidat_cv(tree, x, y, penalite), ids, values, tree))
    affines.sort(key=lambda t: t[0])

    # 3. affinage fin des meilleurs, puis nouveau tri
    n_refine = max(0, getattr(args, "refine", 8))
    finalistes = []
    for score, ids, values, tree in affines[:1 + n_refine]:
        arbre = refine_constants_lm(tree, x, y, max_nfev=300)
        if arbre is not None:
            try:
                arbre = _arrondir_constantes(arbre)
            except Exception:
                pass
            ids, values = tree_to_tokens(arbre)
            tree = arbre
        finalistes.append((score_candidat(tree, x, y, penalite), ids, values, tree))
    finalistes.sort(key=lambda t: t[0])
    best = finalistes[0]
    return best[1], best[2], best[3]


def evaluate(model, args, device, seed, n_samples=100, n_examples=0,
             verbose=True, title="évaluation", samples=None):
    """Sur de nouvelles fonctions : exactitude structurelle + RMSE
    fonctionnel + exemples affichés.

    `samples` permet d'évaluer sur un jeu figé (benchmark) au lieu d'en
    générer un nouveau à chaque fois."""
    rng = random.Random(seed)
    n = struct_ok = exact_ok = found = 0
    rel_errors: list[float] = []
    by_depth: dict[int, list[int]] = {}
    examples = []

    for i in range(n_samples):
        if samples is not None:
            if i >= len(samples):
                break
            s = samples[i]
        else:
            try:
                s = generate_sample(rng, args.n_points, args.max_depth,
                                    0.0, args.x_range, args.max_len,
                                    canonique=getattr(args, "canonique", False))
            except RuntimeError:
                continue
        n += 1
        t_ids, t_vals = tree_to_tokens(s["tree"])
        p_ids, p_vals, p_tree = predict_tree(model, s, args, device)

        ok_struct = p_ids == t_ids
        ok_exact = ok_struct and all(
            abs(p - q) <= 0.05
            for p, q, tid in zip(p_vals, t_vals, t_ids) if tid == CONST_ID)

        rel = relative_rmse(s["tree"], p_tree, args.x_range)
        ok_found = rel < 0.05

        struct_ok += ok_struct
        exact_ok += ok_exact
        found += ok_found
        rel_errors.append(rel)
        d = s["depth"]
        slot = by_depth.setdefault(d, [0, 0])
        slot[0] += 1
        slot[1] += ok_found

        if len(examples) < n_examples:
            examples.append((tree_to_str(s["tree"]),
                             tree_to_str(p_tree) if p_tree else "<invalide>",
                             ok_exact, ok_found, rel))

    if not verbose or n == 0:
        return {"n": n, "struct": struct_ok / max(n, 1),
                "exact": exact_ok / max(n, 1),
                "found": found / max(n, 1),
                "by_depth": {d: {"n": c, "found": f / max(c, 1)}
                             for d, (c, f) in by_depth.items()},
                "median_rel_rmse": float(np.median(rel_errors)) if rel_errors else 1.0}

    print(f"\n── {title} : {n} fonctions nouvelles "
          f"(profondeur ≤ {args.max_depth}, sans bruit) ──")
    print(f"   structure retrouvée    : {100 * struct_ok / n:5.1f} %")
    print(f"   formule exacte (+cst.) : {100 * exact_ok / n:5.1f} %")
    print(f"   fonction retrouvée     : {100 * found / n:5.1f} %   "
          f"(RMSE relative < 5 %)")
    if rel_errors:
        print(f"   RMSE relative médian   : {np.median(rel_errors):.4f}")
    if by_depth:
        print("   par profondeur (fonction retrouvée) :")
        for d in sorted(by_depth):
            c, f = by_depth[d]
            print(f"      profondeur {d} : {100 * f / c:5.1f} %  (n={c})")

    for i, (truth, pred, ok_e, ok_f, rel) in enumerate(examples, 1):
        mark = "✔" if ok_e else ("~" if ok_f else "✘")
        print(f"\n   [{i}] {mark} RMSE rel. = {rel:.3f}")
        print(f"       vérité : {truth}")
        print(f"       prédit : {pred}")
    print()
    return {"n": n, "struct": struct_ok / n, "exact": exact_ok / n,
            "found": found / n,
            "by_depth": {d: {"n": c, "found": f / max(c, 1)}
                         for d, (c, f) in by_depth.items()},
            "median_rel_rmse": float(np.median(rel_errors)) if rel_errors else 1.0}


# ---- jeu de validation figé (benchmark) ----------------------------------

def creer_benchmark(path, n=300, seed=1234, max_depth=3, n_points=64,
                    x_range=4.0, max_len=64, canonique=False):
    """Génère une fois pour toutes un jeu de fonctions de référence.

    Le jeu est ensuite rejoué à l'identique : on peut donc comparer deux
    versions du modèle sans que la différence vienne du tirage."""
    rng = random.Random(seed)
    fonctions = []
    while len(fonctions) < n:
        try:
            s = generate_sample(rng, n_points, max_depth, 0.0, x_range, max_len,
                                canonique=canonique)
        except RuntimeError:
            continue
        ids, vals = tree_to_tokens(s["tree"])
        fonctions.append({
            "ids": ids,
            "vals": [round(float(v), 6) for v in vals],
            "depth": int(s["depth"]),
            "x": [round(float(v), 6) for v in s["x"]],
            "y": [round(float(v), 6) for v in s["y"]],
        })
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"version": 1, "max_depth": max_depth, "x_range": x_range,
                   "n_points": n_points, "canonique": bool(canonique),
                   "fonctions": fonctions}, fh)
    return fonctions


def charger_benchmark(path, canonique=False):
    """Relit le jeu figé et reconstruit les échantillons attendus par
    `evaluate`. `canonique` réécrit les fonctions de référence dans la même
    forme que les cibles d'entraînement."""
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    samples = []
    for f in data["fonctions"]:
        tree = parse_ids(f["ids"], f["vals"])
        if canonique:
            tree = canonical_tree(tree)
        x = np.asarray(f["x"], dtype=np.float64)
        y = np.asarray(f["y"], dtype=np.float64)
        samples.append({"x": x, "y": y, "tree": tree,
                        "depth": int(f["depth"])})
    return samples, data


def _wilson(k, n, z=1.96):
    """Intervalle de confiance de Wilson (proportions)."""
    if n == 0:
        return 0.0, 1.0
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    demi = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - demi), min(1.0, centre + demi)


def run_benchmark(model, args, device, samples, repetitions=3, verbose=True):
    """Rejoue le benchmark plusieurs fois (seule la graine d'inférence change)
    et agrège les résultats avec un intervalle de confiance."""
    resultats = []
    for r in range(max(1, repetitions)):
        a = argparse.Namespace(**vars(args))
        a.seed = int(getattr(args, "seed", 0)) + 1000 * r
        resultats.append(evaluate(model, a, device, a.seed,
                                  n_samples=len(samples), verbose=False,
                                  samples=samples))
    n = sum(st["n"] for st in resultats)
    trouves = sum(round(st["found"] * st["n"]) for st in resultats)
    exacts = sum(round(st["exact"] * st["n"]) for st in resultats)
    structs = sum(round(st["struct"] * st["n"]) for st in resultats)
    global_ = {"n": n, "found": trouves, "exact": exacts, "struct": structs}

    par_profondeur = {}
    for st in resultats:
        for d, info in st["by_depth"].items():
            slot = par_profondeur.setdefault(d, [0, 0])
            slot[0] += info["n"]
            slot[1] += round(info["found"] * info["n"])

    if verbose:
        print(f"\n── benchmark : {len(samples)} fonctions figées × "
              f"{max(1, repetitions)} tirages ──")
        for nom, cle in (("structure retrouvée", "struct"),
                         ("formule exacte (+cst.)", "exact"),
                         ("fonction retrouvée", "found")):
            lo, hi = _wilson(global_[cle], n)
            print(f"   {nom:24s}: {100 * global_[cle] / max(n, 1):5.1f} % "
                  f"[{100 * lo:.1f} ; {100 * hi:.1f}]")
        if par_profondeur:
            print("   par profondeur (fonction retrouvée) :")
            for d in sorted(par_profondeur):
                c, f = par_profondeur[d]
                lo, hi = _wilson(f, c)
                print(f"      profondeur {d} : {100 * f / c:5.1f} % "
                      f"[{100 * lo:.1f} ; {100 * hi:.1f}]  (n={c})")
        print()

    return {"n": n,
            "struct": global_["struct"] / max(n, 1),
            "exact": global_["exact"] / max(n, 1),
            "found": global_["found"] / max(n, 1),
            "by_depth": {d: {"n": c, "found": f / max(c, 1)}
                         for d, (c, f) in par_profondeur.items()}}


# =========================================================================
# 7. ENTRAÎNEMENT
# =========================================================================
def build_scheduler(opt, total, warmup):
    def fn(step):
        if step < warmup:
            return (step + 1) / max(warmup, 1)
        p = (step - warmup) / max(1, total - warmup)
        p = min(p, 1.0)
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * p))
    return torch.optim.lr_scheduler.LambdaLR(opt, fn)


def sample_depth(rng, step, total, max_depth, enable=True):
    """Curriculum : davantage de fonctions simples au début, de plus en plus
    de compositions profondes ensuite. Accélère nettement l'apprentissage des
    expressions composées."""
    if not enable or max_depth <= 1:
        return max_depth
    t = step / max(1, total)
    if t < 0.25:                 # profondeurs {1, 2}
        p_hard, p_tiny = 0.0, 0.45
    elif t < 0.55:               # profondeurs {2, 3}
        p_hard, p_tiny = 0.5, 0.0
    else:                        # surtout la profondeur max
        p_hard, p_tiny = 0.85, 0.0
    r = rng.random()
    if r < p_hard:
        return max_depth
    if r < p_hard + p_tiny:
        return max(1, max_depth - 2)
    return max_depth - 1


def _gen_batch_worker(args_seed, n_samples, n_points, max_depth, noise, x_range, max_len, canonique):
    rng = random.Random(args_seed)
    samples = []
    for _ in range(n_samples):
        try:
            samples.append(generate_sample(
                rng, n_points, max_depth, noise,
                x_range, max_len,
                canonique=canonique,
                n_points_var=False, domaine_var=False))
        except Exception:
            pass
    return samples


def train(model, args, device, start_step=0):
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=0.01, betas=(0.9, 0.98))
    sched = build_scheduler(opt, args.steps, args.warmup)
    for _ in range(start_step):
        sched.step()

    rng = random.Random(args.seed + start_step)
    best = {"found": -1.0}
    t0 = time.time()
    window = 0.0

    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    num_workers = int(getattr(args, "num_workers", 0) or 0)
    executor = None
    if num_workers > 0:
        try:
            ctx = mp.get_context("fork")
            executor = ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx)
            print(f"génération parallèle: {num_workers} workers")
        except Exception:
            executor = None

    print(f"entraînement : {args.steps - start_step} étapes, "
          f"lot={args.batch_size}, d={args.d_model}, "
          f"points={args.n_points}, profondeur≤{args.max_depth}, "
          f"curriculum={'non' if args.no_curriculum else 'oui'}, "
          f"appareil={device}, amp={'oui' if use_amp else 'non'}")

    for step in range(start_step + 1, args.steps + 1):
        samples = []
        if executor is not None and args.batch_size > 1:
            try:
                futures = []
                per = max(1, args.batch_size // num_workers)
                for i in range(num_workers):
                    s = per
                    if i == num_workers - 1:
                        s = args.batch_size - per * (num_workers - 1)
                    if s <= 0:
                        continue
                    futures.append(executor.submit(
                        _gen_batch_worker,
                        args.seed + start_step + step * 131542391 + i,
                        s, args.n_points, args.max_depth, args.noise,
                        args.x_range, args.max_len,
                        getattr(args, "canonique", False)))
                for f in futures:
                    samples.extend(f.result(timeout=30))
            except Exception:
                samples = []
        if len(samples) < args.batch_size:
            while len(samples) < args.batch_size:
                try:
                    samples.append(generate_sample(
                        rng, args.n_points, args.max_depth, args.noise,
                        args.x_range, args.max_len,
                        canonique=getattr(args, "canonique", False),
                        n_points_var=False, domaine_var=False))
                except RuntimeError:
                    continue
        batch = build_batch(samples, device, args.max_len)

        opt.zero_grad(set_to_none=True)
        if use_amp:
            with torch.cuda.amp.autocast():
                logits, const_pred = model(batch["src"], batch["tgt_in"])
                valid = target_valid_mask(batch["tgt_out"])
                masked = logits.masked_fill(~valid, -1e4)
                loss_sym = F.cross_entropy(
                    masked.reshape(-1, V), batch["tgt_out"].reshape(-1),
                    ignore_index=PAD_ID)
                cmask = (batch["tgt_out"] == CONST_ID).float()
                if getattr(args, "loss_const", "mse") == "huber":
                    ecarts = (const_pred - batch["const_tgt"]).abs()
                    perte = torch.where(ecarts <= 0.5, 0.5 * ecarts ** 2,
                                        0.5 * (ecarts - 0.25))
                else:
                    perte = (const_pred - batch["const_tgt"]) ** 2
                loss_c = (perte * cmask).sum() / cmask.sum().clamp(min=1.0)
                loss = loss_sym + args.w_const * loss_c
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
        else:
            logits, const_pred = model(batch["src"], batch["tgt_in"])
            valid = target_valid_mask(batch["tgt_out"])
            masked = logits.masked_fill(~valid, -1e4)
            loss_sym = F.cross_entropy(
                masked.reshape(-1, V), batch["tgt_out"].reshape(-1),
                ignore_index=PAD_ID)
            cmask = (batch["tgt_out"] == CONST_ID).float()
            if getattr(args, "loss_const", "mse") == "huber":
                ecarts = (const_pred - batch["const_tgt"]).abs()
                perte = torch.where(ecarts <= 0.5, 0.5 * ecarts ** 2,
                                    0.5 * (ecarts - 0.25))
            else:
                perte = (const_pred - batch["const_tgt"]) ** 2
            loss_c = (perte * cmask).sum() / cmask.sum().clamp(min=1.0)
            loss = loss_sym + args.w_const * loss_c
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        window += 1

        if step % args.log_every == 0:
            dt = max(time.time() - t0, 1e-9)
            print(f"étape {step:5d}/{args.steps} | "
                  f"loss {loss.item():6.3f} (sym {loss_sym.item():6.3f}, "
                  f"cst {loss_c.item():6.3f}) | "
                  f"{window / dt:5.1f} étapes/s | "
                  f"lr {opt.param_groups[0]['lr']:.2e}")
            t0 = time.time()
            window = 0.0

        if step % args.eval_every == 0 or step == args.steps:
            stats = evaluate(model, args, device, args.seed + 10_000,
                             n_samples=args.eval_samples,
                             n_examples=0, verbose=True,
                             title=f"contrôle à l'étape {step}")
            if stats["found"] >= best["found"]:
                best = dict(stats)
                best["etape"] = step
                save_checkpoint(model, args, step, best)
                print(f"   → checkpoint enregistré ({args.ckpt})")
            model.train()

    return best


def save_checkpoint(model, args, step, best=None):
    torch.save({
        "model": model.state_dict(),
        "hparams": model.hparams,
        "step": step,
        "best": best,
        "vocab": VOCAB,
    }, args.ckpt)


def load_or_build(args, device):
    """Charge le checkpoint si présent/demandé, sinon crée un modèle neuf."""
    if os.path.exists(args.ckpt) and (args.resume or args.eval_only):
        ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
        hp = dict(ckpt["hparams"])
        hp.setdefault("n_points", args.n_points)
        hp.setdefault("max_len", args.max_len)
        model = SymbolicRegressor(**hp).to(device)
        sd = ckpt["model"]
        # compat ancien format (enc_pos/dec_pos)
        sd = {k: v for k, v in sd.items() if not k.endswith("enc_pos.weight")
              and not k.endswith("dec_pos.weight")}
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing or unexpected:
            print(f"checkpoint chargé avec compatibilité : manquants={len(missing)}, inattendus={len(unexpected)}")
        try:
            if os.environ.get("RS_COMPILE", "1") == "1":
                model = torch.compile(model, mode="reduce-overhead" if device.type == "cpu" else "default")
        except Exception:
            pass
        print(f"checkpoint chargé : {args.ckpt} (étape {ckpt.get('step', '?')})")
        return model, int(ckpt.get("step", 0))
    hp = {"d_model": args.d_model,
          "nhead": args.n_heads,
          "enc_layers": args.enc_layers,
          "dec_layers": args.dec_layers,
          "ff": args.ff,
          "dropout": args.dropout,
          "n_points": args.n_points,
          "max_len": args.max_len}
    model = SymbolicRegressor(**hp).to(device)
    try:
        if os.environ.get("RS_COMPILE", "1") == "1" and not args.eval_only:
            model = torch.compile(model, mode="reduce-overhead" if device.type == "cpu" else "default")
    except Exception:
        pass
    return model, 0


# =========================================================================
# 8. PRÉDICTION SUR UN NUAGE UTILISATEUR
# =========================================================================

def load_points_file(path):
    xs, ys = [], []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            for sep in (";", ",", "\t", " "):
                if sep in line:
                    parts = line.replace(";", ",").replace("\t", ",").split(",")
                    break
            else:
                continue
            if len(parts) < 2:
                continue
            try:
                xs.append(float(parts[0]))
                ys.append(float(parts[1]))
            except ValueError:
                continue
    return np.array(xs), np.array(ys)


def parse_inline_points(text):
    """Format attendu : "x1,y1;x2,y2;..." (points séparés par ';')."""
    xs, ys = [], []
    for pair in text.split(";"):
        pair = pair.strip().strip("()")
        a, b = pair.split(",")[0], pair.split(",")[1]
        xs.append(float(a))
        ys.append(float(b))
    return np.array(xs), np.array(ys)


def predict_from_points(model, xs, ys, args, device, source=""):
    if xs.size < 2:
        sys.exit("il faut au moins 2 points.")
    order = np.argsort(xs)
    xs, ys = xs[order], ys[order]
    if not (-args.x_range <= xs.min() and xs.max() <= args.x_range):
        print(f"⚠ x hors de la plage d'entraînement [-{args.x_range}, "
              f"{args.x_range}] : extrapolation incertaine.")

    # le modèle attend exactement n_points abscisses → on rééchantillonne
    xg = np.linspace(xs.min(), xs.max(), args.n_points)
    yg = np.interp(xg, xs, ys)
    sample = {"x": xg, "y": yg, "tree": None, "depth": 0}

    ids, values, tree = predict_tree(model, sample, args, device)
    print(f"\nnuage {source} : {xs.size} points "
          f"x ∈ [{xs.min():.3g}, {xs.max():.3g}]")
    if tree is None:
        print("aucune formule valide trouvée "
              f"(séquence : {' '.join(ID2TOK[i] for i in ids)})")
        return
    brut = tree_to_str(tree)
    if not getattr(args, "no_simplify", False):
        simplifie = simplify_formula(tree, xs, ys)
        if simplifie is not None and simplifie is not tree:
            tree = simplifie
            print(f"(forme brute du réseau : {brut})")
    print(f"formule prédite : y = {tree_to_str(tree)}")
    yp = eval_tree(tree, xs)
    if np.all(np.isfinite(yp)):
        err = np.sqrt(float(np.mean((yp - ys) ** 2)))
        print(f"RMSE sur vos points : {err:.4g}  "
              f"(max |Δ| = {np.max(np.abs(yp - ys)):.4g})")
    else:
        print("la formule prédite n'est pas finie sur vos points.")


# =========================================================================
# 9. CLI
# =========================================================================

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Régression symbolique : un Transformer qui retrouve "
                    "la formule d'une fonction à partir d'un nuage de points.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # modèle (valeurs calibrées pour un CPU avec 4 à 8 cœurs)
    p.add_argument("--d-model", type=int, default=192, help="dimension du modèle")
    p.add_argument("--n-heads", type=int, default=6, help="têtes d'attention")
    p.add_argument("--enc-layers", type=int, default=3, help="couches encodeur")
    p.add_argument("--dec-layers", type=int, default=3, help="couches décodeur")
    p.add_argument("--ff", type=int, default=768, help="dimension interne FFN")
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--n-points", type=int, default=64,
                   help="points du nuage (entrée du modèle)")
    p.add_argument("--max-len", type=int, default=64,
                   help="longueur max de la formule en jetons")
    # inférence
    p.add_argument("--candidates", type=int, default=8,
                   help="formules générées par nuage, on garde la mieux "
                        "ajustée (1 = décodage glouton simple)")
    p.add_argument("--temperature", type=float, default=1.0,
                   help="température d'échantillonnage des candidats")
    p.add_argument("--const-noise", type=float, default=0.15,
                   help="bruit ajouté aux constantes des candidats échantillonnés "
                        "(0 = constantes strictement déterministes)")
    p.add_argument("--refine", type=int, default=8,
                   help="nombre de candidats dont les constantes sont "
                        "affinées finement (moindres carrés)")
    p.add_argument("--refine-steps", type=int, default=300,
                   help="itérations d'affinage par descente de gradient "
                        "(repli si scipy est absent)")
    p.add_argument("--penalite", type=float, default=0.002,
                   help="pénalité de complexité par nœud dans la sélection "
                        "(0 = classement au RMSE brut)")
    p.add_argument("--aleatoire", action="store_true",
                   help="tirage non reproductible (par défaut, un même nuage "
                        "redonne toujours la même formule)")
    p.add_argument("--no-simplify", action="store_true",
                   help="ne pas simplifier symboliquement la formule "
                        "retrouvée (par défaut : simplification via SymPy)")
    # données
    p.add_argument("--max-depth", type=int, default=3, choices=[1, 2, 3, 4, 5],
                   help="profondeur max des formules générées")
    p.add_argument("--x-range", type=float, default=4.0,
                   help="abscisses dans [-R, R]")
    p.add_argument("--noise", type=float, default=0.0,
                   help="bruit gaussien ajouté sur y (fraction de l'écart-type)")
    # entraînement
    p.add_argument("--steps", type=int, default=6000, help="nombre d'étapes")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--w-const", type=float, default=1.0,
                   help="poids de la perte sur les constantes")
    p.add_argument("--loss-const", choices=["mse", "huber"], default="mse",
                   help="perte sur les constantes : mse (historique) ou huber "
                        "(moins sensible aux grandes constantes)")
    p.add_argument("--canonique", action="store_true",
                   help="cibles d'entraînement sous forme canonique "
                        "(une seule écriture par fonction, moins d'entropie)")
    p.add_argument("--no-curriculum", action="store_true",
                   help="désactiver le mélange de difficultés "
                        "(fonctions simples d'abord, profondes ensuite)")
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--eval-samples", type=int, default=100)
    p.add_argument("--examples", type=int, default=6,
                   help="exemples affichés à l'évaluation finale")
    # benchmark (jeu de fonctions figé, pour comparer deux versions)
    p.add_argument("--creer-benchmark", nargs="?", const="benchmark_valide.json",
                   default=None, metavar="fichier.json",
                   help="génère un jeu de validation figé puis quitte")
    p.add_argument("--benchmark", default=None, metavar="fichier.json",
                   help="évalue sur un jeu figé (au lieu d'un tirage aléatoire)")
    p.add_argument("--benchmark-repetitions", type=int, default=3,
                   help="nombre de tirages d'inférence par fonction du benchmark")
    # divers
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ckpt", default="modele_regression.pt",
                   help="fichier de checkpoint")
    p.add_argument("--resume", action="store_true", help="reprendre le checkpoint")
    p.add_argument("--eval-only", action="store_true",
                   help="ne pas entraîner, juste évaluer")
    p.add_argument("--device", default="auto",
                   help="auto | cpu | cuda | cuda:N")
    p.add_argument("--num-workers", type=int, default=0,
                   help="workers multiprocessing pour générer les lots (0 = synchrone)")
    p.add_argument("--predict", default=None, metavar='"x1,y1;x2,y2;..."',
                   help="prédire une formule à partir de points fournis")
    p.add_argument("--predict-file", default=None, metavar="fichier.csv",
                   help="idem, points lus dans un fichier x,y")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # ---- création d'un jeu de validation figé ----
    if args.creer_benchmark:
        fonctions = creer_benchmark(
            args.creer_benchmark, n=300, seed=args.seed,
            max_depth=args.max_depth, n_points=args.n_points,
            x_range=args.x_range, max_len=args.max_len,
            canonique=args.canonique)
        print(f"benchmark créé : {args.creer_benchmark} "
              f"({len(fonctions)} fonctions, profondeur ≤ {args.max_depth})")
        return

    # ---- mode prédiction ----
    if args.predict or args.predict_file:
        if not os.path.exists(args.ckpt):
            sys.exit(f"checkpoint introuvable : {args.ckpt} "
                     f"(entraînez d'abord le modèle).")
        args.eval_only = True
        model, _ = load_or_build(args, device)
        if args.predict:
            xs, ys = parse_inline_points(args.predict)
            predict_from_points(model, xs, ys, args, device, "(inline)")
        else:
            xs, ys = load_points_file(args.predict_file)
            predict_from_points(model, xs, ys, args, device, args.predict_file)
        return

    # ---- entraînement / évaluation ----
    model, start_step = load_or_build(args, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"modèle : {n_params/1e6:.2f} M paramètres, vocabulaire "
          f"{V} jetons, appareil = {device}")

    # ---- évaluation sur jeu figé ----
    if args.benchmark:
        samples, meta = charger_benchmark(args.benchmark,
                                          canonique=args.canonique)
        print(f"benchmark : {len(samples)} fonctions figées "
              f"(profondeur ≤ {meta.get('max_depth', '?')})")
        run_benchmark(model, args, device, samples,
                      repetitions=args.benchmark_repetitions)
        return

    if not args.eval_only:
        if start_step >= args.steps:
            print(f"le checkpoint est déjà à l'étape {start_step} ≥ "
                  f"--steps {args.steps} : rien à faire "
                  f"(augmentez --steps ou ajoutez --eval-only).")
        else:
            best = train(model, args, device, start_step)
            if best.get("etape"):
                print(f"meilleur checkpoint conservé (étape {best['etape']}) "
                      f"dans {args.ckpt}")
            else:                          # aucune évaluation : on sauvegarde
                save_checkpoint(model, args, args.steps)
                print(f"modèle enregistré dans {args.ckpt}")

    evaluate(model, args, device, args.seed + 10_000,
             n_samples=max(args.eval_samples, 100),
             n_examples=args.examples, verbose=True,
             title="évaluation finale")


if __name__ == "__main__":
    main()
