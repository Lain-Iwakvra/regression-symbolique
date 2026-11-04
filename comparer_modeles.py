#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Comparaison de plusieurs modèles de régression symbolique."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from typing import List

import numpy as np
import torch

import regression_symbolique as rs


class Options:
    def __init__(self, hparams, candidats=48, affinage=8):
        self.n_points = int(hparams.get("n_points", 64))
        self.max_len = int(hparams.get("max_len", 64))
        self.candidates = int(candidats)
        self.temperature = 1.0
        self.const_noise = 0.15
        self.refine = int(affinage)
        self.refine_steps = 300
        self.penalite = 0.002
        self.seed = 0
        self.x_range = 4.0


def load_model(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    hp = dict(ckpt["hparams"])
    hp.setdefault("n_points", 64)
    hp.setdefault("max_len", 64)
    model = rs.SymbolicRegressor(**hp).to(device)
    sd = ckpt["model"]
    sd = {k: v for k, v in sd.items() if not k.endswith("enc_pos.weight") and not k.endswith("dec_pos.weight")}
    model.load_state_dict(sd, strict=False)
    model.eval()
    return model, hp, ckpt


def generate_fixed_samples(n_samples: int, seed: int, max_depth: int = 4):
    rng = random.Random(seed)
    samples = []
    for i in range(n_samples):
        try:
            s = rs.generate_sample(rng, 64, max_depth, 0.0, 4.0, 64, canonique=True, n_points_var=False, domaine_var=False)
            samples.append(s)
        except Exception:
            continue
    return samples


def eval_on_samples(model, samples, opt, device):
    n = 0
    found = 0
    exact = 0
    total_rmse_rel = 0.0
    depths = {}
    total = len(samples)
    t0 = time.time()
    for idx, s in enumerate(samples):
        tree_true = s.get("tree")
        if tree_true is None:
            continue
        depth = rs.tree_depth(tree_true)
        depths.setdefault(depth, [0, 0])
        depths[depth][0] += 1
        n += 1
        try:
            ids, vals, tree_pred = rs.predict_tree(model, s, opt, device)
            if tree_pred is None:
                # update bar
                pct = (idx+1)/total
                bar = "█"*int(40*pct) + "░"*(40-int(40*pct))
                sys.stdout.write(f"\r    [{bar}] {idx+1}/{total}")
                sys.stdout.flush()
                continue
            try:
                tree_pred = rs.simplify_formula(tree_pred, s["x"], s["y"])
            except Exception:
                pass
            yp = rs.eval_tree(tree_pred, s["x"])
            yt = rs.eval_tree(tree_true, s["x"])
            fini = np.isfinite(yp) & np.isfinite(yt)
            if not fini.any():
                pct = (idx+1)/total
                bar = "█"*int(40*pct) + "░"*(40-int(40*pct))
                sys.stdout.write(f"\r    [{bar}] {idx+1}/{total}")
                sys.stdout.flush()
                continue
            rmse = np.sqrt(np.mean((yp[fini] - yt[fini]) ** 2))
            echelle = np.std(yt[fini]) if np.std(yt[fini]) > 1e-12 else max(np.max(np.abs(yt[fini])), 1e-12)
            rel = rmse / echelle
            if rel < 0.05:
                found += 1
                depths[depth][1] += 1
            try:
                if rs.trees_equal(tree_pred, tree_true):
                    exact += 1
            except Exception:
                pass
            total_rmse_rel += rel
        except Exception:
            pass
        pct = (idx+1)/total
        bar = "█"*int(40*pct) + "░"*(40-int(40*pct))
        sys.stdout.write(f"\r    [{bar}] {idx+1}/{total}")
        sys.stdout.flush()
    sys.stdout.write("\n")
    sys.stdout.flush()
    return {
        "n": n,
        "found": found / max(n, 1),
        "exact": exact / max(n, 1),
        "mean_rel": total_rmse_rel / max(n, 1),
        "depths": depths
    }


def compare_models(ckpts: List[str], args):
    device = torch.device("cuda" if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("CPU")
    print()

    print(f"Génération de {args.samples} fonctions identiques pour tous les modèles...")
    samples = generate_fixed_samples(args.samples, args.seed, max_depth=args.max_depth)
    print(f"  {len(samples)} fonctions générées avec seed={args.seed}")
    print()

    results = []

    for idx, ckpt_path in enumerate(ckpts):
        if not os.path.exists(ckpt_path):
            print(f"⚠ {ckpt_path} introuvable, ignoré")
            continue

        print(f"[{idx+1}/{len(ckpts)}] {ckpt_path}")
        model, hp, ckpt = load_model(ckpt_path, device)
        step = ckpt.get("step", "?")
        params = sum(p.numel() for p in model.parameters()) / 1e6

        opt = Options(hp, candidats=args.candidates, affinage=args.refine)
        opt.temperature = args.temperature
        opt.penalite = args.penalite
        opt.seed = args.seed
        opt.x_range = 4.0

        print(f"  d_model={hp['d_model']} | enc{hp['enc_layers']}/dec{hp['dec_layers']} | ff={hp.get('ff')} | heads={hp.get('nhead')} | {params:.1f}M | étape {step}")
        print(f"  Test en cours sur {len(samples)} fonctions identiques...")

        t0 = time.time()
        stats = eval_on_samples(model, samples, opt, device)
        dt = time.time() - t0

        print(f"  fonction retrouvée : {100*stats['found']:.1f}%   exactement identique : {100*stats['exact']:.1f}%   temps : {dt:.1f}s")
        print()

        results.append({
            "fichier": os.path.basename(ckpt_path),
            "chemin": ckpt_path,
            "d_model": hp["d_model"],
            "couches": f"{hp['enc_layers']}/{hp['dec_layers']}",
            "ff": hp.get("ff"),
            "tetes": hp.get("nhead"),
            "params_M": round(params, 1),
            "etape": step,
            "trouve": round(100*stats['found'], 1),
            "exact": round(100*stats['exact'], 1),
            "temps_s": round(dt, 1)
        })

    return results


def main():
    p = argparse.ArgumentParser(description="Comparer plusieurs modèles sur des fonctions identiques")
    p.add_argument("checkpoints", nargs="+", help="Fichiers .pt à comparer")
    p.add_argument("--samples", type=int, default=200, help="Nombre de fonctions à tester")
    p.add_argument("--max-depth", type=int, default=4)
    p.add_argument("--candidates", type=int, default=48)
    p.add_argument("--refine", type=int, default=8)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--penalite", type=float, default=0.002)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    p.add_argument("--output", default=None)
    args = p.parse_args()

    print("=" * 80)
    print("COMPARAISON SUR FONCTIONS IDENTIQUES")
    print("=" * 80)
    print(f"Modèles: {len(args.checkpoints)} | Fonctions: {args.samples} | Seed: {args.seed}")
    print(f"candidates={args.candidates}, refine={args.refine}, max_depth={args.max_depth}")
    print("=" * 80)
    print()

    results = compare_models(args.checkpoints, args)

    print("\n" + "=" * 100)
    print("TABLEAU COMPARATIF")
    print("=" * 100)
    print(f"{'Modèle':<32} {'d_model':<9} {'Couches':<10} {'ff':<7} {'Têtes':<7} {'Params(M)':<11} {'Étape':<9} {'Trouvé%':<10} {'Exact%':<10} {'Temps(s)':<10}")
    print("-" * 100)
    for r in results:
        print(f"{r['fichier']:<32} {r['d_model']:<9} {r['couches']:<10} {r['ff']:<7} {r['tetes']:<7} {r['params_M']:<11} {r['etape']:<9} {r['trouve']:<10} {r['exact']:<10} {r['temps_s']:<10}")
    print("=" * 100)

    if args.output:
        if args.output.endswith(".json"):
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump({"seed": args.seed, "samples": args.samples, "results": results}, f, indent=2, ensure_ascii=False)
            print(f"\nSauvegardé: {args.output}")
        elif args.output.endswith(".csv"):
            import csv
            with open(args.output, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=results[0].keys())
                w.writeheader()
                w.writerows(results)
            print(f"\nSauvegardé: {args.output}")


if __name__ == "__main__":
    main()
