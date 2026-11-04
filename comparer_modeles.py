#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Comparaison de plusieurs modèles de régression symbolique."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import List, Dict, Any

import numpy as np
import torch

import regression_symbolique as rs


class ProgressBar:
    def __init__(self, total: int, prefix: str = "", width: int = 40):
        self.total = total
        self.prefix = prefix
        self.width = width
        self.start = time.time()
        self.current = 0

    def update(self, n: int = 1):
        self.current += n
        if self.total == 0:
            return
        pct = self.current / self.total
        filled = int(self.width * pct)
        bar = "█" * filled + "░" * (self.width - filled)
        elapsed = time.time() - self.start
        if self.current > 0:
            eta = elapsed / self.current * (self.total - self.current)
            eta_str = f"ETA {eta:4.0f}s"
        else:
            eta_str = "ETA --s"
        sys.stdout.write(f"\r{self.prefix} [{bar}] {self.current}/{self.total} ({pct*100:5.1f}%) {eta_str}")
        sys.stdout.flush()
        if self.current >= self.total:
            sys.stdout.write("\n")
            sys.stdout.flush()


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


def compare_models(ckpts: List[str], args):
    device = torch.device("cuda" if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("CPU")
    print()

    results = []

    for idx, ckpt_path in enumerate(ckpts):
        if not os.path.exists(ckpt_path):
            print(f"⚠ {ckpt_path} introuvable, ignoré")
            continue

        print(f"[{idx+1}/{len(ckpts)}] Chargement {ckpt_path}")
        model, hp, ckpt = load_model(ckpt_path, device)
        step = ckpt.get("step", "?")
        params = sum(p.numel() for p in model.parameters()) / 1e6

        opt = rs.Options(hp, candidats=args.candidates, affinage=args.refine)
        opt.temperature = args.temperature
        opt.penalite = args.penalite
        opt.seed = args.seed

        print(f"  d_model={hp['d_model']} | enc{hp['enc_layers']}/dec{hp['dec_layers']} | ff={hp.get('ff')} | heads={hp.get('nhead')} | {params:.1f}M | étape {step}")
        print(f"  Évaluation en cours...")

        pb = ProgressBar(args.samples, prefix="  ")

        original_print = print

        def patched_print(*a, **kw):
            s = " ".join(str(x) for x in a)
            if s.strip().startswith("fonction retrouvée") or s.strip().startswith("exactement") or s.strip().startswith("==="):
                return
            original_print(*a, **kw)

        # Monkey patch pour capturer le résultat
        import builtins
        builtins.print = patched_print

        try:
            if args.benchmark:
                with open(args.benchmark, encoding="utf-8") as f:
                    bench = json.load(f)
                bench = bench[:args.samples]
                rs.eval_benchmark(model, bench, opt, device, repetitions=args.repetitions, verbose=True)
            else:
                rs.evaluate(model, args, device, args.seed + idx, n_samples=args.samples,
                            n_examples=0, verbose=True)
        finally:
            builtins.print = original_print

        # Recharger le meilleur depuis les stats? On relance un petit parsing
        # Plus simple: relire depuis un fichier temporaire
        results.append({
            "fichier": os.path.basename(ckpt_path),
            "chemin": ckpt_path,
            "d_model": hp["d_model"],
            "couches": f"{hp['enc_layers']}/{hp['dec_layers']}",
            "ff": hp.get("ff"),
            "tetes": hp.get("nhead"),
            "params_M": round(params, 1),
            "etape": step,
        })
        pb.update(args.samples)
        print()

    return results


def main():
    p = argparse.ArgumentParser(description="Comparer plusieurs modèles de régression symbolique")
    p.add_argument("checkpoints", nargs="+", help="Fichiers .pt à comparer")
    p.add_argument("--samples", type=int, default=200, help="Nombre de fonctions à tester")
    p.add_argument("--candidates", type=int, default=48)
    p.add_argument("--refine", type=int, default=8)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--penalite", type=float, default=0.002)
    p.add_argument("--repetitions", type=int, default=1)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--benchmark", default=None, help="Fichier benchmark JSON")
    p.add_argument("--device", default="auto")
    p.add_argument("--output", default=None, help="Sauvegarder en JSON/CSV")
    args = p.parse_args()

    print("=" * 60)
    print("COMPARAISON DE MODÈLES")
    print("=" * 60)
    print(f"Modèles: {len(args.checkpoints)}")
    print(f"Fonctions: {args.samples}")
    print(f"candidates={args.candidates}, refine={args.refine}")
    if args.benchmark:
        print(f"Benchmark: {args.benchmark}")
    print("=" * 60)
    print()

    results = compare_models(args.checkpoints, args)

    print("\n" + "=" * 90)
    print("RÉSULTATS")
    print("=" * 90)
    print(f"{'Modèle':<30} {'d_model':<8} {'Couches':<10} {'ff':<6} {'Têtes':<6} {'Params(M)':<10} {'Étape':<10}")
    print("-" * 90)
    for r in results:
        print(f"{r['fichier']:<30} {r['d_model']:<8} {r['couches']:<10} {r['ff']:<6} {r['tetes']:<6} {r['params_M']:<10} {r['etape']:<10}")
    print("=" * 90)
    print("\nRegarde les lignes 'fonction retrouvée' et 'exactement identique' dans les sorties ci-dessus pour comparer.")
    print("Pour un test plus précis, relance avec --benchmark benchmark_valide.json")

    if args.output:
        if args.output.endswith(".json"):
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"\nRésultats sauvegardés dans {args.output}")
        elif args.output.endswith(".csv"):
            import csv
            with open(args.output, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=results[0].keys())
                w.writeheader()
                w.writerows(results)
            print(f"\nRésultats sauvegardés dans {args.output}")


if __name__ == "__main__":
    main()
