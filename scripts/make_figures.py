#!/usr/bin/env python3
"""Genera todas las figuras para el paper ACL de Ontogenia Artificial."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).parent.parent
RESULTS = ROOT / "results"
DATA = ROOT / "data"
FIGURES = ROOT / "figures"
FIGURES.mkdir(exist_ok=True)

# ── Colores consistentes por modelo ──────────────────────────────────────────
MODEL_COLORS = {"14m": "#4C72B0", "160m": "#DD8452", "410m": "#55A868"}
MODEL_SIZES = ["14m", "160m", "410m"]

# ── Umbral para H1: caída real (no ruido polinomial) ─────────────────────────
DEPTH_THRESHOLD = 0.05   # ≥ 5 pp de profundidad
PEAK_BEFORE_END = True   # el máximo recuperado ocurre después del mínimo


def compute_h1_stats(topo: pd.DataFrame) -> dict:
    """
    Calcula % de paradigmas no monótonos con umbral de profundidad.

    Criterio: shape == 'u_shape' OR (shape in ['oscillatory','inverted_u']
    AND depth >= 0.05 AND min_step < max_step) → valle antes del pico.
    """
    rows = []
    for model_size in MODEL_SIZES:
        sub = topo[topo["model_size"] == model_size].copy()
        sub = sub[sub["task"] != "blimp"]

        is_u = (
            (sub["shape"] == "u_shape") |
            (
                (sub["shape"].isin(["oscillatory", "inverted_u"])) &
                (sub["depth"] >= DEPTH_THRESHOLD) &
                (sub["min_step"] < sub["max_step"])
            )
        )
        n_total = len(sub)
        n_nonmono = int(is_u.sum())
        rows.append({
            "model_size": model_size,
            "n_total": n_total,
            "n_nonmono": n_nonmono,
            "pct_nonmono": 100 * n_nonmono / n_total if n_total else 0,
        })
    return {r["model_size"]: r for r in rows}


if __name__ == "__main__":
    metrics = pd.read_parquet(RESULTS / "aggregated_metrics.parquet")
    topo = pd.read_parquet(RESULTS / "topology_summary.parquet")
    milestones = pd.read_csv(DATA / "human_milestones.csv")

    h1 = compute_h1_stats(topo)
    print("=== H1 ===")
    for sz, r in h1.items():
        print(f"  {sz}: {r['n_nonmono']}/{r['n_total']} = {r['pct_nonmono']:.1f}% no-monotone (depth≥5pp)")
