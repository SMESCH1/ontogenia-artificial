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


def compute_h1_stats(topo: pd.DataFrame) -> dict:
    """
    Calcula % de paradigmas no monótonos con umbral de profundidad.

    Criterio: shape == 'u_shape' OR (shape in ['oscillatory','inverted_u']
    AND depth >= 0.05 AND min_step < max_step) → valle antes del pico.
    """
    rows = []
    for model_size in MODEL_SIZES:
        sub = topo[topo["model_size"] == model_size].copy()
        # "blimp" es la fila agregada del grupo de tareas; excluir para operar sobre los 67 paradigmas
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


# Paradigmas elegidos para Fig 1 (6 paneles)
REPRESENTATIVE_TASKS = [
    ("blimp_determiner_noun_agreement_1",      "Det-N agreement\n(morfológico regular)"),
    ("blimp_irregular_past_participle_verbs",   "Irregular past participle\n(morfológico irregular ↑ U-curve)"),
    ("blimp_anaphor_number_agreement",          "Anaphor number\n(anáforas)"),
    ("blimp_passive_1",                         "Pasiva\n(estructural)"),
    ("blimp_principle_A_c_command",             "Principio A c-command\n(ligamiento)"),
    ("blimp_wh_questions_object_gap",           "Wh-questions object gap\n(extracción)"),
]


def plot_trajectories(metrics: pd.DataFrame) -> None:
    """Fig 1: grid 2×3 de trayectorias por paradigma."""
    blimp = metrics[
        metrics["task"].astype(str).str.startswith("blimp_") &
        metrics["training_step"].notna() &
        metrics["acc,none"].notna()
    ].copy()

    fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharey=False)
    axes = axes.flatten()

    for ax, (task, label) in zip(axes, REPRESENTATIVE_TASKS):
        for model_size in MODEL_SIZES:
            sub = blimp[
                (blimp["task"] == task) & (blimp["model_size"] == model_size)
            ].sort_values("training_step")
            if sub.empty:
                continue
            x = sub["training_step"].values + 1   # +1 para log-scale (step 0 → 1)
            y = sub["acc,none"].values
            ax.plot(x, y, "o-", color=MODEL_COLORS[model_size],
                    label=model_size, linewidth=1.5, markersize=3, alpha=0.85)

        ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.set_xscale("log")
        ax.set_xlabel("Training steps (log)", fontsize=8)
        ax.set_ylabel("Accuracy", fontsize=8)
        ax.set_title(label, fontsize=8, pad=4)
        ax.set_ylim(0.4, 1.02)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3, linestyle=":")

    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color=c, marker="o", label=sz)
               for sz, c in MODEL_COLORS.items()]
    axes[0].legend(handles=handles, title="Modelo", fontsize=7, title_fontsize=7)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.suptitle(
        "Trayectorias de aprendizaje BLiMP — paradigmas representativos",
        fontsize=10,
    )
    out_png = FIGURES / "fig1_trajectories.png"
    out_pdf = FIGURES / "fig1_trajectories.pdf"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_png}")


CONFIDENCE_COLORS = {"alta": "#2196F3", "media": "#FF9800", "baja": "#9E9E9E"}
CONFIDENCE_LABELS = {"alta": "Alta", "media": "Media", "baja": "Baja"}


def plot_human_alignment_scatter(
    metrics: pd.DataFrame,
    milestones: pd.DataFrame,
    model_size: str = "160m",
) -> None:
    """Fig 2: scatter step_stabilize vs AoA humano (Spearman)."""
    from ontogenia.human_alignment import compute_stabilization_table

    stab = compute_stabilization_table(
        metrics, target_model_size=model_size, metric_col="acc,none"
    )
    stab = stab[stab["task"] != "blimp"]

    merged = stab.merge(milestones[["task", "aoa_months", "confidence"]], on="task", how="inner")
    merged = merged.dropna(subset=["training_step_stabilize", "aoa_months"])

    x = merged["aoa_months"].values
    y = merged["training_step_stabilize"].values + 1  # +1 para log

    rho, pval = spearmanr(x, y)

    fig, ax = plt.subplots(figsize=(7, 5))

    for conf in ["alta", "media", "baja"]:
        mask = merged["confidence"] == conf
        ax.scatter(
            merged.loc[mask, "aoa_months"],
            merged.loc[mask, "training_step_stabilize"] + 1,
            c=CONFIDENCE_COLORS[conf],
            label=f"Confianza {CONFIDENCE_LABELS[conf]} (n={mask.sum()})",
            s=40, alpha=0.75, edgecolors="white", linewidths=0.4, zorder=3,
        )

    ax.set_yscale("log")
    ax.set_xlabel("Edad de adquisición humana (AoA, meses)", fontsize=10)
    ax.set_ylabel(f"Step de estabilización Pythia-{model_size} (log)", fontsize=10)
    ax.set_title(
        f"Alineación Pythia-{model_size} ↔ adquisición humana\n"
        f"Spearman ρ = {rho:.3f}, p = {pval:.3f} (n={len(merged)})",
        fontsize=10,
    )
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3, linestyle=":")

    fig.tight_layout()
    out_png = FIGURES / f"fig2_human_alignment_{model_size}.png"
    out_pdf = FIGURES / f"fig2_human_alignment_{model_size}.pdf"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_png}  (rho={rho:.3f}, p={pval:.3f}, n={len(merged)})")


SHAPE_COLORS = {
    "monotonic":    "#4CAF50",
    "u_shape":      "#F44336",
    "non_monotone": "#FF9800",
    "oscillatory":  "#B0BEC5",
    "inverted_u":   "#9C27B0",
    "single_turn":  "#795548",
}


def _categorize_shape(row: pd.Series) -> str:
    """Clasifica con criterio robusto (depth>=5pp + valle antes del pico)."""
    s = row["shape"]
    if s == "u_shape":
        return "u_shape"
    if s in ("oscillatory", "inverted_u") and row["depth"] >= DEPTH_THRESHOLD and row["min_step"] < row["max_step"]:
        return "non_monotone"
    if s == "monotonic":
        return "monotonic"
    return "oscillatory"


def plot_topology_distribution(topo: pd.DataFrame) -> None:
    """Fig 3: distribución de topologías (criterio robusto) por modelo."""
    sub = topo[topo["task"] != "blimp"].copy()
    sub["shape_robust"] = sub.apply(_categorize_shape, axis=1)

    category_order = ["monotonic", "u_shape", "non_monotone", "oscillatory"]
    category_labels = {
        "monotonic":    "Monótona",
        "u_shape":      "Curva U",
        "non_monotone": "No monótona\n(depth>=5pp)",
        "oscillatory":  "Oscilatoria\n(ruido)",
    }

    counts = (
        sub.groupby(["model_size", "shape_robust"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=category_order, fill_value=0)
    )
    pcts = counts.div(counts.sum(axis=1), axis=0) * 100

    fig, ax = plt.subplots(figsize=(7, 4))
    bottom = np.zeros(len(MODEL_SIZES))
    for cat in category_order:
        vals = [pcts.loc[sz, cat] if sz in pcts.index else 0.0 for sz in MODEL_SIZES]
        ax.bar(MODEL_SIZES, vals, bottom=bottom,
               color=SHAPE_COLORS[cat], label=category_labels[cat], edgecolor="white")
        bottom += np.array(vals)

    ax.axhline(30, color="red", linestyle="--", linewidth=1, alpha=0.7,
               label="Umbral H1 (30%)")
    ax.set_ylabel("% paradigmas", fontsize=10)
    ax.set_xlabel("Modelo Pythia", fontsize=10)
    ax.set_title("Distribución de topologías por modelo (BLiMP, 67 paradigmas)", fontsize=10)
    ax.legend(fontsize=7, loc="upper right", bbox_to_anchor=(1.38, 1))
    ax.set_ylim(0, 105)

    fig.tight_layout()
    out_png = FIGURES / "fig3_topology.png"
    out_pdf = FIGURES / "fig3_topology.pdf"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_png}")


if __name__ == "__main__":
    metrics = pd.read_parquet(RESULTS / "aggregated_metrics.parquet")
    topo = pd.read_parquet(RESULTS / "topology_summary.parquet")
    milestones = pd.read_csv(DATA / "human_milestones.csv")

    h1 = compute_h1_stats(topo)
    print("=== H1 ===")
    for sz, r in h1.items():
        print(f"  {sz}: {r['n_nonmono']}/{r['n_total']} = {r['pct_nonmono']:.1f}% no-monotone (depth>=5pp)")

    plot_trajectories(metrics)
    for sz in MODEL_SIZES:
        plot_human_alignment_scatter(metrics, milestones, model_size=sz)
    plot_topology_distribution(topo)
    print("\nTodas las figuras generadas en figures/")
