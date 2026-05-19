"""Alineación humana: step de estabilización vs AoA (Spearman + bootstrap)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


@dataclass
class HumanAlignmentResult:
    overlap: pd.DataFrame
    stats: pd.DataFrame


def _first_stable_step(
    steps: np.ndarray,
    values: np.ndarray,
    *,
    threshold_ratio: float = 0.9,
    window: int = 3,
) -> float | None:
    if steps.size < window:
        return None
    order = np.argsort(steps)
    x = steps[order]
    y = values[order]
    target = threshold_ratio * np.max(y)
    ok = y >= target
    for idx in range(0, len(ok) - window + 1):
        if np.all(ok[idx : idx + window]):
            return float(x[idx])
    return None


def compute_stabilization_table(
    metrics: pd.DataFrame,
    *,
    metric_col: str = "acc,none",
    task_col: str = "task",
    step_col: str = "training_step",
    model_col: str = "model_size",
    target_model_size: str | None = None,
    threshold_ratio: float = 0.9,
    window: int = 3,
) -> pd.DataFrame:
    required = {metric_col, task_col, step_col, model_col}
    missing = required - set(metrics.columns)
    if missing:
        raise ValueError(f"Faltan columnas en métricas: {sorted(missing)}")

    work = metrics.dropna(subset=[metric_col, task_col, step_col, model_col]).copy()
    if target_model_size is not None:
        work = work[work[model_col] == target_model_size].copy()

    rows: list[dict] = []
    for (model_size, task), g in work.groupby([model_col, task_col], dropna=False):
        stable = _first_stable_step(
            g[step_col].to_numpy(dtype=float),
            g[metric_col].to_numpy(dtype=float),
            threshold_ratio=threshold_ratio,
            window=window,
        )
        rows.append(
            {
                "model_size": model_size,
                "task": task,
                "training_step_stabilize": stable,
                "max_metric": float(np.max(g[metric_col].to_numpy(dtype=float))),
                "n_points": int(len(g)),
            }
        )
    return pd.DataFrame(rows)


def _bootstrap_spearman(
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_boot: int = 10000,
) -> tuple[float, float]:
    n = len(x)
    rng = np.random.default_rng(1234)
    rhos = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if np.unique(x[idx]).size < 2 or np.unique(y[idx]).size < 2:
            continue
        rho, _ = spearmanr(x[idx], y[idx])
        if not np.isnan(rho):
            rhos.append(float(rho))
    if not rhos:
        return float("nan"), float("nan")
    lo, hi = np.percentile(np.array(rhos), [2.5, 97.5])
    return float(lo), float(hi)


def run_human_alignment(
    *,
    metrics_parquet: str | pd.PathLike,
    aoa_csv: str | pd.PathLike,
    metric_col: str = "acc,none",
    target_model_size: str | None = None,
    aoa_task_col: str = "task",
    aoa_months_col: str = "aoa_months",
    n_boot: int = 10000,
) -> HumanAlignmentResult:
    metrics = pd.read_parquet(metrics_parquet)
    aoa = pd.read_csv(aoa_csv)
    if aoa_task_col not in aoa.columns or aoa_months_col not in aoa.columns:
        raise ValueError(
            f"CSV AoA debe contener columnas `{aoa_task_col}` y `{aoa_months_col}`."
        )

    stable = compute_stabilization_table(
        metrics,
        metric_col=metric_col,
        target_model_size=target_model_size,
    )
    overlap = stable.merge(
        aoa[[aoa_task_col, aoa_months_col]].rename(
            columns={aoa_task_col: "task", aoa_months_col: "aoa_months"}
        ),
        on="task",
        how="inner",
    ).dropna(subset=["training_step_stabilize", "aoa_months"])

    if overlap.empty:
        stats = pd.DataFrame(
            [
                {
                    "rho": np.nan,
                    "p_value": np.nan,
                    "ci_lo": np.nan,
                    "ci_hi": np.nan,
                    "n_overlap": 0,
                    "model_size": target_model_size,
                    "metric_col": metric_col,
                }
            ]
        )
        return HumanAlignmentResult(overlap=overlap, stats=stats)

    x = overlap["training_step_stabilize"].to_numpy(dtype=float)
    y = overlap["aoa_months"].to_numpy(dtype=float)
    if np.unique(x).size < 2 or np.unique(y).size < 2:
        rho, p = np.nan, np.nan
    else:
        rho, p = spearmanr(x, y)
    ci_lo, ci_hi = _bootstrap_spearman(x, y, n_boot=n_boot)
    stats = pd.DataFrame(
        [
            {
                "rho": float(rho),
                "p_value": float(p),
                "ci_lo": ci_lo,
                "ci_hi": ci_hi,
                "n_overlap": int(len(overlap)),
                "model_size": target_model_size,
                "metric_col": metric_col,
            }
        ]
    )
    return HumanAlignmentResult(overlap=overlap, stats=stats)

