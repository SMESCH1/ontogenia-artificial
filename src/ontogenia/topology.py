"""Clasificación topológica de trayectorias (monótona / U / U invertida / oscilatoria)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class TopologyResult:
    shape: str
    sign_changes: int
    min_step: float | None
    max_step: float | None
    depth: float | None
    peak_height: float | None


def _sign_changes(values: np.ndarray) -> int:
    signs = np.sign(values)
    # ignora ceros para no inflar cambios por ruido
    signs = signs[signs != 0]
    if signs.size <= 1:
        return 0
    return int(np.sum(signs[1:] != signs[:-1]))


def classify_series(
    steps: np.ndarray,
    scores: np.ndarray,
    *,
    smooth_points: int = 200,
) -> TopologyResult:
    if steps.size < 3:
        return TopologyResult("insufficient_points", 0, None, None, None, None)

    # orden estable por paso
    order = np.argsort(steps)
    x = steps[order].astype(float)
    y = scores[order].astype(float)

    if x.size <= 7:
        # para series cortas evitamos oscilaciones artificiales del polyfit.
        grid = x
        pred = y
        d1 = np.diff(pred)
    else:
        deg = min(5, x.size - 1)
        poly = np.poly1d(np.polyfit(x, y, deg=deg))
        grid = np.linspace(float(x.min()), float(x.max()), smooth_points)
        pred = poly(grid)
        d1 = np.gradient(pred, grid)
    changes = _sign_changes(d1)

    min_idx = int(np.argmin(pred))
    max_idx = int(np.argmax(pred))
    min_step = float(grid[min_idx])
    max_step = float(grid[max_idx])
    depth = float(np.max(pred) - np.min(pred))

    if changes == 0:
        shape = "monotonic"
    elif changes == 1:
        # decreciente->creciente => U ; creciente->decreciente => inverted U
        split = max(len(d1) // 2, 1)
        left = np.mean(d1[:split])
        right = np.mean(d1[split:])
        if left < 0 and right > 0:
            shape = "u_shape"
        elif left > 0 and right < 0:
            shape = "inverted_u"
        else:
            shape = "single_turn"
    else:
        shape = "oscillatory"

    peak_height = float(np.max(pred) - pred[0])
    return TopologyResult(shape, changes, min_step, max_step, depth, peak_height)


def classify_all_tasks(
    frame: pd.DataFrame,
    *,
    metric_col: str = "acc,none",
    task_col: str = "task",
    step_col: str = "training_step",
    model_col: str = "model_size",
) -> pd.DataFrame:
    required = {metric_col, task_col, step_col, model_col}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Faltan columnas para topología: {sorted(missing)}")

    rows: list[dict] = []
    valid = frame.dropna(subset=[metric_col, task_col, step_col, model_col]).copy()
    for (model_size, task), g in valid.groupby([model_col, task_col], dropna=False):
        steps = g[step_col].to_numpy(dtype=float)
        scores = g[metric_col].to_numpy(dtype=float)
        topo = classify_series(steps, scores)
        rows.append(
            {
                "model_size": model_size,
                "task": task,
                "n_points": int(len(g)),
                "shape": topo.shape,
                "sign_changes": topo.sign_changes,
                "min_step": topo.min_step,
                "max_step": topo.max_step,
                "depth": topo.depth,
                "peak_height": topo.peak_height,
            }
        )
    return pd.DataFrame(rows)

