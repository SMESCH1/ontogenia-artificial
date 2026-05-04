"""Consolidar JSON de `results/` en tablas largas (Parquet) para notebooks."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

_STEP_RE = re.compile(r"step(\d+)$")


def _parse_step_from_meta(meta: dict[str, Any]) -> int | None:
    rev = meta.get("revision", "")
    if isinstance(rev, str) and rev.startswith("step"):
        try:
            return int(rev.replace("step", "", 1))
        except ValueError:
            return None
    return None


def _scalar_metrics(task_metrics: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in task_metrics.items():
        if isinstance(v, (bool, int, float, str)) or v is None:
            out[k] = v
    return out


def collect_metrics_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        meta = data.get("meta") or {}
        lm = data.get("lm_eval") or {}
        results = lm.get("results") or {}
        step = _parse_step_from_meta(meta)
        if step is None:
            m = _STEP_RE.search(path.stem)
            step = int(m.group(1)) if m else None
        for task_name, task_metrics in results.items():
            if task_name in ("groups",) or not isinstance(task_metrics, dict):
                continue
            flat = _scalar_metrics(task_metrics)
            row = {
                "source_file": str(path),
                "source_subdir": path.parent.name,
                "model_size": meta.get("model_size"),
                "pretrained": meta.get("pretrained"),
                "revision": meta.get("revision"),
                "training_step": step,
                "task": task_name,
                **flat,
            }
            rows.append(row)
    return rows


def aggregate_results_dir(
    results_root: Path,
    *,
    out_parquet: Path,
    include_smoke: bool = True,
    include_sweep: bool = True,
) -> Path:
    """Lee `results/sweep/*.json` (y opc. `smoke/`), escribe un Parquet largo."""
    paths: list[Path] = []
    if include_sweep:
        paths.extend(sorted((results_root / "sweep").glob("*.json")))
    if include_smoke:
        smoke_dir = results_root / "smoke"
        if smoke_dir.is_dir():
            paths.extend(sorted(smoke_dir.glob("*.json")))
    rows = collect_metrics_rows(paths)
    frame = pd.DataFrame(rows)
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out_parquet, index=False)
    return out_parquet
