"""Consolidar JSON de `results/` en tablas largas (Parquet) para notebooks."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import pandas as pd

from ontogenia.metrics import pairwise_slln_lp_margin

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


def _infer_token_count(choice_text: str, token_logprobs: list[float] | None) -> int:
    if token_logprobs:
        return max(len(token_logprobs), 1)
    # fallback aproximado: split por espacios (mejor que 0 para evitar excepciones)
    return max(len(choice_text.split()), 1)


def _sample_to_slln_rows(
    sample: dict[str, Any],
    *,
    model_size: str | None,
    pretrained: str | None,
    revision: str | None,
    training_step: int | None,
    source_file: str,
    source_subdir: str,
    alpha: float,
) -> dict[str, Any] | None:
    doc = sample.get("doc")
    if not isinstance(doc, dict):
        return None
    sentence_good = doc.get("sentence_good")
    sentence_bad = doc.get("sentence_bad")
    if not isinstance(sentence_good, str) or not isinstance(sentence_bad, str):
        return None

    # lm-eval en multiple_choice guarda 2 reqs (una por opción); cada req.resps
    # suele venir como [(sum_logprob, is_greedy), ...].
    req_resps = sample.get("resps")
    if not isinstance(req_resps, list) or len(req_resps) < 2:
        return None

    def _sum_logprob(req_idx: int) -> float | None:
        req = req_resps[req_idx]
        if not isinstance(req, list) or not req:
            return None
        first = req[0]
        if isinstance(first, (list, tuple)) and first:
            val = first[0]
            if isinstance(val, (int, float)):
                return float(val)
        return None

    lp_good = _sum_logprob(0)
    lp_bad = _sum_logprob(1)
    if lp_good is None or lp_bad is None:
        return None

    args = sample.get("arguments")
    token_logprobs_good = None
    token_logprobs_bad = None
    if isinstance(args, list) and len(args) >= 2:
        # no hay garantía de token-level logprobs aquí; conservamos fallback robusto
        token_logprobs_good = None
        token_logprobs_bad = None
    n_good = _infer_token_count(sentence_good, token_logprobs_good)
    n_bad = _infer_token_count(sentence_bad, token_logprobs_bad)
    margin = pairwise_slln_lp_margin(lp_good, n_good, lp_bad, n_bad, alpha=alpha)

    return {
        "source_file": source_file,
        "source_subdir": source_subdir,
        "model_size": model_size,
        "pretrained": pretrained,
        "revision": revision,
        "training_step": training_step,
        "task": sample.get("task", None),
        "doc_id": sample.get("doc_id"),
        "sentence_good": sentence_good,
        "sentence_bad": sentence_bad,
        "logprob_good": lp_good,
        "logprob_bad": lp_bad,
        "n_tokens_good": n_good,
        "n_tokens_bad": n_bad,
        "slln_margin_alpha": alpha,
        "slln_margin": margin,
        "acc": sample.get("acc"),
    }


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


def collect_slln_sample_rows(
    result_json_paths: list[Path],
    *,
    samples_dir: Path,
    alpha: float = 0.5,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in result_json_paths:
        if not path.is_file():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        meta = data.get("meta") or {}
        step = _parse_step_from_meta(meta)
        if step is None:
            m = _STEP_RE.search(path.stem)
            step = int(m.group(1)) if m else None
        sidecar = samples_dir / f"{path.stem}_samples.json"
        if not sidecar.is_file():
            continue
        samples_by_task = json.loads(sidecar.read_text(encoding="utf-8"))
        if not isinstance(samples_by_task, dict):
            continue
        for task_name, task_samples in samples_by_task.items():
            if not isinstance(task_samples, list):
                continue
            for sample in task_samples:
                if not isinstance(sample, dict):
                    continue
                row = _sample_to_slln_rows(
                    sample | {"task": task_name},
                    model_size=meta.get("model_size"),
                    pretrained=meta.get("pretrained"),
                    revision=meta.get("revision"),
                    training_step=step,
                    source_file=str(path),
                    source_subdir=path.parent.name,
                    alpha=alpha,
                )
                if row is not None:
                    # descarta márgenes no finitos por seguridad
                    if isinstance(row["slln_margin"], float) and math.isfinite(row["slln_margin"]):
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


def aggregate_samples_dir(
    *,
    results_root: Path,
    samples_dir: Path,
    out_parquet: Path,
    include_smoke: bool = True,
    include_sweep: bool = True,
    alpha: float = 0.5,
) -> Path:
    """Consolida sidecars `*_samples.json` en Parquet por ítem (SLLN margin)."""
    paths: list[Path] = []
    if include_sweep:
        paths.extend(sorted((results_root / "sweep").glob("*.json")))
    if include_smoke:
        smoke_dir = results_root / "smoke"
        if smoke_dir.is_dir():
            paths.extend(sorted(smoke_dir.glob("*.json")))
    rows = collect_slln_sample_rows(paths, samples_dir=samples_dir, alpha=alpha)
    frame = pd.DataFrame(rows)
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out_parquet, index=False)
    return out_parquet
