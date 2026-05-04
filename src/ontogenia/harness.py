"""Ejecución de lm-eval y escritura de artefactos JSON estandarizados."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import lm_eval
from lm_eval import simple_evaluate
from lm_eval.utils import handle_non_serializable


def run_lm_eval(
    *,
    model_args: str,
    tasks: list[str],
    batch_size: int | str = "auto",
    device: str | None = None,
    limit: int | float | None = None,
    log_samples: bool = False,
    bootstrap_iters: int = 0,
) -> dict[str, Any]:
    """
    Evalúa el modelo `hf` con la lista de tareas (nombres lm-eval).
    `bootstrap_iters=0` desactiva stderr en pruebas rápidas.
    """
    return simple_evaluate(
        model="hf",
        model_args=model_args,
        tasks=tasks,
        batch_size=batch_size,
        device=device,
        limit=limit,
        log_samples=log_samples,
        bootstrap_iters=bootstrap_iters,
    )


def build_result_envelope(
    raw: dict[str, Any],
    *,
    model_size: str,
    pretrained: str,
    revision: str,
    tasks: list[str],
) -> dict[str, Any]:
    """Envuelve la salida de lm-eval con metadatos del experimento."""
    return {
        "meta": {
            "model_size": model_size,
            "pretrained": pretrained,
            "revision": revision,
            "tasks": tasks,
            "lm_eval_version": getattr(lm_eval, "__version__", "unknown"),
        },
        "lm_eval": raw,
    }


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=handle_non_serializable, ensure_ascii=False)
    # second line: ensure file ends with newline
    with path.open("a", encoding="utf-8") as f:
        f.write("\n")
