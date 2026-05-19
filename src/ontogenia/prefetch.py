"""Pre-descarga de checkpoints Pythia para correr sweep sin esperas de red."""

from __future__ import annotations

from dataclasses import dataclass

from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from ontogenia.checkpoints import PYTHIA_MODEL_IDS, revision_for_step


@dataclass
class PrefetchResult:
    model_size: str
    revision: str
    ok: bool
    error: str | None = None


def prefetch_checkpoints(
    *,
    sizes: list[str],
    steps: list[int],
    cache_dir: str | None = None,
) -> list[PrefetchResult]:
    results: list[PrefetchResult] = []
    total = len(sizes) * len(steps)
    for size in sizes:
        if size not in PYTHIA_MODEL_IDS:
            raise ValueError(f"Tamaño desconocido: {size}. Usar {sorted(PYTHIA_MODEL_IDS)}")
        model_id = PYTHIA_MODEL_IDS[size]
        for step in tqdm(steps, total=len(steps), desc=f"prefetch {size}"):
            revision = revision_for_step(step)
            try:
                AutoTokenizer.from_pretrained(
                    model_id,
                    revision=revision,
                    cache_dir=cache_dir,
                )
                AutoModelForCausalLM.from_pretrained(
                    model_id,
                    revision=revision,
                    cache_dir=cache_dir,
                    torch_dtype="auto",
                )
                results.append(PrefetchResult(size, revision, True, None))
            except Exception as exc:  # pragma: no cover - depende de red/HF
                results.append(PrefetchResult(size, revision, False, str(exc)))
    _ = total  # conserva compatibilidad para futuros logs
    return results

