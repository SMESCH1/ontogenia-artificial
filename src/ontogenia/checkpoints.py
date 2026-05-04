"""Vector de checkpoints Pythia y utilidades HuggingFace `revision=stepN`."""

from __future__ import annotations

from collections.abc import Iterator

# 24 checkpoints — `docs/04_experimental_design.md` (densos al inicio, lineales al final)
CHECKPOINT_STEPS: tuple[int, ...] = (
    0,
    1,
    2,
    4,
    8,
    16,
    32,
    64,
    128,
    256,
    512,
    1000,
    2000,
    4000,
    8000,
    16000,
    32000,
    48000,
    64000,
    80000,
    96000,
    112000,
    128000,
    143000,
)

# Repositorios HuggingFace (familia deduplicada)
PYTHIA_MODEL_IDS: dict[str, str] = {
    "14m": "EleutherAI/pythia-14m-deduped",
    "160m": "EleutherAI/pythia-160m-deduped",
    "410m": "EleutherAI/pythia-410m-deduped",
}


def revision_for_step(step: int) -> str:
    """Tag de revisión Pythia en el Hub (`step0`, `step143000`, ...)."""
    return f"step{step}"


def iter_checkpoint_triplets(
    sizes: tuple[str, ...] = ("14m", "160m", "410m"),
    steps: tuple[int, ...] = CHECKPOINT_STEPS,
) -> Iterator[tuple[str, str, str]]:
    """Produce `(size_key, model_id, revision)` por cada combinación."""
    for size in sizes:
        if size not in PYTHIA_MODEL_IDS:
            raise KeyError(f"Tamaño desconocido: {size}. Usar {tuple(PYTHIA_MODEL_IDS)}")
        mid = PYTHIA_MODEL_IDS[size]
        for step in steps:
            yield size, mid, revision_for_step(step)


def model_args_for_pythia(
    size_or_id: str,
    step: int,
    *,
    dtype: str = "float16",
    trust_remote_code: bool = False,
) -> str:
    """Cadena `model_args` para lm-eval `--model hf`."""
    pretrained = PYTHIA_MODEL_IDS.get(size_or_id, size_or_id)
    parts = [
        f"pretrained={pretrained}",
        f"revision={revision_for_step(step)}",
        f"dtype={dtype}",
    ]
    if trust_remote_code:
        parts.append("trust_remote_code=True")
    return ",".join(parts)
