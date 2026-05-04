# Instrucciones para agentes

## Orden sugerido de lectura

`ROADMAP.md` → `docs/04_experimental_design.md` → `docs/05_human_alignment.md` → código en `src/ontogenia/`.

## Qué no versionar

- `data/raw/`, `data/processed/`, `results/`, caches HF (`hf_cache/`, `.cache/`), `.venv/`.

## Convenciones de resultados

- Smoke: `results/smoke/{size}_step{N}_{task}.json`
- Sweep: `results/sweep/{size}_step{N}.json` (tareas listadas en `meta.tasks` dentro del archivo)

Envelope común: `meta` + `lm_eval` (dict completo devuelto por `lm_eval.simple_evaluate`).

Muestras por ítem (`doc`, `resps`, métricas por ejemplo): con `--samples-dir` se escribe un archivo paralelo `*_samples.json` y se elimina la clave `samples` del dict principal antes de guardar.

## No expandir el proyecto hacia

Interpretabilidad mecanicista profunda, Singular Learning Theory operativa, ToM, fine-tuning masivo — salvo mención breve como trabajo futuro en el paper.
