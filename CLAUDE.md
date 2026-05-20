# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Proyecto

Research Project de NLP (UDESA, 2026). Estudia si Pythia exhibe curvas-U sintácticas análogas a la adquisición infantil, y si el orden de emergencia por paradigma correlaciona con edad de adquisición humana (Wordbank). Deadline: **26/05/2026**. Paper en formato ACL, en **español**.

Trayectorias sintácticas en **Pythia** (checkpoints `revision=stepN`) sobre **BLiMP + Zorro**, métrica **SLLN-LP** (α=0.5), correlación **Spearman** entre orden de estabilización y AoA humana. H1 ≥30% paradigmas no monótonos; H2 ρ>0.3; H3 curvas-U más profundas en morfológico-irregular. **Checkpoints:** 24 pasos en `src/ontogenia/checkpoints.py` (`CHECKPOINT_STEPS`).

Documentación completa en `docs/` — leer `01_project_overview.md` para contexto general, `04_experimental_design.md` para las hipótesis operacionalizadas.

## Comandos

```bash
# Setup
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .

# Tests
python -m unittest discover -s tests -v

# Linting
ruff check src/ && ruff format src/

# Pipeline
./scripts/run_smoke.sh --limit 20                    # prueba rápida → results/smoke/
./scripts/run_full_sweep.sh                          # sweep completo → results/sweep/ (GPU, ~24h/modelo)
./scripts/prefetch_checkpoints.sh --sizes 14m,160m,410m --steps default
./scripts/run_post_analysis.sh                       # aggregate + topology + human-alignment

# Subcomandos programáticos
python -m ontogenia sweep --sizes 160m --steps 0,512,2000 --tasks blimp --limit 50
python -m ontogenia aggregate --results-dir results --output-parquet results/aggregated_metrics.parquet
python -m ontogenia prefetch --sizes 14m,160m,410m --steps default
python -m ontogenia topology --metrics-parquet results/aggregated_metrics.parquet --output-parquet results/topology_summary.parquet
python -m ontogenia human-alignment --metrics-parquet results/aggregated_metrics.parquet --aoa-csv data/human_milestones.csv --model-size 160m
```

**Salidas:** JSON bajo `results/` con `{ "meta": {...}, "lm_eval": <salida harness> }`. Carpeta gitignored. Para guardar pares por ítem usar `--samples-dir DIR` (vuelca `lm_eval["samples"]` a `DIR/<stem>_samples.json`).

## Arquitectura del pipeline

```
HuggingFace (revision=stepN)
        │
        ▼
lm-evaluation-harness  ←── configs/*.yaml (tareas custom BLiMP/Zorro)
        │
        ▼  JSON por (modelo, checkpoint, paradigma)  →  results/
        │
        ▼
src/ontogenia/aggregate.py   → results/aggregated_metrics.parquet
        │
        ├─▶ src/ontogenia/topology.py       → topology_summary.parquet
        └─▶ src/ontogenia/human_alignment.py → Spearman + bootstrap vs. Wordbank AoA
        │
        ▼
notebooks/02_trajectories.ipynb       ← polinomio grado 5, clasificación topológica
notebooks/03_human_correlation.ipynb  ← scatter ranking Pythia vs. ranking humano
```

Módulos en `src/ontogenia/`:
- `checkpoints.py` — vector de 24 steps, `model_args` para HuggingFace, `iter_checkpoint_triplets`
- `metrics.py` — SLLN-LP (α=0.5), mean_LP
- `harness.py` — wrapper sobre lm-evaluation-harness
- `cli.py` — subcomandos `smoke`, `sweep`, `eval`, `aggregate`, `prefetch`, `topology`, `human-alignment`
- `aggregate.py` — JSON → parquet
- `topology.py` — ajuste polinomial grado 5, clasificador monótona/U/U-inv/oscilatoria
- `human_alignment.py` — parsing Wordbank + Spearman + bootstrap
- `prefetch.py` — descarga anticipada de checkpoints
- `sae_probes.py` — análisis representacional vía SAEs (**deseable, no priorizado**, ver `docs/06_sae_representational_probes.md`)

## Decisiones clave de diseño

**Métrica principal — SLLN-LP, no accuracy:** la log-probabilidad conjunta sesga hacia oraciones cortas. SLLN-LP normaliza sub-linealmente (`log P(s) / |s|^α`, α=0.5) para evitar ese artefacto en pares mínimos. Ver `docs/02_theoretical_framework.md §3`.

**Checkpoints — escala logarítmica en fase temprana:** los primeros 512 pasos se muestrean densos (step0…step512 en potencias de 2) porque la emergencia ocurre rápido ahí. La fase tardía usa intervalos lineales de 16k pasos. 24 checkpoints en total por modelo.

**Modelos — sólo deduplicados:** `pythia-{14m,160m,410m}-deduped`. La versión deduplicada tiene el orden exacto del corpus controlado, requisito para análisis longitudinal limpio.

**lm-evaluation-harness como motor:** garantiza entropía cruzada determinista y soporta `revision=stepN` de HuggingFace nativo. Las tareas custom van en `configs/*.yaml`.

**Resultados y pesos — gitignored:** `results/`, `data/raw/`, `data/processed/`, caches de HuggingFace.

## Lectura rápida al retomar

1. `ROADMAP.md`
2. `docs/04_experimental_design.md`
3. `docs/05_human_alignment.md`
4. `src/ontogenia/cli.py` — punto de entrada de todos los subcomandos

**Fuera de scope:** SLT/LLC, interpretabilidad de circuitos, ToM, entrenar desde cero.

## SAEs (análisis representacional — deseable)

`src/ontogenia/sae_probes.py` usa `sparsify` (EleutherAI) para proyectar activaciones de Pythia a través de SAEs pre-entrenados y detectar features sintácticos. **No instalar `sparsify` por defecto** — está comentado en `requirements.txt`. Activar sólo si se implementa H4. Documentación: `docs/06_sae_representational_probes.md`.

## Paper

`paper/main.tex` con plantilla ACL (`acl.sty`). Compilar con `pdflatex` o `lualatex`. Bibliografía en `paper/custom.bib`. Los artefactos de compilación están en `.gitignore`.
