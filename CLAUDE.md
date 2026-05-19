# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Proyecto

Research Project de NLP (UDESA, 2026). Estudia si Pythia exhibe curvas-U sintácticas análogas a la adquisición infantil, y si el orden de emergencia por paradigma correlaciona con edad de adquisición humana (Wordbank). Deadline: **26/05/2026**. Paper en formato ACL, en **español**.

Documentación completa en `docs/` — leer `01_project_overview.md` para contexto general, `04_experimental_design.md` para las hipótesis operacionalizadas (H1/H2/H3).

## Comandos

```bash
# Instalar en modo editable
pip install -e .

# Linting (ruff, configurado en pyproject.toml)
ruff check src/
ruff format src/

# Smoke test del pipeline (Semana 2 en adelante)
python scripts/run_full_sweep.sh  # aún no implementado
```

El proyecto no tiene suite de tests formales — la validación es empírica (sanity checks en notebooks).

## Arquitectura del pipeline

```
HuggingFace (revision=stepN)
        │
        ▼
lm-evaluation-harness  ←── configs/*.yaml (tareas custom BLiMP/Zorro)
        │
        ▼  JSON por (modelo, checkpoint, paradigma)
results/*.json  (gitignored)
        │
        ▼
src/ontogenia/metrics.py     ← SLLN-LP, mean_LP, accuracy
        │
        ▼
results/aggregated.parquet   (gitignored)
        │
        ▼
notebooks/02_trajectories.ipynb  ← polinomio grado 5, clasificación topológica
notebooks/03_human_correlation.ipynb  ← Spearman con bootstrap vs. Wordbank AoA
```

Módulos en `src/ontogenia/` (implementación: Semanas 2–3):
- `checkpoints.py` — iterador sobre `revision=stepN` con cache local
- `metrics.py` — SLLN-LP (α=0.5), mean_LP
- `eval_runner.py` — wrapper programático sobre lm-eval
- `topology.py` — ajuste polinomial grado 5, clasificador monótona/U/U-inv/oscilatoria
- `human_alignment.py` — parsing Wordbank + Spearman + bootstrap
- `sae_probes.py` — análisis representacional vía SAEs (**deseable, no priorizado**, ver `docs/06_sae_representational_probes.md`)

## Decisiones clave de diseño

**Métrica principal — SLLN-LP, no accuracy:** la log-probabilidad conjunta sesga hacia oraciones cortas. SLLN-LP normaliza sub-linealmente (`log P(s) / |s|^α`, α=0.5) para evitar ese artefacto en pares mínimos. Ver `docs/02_theoretical_framework.md §3`.

**Checkpoints — escala logarítmica en fase temprana:** los primeros 512 pasos se muestrean densos (step0…step512 en potencias de 2) porque la emergencia ocurre rápido ahí. La fase tardía usa intervalos lineales de 16k pasos. 24 checkpoints en total por modelo.

**Modelos — sólo deduplicados:** `pythia-{14m,160m,410m}-deduped`. La versión deduplicada tiene el orden exacto del corpus controlado, requisito para análisis longitudinal limpio.

**lm-evaluation-harness como motor:** garantiza entropía cruzada determinista y soporta `revision=stepN` de HuggingFace nativo. Las tareas custom van en `configs/*.yaml`.

**Resultados y pesos — gitignored:** `results/`, `data/raw/`, `data/processed/`, caches de HuggingFace. Los resultados grandes van como release de GitHub o git-lfs.

## SAEs (análisis representacional — deseable)

`src/ontogenia/sae_probes.py` usa `sparsify` (EleutherAI) para proyectar activaciones de Pythia a través de SAEs pre-entrenados y detectar features sintácticos. **No instalar `sparsify` por defecto** — está comentado en `requirements.txt`. Activar sólo si se implementa H4. Documentación: `docs/06_sae_representational_probes.md`.

## Paper

`paper/main.tex` con plantilla ACL (`acl.sty`). Compilar con `pdflatex` o `lualatex`. Bibliografía en `paper/custom.bib`. Los artefactos de compilación están en `.gitignore`.
