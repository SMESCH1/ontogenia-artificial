# Roadmap — Ontogenia Artificial

> Research Project para NLP (UDESA, 1er semestre 2026, Luciano Del Corro).
> **Deadline: 26 de mayo de 2026** (presentación, Clase 12).

## Contexto

Investigamos la emergencia ontogenética de capacidades sintácticas en LLMs evaluando los 154 checkpoints intermedios de la suite **Pythia** (EleutherAI) sobre baterías psicolingüísticas (**BLiMP**, **Zorro**). Ángulo diferencial: correlacionar el orden de estabilización de paradigmas sintácticos con la **edad de adquisición humana** reportada en Wordbank / literatura de psicolingüística del desarrollo.

**Hito principal:** sintaxis / curvas-U. **Deseable si alcanza:** léxico / mundo físico (EWoK).

**Fuera de scope:** SLT/LLC, interpretabilidad mecanicista, ToM, razonamiento lógico, entrenamiento desde cero.

## Equipo y recursos

- **Equipo:** Sebastián, Leandro, Hugo.
- **Cómputo:** GPU local tipo RTX 3090/4090.
- **Idioma de entregables:** español (paper en formato ACL).
- **Entregables:** (1) paper ACL en PDF, (2) repo GitHub reproducible, (3) presentación.

## Hipótesis centrales

- **H1 — Existencia de curvas no monótonas.** ≥30% de los paradigmas de BLiMP/Zorro muestran caída ≥5 pp seguida de recuperación en Pythia-160m/410m.
- **H2 — Alineación ordinal con humanos.** Spearman ρ > 0.3 (p < 0.05) entre ranking de estabilización en Pythia y ranking humano de edad de adquisición.
- **H3 — Excepciones morfológicas.** Las curvas-U son más profundas en paradigmas con excepciones morfológicas que en paradigmas puramente estructurales.

Detalle operacional en [`docs/04_experimental_design.md`](docs/04_experimental_design.md).

## Calendario de 5 semanas

### Semana 1 (21/04 – 27/04) — Research + scaffolding
**Objetivo:** cerrar marco teórico, hipótesis operacionalizadas y repo organizado. **Sin código todavía.**

- [x] Scaffolding del repositorio (estructura, `.gitignore`, `pyproject.toml`, `requirements.txt`, README).
- [x] Plantilla ACL (`paper/`) con `main.tex` esqueleto en español.
- [x] `docs/01_project_overview.md` — resumen ejecutivo, hipótesis, scope.
- [x] `docs/02_theoretical_framework.md` — curvas-U, SLLN-LP, psicolingüística.
- [x] `docs/03_methodology_and_tech_stack.md` — modelos, datasets, stack.
- [x] `docs/04_experimental_design.md` — H1/H2/H3 operacionalizadas.
- [x] `docs/05_human_alignment.md` — decisión Wordbank + tabla inicial.
- [x] Plantillas `docs/papers/01-05_*.md` para notas de lectura.
- [ ] **Pendiente de la semana:** Leandro y Hugo confirman frentes y empiezan la lectura dirigida de los 5 papers críticos.
- [ ] Expandir tabla de `05_human_alignment.md` a ≥15 pares con consenso del equipo.
- [ ] Descarga del snapshot CSV de Wordbank (inglés) a `data/raw/`.

### Semana 2 (28/04 – 04/05) — Pipeline mínimo funcional
- [x] `src/ontogenia/checkpoints.py` — vector de 24 `stepN`, `model_args` HuggingFace, `iter_checkpoint_triplets`.
- [x] Integrar **`lm-evaluation-harness`** + tarea BLiMP vía CLI (`ontogenia smoke|sweep|eval`); Zorro: guía en [`docs/integracion_zorro.md`](docs/integracion_zorro.md) (YAML / BabyLM pipeline — pendiente de portar).
- [x] **Smoke test** — JSON bajo `results/smoke/` (corrida local: `pythia-14m-deduped` @ `step2000`, `blimp_anaphor_number_agreement`, **acc = 0.75** sobre **20/1000** ítems, `cuda:0`, batch auto → 64). Plot de trayectoria: pendiente en notebook.
- [x] `src/ontogenia/metrics.py` — SLLN-LP + tests unitarios; **validación numérica vs cifra publicada ZhoBLiMP** sigue opcional (sanity del paper).
- [x] Vector de checkpoints — **24 pasos** alineados con `04_experimental_design.md` (código + docs).
- [x] Post-proceso mínimo — `ontogenia aggregate` → `results/aggregated_metrics.parquet`; opción `--samples-dir` para sidecar `*_samples.json` (por ítem).
- [x] Pre-descarga disponible vía `ontogenia prefetch` / `scripts/prefetch_checkpoints.sh` (ejecución completa pendiente según espacio en disco y ventana de cómputo).

### Semana 3 (05/05 – 11/05) — Corrida completa
- [ ] Sweep Pythia-14m/160m/410m × 24 checkpoints × **BLiMP** (`ontogenia sweep --tasks blimp`) + Zorro cuando esté integrado (~80 paradigmas).
- [x] `results/` con naming — `smoke/{size}_step{N}_{task}.json`, `sweep/{size}_step{N}.json` (ver `CLAUDE.md` / `AGENTS.md`).
- [x] `notebooks/02_trajectories.ipynb` operativo con `aggregated_metrics.parquet` + export `topology_summary.parquet`.
- [x] Clasificación topológica implementada en `src/ontogenia/topology.py` + comando `ontogenia topology`.
- [ ] Buffer para re-correr ante artefactos.

**Estimación de tiempo (orden de magnitud):** el diseño en `04_experimental_design.md` asume **≲24 h por modelo** en 3090/4090 para el sweep BLiMP completo (3 tamaños × 24 checkpoints × grupo `blimp`). En la práctica depende de GPU, drivers y caché HF; conviene **cronometrar una** corrida `14m` × `step0` × `blimp` sin `--limit` y extrapolar ×72.

### Semana 4 (12/05 – 18/05) — Análisis + escritura
- [x] `notebooks/03_human_correlation.ipynb` operativo + módulo `src/ontogenia/human_alignment.py` y comando `ontogenia human-alignment` (Spearman + bootstrap).
- [ ] Borrador v1 del paper (6–8 páginas): abstract, intro, related work, método, resultados, discusión.
- [ ] **Si queda margen:** piloto de EWoK sobre el mejor Pythia identificado.
- [ ] Figuras preliminares para la presentación.

### Semana 5 (19/05 – 26/05) — Pulido + presentación
- [ ] Revisión interna cruzada entre Sebastián, Leandro y Hugo.
- [ ] Paper v2 (final), figuras con calidad publicable.
- [ ] Slides (~15 min).
- [ ] README reproducible con instrucciones end-to-end.
- [ ] Tag `v1.0-presentacion` el 25/05.
- [ ] **26/05: presentación.**

## Plan de contingencia

Si en Semana 3 el sweep completo no es viable, degradamos **en este orden**:
1. Reducir a 1 modelo (`pythia-160m-deduped`).
2. Reducir a sólo Zorro (13 paradigmas).
3. Reducir a 15 checkpoints.
4. Dejar H2 (alineación humana) como discusión cualitativa y sostener H1 + H3 cuantitativas.

El peor caso sigue siendo un experimento publicable con una pregunta clara.

## Navegación del repo

| Carpeta / archivo | Qué hay |
| --- | --- |
| [`docs/`](docs/) | Marco teórico, diseño experimental, notas de papers. |
| [`paper/`](paper/) | Plantilla ACL + `main.tex` del manuscrito (español). |
| `src/ontogenia/` | Pipeline: `cli` (smoke / sweep / eval / **aggregate**), `checkpoints`, `metrics`, `harness`, `aggregate`. |
| [`CLAUDE.md`](CLAUDE.md), [`AGENTS.md`](AGENTS.md) | Handoff, convenciones `results/`, `--samples-dir`. |
| [`docs/enlaces_herramientas.md`](docs/enlaces_herramientas.md), [`docs/deep_research_highlights.md`](docs/deep_research_highlights.md), [`docs/integracion_zorro.md`](docs/integracion_zorro.md) | Enlaces, resumen SoTA, plan Zorro. |
| `configs/` | Placeholder para YAMLs custom (Zorro u otras tareas). |
| `notebooks/` | `02_trajectories`, `03_human_correlation` (esqueleto + nota Parquet). |
| `scripts/` | `run_smoke.sh`, `run_full_sweep.sh`, `prefetch_checkpoints.sh`, `run_post_analysis.sh`. |
| `data/` | Datasets crudos (gitignored) + `human_milestones.csv`. |
| `results/` | JSONs de evaluación (gitignored). |

## Referencias principales

- Biderman et al. (2023) — Pythia.
- Bunzeck & Zarrieß (2024) — Fifty shapes of BLiMP.
- Bunzeck et al. (2024) — ZhoBLiMP / SLLN-LP.
- Frank et al. (2017) — Wordbank.
- Hu et al. (2024) — BabyLM Challenge 2024.
- Kendiukhov (2025) — Developmental Interpretability review.

Bibliografía completa en [`docs/deep-research.md`](docs/deep-research.md).
