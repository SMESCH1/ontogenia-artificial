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
**Objetivo:** cerrar marco teórico, hipótesis operacionalizadas y repo organizado.

- [x] Scaffolding del repositorio (estructura, `.gitignore`, `pyproject.toml`, `requirements.txt`, README).
- [x] Plantilla ACL (`paper/`) con `main.tex` esqueleto en español.
- [x] `docs/01_project_overview.md` — resumen ejecutivo, hipótesis, scope.
- [x] `docs/02_theoretical_framework.md` — curvas-U, SLLN-LP, psicolingüística (+ explicación intuitiva del mecanismo en Pythia).
- [x] `docs/03_methodology_and_tech_stack.md` — modelos, datasets, stack.
- [x] `docs/04_experimental_design.md` — H1/H2/H3 operacionalizadas (+ notas "en concreto" para H1/H3).
- [x] `docs/05_human_alignment.md` — decisión Wordbank + tabla inicial (10 pares).
- [x] Plantillas `docs/papers/01-05_*.md` con estructura de notas de lectura.
- [x] `CLAUDE.md` — guía de arquitectura y comandos para Claude Code.
- [ ] Lectura dirigida de los 5 papers críticos y completar notas (templates creados, contenido pendiente).
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
- [x] `results/` con naming — `smoke/{size}_step{N}_{task}.json`, `sweep/{size}_step{N}.json` (ver `CLAUDE.md` / `AGENTS.md`).
- [x] `notebooks/02_trajectories.ipynb` operativo con `aggregated_metrics.parquet` + export `topology_summary.parquet`.
- [x] Clasificación topológica implementada en `src/ontogenia/topology.py` + comando `ontogenia topology`.
- [x] **Sweep completo BLiMP** — 72 archivos en `results/sweep/` (3 modelos × 24 checkpoints, sin `--limit`). Skip-existing implementado en CLI (`--force` para forzar re-run).
- [x] **Post-análisis** — `./scripts/run_post_analysis.sh` corrido: `aggregated_metrics.parquet` (4899 filas) + `topology_summary.parquet` generados.
- [ ] **Integrar Zorro** — portar YAML de tareas al formato de lm-eval (guía en `docs/integracion_zorro.md`); si no da el tiempo, dejarlo fuera del sweep y documentarlo como limitación.
- [ ] Buffer para re-correr ante artefactos.

### Semana 4 (12/05 – 18/05) — Análisis + escritura
- [x] `notebooks/03_human_correlation.ipynb` operativo + módulo `src/ontogenia/human_alignment.py` y comando `ontogenia human-alignment` (Spearman + bootstrap).
- [x] **`data/human_milestones.csv`** completo — 67 paradigmas BLiMP con AoA (meses), fuente y confianza (4 alta / 17 media / 46 baja).
- [x] **Paper v1** (`paper/main.tex`) — 6+ páginas, 0 placeholders, resultados reales H1/H2/H3, secciones completas.
- [x] **Figuras publicables** — `scripts/make_figures.py`: Fig1 trayectorias (2×3), Fig2 scatter AoA×3 modelos, Fig3 distribución topologías. PNG+PDF en `figures/`.
- [ ] **Si queda margen:** piloto de EWoK sobre el mejor Pythia identificado.

### Semana 5 (19/05 – 26/05) — Pulido + presentación

- [x] Pipeline completo mergeado a main (sweep BLiMP, figuras, paper v1).
- [ ] **Validar mapeo AoA** con Leandro y Hugo — 46/67 paradigmas tienen confianza "baja"; revisar especialmente los morfológicos para H3. Fuentes en `docs/05_human_alignment.md`.
- [ ] **Bibliografía** — completar `paper/custom.bib` con BibTeX reales (Biderman 2023, Bunzeck 2024, Frank 2017, Warstadt 2020 BLiMP, Hu 2024 BabyLM, Kendiukhov 2025).
- [ ] Instalar TeX (`sudo apt install texlive-full`) y compilar `paper/main.tex` a PDF; alternativa: Overleaf.
- [ ] Revisión interna cruzada del paper (abstract, resultados H2 negativo, discusión).
- [ ] Paper v2 (final) con bibliografía, figuras ajustadas post-validación AoA.
- [ ] Slides (~15 min).
- [ ] README reproducible con instrucciones end-to-end.
- [ ] Tag `v1.0-presentacion` el 25/05.
- [ ] **26/05: presentación.**

## Análisis representacional vía SAEs (deseable, no priorizado)

> Abordar sólo si el sweep conductual termina con margen. Estimación: ~10h.
> Documentación completa: [`docs/06_sae_representational_probes.md`](docs/06_sae_representational_probes.md).

**Hipótesis adicional (H4):** durante el valle de la curva-U conductual, los features
SAE sintácticos no decrecen — evidencia de disociación competencia/performance.
Referente metodológico: Kharazi et al. (2025) "The Birth of Knowledge" (arxiv 2505.19440).

- [x] `docs/06_sae_representational_probes.md` — motivación, workflow, estimación de esfuerzo.
- [x] `src/ontogenia/sae_probes.py` — scaffold: `load_sae`, `extract_activations`, `contrastive_features`, `track_features_across_checkpoints`.
- [ ] Fase 0 (~1-2h): instalar `sparsify` + identificar features sintácticos en checkpoint final.
- [ ] Fase 1 (~3-6h GPU): sweep longitudinal con `track_features_across_checkpoints()`.
- [ ] Fase 2 (~1h): plot conductual vs. representacional superpuestos + sección en paper.

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
