# Ontogenia Artificial

**Evaluación longitudinal de la emergencia sintáctica en Pythia y su alineación con la adquisición humana del lenguaje.**

Research Project para la materia **Procesamiento de Lenguaje Natural** (UDESA, 1er semestre 2026, Luciano Del Corro).

## Pregunta de investigación

¿Las trayectorias de adquisición sintáctica de un modelo de lenguaje auto-regresivo (Pythia) a lo largo de su pre-entrenamiento exhiben curvas no monótonas (incluyendo curvas en U) análogas a las documentadas en la adquisición del lenguaje infantil? ¿El orden relativo de emergencia de distintos fenómenos sintácticos en el modelo correlaciona con el orden de adquisición en niños?

## Enfoque

- **Modelo:** Suite Pythia (EleutherAI) — 154 checkpoints públicos, mismo orden exacto de datos. Variantes: `pythia-14m`, `pythia-160m`, `pythia-410m` (deduplicadas).
- **Evaluación:** Pares mínimos sintácticos (BLiMP + Zorro) corridos vía `lm-evaluation-harness`.
- **Métrica principal:** *Pairwise Accuracy* estándar (`acc,none`) del harness (log-probabilidades conjuntas).
- **Métrica de robustez y control:** *Sub-linear Length Normalized Log-Probabilities* (SLLN-LP) con α = 0.5 (adaptada de Bunzeck & Zarrieß 2024), implementada a nivel de ítem para análisis de sensibilidad y control de longitud.
- **Alineación humana:** correlación Spearman entre el orden de estabilización de paradigmas en Pythia y la edad de adquisición reportada en CHILDES / Wordbank.

## Estado

**Proyecto completo — presentado el 26/05/2026.**

Pipeline completo: sweep BLiMP (3 modelos × 24 checkpoints × 67 paradigmas), clasificación topológica, correlación Spearman con AoA humana, figuras publicables y paper ACL finalizado. Resultados: H1 confirmada (76–91 % paradigmas no monótonos), H2 con señal positiva en subconjunto refinado (ρ = +0.169, N=31), H3 confirmada con alta significancia (Mann-Whitney p < 0.005 en 410M).

## Estructura

```
docs/papers/    Notas de lectura de los 5 papers clave
docs/slides_*   Slides de la presentación (HTML + Marp)
paper/          Manuscrito LaTeX finalizado (formato ACL, español)
src/ontogenia/  Pipeline: CLI, checkpoints Pythia, SLLN-LP, topología, alineación humana
configs/        YAMLs de tareas custom para lm-evaluation-harness
notebooks/      Trayectorias (02) y correlación humana (03)
scripts/        run_smoke.sh, run_full_sweep.sh, run_post_analysis.sh, make_figures.py
tests/          Tests unitarios (métricas)
data/           human_milestones.csv (AoA refinado) + human_milestones_refined.csv
figures/        Figuras PNG/PDF generadas para el paper
results/        JSONs + Parquet de evaluación (gitignored)
```

## Autores

Sebastián Mesch Henriques, Leandro Miguel Carcagno, Hugo Alejandro Cabaña.
Materia NLP — UDESA, 1er semestre 2026 (prof. Luciano Del Corro).

## Referencias clave

- Bunzeck & Zarrieß (2024). *Fifty shapes of BLiMP: syntactic learning curves in language models are not uniform, but sometimes unruly.* CLASP.
- Biderman et al. (2023). *Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling.* ICML.
- Hu et al. (2024). *Findings of the Second BabyLM Challenge.* CoNLL.
- Bibliografía completa en [`docs/deep-research.md`](docs/deep-research.md).
