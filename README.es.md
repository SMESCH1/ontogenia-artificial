# Ontogenia Artificial

🌐 **Idioma:** [English](README.md) · **Español**

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

## Resultados principales

| Hipótesis | Resultado |
| --- | --- |
| **H1** — Existencia de curvas no monótonas | ✅ Confirmada: **76–91 %** de los paradigmas son no monótonos |
| **H2** — Alineación ordinal con humanos | 🟡 Señal positiva en subconjunto refinado: **ρ = +0.169** (N = 31) |
| **H3** — Curvas-U más profundas en morfología irregular | ✅ Confirmada con alta significancia: **Mann-Whitney p < 0.005** (410M) |

📄 **Paper completo (formato ACL, español):** [`paper/main.pdf`](paper/main.pdf)

![Trayectorias sintácticas](figures/fig1_trajectories.png)

## Cómo reproducir

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .

# Prueba rápida
./scripts/run_smoke.sh --limit 20            # → results/smoke/

# Sweep completo (GPU, ~24h/modelo)
./scripts/run_full_sweep.sh                  # → results/sweep/

# Post-análisis: agregación + topología + alineación humana
./scripts/run_post_analysis.sh

# Tests
python -m unittest discover -s tests -v
```

## Estructura

```
docs/           Marco teórico y diseño experimental (documentos de la fase de planificación)
docs/papers/    Notas de lectura de los 5 papers clave
paper/          Manuscrito LaTeX finalizado (formato ACL, español) + main.pdf
src/ontogenia/  Pipeline: CLI, checkpoints Pythia, SLLN-LP, topología, alineación humana
configs/        YAMLs de tareas custom para lm-evaluation-harness
notebooks/      Trayectorias (02) y correlación humana (03)
scripts/        run_smoke.sh, run_full_sweep.sh, run_post_analysis.sh, make_figures.py
tests/          Tests unitarios (métricas, agregación, topología, alineación)
data/           human_milestones.csv (AoA) + wordbank_item_data.csv
figures/        Figuras PNG/PDF generadas para el paper
results/        JSONs + Parquet de evaluación (gitignored)
```

> **Nota:** los documentos en `docs/0X_*.md` son de la **fase de planificación** y reflejan el diseño original (donde SLLN-LP figuraba como métrica principal). La decisión metodológica final —Pairwise Accuracy como métrica principal y SLLN-LP como control de robustez— está documentada en el [paper](paper/main.pdf).

## Hipótesis y diseño experimental

Ver [`ROADMAP.md`](ROADMAP.md) y [`docs/04_experimental_design.md`](docs/04_experimental_design.md) para las hipótesis operacionalizadas.

## Autores

Sebastián Mesch Henriques, Leandro Miguel Carcagno, Hugo Alejandro Cabaña.
Materia NLP — UDESA, 1er semestre 2026 (prof. Luciano Del Corro).

## Referencias clave

- Bunzeck & Zarrieß (2024). *Fifty shapes of BLiMP: syntactic learning curves in language models are not uniform, but sometimes unruly.* CLASP.
- Biderman et al. (2023). *Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling.* ICML.
- Hu et al. (2024). *Findings of the Second BabyLM Challenge.* CoNLL.
- Frank et al. (2017). *Wordbank: An open repository for developmental vocabulary data.*
- Bibliografía completa en [`docs/deep-research.md`](docs/deep-research.md).

## Licencia

Código bajo licencia [MIT](LICENSE). Los datasets de terceros (BLiMP, Zorro, Wordbank/CHILDES) y los modelos Pythia conservan sus respectivas licencias. El material de cátedra no se incluye en este repositorio.
