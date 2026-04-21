# Ontogenia Artificial

**Evaluación longitudinal de la emergencia sintáctica en Pythia y su alineación con la adquisición humana del lenguaje.**

Research Project para la materia **Procesamiento de Lenguaje Natural** (UDESA, 1er semestre 2026, Luciano Del Corro).

## Pregunta de investigación

¿Las trayectorias de adquisición sintáctica de un modelo de lenguaje auto-regresivo (Pythia) a lo largo de su pre-entrenamiento exhiben curvas no monótonas (incluyendo curvas en U) análogas a las documentadas en la adquisición del lenguaje infantil? ¿El orden relativo de emergencia de distintos fenómenos sintácticos en el modelo correlaciona con el orden de adquisición en niños?

## Enfoque

- **Modelo:** Suite Pythia (EleutherAI) — 154 checkpoints públicos, mismo orden exacto de datos. Variantes: `pythia-14m`, `pythia-160m`, `pythia-410m` (deduplicadas).
- **Evaluación:** Pares mínimos sintácticos (BLiMP + Zorro) corridos vía `lm-evaluation-harness`.
- **Métrica principal:** *Sub-linear Length Normalized Log-Probabilities* (SLLN-LP), adaptada de Bunzeck & Zarrieß 2024 y ZhoBLiMP.
- **Alineación humana:** correlación Spearman entre el orden de estabilización de paradigmas en Pythia y la edad de adquisición reportada en CHILDES / Wordbank.

## Estado

🚧 **Semana 1 de 5** — research + scaffolding. Aún sin código ejecutable.

Ver [`docs/`](docs/) para el marco teórico, el diseño experimental y el plan semanal.

## Estructura

```
docs/           Marco teórico, diseño experimental, notas bibliográficas
paper/          Manuscrito LaTeX (formato ACL, español)
src/ontogenia/  Código del pipeline de evaluación (próximamente)
configs/        YAMLs de tareas custom para lm-evaluation-harness
notebooks/      Análisis exploratorio y visualización de trayectorias
scripts/        Orquestación del sweep sobre checkpoints
data/           Datasets humanos curados (gitignored en /raw y /processed)
results/        JSONs de evaluación por (modelo, checkpoint, tarea) (gitignored)
```

## Deadline

Presentación final: **26 de mayo de 2026**.

## Autores

Por completar.

## Referencias clave

- Bunzeck & Zarrieß (2024). *Fifty shapes of BLiMP: syntactic learning curves in language models are not uniform, but sometimes unruly.* CLASP.
- Biderman et al. (2023). *Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling.* ICML.
- Hu et al. (2024). *Findings of the Second BabyLM Challenge.* CoNLL.
- Bibliografía completa en [`docs/deep-research.md`](docs/deep-research.md).
