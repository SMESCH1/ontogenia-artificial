---
cita: Hu, M. Y., Mueller, A. et al. (2024). *Findings of the Second BabyLM Challenge: Sample-Efficient Pretraining on Developmentally Plausible Corpora.* CoNLL.
link: https://arxiv.org/html/2412.05149v1
leído_por: _
fecha: _
prioridad: 🟡 media — pipeline de evaluación
---

# Notas: BabyLM Challenge 2024

## Contribución central
- Competencia de pre-entrenamiento bajo restricciones de datos bio-plausibles (10M / 100M palabras).
- Corpus centrado en lenguaje dirigido al niño (CDS), historias simples.
- Métrica estandarizada: perplejidad + precisión en BLiMP/Zorro/EWoK.

## Qué tomamos
1. El **pipeline de evaluación** (fork de LM-Eval Harness con Zorro y EWoK integrados).
2. Conjunto de hiperparámetros y batch sizes para modelos pequeños.
3. Marcos de evaluación ya validados (no reimplementar).

## Qué NO tomamos
- Entrenamiento desde cero de modelos BabyLM (fuera de scope).
- Restricciones de datos (nuestro objeto es Pythia, entrenado en web abierta).

## EWoK (Elements of World Knowledge)
- Evalúa física intuitiva, referencias espaciales, pragmática.
- Formato multiple-choice zero-shot con log-probs.
- **Uso en nuestro proyecto:** piloto opcional si el tiempo lo permite.

## Cita sugerida
> El pipeline de evaluación del BabyLM Challenge 2024 (Hu et al., 2024) integra nativamente Zorro y EWoK como tareas zero-shot sobre log-probabilidades, constituyendo el estándar actual para comparar modelos pre-entrenados en régimen dev-plausible.

## Pendiente
- ¿Su fork incluye Zorro listo o hay que portarlo a LM-Eval Harness vanilla?
- ¿Hay configs YAML reutilizables?
