# Diseño Experimental

## 1. Variables del experimento

- **Variable independiente principal:** `training_step` de Pythia (1 ≤ step ≤ 143000, con 154 revisions públicas).
- **Variables independientes secundarias:** tamaño del modelo (`{14m, 160m, 410m}`), familia de tarea (`{BLiMP, Zorro}`), paradigma sintáctico (p. ej. `anaphor_agreement`, `irregular_past_tense`).
- **Variable dependiente:** SLLN-LP relativa entre el par mínimo gramatical y agramatical, agregada por paradigma.

## 2. Grilla experimental

| Dimensión | Valores |
| --- | --- |
| Modelos | `pythia-14m-deduped`, `pythia-160m-deduped`, `pythia-410m-deduped` |
| Checkpoints | `{step0, step1, step2, step4, step8, step16, step32, step64, step128, step256, step512, step1000, step2000, step4000, step8000, step16000, step32000, step48000, step64000, step80000, step96000, step112000, step128000, step143000}` — 24 checkpoints, densos en fase temprana (log2) y lineales en fase tardía |
| Tareas | BLiMP (67 paradigmas) + Zorro (13 paradigmas) = 80 paradigmas × ~1000 items cada uno |
| Métrica principal | SLLN-LP con α=0.5 |
| Métrica secundaria | mean_LP; accuracy |

**Volumen total:** 3 modelos × 24 checkpoints × 80 paradigmas × ~1000 items ≈ **5.8M forward passes**, pero con prompts cortos y batch_size=16-32 estimamos <24h por modelo en 3090/4090 (a estimar empíricamente en Semana 2).

## 3. Hipótesis operacionalizadas

### H1 — Existencia de curvas no monótonas

**Definición operacional.** Para cada paradigma `p` y modelo `m`, se computa la serie temporal $a_p(t)$ = accuracy del paradigma p en el checkpoint t. Se ajusta un polinomio de grado 5 a $a_p(t)$ sobre el eje log-tokens (siguiendo Bunzeck & Zarrieß 2024). El paradigma se clasifica como:

- **No monótono** si: existen puntos $t_1 < t_2 < t_3$ tales que $a_p(t_1) < a_p(t_2) - 0.05$ y $a_p(t_2) - a_p(t_3) \geq 0.05$ (curva U invertida), o simétricamente para curva en U.
- **Monótono** en caso contrario.

**Predicción.** En `pythia-160m-deduped` y `pythia-410m-deduped`, al menos el **30%** de los paradigmas son clasificados como no monótonos. **Falsado** si la proporción es ≤ 20%.

**Test estadístico.** Prueba binomial contra H₀ = 20%.

### H2 — Alineación ordinal con adquisición humana

**Definición operacional.**
- Para cada paradigma con análogo humano disponible (ver `05_human_alignment.md`), definimos `step_stabilize(p, m)` = el primer checkpoint en el que el paradigma alcanza ≥90% de su valor asintótico y se mantiene (ventana de 3 checkpoints consecutivos).
- Definimos `AoA(p)` = edad en meses a la que ≥50% de los niños dominan el fenómeno análogo según Wordbank o la fuente equivalente.
- Se rankean ambos vectores y se calcula **Spearman ρ** entre los rankings.

**Predicción.** ρ > 0.3 con p < 0.05, en al menos uno de los dos modelos más grandes (160m, 410m).

**Intervalos de confianza.** Bootstrap no paramétrico con 10 000 remuestreos.

### H3 — Excepciones morfológicas más propensas a U

**Definición operacional.** Se parte la muestra de paradigmas en dos grupos:
- **Grupo M (morfológico-irregular):** paradigmas que involucran excepciones morfológicas (p. ej. `irregular_past_tense`, `irregular_plural_subject_verb_agreement`).
- **Grupo S (estructural):** paradigmas puramente estructurales (p. ej. `anaphor_gender_agreement`, `wh_questions_object_gap`).

Para cada paradigma no monótono se calcula la **profundidad de U** `depth_U(p)` = diferencia entre el máximo local previo a la caída y el mínimo local. Se testea $H_0: \text{depth}_M = \text{depth}_S$ con U de Mann-Whitney.

**Predicción.** `depth_M > depth_S` con p < 0.05.

## 4. Plan de análisis

1. **Pre-procesamiento.** Cargar `results/*.json`, parsear log-probs, computar SLLN-LP por item, agregar a nivel paradigma por checkpoint.
2. **EDA.** `notebooks/01_eda_checkpoints.ipynb`: inspección de sanity checks (accuracy por checkpoint debería crecer en promedio; verificar que step0 ≈ chance).
3. **Trayectorias.** `notebooks/02_trajectories.ipynb`: una figura por paradigma, 3 líneas (una por tamaño), eje X log-tokens, eje Y SLLN-LP. Fit polinómico grado 5. Clasificación topológica.
4. **Alineación humana.** `notebooks/03_human_correlation.ipynb`: join con tabla de AoA, rankings, Spearman con bootstrap, scatter-plot ranking-Pythia vs ranking-humano.
5. **Tabla resumen.** Paradigma × tipo de curva × step de estabilización × AoA humano análogo × fuente.

## 5. Reproducibilidad

- Semillas fijas donde aplique (el sweep principal es determinista: no hay sampling).
- Versiones congeladas en `requirements.txt`.
- Resultados crudos (JSON) versionados por `git-lfs` si no exceden 2 GB; de lo contrario, release de GitHub con los artefactos.
- Un único script `scripts/run_full_sweep.sh` que reproduce todo desde pesos descargados.

## 6. Plan de contingencia (scope degradable)

Si en Semana 3 el sweep completo no es viable (por tiempo de cómputo, bug persistente, caída de miembros del equipo), degradamos **en este orden**:

1. Reducir a **1 modelo** (pythia-160m-deduped) — ahorra ~67% del cómputo.
2. Reducir a **Zorro solamente** (13 paradigmas vs 80) — ahorra ~80% adicional.
3. Reducir a **15 checkpoints** (vs 24) — ahorra ~40% adicional.
4. Si ninguna de las anteriores alcanza: restringir a **H1 + H3** (no-monotonicidad y excepciones morfológicas), dejando H2 (alineación humana) como discusión cualitativa.

El peor caso (todas las degradaciones aplicadas) sigue siendo un experimento publicable con una pregunta clara.

## 7. Entregables de esta fase de diseño

- [ ] Tabla definitiva de paradigmas a incluir en H3 (partición M vs S), firmada por el equipo.
- [ ] Vector final de checkpoints (24 propuestos; puede ajustarse tras el smoke test).
- [ ] Tabla inicial de mapeo paradigma ↔ hito humano en `05_human_alignment.md`.
