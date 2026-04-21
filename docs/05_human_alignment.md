# Alineación con la adquisición humana del lenguaje

## 1. Decisión de fuente

Evaluamos tres candidatos para obtener **Edad de Adquisición (AoA)** por fenómeno:

| Fuente | Formato | Cobertura | Costo de uso | Idoneidad |
| --- | --- | --- | --- | --- |
| **Wordbank** (Frank et al. 2017) | Tablas tabulares por ítem, % de niños que producen/comprenden por edad | Fuerte en vocabulario; limitada en sintaxis | Bajo — CSVs descargables o `wordbankr` en R | **Alta** para ítems léxicos; **media** para sintaxis |
| **CHILDES** (MacWhinney 2000) | Transcripciones longitudinales crudas | Muy amplia; requiere análisis para extraer AoA | Alto — hay que construir el análisis | Alta pero costosa |
| **Braginsky et al. (2019)** | Tabla agregada multilingüe sobre Wordbank | Síntesis cuantitativa ya hecha | Bajo | Alta — ideal para importar AoA directo |

**Decisión operativa.** Partimos de **Wordbank (inglés)** como fuente principal, complementado con hallazgos específicos de la literatura de psicolingüística para los fenómenos que Wordbank no cubre directamente (p. ej. acuerdo sujeto-verbo, anáforas). Cada entrada en la tabla de mapeo lleva **cita bibliográfica** explícita.

No usamos CHILDES crudo en esta iteración; queda como línea de profundización.

## 2. Problema metodológico: BLiMP es sintaxis, Wordbank es léxico

BLiMP evalúa competencia sintáctica mediante juicios de aceptabilidad, mientras que Wordbank mide producción léxica. El cruce directo es imperfecto por definición. Nuestra estrategia tiene tres pistas:

1. **Proxy léxico para fenómenos morfológicos.** Para los paradigmas que involucran morfología irregular (p. ej. `irregular_past_tense`), usamos la AoA del ítem irregular específico (p. ej. edad a la que el niño produce *went*) como proxy. La justificación es que los niños no producen la regla antes de dominar los ítems frecuentes.
2. **Literatura específica para fenómenos estructurales.** Para acuerdo, anáforas, islas, citamos estudios longitudinales específicos (CHILDES-based) que reportan rangos de edad de emergencia.
3. **Clasificación gruesa donde no hay dato fino.** Cuando sólo hay evidencia cualitativa ("emerge alrededor de los 3-4 años"), usamos la mediana del rango como estimador, pero marcamos la entrada como **baja confianza** para excluirla de análisis robustez.

## 3. Tabla de mapeo (borrador inicial — a completar en Semana 1)

**Nota:** tabla inicial con 10 pares de ejemplo. Se expandirá a ≥15 en la versión final y se validará cruzadamente entre miembros del equipo.

| # | Paradigma (BLiMP/Zorro) | Fenómeno lingüístico | AoA humano (meses) | Fuente | Confianza |
| --- | --- | --- | --- | --- | --- |
| 1 | `irregular_past_tense` (BLiMP) | Morfología verbal irregular | ~36–48 | Marcus et al. 1992; Wordbank (ítems *went, ate, ran*) | Alta |
| 2 | `regular_plural_subject_verb_agreement` (BLiMP) | Concordancia SV plural regular | ~30–36 | Brown 1973; CHILDES | Media |
| 3 | `irregular_plural_subject_verb_agreement` (BLiMP) | Concordancia SV plural irregular | ~48–60 | Rispoli 2005 | Media |
| 4 | `anaphor_gender_agreement` (BLiMP) | Concordancia género con anáfora | ~40–54 | Wexler 1994 | Baja |
| 5 | `anaphor_number_agreement` (BLiMP) | Concordancia número con anáfora | ~36–48 | Wexler 1994 | Baja |
| 6 | `determiner_noun_agreement_1` (BLiMP) | Concordancia determinante-sustantivo | ~24–30 | Brown 1973 | Media |
| 7 | `wh_questions_object_gap` (BLiMP) | Formación de preguntas wh- | ~36–42 | Stromswold 1995 | Media |
| 8 | `principle_A_c_command` (BLiMP) | Principio A (ligamiento) | ~60+ | Chien & Wexler 1990 | Baja |
| 9 | `subject-verb agreement` (Zorro) | Concordancia SV (vocabulario infantil) | ~30–36 | Brown 1973; Wordbank ítems verbales | Alta |
| 10 | `filler-gap dependencies` (Zorro) | Dependencias de larga distancia | ~48–60 | Goodluck 1991 | Baja |

## 4. Tratamiento estadístico de la tabla

- Cuando haya rango (p. ej. 36–48 meses), usamos el **punto medio** para el ranking principal y reportamos la robustez re-corriendo con los extremos.
- Se computa **Spearman ρ** sobre el ranking completo **y sobre el subconjunto de confianza alta+media únicamente** (análisis de sensibilidad).
- Se excluyen de H2 los paradigmas sin análogo humano identificable; se reportan cuántos quedan y por qué.

## 5. Fuentes y links

- Wordbank: <https://wordbank.stanford.edu>
- Braginsky, M. et al. (2019). *Consistency and variability in children's word learning across languages*. Open Mind.
- Brown, R. (1973). *A First Language: The Early Stages*. Harvard.
- Chien, Y.-C. & Wexler, K. (1990). Lang Acq.
- Marcus, G. et al. (1992). *Overregularization in language acquisition*. Monographs SRCD.
- Rispoli, M. (2005). *When children reach beyond their grasp: Why some children make pronoun case errors and others don't*. Journal of Child Language.
- Stromswold, K. (1995). *The acquisition of subject and object wh-questions*. Language Acquisition.
- Wexler, K. (1994). *Optional infinitives, head movement, and the economy of derivations in child grammar*.

## 6. Tareas pendientes de esta fase

- [ ] Validar acceso efectivo a Wordbank CSV y/o `wordbankr`.
- [ ] Expandir tabla a ≥15 pares con consenso del equipo.
- [ ] Generar `data/human_milestones.csv` versionado con las columnas: `paradigm, phenomenon, aoa_months_low, aoa_months_high, source, confidence`.
