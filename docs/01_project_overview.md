# Project Overview — Ontogenia Artificial

## Resumen ejecutivo

Este proyecto investiga **cuándo y cómo emergen capacidades sintácticas específicas** durante el pre-entrenamiento de un modelo de lenguaje auto-regresivo, y **en qué medida el orden de esa emergencia se asemeja al de la adquisición humana del lenguaje**. Lo hace mediante una evaluación longitudinal sistemática de los checkpoints intermedios de la suite **Pythia** (EleutherAI) sobre baterías de pares mínimos sintácticos (BLiMP, Zorro), y un cruce posterior con datos humanos de edad de adquisición (AoA) provenientes de **CHILDES / Wordbank**.

Se enmarca en el campo de la ***Developmental Interpretability*** — el análisis dinámico y continuo del proceso de pre-entrenamiento, en contraposición a la evaluación estática *post-hoc* del modelo convergido (Kendiukhov 2025). Trabajos recientes (Bunzeck & Zarrieß 2024; ZhoBLiMP 2024) han demostrado que el promedio del rendimiento en la época final oculta trayectorias individuales "rebeldes": curvas en U, U invertida, oscilaciones prolongadas. Este proyecto replica esa observación sobre Pythia, pero suma un componente de **alineación con el desarrollo humano** que, hasta donde sabemos, no ha sido sistematizado en la misma grilla experimental.

## Problema

La evaluación dominante de LLMs reporta accuracy promedio al final del entrenamiento. Esto tiene dos costos:

1. **Oculta no-monotonicidades.** Una capacidad puede aparecer, degradarse por sobre-regularización y reaparecer — exactamente el patrón clásico de las curvas en U del desarrollo infantil (p. ej. *went → goed → went* en niños angloparlantes). Reportar sólo el valor final borra esa dinámica.
2. **Desvincula el modelado del lenguaje de la psicolingüística.** Las métricas usuales (accuracy, perplexity agregada) no son comparables con los datos longitudinales que la psicología del desarrollo genera sobre niños. Sin ese puente, el "paralelismo cognitivo" entre LLMs y humanos queda en el nivel anecdótico.

## Pregunta de investigación

> **¿Las trayectorias de adquisición sintáctica de Pythia a lo largo del pre-entrenamiento son no-monótonas a nivel de paradigma individual, y el orden relativo de emergencia de esos paradigmas correlaciona con el orden de adquisición documentado en niños humanos?**

## Hipótesis

Tres predicciones falsables (detalle y métricas en `04_experimental_design.md`):

- **H1 — Existencia de curvas no monótonas.** ≥30% de los paradigmas de BLiMP/Zorro muestran una caída ≥5 puntos porcentuales seguida de recuperación en Pythia-160m y/o Pythia-410m.
- **H2 — Alineación ordinal con humanos.** El ranking de paradigmas por *step de estabilización* en Pythia correlaciona (Spearman ρ > 0.3, p < 0.05) con el ranking humano de AoA para los fenómenos análogos.
- **H3 — Excepciones morfológicas.** La magnitud de la curva-U es mayor en paradigmas que involucran excepciones morfológicas (verbos irregulares, plurales irregulares) que en paradigmas puramente estructurales (acuerdo sujeto-verbo, ligamiento).

## Aporte esperado

1. Una **tabla de mapeo** reproducible entre paradigmas de BLiMP/Zorro e hitos de adquisición infantil, con referencia bibliográfica y fuente de AoA (contribución metodológica).
2. Curvas de aprendizaje por paradigma (eje X = tokens ingeridos, eje Y = SLLN-LP) para 3 tamaños de Pythia × ~30-40 checkpoints, clasificadas por topología (monótona, U, U invertida, oscilatoria) replicando Bunzeck & Zarrieß.
3. Un **coeficiente de correlación Pythia ↔ humano** con intervalos de confianza por bootstrap, como medida cuantitativa — y sin pretensiones de equivalencia temporal — de la analogía ontogenética.
4. Repo reproducible (`lm-evaluation-harness` + configs YAML + notebooks) y paper corto formato ACL en español.

## Scope

**Dentro del scope:**
- Pythia-14m, 160m, 410m deduplicadas; ~30-40 checkpoints por modelo.
- Sintaxis inglesa vía pares mínimos (BLiMP completo + subsets de Zorro).
- Correlación con AoA de Wordbank (inglés) y/o subsets de CHILDES.
- **Extensión deseable si da el tiempo:** piloto sobre EWoK (adquisición léxica / mundo físico) sobre el mejor Pythia identificado.

**Fuera del scope (explícitamente excluido):**
- Singular Learning Theory, estimación del Coeficiente de Aprendizaje Local (LLC). Se mencionará como línea futura.
- Interpretabilidad mecanicista de circuitos (induction heads, IOI).
- Teoría de la Mente / BigToM.
- Razonamiento lógico proposicional (SimpleLogic).
- Entrenamiento de modelos desde cero o fine-tuning.

## Equipo y cronograma

- **Equipo:** 2–3 personas.
- **Cómputo:** GPU local tipo RTX 3090/4090.
- **Deadline final:** 26 de mayo de 2026 (presentación, Clase 12).
- **Idioma de entregables:** español (paper, slides, docs).
- **Roadmap completo:** ver plan en `/home/sebas/.claude/plans/onbordeate-a-este-proyecto-precious-honey.md` o resumen en `03_methodology_and_tech_stack.md`.

## Navegación de la documentación

| Doc | Contenido |
| --- | --- |
| `01_project_overview.md` *(este archivo)* | Resumen ejecutivo, problema, hipótesis, scope. |
| `02_theoretical_framework.md` | Marco teórico: curvas-U, psicolingüística del desarrollo, breve contexto de SLT. |
| `03_methodology_and_tech_stack.md` | Decisiones concretas: modelos, datasets, librerías, equipo. |
| `04_experimental_design.md` | Operacionalización de H1/H2/H3, métricas, análisis estadístico. |
| `05_human_alignment.md` | Fuente de AoA y tabla de mapeo paradigma ↔ hito infantil. |
| `deep-research.md` | Informe de deep research con SoTA completo y bibliografía. |
| `papers/` | Notas de lectura por paper crítico. |
