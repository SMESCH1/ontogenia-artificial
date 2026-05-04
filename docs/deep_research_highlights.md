# Highlights de `deep-research.md` — Ontogenia Artificial

Resumen operativo de [deep-research.md](deep-research.md) para este proyecto (longitudinal, Pythia, BLiMP/Zorro, sin SLT/ToM en scope).

## Marco

1. **Developmental interpretability:** dejar de evaluar solo el modelo final y estudiar **cómo y cuándo** emergen capacidades a lo largo del pre-entrenamiento (síntesis Kendiukhov 2025 en el informe).
2. **Pythia como oro estándar:** mismo **orden de datos** (The Pile), **154 checkpoints** por tamaño; las diferencias entre `revision=stepN` se atribuyen a la dinámica de aprendizaje, no a reordenar el corpus (Biderman et al. 2023).

## Evidencia directa para H1 / diseño BLiMP

3. **Fifty shapes of BLiMP (Bunzeck & Zarrieß, ACL 2024):** trayectorias por paradigma en checkpoints de Pythia (y otros) — **no monótonas**, curvas en U, invertidas y oscilaciones; el rendimiento **promedio al final** oculta esa estructura.
4. **ZhoBLiMP / SLLN-LP (Bunzeck et al., 2024):** métrica de log-prob **normalizada sub-linealmente por longitud** para pares mínimos donde las dos oraciones difieren en tokens; reduce sesgo de longitud frente a prob. cruda o a normalización lineal pura.

## Alineación humana y benchmarks (contexto H2 / extensiones)

5. Literatura que relaciona **surprisal / estabilización en LMs** con **adquisición humana** (referencias Chang & Bergen, Evanson, Haga citadas en el marco teórico del repo).
6. **Wordbank / CHILDES** como fuentes cuantitativas de AoA (tabla de mapeo en `05_human_alignment.md`).
7. **BabyLM 2024** y pipeline de evaluación: referencia para **Zorro / EWoK** y evaluación zero-shot reproducible (Hu et al. 2024; repo `babylm/evaluation-pipeline-2024`).

## Fuera del scope de Ontogenia (pero citados en deep-research)

- **SLT / LLC** y compresibilidad–MDL: marco complementario; el curso explícitamente no estima LLC.
- **Interpretabilidad mecanicista** (induction heads, data attribution): útil como contexto, no como deliverable.
- **ToM / BigToM, SimpleLogic, DevBench:** benchmarks vecinos, no parte de la grilla principal.

## Papers “ancla” (URLs en [enlaces_herramientas.md](enlaces_herramientas.md))

| Rol | Documento |
| --- | --- |
| Suite y checkpoints | Pythia (ICML 2023) |
| Curvas no monótonas BLiMP | Fifty shapes of BLiMP (ACL 2024) |
| Métrica SLLN-LP | ZhoBLiMP (arXiv 2411.06096) |
| Campo | Developmental interpretability review (arXiv 2508.15841) |

Integración práctica de Zorro: [integracion_zorro.md](integracion_zorro.md).
