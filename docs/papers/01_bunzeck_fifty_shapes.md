---
cita: Bunzeck, B. & Zarrieß, S. (2024). *Fifty shapes of BLiMP: syntactic learning curves in language models are not uniform, but sometimes unruly.* CLASP.
link: https://aclanthology.org/2024.clasp-1.7.pdf
leído_por: _
fecha: _
prioridad: 🔴 crítica — base metodológica directa
---

# Notas: Fifty Shapes of BLiMP

## Contribución central
(Completar al leer)

## Método
- Modelos usados: Pythia (hasta 1.4B) + BabyLlama.
- Spacing de checkpoints: logarítmico.
- Métrica: ___
- Fit polinomial: grado 5 sobre eje log-tokens.
- Clasificación topológica: ___

## Hallazgo principal
- El promedio oculta trayectorias individuales rebeldes.
- Tipos de curva identificados: monótona, U, U invertida, oscilatoria.

## Cifras clave
- Proporción de paradigmas no monótonos: ___
- Paradigmas más propensos a U: ___

## Limitaciones reportadas
- ___

## Qué tomamos de acá
1. Fit polinomial grado 5 (criterio de clasificación).
2. Spacing logarítmico de checkpoints.
3. Análisis por paradigma individual, no agregado.

## Qué NO tomamos
- ___

## Cita sugerida en paper
> Bunzeck & Zarrieß (2024) mostraron que ajustar polinomios de grado 5 sobre curvas de aprendizaje sintáctico revela trayectorias no monótonas que el promedio temporal oculta.
