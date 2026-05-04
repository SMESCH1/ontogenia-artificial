# Integración de Zorro

**Estado:** en `lm-eval` “vanilla” de EleutherAI **no** aparece un grupo de tareas `zorro` listado por defecto. BLiMP sí está integrado (`blimp`, `blimp_*`).

## Opciones recomendadas

### A) Portar YAMLs desde BabyLM 2024 (preferido para homogeneizar)

El repositorio [babylm/evaluation-pipeline-2024](https://github.com/babylm/evaluation-pipeline-2024) incluye configuraciones de evaluación usadas en el challenge, incluyendo tareas orientadas a adquisición. Pasos típicos:

1. Clonar el repo del pipeline y localizar definiciones de tarea bajo su árbol `lm_eval/tasks/` (o equivalente en esa versión).
2. Copiar/adaptar los YAML + rutas de datos al directorio `configs/` de Ontogenia y registrar la tarea según la [guía de nuevas tareas](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/new_task_guide.md) del harness (variable de entorno `LM_EVAL_TASKS_PATH` o instalación editable con tasks extra).
3. Validar con `lm-eval ls tasks` y un `ontogenia smoke --task <nueva_tarea> --limit 5`.

### B) Ejecutar evaluaciones Zorro desde el fork BabyLM

Mantener dos entornos o scripts: uno con el pipeline BabyLM solo para Zorro, y consolidar métricas a mano en el mismo esquema JSON (`meta` + resultados) que usa Ontogenia.

### C) Dataset original

Los estímulos y metadatos están en [phueb/Zorro](https://github.com/phueb/Zorro). Cualquier tarea custom debe apuntar a los mismos pares mínimos y documentar la versión del commit usada.

## Nota

Hasta que Zorro esté registrado en el harness del proyecto, el comando `ontogenia sweep --tasks blimp` cubre la parte principal del diseño experimental; Zorro queda como extensión explícita del roadmap.
