---
cita: Biderman, S., Schoelkopf, H. et al. (2023). *Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling.* ICML.
link: https://arxiv.org/pdf/2304.01373
leído_por: _
fecha: _
prioridad: 🔴 crítica — infraestructura base
---

# Notas: Pythia Suite

## Contribución central
Suite de modelos de 14M a 12B con:
- **Mismo orden exacto de datos** (The Pile).
- Uniformidad de hiperparámetros.
- **154 checkpoints** públicos por variante.
- Variantes deduplicadas y no-deduplicadas.

## Ventajas para nuestra investigación
1. Control sobre la variable temporal: cualquier diferencia entre checkpoints se debe a aprendizaje, no al orden.
2. Acceso vía HuggingFace con `revision=stepN`.
3. Tamaños pequeños accesibles en GPU única.

## Parámetros relevantes
- Pasos totales: 143 000.
- Checkpoints tempranos (denso): step0, step1, step2, step4, ..., step512 (log2).
- Checkpoints tardíos (lineal): cada 1000 pasos.
- Tokens por paso × batch_size: ___ (completar al leer).

## Detalles técnicos de reproducibilidad
- Seeds: ___
- Tokenizer: GPT-NeoX-20B tokenizer.
- Formato: SafeTensors + config.json estándar de HF.

## Cita sugerida
> La suite Pythia (Biderman et al., 2023) provee los checkpoints necesarios para un análisis longitudinal determinista, al conservar el orden exacto del corpus de pre-entrenamiento y exponer 154 revisiones por tamaño.

## Pitfalls documentados
- ___
