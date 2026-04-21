---
cita: Bunzeck et al. (2024). *ZhoBLiMP: A Systematic Assessment of Language Models with Linguistic Minimal Pairs in Chinese.* ArXiv 2411.06096.
link: https://arxiv.org/html/2411.06096v2
leído_por: _
fecha: _
prioridad: 🔴 crítica — fórmula exacta de SLLN-LP
---

# Notas: ZhoBLiMP / SLLN-LP

## Contribución central
- BLiMP-style para chino (morfológicamente distinto al inglés).
- Análisis longitudinal sobre 47 checkpoints de Pythia.
- **Propone SLLN-LP** — métrica que neutraliza sesgos de longitud en pares mínimos.

## SLLN-LP (fórmula exacta — VERIFICAR al leer)
$$\text{SLLN-LP}(s) = \frac{\log P(s)}{|s|^{\alpha}}$$

- $\alpha \in (0, 1)$, recomendado ~0.5.
- Corrige tanto la probabilidad cruda ($\alpha=0$) como la probabilidad normalizada por longitud completa ($\alpha=1$), que introducen sesgos opuestos.

## Por qué importa para nosotros
Pares mínimos donde las dos sentencias difieren en cantidad de tokens (caso típico con morfología irregular: *went* es 1 token, *goed* pueden ser 2) tienen sesgo de longitud. SLLN-LP lo mitiga.

## Validación que haremos
- Implementar SLLN-LP en `src/ontogenia/metrics.py`.
- Reproducir al menos **una** cifra del paper de ZhoBLiMP como sanity check.
- Análisis de sensibilidad: correr con $\alpha \in \{0.3, 0.5, 0.7\}$ y reportar.

## Cita sugerida
> Para mitigar el sesgo de longitud entre pares mínimos que difieren en cantidad de tokens, utilizamos la métrica SLLN-LP propuesta por Bunzeck et al. (2024), una normalización sub-lineal de la log-probabilidad que corrige simultáneamente las patologías de la probabilidad cruda y de la probabilidad per-token.

## Dudas a resolver al leer
- ¿Cómo eligen α empíricamente?
- ¿Validan SLLN-LP contra BLiMP inglés también?
