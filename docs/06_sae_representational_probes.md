# Análisis Representacional vía SAEs (Deseable)

> **Estado:** Deseable, no priorizado. Se puede abordar si queda margen después del
> sweep conductual (BLiMP/Zorro) de Semana 3.
> Estimación total: ~10h (GPU + análisis). Ver sección "Estimación de esfuerzo".

## Motivación

El análisis conductual (BLiMP/Zorro) mide **comportamiento externo**: si Pythia
asigna mayor log-probabilidad a la oración gramatical que a la agramatical. Pero no
responde: ¿el modelo **codifica internamente** información sintáctica incluso cuando
falla en el output?

Esta pregunta es el corazón del debate competencia/performance en adquisición del
lenguaje. Durante la sobreregularización (el "valle" de la curva-U), ¿el niño olvidó
la forma irregular, o la tiene almacenada pero algo en la producción interfiere? Los
**Sparse Autoencoders (SAEs)** permiten atacar la misma pregunta en Pythia: descomponen
las activaciones internas en features interpretables, y se puede verificar si los
features asociados a un paradigma sintáctico permanecen activos aunque el comportamiento
externo baje.

Referencia conceptual: Anthropic publicó los *Natural Language Autoencoders* (NLAEs,
transformer-circuits.pub/2026/nla), que hacen algo más ambicioso (traducen activaciones
directamente a texto). Los NLAEs están disponibles para Llama y Gemma en Neuronpedia
(neuronpedia.org/nla) pero **no para Pythia**. Los SAEs de EleutherAI son la alternativa
más cercana y están disponibles para la suite de modelos que usamos.

---

## Hipótesis adicional

**H4 (representacional):** Durante el valle de la curva-U conductual, los features SAE
asociados al paradigma sintáctico permanecen activos (o no decrecen), mientras que el
comportamiento externo falla. Esto evidenciaría una disociación competencia/performance
análoga a la observada en adquisición humana.

---

## Recursos disponibles (sin entrenamiento adicional)

EleutherAI publicó SAEs pre-entrenados sobre los outputs de MLP de cada capa:

| Modelo | Repo SAE en HuggingFace |
|--------|------------------------|
| pythia-70m-deduped | `EleutherAI/sae-pythia-70m-deduped-32k` |
| pythia-160m | `EleutherAI/sae-pythia-160m-32k` |
| pythia-160m-deduped | `EleutherAI/sae-pythia-160m-deduped-32k` |
| pythia-410m | `EleutherAI/sae-pythia-410m-65k` |
| pythia-410m-deduped | `EleutherAI/sae-pythia-410m-deduped-65k` |

Librería: `pip install sparsify` (EleutherAI).
Repo: https://github.com/EleutherAI/sae

Para etiquetado automático de features en lenguaje natural (AutoInterp):
https://blog.eleuther.ai/autointerp/

### Limitación crítica: SAEs del checkpoint final

Los SAEs de EleutherAI están entrenados sobre el **checkpoint final** de cada modelo.
Proyectar activaciones de checkpoints intermedios a través de ese SAE es una
**aproximación**: el espacio de activación cambia durante el entrenamiento.

Esto no invalida el análisis, pero debe reportarse como limitación metodológica.
La alternativa correcta sería entrenar un SAE separado por checkpoint, lo que es
computacionalmente costoso (fuera de scope para este trabajo).

**Referente metodológico:** "The Birth of Knowledge" (Kharazi et al., 2025 —
arxiv 2505.19440) hizo esto correctamente para Pythia-12B con 25 checkpoints, pero
analizó features **semánticos** (disciplinas, conceptos), no sintácticos. Nuestro
análisis sintáctico sería complementario y novedoso respecto a ese trabajo.

---

## Workflow propuesto

### Fase 0 — Identificar features sintácticos (checkpoint final, ~1-2h)

1. Instalar `sparsify` y verificar que carga correctamente.
2. Cargar el SAE de `pythia-160m-deduped` en una capa media (recomendadas: 4–8).
3. Tomar 200–500 pares mínimos de BLiMP para un paradigma focal
   (p.ej. `subject_verb_agreement` o `irregular_verb_forms`).
4. Extraer activaciones del modelo final para esas oraciones.
5. Correr `contrastive_features()` (ver `src/ontogenia/sae_probes.py`): identifica
   qué features se activan diferencialmente en gramaticales vs. agramaticales.
6. Conservar los top-20 features como "features sintácticos del paradigma".
7. Opcional: pasar esos feature indices por AutoInterp para obtener etiquetas en
   lenguaje natural ("concordancia sujeto-verbo plural", etc.).

### Fase 1 — Sweep longitudinal (~3-6h GPU)

Para cada checkpoint del vector longitudinal (los mismos que en el sweep BLiMP/Zorro):
1. Cargar `pythia-160m-deduped` en ese checkpoint (`revision=stepN`).
2. Extraer activaciones para las mismas 200-500 oraciones.
3. Proyectar por el SAE del checkpoint final (aproximación anotada).
4. Registrar activación media de los top-20 features sintácticos.

Resultado: curva temporal de activación de features sintácticos.

### Fase 2 — Comparación conductual vs. representacional (~1h)

Superponer en un mismo plot:
- **Eje izquierdo:** accuracy BLiMP para el paradigma (curva conductual, ya calculada).
- **Eje derecho:** activación media de features SAE sintácticos (curva representacional).

Preguntas: ¿la curva representacional es más suave que la conductual? ¿Hay disociación
durante la U (activación interna alta, accuracy bajo)?

Este plot sería la Figura X del paper si H4 se confirma.

---

## Integración con el pipeline existente

El módulo `src/ontogenia/sae_probes.py` implementa el scaffold completo:

| Función | Descripción |
|---------|-------------|
| `load_sae(model_name, layer)` | Carga SAE pre-entrenado desde HuggingFace |
| `extract_activations(sentences, model, revision, layer)` | Extrae hidden states de Pythia |
| `get_feature_activations(activations, sae)` | Proyecta activaciones por el SAE |
| `contrastive_features(good_acts, bad_acts, sae)` | Rankea features por diferencia gramatical |
| `track_features_across_checkpoints(...)` | Pipeline longitudinal completo |

Ver también `notebooks/04_sae_probes.ipynb` (por crear) para análisis exploratorio.

---

## Estimación de esfuerzo

| Tarea | Tiempo estimado | Dependencia |
|-------|----------------|-------------|
| Instalar sparsify + smoke test | 30 min | ninguna |
| Fase 0: identificar features (checkpoint final) | 1–2h | pipeline activaciones |
| Fase 1: sweep longitudinal (35 ckpts × 1 paradigma) | 3–6h GPU | Fase 0 |
| Fase 2: plots comparativos | 1h | Fases 0 + 1 |
| Sección en el paper (si H4 confirmada) | 1–2h | Fases 0 + 1 + 2 |
| **Total** | **~7–12h** | — |

La Fase 0 es la de menor riesgo: no depende del sweep principal y da un resultado
útil por sí sola (mapa de features sintácticos en el checkpoint final).

---

## Referencias

- EleutherAI SAEs: https://github.com/EleutherAI/sae
- HuggingFace collection: https://huggingface.co/collections/EleutherAI/sparse-autoencoders
- AutoInterp (EleutherAI): https://blog.eleuther.ai/autointerp/
- Kharazi et al. (2025) "The Birth of Knowledge": https://arxiv.org/abs/2505.19440
  — Referente metodológico: SAEs longitudinales sobre checkpoints de Pythia-12B.
- Anthropic NLAEs: https://transformer-circuits.pub/2026/nla/
  — Motivación conceptual; disponibles para Llama/Gemma en neuronpedia.org/nla.
