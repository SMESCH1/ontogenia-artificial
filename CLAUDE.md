# Ontogenia Artificial — contexto para sesiones

Proyecto NLP (UDESA, 2026): trayectorias sintácticas en **Pythia** (checkpoints `revision=stepN`) sobre **BLiMP + Zorro**, métrica **SLLN-LP** (α=0.5), correlación **Spearman** entre orden de estabilización y AoA humana (**Wordbank** / tabla en `docs/05_human_alignment.md`).

**Pregunta:** ¿Las curvas por paradigma son no monótonas (p. ej. U) y ¿el orden de emergencia correlaciona con adquisición infantil?

**Hipótesis:** H1 ≥30% paradigmas no monótonos; H2 ρ>0.3; H3 curvas-U más profundas en morfológico-irregular vs estructural (`docs/04_experimental_design.md`).

**Modelos:** `pythia-14m-deduped`, `pythia-160m-deduped`, `pythia-410m-deduped`. **Checkpoints:** 24 pasos en `src/ontogenia/checkpoints.py` (`CHECKPOINT_STEPS`).

**Deadline:** presentación 26 mayo 2026 (`ROADMAP.md`).

## Comandos

```bash
python -m venv .venv && source .venv/bin/activate   # o .\venv\Scripts\activate en Windows
pip install -r requirements.txt && pip install -e .
python -m unittest discover -s tests -v               # métricas SLLN-LP
./scripts/run_smoke.sh --limit 20                   # Pythia × pocos steps × 1 paradigma BLiMP → results/smoke/
./scripts/run_full_sweep.sh                         # 3×24×blimp → results/sweep/ (muy largo; usar GPU)
python -m ontogenia sweep --sizes 160m --steps 0,512,2000 --tasks blimp --limit 50   # contingencia
python -m ontogenia aggregate --results-dir results --output-parquet results/aggregated_metrics.parquet
```

**Salidas:** JSON bajo `results/` con `{ "meta": {...}, "lm_eval": <salida harness> }`. Carpeta **gitignored**.

**Por ítem (oraciones del dataset, no texto generado):** BLiMP/Zorro en el harness son *scoring* de log-probs entre opciones; no hay “inferencia” de oraciones nuevas. Para guardar cada par (`sentence_good` / `sentence_bad`) y las respuestas del modelo, usar `--log-samples` (JSON grande) y/o `--samples-dir DIR` para volcar `lm_eval["samples"]` a `DIR/<stem>_samples.json` y dejar el JSON principal liviano.

**Tabla de enlaces:** `docs/enlaces_herramientas.md`. **Resumen deep-research:** `docs/deep_research_highlights.md`. **Zorro:** `docs/integracion_zorro.md`.

## Lectura rápida al retomar

1. `ROADMAP.md`
2. `docs/04_experimental_design.md`
3. `docs/05_human_alignment.md`
4. `src/ontogenia/cli.py` (subcomandos `smoke`, `sweep`, `eval`, `aggregate`)

**Fuera de scope:** SLT/LLC agresivo, interpretabilidad de circuitos, ToM, entrenar desde cero.
