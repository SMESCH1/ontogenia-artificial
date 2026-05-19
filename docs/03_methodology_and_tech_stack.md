# Metodología y Stack Tecnológico

## 1. Diseño experimental (resumen)

No se preentrena ningún modelo. El diseño es **ex-post sobre modelos open-source que publican checkpoints intermedios**, con un eje longitudinal definido por el número de pasos de entrenamiento.

- **Familia de modelos:** Pythia (EleutherAI), variantes **deduplicadas** `pythia-14m-deduped`, `pythia-160m-deduped`, `pythia-410m-deduped`. Elección justificada por: (i) 154 checkpoints públicos, (ii) orden exacto del corpus The Pile conservado, (iii) suficientemente pequeños para caber en una RTX 3090/4090 sin optimizaciones especiales. Se descartan tamaños ≥1B por presupuesto de cómputo y tiempo.
- **Baterías de evaluación:**
  - **BLiMP** (Warstadt et al., 2020) — cobertura sintáctica amplia en inglés general.
  - **Zorro** (Huebner et al., 2021) — pares mínimos con vocabulario restringido al de un niño, apto para comparar con adquisición temprana.
- **Métrica principal:** SLLN-LP (α=0.5 inicial; se sensibiliza en el análisis). Secundaria: mean_LP (log-prob promedio por token). Accuracy sólo para lectura rápida.
- **Eje longitudinal:** vector no lineal de ~35 checkpoints (decisión final en Semana 2, ver `04_experimental_design.md`).

### 1.1 Arquitectura del modelo (Pythia)

Pythia es una familia de **transformers causales** (solo decodificador), en la línea **GPT-NeoX** de EleutherAI: bloques con atención multi-cabeza enmascarada (auto-regresiva), FFN, normalización y embeddings de token/posición según la configuración publicada por cada tamaño. Los checkpoints de una misma variante comparten **código e hiperparámetros de entrenamiento**; lo que cambía entre 14M, 160M, 410M, etc. es la **profundidad y el ancho** (capas, dimensión oculta, cabezas). Tablas exactas por tamaño: Biderman et al. (2023), [Pythia (ICML)](https://arxiv.org/abs/2304.01373); configuración en Hugging Face bajo `EleutherAI/pythia-*-deduped`.

## 2. Pipeline de ejecución (alto nivel)

```
1. Para cada (modelo_tamaño, checkpoint_step):
   a. Cargar pesos vía HuggingFace revision=step{N}.
   b. Correr BLiMP + Zorro con lm-evaluation-harness (zero-shot, sin decoding).
   c. Guardar JSON agregado por tarea en `results/sweep/` (y opcionalmente **muestras por ítem** con `--log-samples` y/o `--samples-dir`; ver `CLAUDE.md`).
   d. Liberar VRAM.
2. Post-proceso: calcular SLLN-LP, accuracy por paradigma, y persistir en
   results/aggregated.parquet.
3. Análisis: trazar trayectorias, ajustar polinomios de 5° grado, clasificar
   topología (monótona / U / U invertida / oscilatoria).
4. Alineación humana: joinear ranking de steps-de-estabilización con AoA de
   Wordbank; calcular Spearman ρ con bootstrap.
```

## 3. Stack tecnológico

### Core
- **Python 3.10+**
- **PyTorch** ≥ 2.1 (CUDA 12.x)
- **HuggingFace `transformers`** ≥ 4.40 — carga de Pythia por `revision=stepN`.
- **HuggingFace `datasets`** — acceso a BLiMP; Zorro se ingiere manualmente desde GitHub (`phueb/Zorro`).

### Evaluación
- **`lm-evaluation-harness`** (EleutherAI, fork `main`). Se usa como motor principal: garantiza entropía cruzada determinista, soporta multiple-choice zero-shot e inyección de `revision`. Tareas custom definidas en `configs/*.yaml`.
- **`babylm/evaluation-pipeline-2024`** — fork curado con Zorro integrado; se evaluará si conviene usarlo directo o portar los configs a la versión de EleutherAI para homogeneizar.

### Análisis y visualización
- **NumPy / SciPy / Pandas** para agregación y estadística.
- **`scipy.stats.spearmanr`** y bootstrap manual para los intervalos.
- **`numpy.polyfit`** (grado 5) para clasificación de topología, replicando Bunzeck & Zarrieß.
- **Matplotlib / Seaborn** para las figuras del paper.

### Adquisición humana
- **Wordbank** (<https://wordbank.stanford.edu>): descarga manual de los CSV o uso del paquete R `wordbankr` vía `rpy2` si se necesita. Alternativa ligera: snapshot CSV versionado en `data/human_milestones.csv`.
- **CHILDES**: si se necesita análisis de producción libre, usar `childespy` (Python) o `childesr` (R). Probablemente no en esta iteración.
- **Braginsky et al. (2019)** — trabajo de referencia para consolidar AoA multilingüe.

### Paper y presentación
- **LaTeX** con la plantilla ACL (`acl_latex.tex` y `acl.sty`). Idioma: **español**.
- **Overleaf** recomendado para colaboración del equipo.
- **Slides:** Google Slides o Beamer.

### Entorno y reproducibilidad
- `requirements.txt` y `pyproject.toml` en la raíz del repo.
- Cómputo local: **RTX 3090 / 4090**. Pythia-410m con batch_size=16 en FP16 entra cómoda.
- Semillas fijas en los evaluadores donde aplique.

## 4. Organización del código (esqueleto; se implementa en Semanas 2-3)

```
src/ontogenia/
├── checkpoints.py      # vector de steps y model_args para HuggingFace
├── metrics.py          # SLLN-LP, margen en pares
├── harness.py          # simple_evaluate + guardado JSON
├── aggregate.py        # consolidar métricas y muestras por-ítem → parquet
├── topology.py         # clasificación topológica (monótona/U/invertida/oscilatoria)
├── human_alignment.py  # step de estabilización + Spearman + bootstrap
├── prefetch.py         # pre-descarga de checkpoints al cache HF
└── cli.py              # smoke | sweep | eval | aggregate | prefetch | topology | human-alignment
```

Los **notebooks** en `notebooks/` no llevan lógica pesada: importan de `src/ontogenia/` y producen figuras.

## 5. Equipo y responsabilidades (borrador inicial — a discutir con el equipo)

- **Sebastián** — coordinación general, redacción del paper, diseño experimental.
- **Leandro** — por asignar (sugerido: pipeline de evaluación + sweep de checkpoints sobre Pythia).
- **Hugo** — por asignar (sugerido: alineación humana, curación de `human_milestones.csv`, análisis estadístico con Spearman + bootstrap).

Las asignaciones son sólo una propuesta basada en un reparto natural de frentes (ingeniería de modelos / datos humanos + estadística / redacción). Se reasignan según preferencias y disponibilidad del equipo.

## 6. Control de versiones

- **GitHub** para el repo final (público en la presentación).
- Ramas: `main` protegida, trabajo en `feat/*`.
- Commits descriptivos; no commitear pesos ni resultados grandes (`.gitignore` ya lo cubre).
- Un **tag** `v1.0-presentacion` el 25/05/2026 para congelar el estado entregado.

## 7. Riesgos técnicos y mitigaciones

| Riesgo | Probabilidad | Mitigación |
| --- | --- | --- |
| Descarga lenta de 35+ checkpoints de Pythia-410m | Alta | Pre-descarga nocturna en Semana 2; cache local persistente |
| Tokenización distinta entre Zorro/BLiMP y Pythia introduce artefactos | Media | Sanity check en Semana 2 contra un caso publicado de Bunzeck |
| SLLN-LP mal normalizado (valor de α sensible) | Media | Sensibilidad: correr con α ∈ {0.3, 0.5, 0.7} y reportar |
| Wordbank no cubre los fenómenos sintácticos de BLiMP directamente | Alta | Plan B: usar subset de pares mínimos con análogos léxicos en Wordbank; plan C: narrowing a un solo fenómeno bien documentado (irregulares) |
| Equipo cae de 3 a 1 persona | Media | Scope degradable a 1 tamaño de Pythia × 10 paradigmas (ver `04_experimental_design.md`) |

## 8. Referencias técnicas

- EleutherAI Pythia: <https://github.com/EleutherAI/pythia>
- LM Evaluation Harness: <https://github.com/EleutherAI/lm-evaluation-harness>
- BabyLM 2024 pipeline: <https://github.com/babylm/evaluation-pipeline-2024>
- Zorro dataset: <https://github.com/phueb/Zorro>
- Wordbank: <https://wordbank.stanford.edu>
