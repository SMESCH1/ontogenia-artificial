---
marp: true
theme: gaia
_class: lead
paginate: true
backgroundColor: #0A0E17
color: #F8FAFC
style: |
  section {
    font-family: 'Helvetica Neue', Arial, sans-serif;
    padding: 40px;
    background-color: #0A0E17;
    color: #F8FAFC;
  }
  h1 {
    font-family: 'Georgia', sans-serif;
    color: #00E5FF;
  }
  h2 {
    font-family: 'Georgia', sans-serif;
    color: #3B82F6;
    border-bottom: 2px solid #1C253C;
  }
  h3 {
    color: #00E5FF;
  }
  footer {
    color: #94A3B8;
    font-size: 0.5em;
  }
  .highlight {
    color: #00E5FF;
    font-weight: bold;
  }
  .confirm {
    color: #10B981;
    font-weight: bold;
  }
  .fail {
    color: #EF4444;
    font-weight: bold;
  }
  .card {
    background-color: #131A2A;
    border: 1px solid #1C253C;
    border-radius: 8px;
    padding: 15px;
    margin: 10px 0;
  }
  .grid-2 {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 20px;
  }
---

# Ontogenia Artificial

### Trayectorias de adquisición sintáctica en Pythia y su alineación con la adquisición humana del lenguaje

**Sebastián Mesch Henriques · Leandro · Hugo**
Universidad de San Andrés (UDESA) · NLP 2026

*Modelos: 3  |  Checkpoints: 24  |  Paradigmas BLiMP: 67*

---

## 1. Motivación y Marco Teórico

¿Los LLMs replican la fenomenología del aprendizaje infantil?

<div class="grid-2">
<div class="card">

### Adquisición en Niños (Curva en U)
1. **Memorización**: Imitación de formas irregulares correctas (*went*, *ate*).
2. **Abstracción**: Inducción de regla morfológica general (-ed) → Sobreregularización (*goed*, *eated*) [Valle de la U].
3. **Integración**: Lexicalización de excepciones y dominio.
</div>

<div class="card">

### Dinámica en LLMs
* Modelamos la **Pairwise Accuracy** a lo largo del pre-entrenamiento.
* Estudiamos la transición entre memorización superficial de n-gramas y generalización de reglas sintácticas.
* Hipótesis: Morfología irregular genera valles de curva en U más marcados y severos.
</div>
</div>

---

## 2. Hipótesis de Investigación

* **H1: Existencia de curvas no monótonas** <span class="confirm">✓ CONFIRMADA</span>
  $\ge 30\%$ de los paradigmas BLiMP exhiben trayectorias no monótonas (caída $\ge 5$ pp con recuperación posterior).
  
* **H2: Alineación ordinal infantil ↔ IA** <span class="fail">✗ NO CONFIRMADA</span>
  Existe correlación positiva y significativa (Spearman $\rho > 0.3$) entre la edad de adquisición infantil (AoA) y el step de estabilización en Pythia.

* **H3: Conflicto Morfológico (Irregulares)** <span class="confirm">✓ CONFIRMADA</span>
  Los paradigmas morfológicos irregulares exhiben valles significativamente más profundos que los sintácticos estructurales puros.

---

## 3. Método y Pipeline Experimental

* **Modelos**: Pythia (14M, 160M, 410M deduped). 24 checkpoints log-lineales (steps 0 a 143k).
* **Dataset**: BLiMP (67 paradigmas, 1.000 pares mínimos c/u).
* **Métrica Principal**: Pairwise Accuracy.
* **Control de Robustez**: Pipeline SLLN-LP (Length Normalized $\alpha=0.5$).
* **Alineación Humana**: Mapeo AoA (Wordbank) + Spearman bootstrap (10.000 it).
* **Análisis H3**: Test unilateral U de Mann-Whitney (Irregular > Estructural).

---

## 4. Resultados H1: Prevalencia de No-Monotonicidad

La no-monotonicidad es el estándar de optimización, superando ampliamente el umbral del 30%:

* **Pythia-14M**: <span class="highlight">76,1%</span> (51 / 67 paradigmas) exhiben curva no monótona.
* **Pythia-160M**: <span class="highlight">85,1%</span> (57 / 67 paradigmas) exhiben curva no monótona.
* **Pythia-410M**: <span class="highlight">91,0%</span> (61 / 67 paradigmas) exhiben curva no monótona.

*La no-monotonicidad aumenta con la escala: modelos grandes memorizan excepciones antes de generalizar.*

*(Ver figura de Distribución de Topologías en `figures/fig3_topology.png`)*

---

## 5. Resultados H2: Correlación con Adquisición Humana

* **Spearman $\rho$ (Pythia-160M)**: <span class="fail">−0,180</span> (dirección contraria)
* **p-value**: <span class="fail">0,168</span> (estadísticamente no significativo)

### ¿Por qué ocurre la discrepancia?
1. **Sesgo de Dominio**: Wordbank AoA mide producción léxica escolar inicial. BLiMP evalúa conocimiento de sintaxis abstracta.
2. **Divergencia de Curriculum**: Pythia entrena con corpus web masivo y desordenado de adultos. Los niños reciben input controlado (CHILDES) e interactivo.
3. **Incertidumbre**: 46 de 67 paradigmas mapeados tienen baja confianza.

*(Ver figura de dispersión en `figures/fig2_human_alignment_160m.png`)*

---

## 6. Resultados H3: Morfología Irregular

Los paradigmas de excepciones morfológicas (irregulares) presentan valles marcadamente más severos que la sintaxis estructural:

| Modelo | Prof. Valle Irregular (M) | Prof. Valle Estructural (S) | Mann-Whitney U |
| :--- | :---: | :---: | :---: |
| **Pythia-14M** | $d = 0,322$ | $d = 0,298$ | $p = 0,186$ |
| **Pythia-160M** | **$d = 0,437$** | $d = 0,349$ | **$p = 0,022$** <span class="confirm">✓</span> |
| **Pythia-410M** | **$d = 0,482$** | $d = 0,338$ | **$p = 0,003$** <span class="confirm">✓✓</span> |

**Interpretación**: Respalda la teoría conexionista de curvas en U. El modelo primero memoriza de forma episódica, luego colapsa al intentar inducir la regla de regularización, y finalmente lexicaliza la excepción.

---

## 7. Discusión y Limitaciones

* **Dinámica de Optimización**: La optimización de redes autorregresivas grandes mediante descenso de gradiente en CommonCrawl genera de forma innata fenomenología U-shaped.
* **Limitación Discriminativa**: BLiMP evalúa a nivel discriminativo (juicios de aceptabilidad en pares mínimos) mas no generativo.
* **Avenidas de Control**: Integrar el vocabulario controlado infantil de **Zorro** y datasets de comprensión sintáctica real es el siguiente paso requerido.

---

## 8. Conclusiones y Trabajo Futuro

* **H1 <span class="confirm">✓</span>**: Alta prevalencia de no-monotonicidad en pre-entrenamiento.
* **H2 <span class="fail">✗</span>**: Nula alineación temporal directa con hitos léxicos infantiles.
* **H3 <span class="confirm">✓</span>**: Conflictos morfológicos de excepciones producen valles más profundos.

### Trabajo Futuro (H4):
1. **Curriculum Infantil**: Entrenar sobre CHILDES / BabyLM.
2. **Control Metodológico**: Integrar Zorro y AoA sintáctica real.
3. **Mecanicismo Temporal (Interpretabilidad)**: Probes SAE (Sparse Autoencoders) para rastrear la transición de memoria a regla (H4).
