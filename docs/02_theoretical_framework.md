# Marco Teórico

## 1. Developmental Interpretability: evaluar el *cómo*, no sólo el *qué*

Durante la última década, la evaluación de modelos de lenguaje se ha centrado abrumadoramente en análisis estáticos *post-hoc*: se mide el rendimiento del modelo en su estado final, tras la convergencia de la función de pérdida. Este paradigma tiene valor operativo pero oculta una pregunta más interesante desde el punto de vista científico: **¿cómo y cuándo aparecen las capacidades?**

La **Developmental Interpretability** (Kendiukhov, 2025) propone responder esa pregunta analizando la *trayectoria* del modelo a lo largo del pre-entrenamiento, no sólo su destino. Para hacerlo, requiere dos condiciones técnicas:

1. **Acceso a checkpoints intermedios** con control sobre el orden de los datos (para aislar tiempo de entrenamiento de otras variables). La familia Pythia (Biderman et al., 2023) satisface esta condición con 154 checkpoints públicos.
2. **Baterías de evaluación aplicables zero-shot** — porque los modelos tempranos no son instruccionales, cualquier protocolo que requiera fine-tuning contamina el análisis. El estándar es medir log-verosimilitud de terminaciones alternativas sobre pares mínimos (Warstadt et al., 2020 — BLiMP).

Este proyecto se ubica en esta tradición, con foco restringido a la emergencia sintáctica.

## 2. Curvas en U: una firma del aprendizaje por estructura

La **curva en U** es un patrón clásico en la psicolingüística del desarrollo. Para los verbos irregulares del inglés, por ejemplo, los niños exhiben tres fases:

1. **Memorización correcta temprana:** producen *went* como forma aislada, tratándola como un ítem léxico.
2. **Sobre-regularización:** al consolidar la regla *-ed*, generalizan y producen *goed* o *wented*, temporalmente degradando el rendimiento en estas formas.
3. **Dominio maduro:** reconcilian la regla con las excepciones y recuperan *went*, ahora como excepción lexicalizada.

La interpretación canónica (Rumelhart & McClelland 1986; Marcus et al. 1992) es que estas curvas son la firma conductual de una transición desde memoria episódica hacia inducción algorítmica. No son un "error" del aprendiz: son evidencia de que el sistema está re-estructurando su representación.

**Observación empírica clave de los últimos dos años:** los LLMs auto-regresivos también exhiben curvas en U — y variantes aún más complejas — a nivel de paradigma sintáctico individual. Bunzeck & Zarrieß (2024), en *"Fifty shapes of BLiMP"*, ajustaron polinomios de quinto grado a las trayectorias de cada paradigma de BLiMP sobre checkpoints de Pythia y BabyLlama, y encontraron:

- Curvas **monótonas** (la mayoría, pero no todas).
- Curvas **en U** (caída y recuperación).
- Curvas **en U invertida** (pico temprano y decaimiento).
- **Oscilaciones prolongadas** previas a la convergencia.

El hallazgo crítico es metodológico: **promediar el rendimiento por época final borra esta estructura**. La accuracy del último checkpoint no distingue entre un paradigma que se aprendió monótonamente y uno que atravesó tres fases de sobre-regularización.

## 3. Por qué SLLN-LP y no accuracy

La métrica natural para pares mínimos es: *"¿el modelo asigna mayor log-probabilidad a la sentencia gramatical que a la agramatical?"*. Sin embargo, la probabilidad conjunta sufre un sesgo de longitud: una sentencia más corta tiene, todo lo demás constante, mayor probabilidad. En pares mínimos donde las dos sentencias difieren sólo marginalmente en cantidad de tokens (por ejemplo por distinta morfología), este sesgo introduce artefactos que no reflejan competencia gramatical.

**ZhoBLiMP** (Bunzeck et al., 2024, ArXiv 2411.06096) propone la métrica **SLLN-LP** (*Sub-linear Length Normalized Log-Probability*):

$$\text{SLLN-LP}(s) = \frac{\log P(s)}{|s|^{\alpha}}$$

con $\alpha \in (0, 1)$ — típicamente ~0.5 — que normaliza de manera sub-lineal, evitando la sobre-corrección de la normalización por longitud completa (que introduce su propio sesgo en sentido opuesto). Esta es la métrica que usaremos como principal, complementándola con *mean_LP* (log-probabilidad promedio por token) para reproducir resultados de Zorro que la emplean (Huebner et al., 2021).

## 4. BLiMP y Zorro: dos corpus de pares mínimos

- **BLiMP** (Warstadt et al., 2020) contiene 67 paradigmas × 1000 pares mínimos cada uno, cubriendo sintaxis inglesa amplia (acuerdo, ligamiento, islas, etc.). Vocabulario no restringido; contiene sentencias con palabras de baja frecuencia que niños no conocerían.
- **Zorro** (Huebner et al., 2021) fue diseñado para modelos entrenados con vocabulario restringido al de un niño de ~5 años. Es más apto para comparaciones con adquisición temprana y para modelos pequeños tipo BabyLM. Cubre subsets de los paradigmas de BLiMP pero en lexicón infantil.

Usaremos ambos: **BLiMP** como referencia comparable con la literatura adulta y **Zorro** para la comparación directa con adquisición infantil.

## 5. Adquisición humana del lenguaje como contraste

La psicolingüística del desarrollo ha producido, a lo largo de décadas, dos recursos fundamentales que permiten obtener una **métrica cuantitativa de edad de adquisición (AoA)** por fenómeno:

- **CHILDES** (MacWhinney, 2000): base de datos longitudinal de transcripciones del habla infantil y del lenguaje dirigido al niño. Permite estimar cuándo un fenómeno emerge espontáneamente en la producción.
- **Wordbank** (Frank et al., 2017): agregación de datos del MacArthur-Bates CDI (Communicative Development Inventory). Reporta proporción de niños que producen/comprenden cada ítem a cada edad (en meses), lo que permite calcular un AoA probabilístico (típicamente, la edad a la que ≥50% de los niños lo produce).

Para este proyecto, **Wordbank es más directamente utilizable** porque entrega tablas cuantitativas por ítem léxico y estructura morfo-sintáctica (cuando hay items relevantes), mientras que CHILDES requiere un paso de análisis no trivial. La decisión operativa se cierra en `05_human_alignment.md`.

Trabajos recientes han comenzado a usar AoA como ancla comparativa para LLMs: Chang & Bergen (2022), Evanson et al. (2023) y Haga et al. (2024) argumentan que la *surprisal* y el orden de estabilización en modelos pre-entrenados se correlacionan — con intensidades variables — con curvas de adquisición humana. Nuestro aporte es aplicar esta estrategia al régimen dinámico (no sólo el modelo final) y sobre la grilla de 154 checkpoints de Pythia.

## 6. Contexto breve: SLT y por qué queda fuera de scope

Una línea muy activa (Timaeus, Hoogland et al. 2024-2025) propone que las transiciones de fase en el aprendizaje de redes profundas se pueden cuantificar geométricamente mediante el **Coeficiente de Aprendizaje Local (LLC)**, derivado de la **Singular Learning Theory** (Watanabe, 2009). La idea central: los modelos singulares (los transformers lo son) atraviesan el paisaje de pérdida no de manera monótona, sino a través de transiciones de fase en las que el modelo reestructura sus representaciones — y una caída en el LLC coincide temporalmente con la emergencia conductual de una capacidad.

Esta línea es complementaria y atractiva, pero la estimación empírica del LLC requiere SGMCMC sobre ventanas estrechas de checkpoints, lo cual excede razonablemente las 5 semanas disponibles. La mencionamos explícitamente como **línea futura** en el paper para dar crédito al marco, sin asumir el costo experimental.

## 7. Hipótesis y predicciones (remiten a 04)

Las hipótesis operacionalizadas y sus métricas precisas están en `04_experimental_design.md`. En resumen:

- **H1 — No-monotonicidad:** ≥30% de los paradigmas muestran curvas no monótonas en Pythia.
- **H2 — Alineación ordinal humana:** Spearman ρ > 0.3 entre ranking de estabilización del modelo y ranking de AoA humano.
- **H3 — Morfología irregular:** las curvas-U son más marcadas en paradigmas con excepciones morfológicas que en paradigmas puramente estructurales.

## Referencias (abreviadas; bibliografía completa en `deep-research.md`)

- Biderman et al. (2023). *Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling*. ICML.
- Bunzeck & Zarrieß (2024). *Fifty shapes of BLiMP*. CLASP.
- Bunzeck et al. (2024). *ZhoBLiMP: A Systematic Assessment of Language Models with Linguistic Minimal Pairs in Chinese*. ArXiv 2411.06096.
- Chang & Bergen (2022). *Word Acquisition in Neural Language Models*. TACL.
- Frank et al. (2017). *Wordbank: An open repository for developmental vocabulary data*. Journal of Child Language.
- Hoogland, J. et al. (2024-2025). Trabajos del instituto Timaeus sobre SLT aplicada a LLMs.
- Huebner et al. (2021). *BabyBERTa* / Zorro.
- Kendiukhov (2025). *A Review of Developmental Interpretability in Large Language Models*. ArXiv 2508.15841.
- MacWhinney (2000). *The CHILDES Project*.
- Marcus et al. (1992). *Overregularization in language acquisition*. Monographs SRCD.
- Warstadt et al. (2020). *BLiMP: The Benchmark of Linguistic Minimal Pairs for English*. TACL.
