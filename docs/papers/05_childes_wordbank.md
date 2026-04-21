---
cita: Frank, M. C., Braginsky, M., Yurovsky, D., & Marchman, V. A. (2017). *Wordbank: An open repository for developmental vocabulary data.* Journal of Child Language.
link: https://wordbank.stanford.edu
complementaria: Braginsky et al. (2019) / MacWhinney (2000) CHILDES
leído_por: _
fecha: _
prioridad: 🟠 alta — fuente de AoA humano
---

# Notas: Wordbank y CHILDES

## Wordbank

### Qué es
Agregación online del MacArthur-Bates CDI. Por cada ítem (palabra o frase), reporta la proporción de niños que lo producen/comprenden a cada edad (en meses).

### Formato
- Tablas descargables (CSV) desde la UI o vía `wordbankr` (R).
- Columnas clave: `item`, `age_months`, `proportion_producing`, `proportion_understanding`, `language`.

### Edad de adquisición (AoA)
Definición operativa estándar: **edad a la que 50% de los niños producen el ítem**. Algunos trabajos usan 30% o 75% según el caso.

### Cobertura
- Fuerte: vocabulario (sustantivos, verbos, adjetivos comunes).
- Débil: sintaxis abstracta (acuerdo, anáforas, islas).

## CHILDES

### Qué es
Base de datos longitudinal de transcripciones del habla de niños y del lenguaje dirigido al niño. Formato CHAT.

### Acceso
- `childespy` (Python) o `childesr` (R).
- Requiere parsing para extraer AoA de un fenómeno sintáctico (no viene tabulado).

### Cuándo usarlo
Solo si algún paradigma clave de BLiMP no tiene análogo en Wordbank. En esta iteración, probablemente no lo activamos.

## Braginsky et al. (2019)

Síntesis cuantitativa multilingüe sobre Wordbank. Tablas agregadas que pueden usarse directamente como fuente de AoA.

## Plan operativo para nuestro proyecto

1. Descargar snapshot CSV de Wordbank (inglés) — `data/raw/wordbank_en.csv`.
2. Curar a `data/human_milestones.csv` con columnas `paradigm, phenomenon, aoa_months_low, aoa_months_high, source, confidence`.
3. Complementar con citas de literatura específica (Brown 1973, Marcus et al. 1992, Wexler 1994) para fenómenos no cubiertos.

## Citas sugeridas
> Usamos Wordbank (Frank et al., 2017) como fuente primaria de edad de adquisición léxica, complementada con estudios específicos de adquisición sintáctica (Brown, 1973; Marcus et al., 1992) cuando los fenómenos evaluados no son directamente léxicos.

## Dudas a resolver
- ¿La API de Wordbank entrega proporciones por mes sin agregación adicional?
- ¿Hay versiones validadas en español que podamos usar si algún día extendemos a español?
