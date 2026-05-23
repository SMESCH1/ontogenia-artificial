import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu

# Cargar topologías
topo_path = pd.read_parquet("/home/sebastian-mesch-henriques/Escritorio/MIA/NLP/ontogenia-artificial/results/topology_summary.parquet")

# Definir tareas del grupo M (morfológicas irregulares) y S (estructurales)
grupo_m_tasks = [
    "blimp_irregular_past_participle_verbs",
    "blimp_irregular_past_participle_adjectives",
    "blimp_irregular_plural_subject_verb_agreement_1",
    "blimp_irregular_plural_subject_verb_agreement_2",
    "blimp_determiner_noun_agreement_irregular_1",
    "blimp_determiner_noun_agreement_irregular_2",
    "blimp_determiner_noun_agreement_with_adj_irregular_1",
    "blimp_determiner_noun_agreement_with_adj_irregular_2"
]

# Las estructurales serán las de BLiMP excluyendo las morfológicas irregulares, las morfológicas regulares y la agregada "blimp"
exclusiones = grupo_m_tasks + [
    "blimp", 
    "blimp_determiner_noun_agreement_1", 
    "blimp_determiner_noun_agreement_2",
    "blimp_determiner_noun_agreement_with_adjective_1",
    "blimp_determiner_noun_agreement_with_adj_2",
    "blimp_regular_plural_subject_verb_agreement_1",
    "blimp_regular_plural_subject_verb_agreement_2"
]

all_blimp_tasks = [t for t in topo_path["task"].unique() if t.startswith("blimp")]
grupo_s_tasks = [t for t in all_blimp_tasks if t not in exclusiones]

print(f"Número de tareas en Grupo M: {len(grupo_m_tasks)}")
print(f"Número de tareas en Grupo S: {len(grupo_s_tasks)}")

# Analizar por tamaño de modelo
for sz in ["14m", "160m", "410m"]:
    sub = topo_path[topo_path["model_size"] == sz].copy()
    
    # Tomar la profundidad de curvas no monótonas
    # H3 especifica medir la profundidad en las U-curves o curvas no monótonas
    # Criterio robusto de H1 para clasificar como curva en U / no monótona:
    is_nonmono = (
        (sub["shape"] == "u_shape") |
        (
            (sub["shape"].isin(["oscillatory", "inverted_u"])) &
            (sub["depth"] >= 0.05) &
            (sub["min_step"] < sub["max_step"])
        )
    )
    
    sub_nonmono = sub[is_nonmono].copy()
    
    depths_m = sub_nonmono[sub_nonmono["task"].isin(grupo_m_tasks)]["depth"].values
    depths_s = sub_nonmono[sub_nonmono["task"].isin(grupo_s_tasks)]["depth"].values
    
    print(f"\n--- Modelo {sz} ---")
    print(f"  Grupo M (n_u={len(depths_m)}): media = {np.mean(depths_m):.3f} if len else nan, std = {np.std(depths_m):.3f} if len else nan")
    print(f"  Grupo S (n_u={len(depths_s)}): media = {np.mean(depths_s):.3f} if len else nan, std = {np.std(depths_s):.3f} if len else nan")
    
    if len(depths_m) > 0 and len(depths_s) > 0:
        stat, pval = mannwhitneyu(depths_m, depths_s, alternative="greater")
        print(f"  Mann-Whitney U (unilateral mayor M): U={stat:.1f}, p={pval:.4f}")
    else:
        print("  Muestra insuficiente para correr test.")
