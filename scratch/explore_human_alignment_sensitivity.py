#!/usr/bin/env python3
"""
Script de exploración de alineación humana (H2).
Calcula la correlación de Spearman para los tres modelos Pythia (14m, 160m, 410m)
y realiza un análisis de sensibilidad filtrando por el nivel de confianza de los hitos.
"""

import sys
import os
import pandas as pd
import numpy as np
from scipy.stats import spearmanr

# Asegurar que podemos importar src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from ontogenia.human_alignment import run_human_alignment

def main():
    metrics_parquet = "results/aggregated_metrics.parquet"
    aoa_csv = "data/human_milestones.csv"

    if not os.path.exists(metrics_parquet):
        print(f"⚠️ Error: No se encontró el archivo de métricas en {metrics_parquet}")
        return

    if not os.path.exists(aoa_csv):
        print(f"⚠️ Error: No se encontró el archivo de hitos humanos en {aoa_csv}")
        return

    print("=============================================================")
    print("🧠 ANÁLISIS DE SENSIBILIDAD: ALINEACIÓN HUMANA (H2) 🧠")
    print("=============================================================\n")

    # Leer archivos base para reporte
    aoa_df = pd.read_csv(aoa_csv)
    print(f"Distribución de confianza en el mapeo AoA original ({len(aoa_df)} tareas):")
    print(aoa_df["confidence"].value_counts().to_string())
    print("\n-------------------------------------------------------------")

    models = ["14m", "160m", "410m"]
    
    # 1. Correlación general (Todos los paradigmas)
    print("\n📊 CASO A: Todos los paradigmas (Incluye Islas y NPI - confianza baja)")
    for model in models:
        res = run_human_alignment(
            metrics_parquet=metrics_parquet,
            aoa_csv=aoa_csv,
            target_model_size=model,
            n_boot=1000
        )
        stats = res.stats.iloc[0]
        print(f"🔹 Pythia-{model:<5} | N={stats['n_overlap']:<2} | rho={stats['rho']:.4f} | p={stats['p_value']:.4f} | IC 95%: [{stats['ci_lo']:.3f}, {stats['ci_hi']:.3f}]")

    print("\n-------------------------------------------------------------")

    # 2. Correlación filtrando por confianza (Solo alta y media)
    print("\n🎯 CASO B: Filtrando tareas de confianza Baja (Excluye Islas y NPI)")
    
    # Crear un CSV temporal de AoA sin las tareas de confianza baja
    temp_aoa_csv = "data/human_milestones_filtered.csv"
    filtered_aoa = aoa_df[aoa_df["confidence"] != "baja"]
    filtered_aoa.to_csv(temp_aoa_csv, index=False)

    for model in models:
        res = run_human_alignment(
            metrics_parquet=metrics_parquet,
            aoa_csv=temp_aoa_csv,
            target_model_size=model,
            n_boot=1000
        )
        stats = res.stats.iloc[0]
        print(f"🔥 Pythia-{model:<5} | N={stats['n_overlap']:<2} | rho={stats['rho']:.4f} | p={stats['p_value']:.4f} | IC 95%: [{stats['ci_lo']:.3f}, {stats['ci_hi']:.3f}]")

    # Eliminar temporal
    if os.path.exists(temp_aoa_csv):
        os.remove(temp_aoa_csv)

    print("\n=============================================================")

if __name__ == "__main__":
    main()
