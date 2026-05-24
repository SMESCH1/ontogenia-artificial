#!/usr/bin/env python3
"""
Script interactivo para comparar dos tablas de mapeo AoA (original vs. refinada).
Permite evaluar si los cambios metodológicos en el mapeo humano mejoran o alteran
la correlación de Spearman de la H2 para los tres modelos Pythia.
"""

import sys
import os
import pandas as pd

# Asegurar importación de src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from ontogenia.human_alignment import run_human_alignment

def report_table_stats(metrics_parquet, aoa_csv, label):
    models = ["14m", "160m", "410m"]
    rows = []
    for model in models:
        # Caso Completo
        res_all = run_human_alignment(
            metrics_parquet=metrics_parquet,
            aoa_csv=aoa_csv,
            target_model_size=model,
            n_boot=500
        )
        s_all = res_all.stats.iloc[0]
        
        # Caso Filtrado (Sin baja confianza)
        aoa_df = pd.read_csv(aoa_csv)
        temp_filtered = f"data/temp_filtered_{model}.csv"
        aoa_df[aoa_df["confidence"] != "baja"].to_csv(temp_filtered, index=False)
        
        res_filt = run_human_alignment(
            metrics_parquet=metrics_parquet,
            aoa_csv=temp_filtered,
            target_model_size=model,
            n_boot=500
        )
        s_filt = res_filt.stats.iloc[0]
        
        if os.path.exists(temp_filtered):
            os.remove(temp_filtered)
            
        rows.append({
            "Modelo": f"Pythia-{model}",
            "N_All": s_all["n_overlap"],
            "rho_All": s_all["rho"],
            "p_All": s_all["p_value"],
            "N_Filt": s_filt["n_overlap"],
            "rho_Filt": s_filt["rho"],
            "p_Filt": s_filt["p_value"]
        })
    
    df = pd.DataFrame(rows)
    print(f"\n==========================================")
    print(f"📊 TABLA: {label} ({aoa_csv})")
    print(f"==========================================")
    print(df.to_string(index=False, formatters={
        "rho_All": "{:,.4f}".format, "p_All": "{:,.4f}".format,
        "rho_Filt": "{:,.4f}".format, "p_Filt": "{:,.4f}".format
    }))

def main():
    metrics_parquet = "results/aggregated_metrics.parquet"
    aoa_orig = "data/human_milestones.csv"
    aoa_refined = "data/human_milestones_refined.csv"
    
    if not os.path.exists(metrics_parquet):
        print(f"⚠️ Error: No se encontró el archivo de métricas en {metrics_parquet}")
        return

    # Reportar original
    report_table_stats(metrics_parquet, aoa_orig, "ORIGINAL")
    
    # Reportar refinada si existe
    if os.path.exists(aoa_refined):
        report_table_stats(metrics_parquet, aoa_refined, "REFINADA / EXPERTA")
    else:
        print(f"\n💡 Nota: No se encontró la tabla refinada en '{aoa_refined}'.")
        print(f"   Duplica '{aoa_orig}' con ese nombre y modifícala manualmente para ver el impacto.")

if __name__ == "__main__":
    main()
