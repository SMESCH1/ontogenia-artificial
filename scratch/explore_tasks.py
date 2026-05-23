import pandas as pd
from pathlib import Path

# Cargar el resumen de topologías
topo_path = Path("/home/sebastian-mesch-henriques/Escritorio/MIA/NLP/ontogenia-artificial/results/topology_summary.parquet")
if not topo_path.exists():
    print("No se encuentra el archivo de topologías.")
    exit(1)

df = pd.read_parquet(topo_path)
print("Columnas:", df.columns)
print("Modelos únicos:", df["model_size"].unique())
print("Total de tareas:", df["task"].nunique())

# Listar todas las tareas blimp disponibles
blimp_tasks = [t for t in df["task"].unique() if t.startswith("blimp")]
print("\nEjemplos de tareas BLiMP:")
for t in sorted(blimp_tasks)[:15]:
    print(f"  - {t}")
