#!/usr/bin/env bash
# Post-proceso Semana 3-4:
# 1) agrega métricas de JSON -> parquet
# 2) clasifica topologías por tarea
# 3) si existe data/human_milestones.csv, corre alineación humana
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
if [[ -f .venv/bin/activate ]]; then
  # shellcheck source=/dev/null
  source .venv/bin/activate
fi

python -m ontogenia aggregate \
  --results-dir results \
  --output-parquet results/aggregated_metrics.parquet \
  "$@"

python -m ontogenia topology \
  --metrics-parquet results/aggregated_metrics.parquet \
  --output-parquet results/topology_summary.parquet

if [[ -f data/human_milestones.csv ]]; then
  python -m ontogenia human-alignment \
    --metrics-parquet results/aggregated_metrics.parquet \
    --aoa-csv data/human_milestones.csv \
    --model-size 160m \
    --output-stats-json results/human_alignment_stats.json \
    --output-overlap-parquet results/human_alignment_overlap.parquet
else
  echo "Aviso: falta data/human_milestones.csv; se omite human-alignment."
fi

