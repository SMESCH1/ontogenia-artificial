#!/usr/bin/env bash
# Corrida Semana 3: 3 modelos × 24 checkpoints × tarea(s) lm-eval (p. ej. grupo `blimp`).
# Requiere GPU y varias horas. Contingencia: --sizes 160m --steps 0,2000,8000,...
# Ver plan de degradación en ROADMAP.md.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
if [[ -f .venv/bin/activate ]]; then
  # shellcheck source=/dev/null
  source .venv/bin/activate
fi
export TOKENIZERS_PARALLELISM=false
python -m ontogenia sweep --output-dir results "$@"
