#!/usr/bin/env bash
# Smoke test: Pythia-14m × 3 checkpoints × 1 paradigma BLiMP (acotado con --limit).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
if [[ -f .venv/bin/activate ]]; then
  # shellcheck source=/dev/null
  source .venv/bin/activate
fi
export TOKENIZERS_PARALLELISM=false
python -m ontogenia smoke --output-dir results "$@"
