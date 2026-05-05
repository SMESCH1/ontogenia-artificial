#!/usr/bin/env bash
# Pre-descarga checkpoints Pythia para evitar esperas de red durante sweep.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
if [[ -f .venv/bin/activate ]]; then
  # shellcheck source=/dev/null
  source .venv/bin/activate
fi
python -m ontogenia prefetch "$@"

