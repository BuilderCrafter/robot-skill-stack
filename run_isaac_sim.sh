#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ISAAC_SIM="${ISAAC_SIM:-/home/etfrobotics/isaacsim}"

if [[ ! -d "$ROOT/.deps/numpy" ]]; then
  echo "ERROR: $ROOT/.deps/numpy is missing." >&2
  echo "Install repo dependencies first:" >&2
  echo "  ./run_isaac_python.sh -m pip install --target .deps -r requirements.txt" >&2
  exit 1
fi

# Kit's embedded Python ignores PYTHONPATH during interpreter initialization.
# Add repo-local dependencies before Kit extensions import NumPy.
exec "$ISAAC_SIM/isaac-sim.sh" \
  --/app/python/extraPaths/0="$ROOT/.deps" \
  --/app/python/extraPaths/1="$ROOT" \
  --ext-folder "$ROOT/exts" \
  --enable robot_skill_stack.ui \
  "$@"
