#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ISAAC_SIM="${ISAAC_SIM:-/home/etfrobotics/isaacsim}"
export ISAAC_SIM

if [[ ! -d "$ROOT/.deps/numpy" ]]; then
  echo "ERROR: $ROOT/.deps/numpy is missing." >&2
  echo "Install repo dependencies first:" >&2
  echo "  ./run_isaac_python.sh -m pip install --target .deps -r requirements.txt" >&2
  exit 1
fi

# Start the full GUI from Isaac's python.sh instead of isaac-sim.sh.
# python.sh honors our repo-local PYTHONPATH, so NumPy 1.26.4 is imported
# before Kit and its extensions initialize.
exec "$ROOT/run_isaac_python.sh"   "$ROOT/scripts/launch_isaac_gui.py"   "$@"
