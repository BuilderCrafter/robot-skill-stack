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

# Isaac Sim 5.1.0-rc.19's python.sh does not support --no-ros-env.
# launch_isaac_gui.py disables ROS bridge startup in the full experience.
exec "$ROOT/run_isaac_python.sh" "$ROOT/scripts/launch_isaac_gui.py" "$@"
