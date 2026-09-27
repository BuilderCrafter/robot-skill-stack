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

# We do not use ROS in the course runtime. --no-ros-env prevents python.sh
# from injecting its bundled ROS environment; launch_isaac_gui.py also
# overrides the full experience so the ROS 2 bridge itself is not started.
exec "$ROOT/run_isaac_python.sh"   --no-ros-env   "$ROOT/scripts/launch_isaac_gui.py"   "$@"
