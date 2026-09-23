#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ISAAC_SIM="${ISAAC_SIM:-/home/etfrobotics/isaacsim}"

export PYTHONPATH="$ROOT/.deps${PYTHONPATH:+:$PYTHONPATH}"

exec "$ISAAC_SIM/isaac-sim.sh" \
  --ext-folder "$ROOT/exts" \
  --enable robot_skill_stack.ui \
  "$@"