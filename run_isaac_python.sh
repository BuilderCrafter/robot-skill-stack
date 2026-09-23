#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ISAAC_SIM="${ISAAC_SIM:-/home/etfrobotics/isaacsim}"

export PYTHONPATH="$ROOT/.deps${PYTHONPATH:+:$PYTHONPATH}"

exec "$ISAAC_SIM/python.sh" "$@"