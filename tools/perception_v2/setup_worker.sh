#!/usr/bin/env bash
# NEW workstation ML environment only. Never install these into Isaac or .deps.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1
PYTHON="${PYTHON:-/usr/bin/python3}"
VENV="${VENV:-$HOME/.venvs/robot-vision}"
"$PYTHON" -c 'import sys; assert sys.version_info[:2] in ((3,11),(3,12)), "Use system Python 3.11 or 3.12, not Isaac or Python 3.13+"; assert "isaacsim" not in sys.executable.lower(), "Do not use Isaac Python"'
[[ -x "$VENV/bin/python" ]] || "$PYTHON" -m venv "$VENV"
P="$VENV/bin/python"
"$P" -c 'import sys; assert sys.prefix != sys.base_prefix and sys.version_info[:2] in ((3,11),(3,12))'
"$P" -m pip install --upgrade pip==25.1.1
"$P" -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126
"$P" -m pip install -r "$ROOT/tools/perception_v2_data/requirements-training.txt"
"$P" -m pip check
"$P" -c 'import sys, torch, numpy, ultralytics; print("Worker Python:", sys.version); print("torch:", torch.__version__, "NumPy:", numpy.__version__, "Ultralytics:", ultralytics.__version__); print("CUDA available:", torch.cuda.is_available())'
"$P" -m pip freeze > "$VENV/worker-environment-lock.txt"
printf '\nWorker environment ready: %s\nNext: launch serve.py with your trusted trained best.pt. No model was downloaded.\n' "$P"
