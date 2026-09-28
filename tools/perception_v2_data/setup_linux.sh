#!/usr/bin/env bash
# Optional HOME-PC Linux setup. Do not run this inside Isaac's environment.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
PYTHON="${PYTHON:-python3.11}"
"$PYTHON" -c 'import sys; assert sys.version_info[:2] == (3,11)'
[[ -x .venv/bin/python ]] || "$PYTHON" -m venv .venv
P=.venv/bin/python
"$P" -c 'import sys; assert sys.version_info[:2] == (3,11) and sys.prefix != sys.base_prefix'
"$P" -m pip install --upgrade pip==25.1.1
"$P" -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126
"$P" -m pip install -r requirements-training.txt
"$P" -m pip check
"$P" check_environment.py --download-model
"$P" -m pip freeze > training-environment-lock.txt
