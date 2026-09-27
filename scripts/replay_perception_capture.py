#!/usr/bin/env python3
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from robot_skill_stack.world.perception.diagnostics import replay_capture


def main():
    parser = argparse.ArgumentParser(description='Replay a GUI Capture without Isaac Sim.')
    parser.add_argument('capture', type=Path)
    args = parser.parse_args()
    print(json.dumps(replay_capture(args.capture), indent=2))


if __name__ == '__main__':
    main()
