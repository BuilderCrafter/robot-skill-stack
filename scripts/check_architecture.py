from __future__ import annotations

import ast
import py_compile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "robot_skill_stack"
GENERIC = [
    PACKAGE / "common",
    PACKAGE / "runtime",
    PACKAGE / "world",
    PACKAGE / "manipulation",
    PACKAGE / "orchestration",
    PACKAGE / "presentation",
]
FORBIDDEN = ("isaacsim", "omni", "pxr")


def imports(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            yield from (alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            yield node.module


def main():
    violations = []
    for folder in GENERIC:
        for path in folder.rglob("*.py"):
            for name in imports(path):
                if name.startswith(FORBIDDEN):
                    violations.append((path.relative_to(ROOT), name))

    for path in ROOT.rglob("*.py"):
        if any(part in {".deps", "__pycache__"} for part in path.parts):
            continue
        py_compile.compile(str(path), doraise=True)

    if violations:
        print("Architecture boundary violations:")
        for path, name in violations:
            print(f"  {path}: {name}")
        return 1

    print("Architecture boundary: PASS")
    print("Python syntax: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
