"""Strict read-only observation/planner contract; never submits robot commands."""
from scripts.benchmark_multiview_grasp_readiness import cli


if __name__ == '__main__':
    cli(strict=True)
