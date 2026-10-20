"""Actual native depth -> provider -> updater -> WorldModel -> unchanged planner.

Mandatory legacy90mm hidden extent and fresh complete/incomplete/frozen/restored
eligibility regression. No robot or backend is constructed or commanded.
"""
import sys

from scripts.benchmark_multiview_observability import cli


if __name__ == '__main__':
    sys.argv.extend(['--scenario', 'partial_box_both', '--scenario', 'temporal_eligibility'])
    cli(strict=True)
