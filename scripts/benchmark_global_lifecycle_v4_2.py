#!/usr/bin/env python3
"""Run the established V4.1 lifecycle fixture with an explicit global model."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys
from unittest.mock import patch

from scripts import benchmark_global_acceptance_v4_1 as acceptance
from scripts.benchmark_global_models_v4_2 import model_profile, project_sources


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--collision-model', choices=('stock', 'enhanced'), required=True)
    selected, remaining = parser.parse_known_args(arguments)
    original_profile = acceptance.fixture.write_profile
    original_hashes = acceptance.source_hashes
    original_observer = acceptance.controlled.make_observer

    def hashes():
        values = project_sources(original_hashes)
        path = Path(__file__)
        values[str(path.relative_to(acceptance.fixture.ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
        return values

    def observer(bundle):
        actual = getattr(bundle.backend.motion_strategy, 'collision_model', None)
        if actual != selected.collision_model or bundle.config.motion.global_collision_model != actual:
            raise RuntimeError('Lifecycle fixture selected a different collision model')
        result = original_observer(bundle)
        result.metadata['verified_global_collision_model'] = actual
        return result

    with patch.object(acceptance.fixture, 'write_profile',
                lambda options: model_profile(options, original_profile, selected.collision_model)), \
            patch.object(acceptance, 'source_hashes', hashes), \
            patch.object(acceptance.controlled, 'make_observer', observer):
        return acceptance.main(remaining)


if __name__ == '__main__':
    raise SystemExit(0 if main() else 1)
