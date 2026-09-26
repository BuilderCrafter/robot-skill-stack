# Lower-level validation

Validated on the Isaac Sim workstation on 2026-09-26.

- Perception V1: complete suite PASS; single-cube localization approximately 2 mm and same-ID reacquisition after Pick/Place.
- Robust execution (Phase A): all five stages PASS; deterministic workspace benchmark mean localization error about 1.85 mm, worst observed about 2.41 mm; 4/4 Pick+Place benchmark trials; mean physical placement error about 4.93 mm; reactive Pick, BT recovery and multi-object manipulation PASS.
- Grasp planning (Phase B): unit and Isaac stages PASS; 90 mm cube rejected before motion with `NO_VALID_GRASP`; normal 50 mm perception-driven cube still Pick+Place successful with about 6 mm physical target error.

The regression suites are retained under `tests/` and runners under `scripts/`.
