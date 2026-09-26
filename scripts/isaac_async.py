from __future__ import annotations

from omni.kit.async_engine import run_coroutine


def run_kit_coroutine(coroutine, simulation_app):
    """Run a Kit coroutine from a standalone SimulationApp script."""
    task = run_coroutine(coroutine)
    while not task.done():
        simulation_app.update()
    return task.result()
