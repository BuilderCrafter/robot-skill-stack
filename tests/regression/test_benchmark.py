# Match the GUI launch contract before Isaac/Kit loads native extensions.
import numpy as np

if np.__version__ != '1.26.4':
    raise RuntimeError(f'Expected NumPy 1.26.4, got {np.__version__} from {np.__file__}')
print(f'Test NumPy: {np.__version__} from {np.__file__}', flush=True)

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import argparse
import sys
import time
import traceback
from contextlib import ExitStack, contextmanager
from pathlib import Path
from unittest.mock import patch


from robot_skill_stack.common.types import Pose
from robot_skill_stack.integrations.isaac.runtime_lifecycle import PhysicsCallbackScope
from tests.support.isaac_async import run_kit_coroutine
from tests.support.control_trace import TraceJournal, record_control
from tests.support.results import write_result
from tests.support.isaac_helpers import (
    playground_cube,
    build_perception_runtime,
    choose_object,
    object_snapshot,
    open_playground,
    set_cube_pose,
    settle,
)

PERCEPTION_POSITIONS = [
    [0.40, -0.12, 0.025],
    [0.44, -0.12, 0.025],
    [0.48, -0.12, 0.025],
    [0.48, -0.04, 0.025],
    [0.44, -0.04, 0.025],
    [0.40, -0.04, 0.025],
    [0.40,  0.04, 0.025],
    [0.44,  0.04, 0.025],
    [0.48,  0.04, 0.025],
    [0.48,  0.12, 0.025],
    [0.44,  0.12, 0.025],
    [0.40,  0.12, 0.025],
]

MANIPULATION_TRIALS = [
    ([0.40,  0.12, 0.025], [0.44,  0.24, 0.025]),
    ([0.48,  0.10, 0.025], [0.48, -0.08, 0.025]),
    ([0.42, -0.10, 0.025], [0.42,  0.16, 0.025]),
    ([0.46,  0.02, 0.025], [0.46, -0.16, 0.025]),
]


def stats(values):
    values = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
        "min": float(np.min(values)),
    }


def trial_snapshot(bundle, object_id):
    now = time.monotonic()
    obj = bundle.world_model.get(object_id)
    return {
        "monotonic_s": now,
        "simulation_time_s": float(bundle.world.current_time),
        "world_updates": bundle.world_updater.update_count,
        "updater_error": bundle.world_updater.last_error,
        "last_seen_s": None if obj is None else obj.last_seen,
        "observation_age_s": None if obj is None or obj.last_seen is None else now-obj.last_seen,
        "object": None if obj is None else object_snapshot(obj),
        "held_object_id": bundle.world_model.held_object_id,
        "gripper_state": bundle.world_model.gripper_state.value,
    }


def physical_snapshot(bundle, cube):
    # Read-only diagnostics: never step the world or change a control target.
    try:
        position, orientation = cube.get_world_pose()
        tcp = bundle.backend.get_end_effector_pose()
        return {
            "monotonic_s": time.monotonic(),
            "simulation_time_s": float(bundle.world.current_time),
            "physics_step": int(bundle.world.current_time_step_index),
            "cube_position": np.asarray(position).tolist(),
            "cube_orientation": np.asarray(orientation).tolist(),
            "tcp_position": tcp.position.tolist(),
            "tcp_orientation": None if tcp.orientation is None else tcp.orientation.tolist(),
        }
    except Exception as exc:
        # A diagnostic read must not mask a command's original result/exception.
        return {"diagnostic_error": f"{type(exc).__name__}: {exc}"}


@contextmanager
def trace_backend_commands(bundle, records, cube):
    def wrap(name, command):
        async def traced(*args, **kwargs):
            row = {"command": name, "before": physical_snapshot(bundle, cube)}
            if name == "move_to_pose_async":
                target = args[0] if args else kwargs["target"]
                row["target"] = {
                    "position": target.position.tolist(),
                    "orientation": None if target.orientation is None else target.orientation.tolist(),
                }
            records.append(row)
            try:
                result = await command(*args, **kwargs)
                row["result"] = {
                    "ok": result.ok, "timed_out": result.timed_out,
                    "message": result.message, "details": dict(result.details),
                }
                return result
            except BaseException as exc:
                row["exception"] = f"{type(exc).__name__}: {exc}"
                raise
            finally:
                row["after"] = physical_snapshot(bundle, cube)
        return traced

    with ExitStack() as stack:
        for name in ("move_to_pose_async", "open_gripper_async", "close_gripper_async", "home_async"):
            stack.enter_context(patch.object(bundle.backend, name, new=wrap(name, getattr(bundle.backend, name))))
        yield


def _run_benchmark(metrics, result_path=None, artifacts_dir=None, journal=None):
    out = Path(
        artifacts_dir
        or Path("outputs") / "phase_a_benchmark_tmp"
    )
    out.mkdir(parents=True, exist_ok=True)

    if journal is not None:
        journal.mark('open_playground')
    print("[1] Opening playground...", flush=True)
    open_playground(simulation_app)

    if journal is not None:
        journal.mark('build_runtime')
    print("[2] Building perception runtime...", flush=True)
    bundle = build_perception_runtime(simulation_app, out)
    cube = playground_cube()
    with ExitStack() as stack:
        set_stage = lambda name: None
        if journal is not None:
            set_stage = stack.enter_context(record_control(
                bundle.backend, PhysicsCallbackScope(bundle.world), journal, cube=cube,
            ))

        def stage(name):
            if journal is not None:
                journal.mark(name)
            print('Benchmark stage:', name, flush=True)
            set_stage(name)

        _benchmark_body(bundle, cube, metrics, result_path, stage)


def _benchmark_body(bundle, cube, metrics, result_path, stage):
    stage('initial_settle')
    settle(bundle.world, 180)

    obj = choose_object(bundle.world_model)
    object_id = obj.object_id

    print("[3] Deterministic workspace perception benchmark...")
    position_errors = []
    size_max_errors = []

    for index, target in enumerate(PERCEPTION_POSITIONS, 1):
        stage(f'perception_{index}')
        truth = set_cube_pose(
            cube,
            target,
            bundle.world,
            settle_steps=48,
        )
        tracked = bundle.world_model.get(object_id)
        if tracked is None or not tracked.visible or tracked.pose is None:
            raise RuntimeError(
                f"Lost persistent object {object_id} at benchmark pose {index}"
            )

        pos_error = float(
            np.linalg.norm(tracked.pose.position - truth)
        )
        size_error = (
            None
            if tracked.size is None
            else np.abs(
                tracked.size - np.array([0.05, 0.05, 0.05])
            )
        )

        position_errors.append(pos_error * 1000)
        size_max_errors.append(
            float(np.max(size_error) * 1000)
            if size_error is not None
            else float("inf")
        )

        row = {
            "index": index,
            "target": list(target),
            "truth": truth.tolist(),
            "perceived": tracked.pose.position.tolist(),
            "position_error_mm": pos_error * 1000,
            "size_error_mm": (
                None
                if size_error is None
                else (size_error * 1000).tolist()
            ),
            "object_id": tracked.object_id,
        }
        metrics["perception_trials"].append(row)
        print(
            f"  pose {index:02d}: "
            f"pos={row['position_error_mm']:.2f} mm "
            f"size_max={size_max_errors[-1]:.2f} mm "
            f"id={tracked.object_id}"
        )

    metrics["perception_position_error_mm"] = stats(
        position_errors
    )
    metrics["perception_size_max_error_mm"] = stats(
        size_max_errors
    )
    metrics["persistent_id"] = object_id

    if metrics["perception_position_error_mm"]["max"] >= 15.0:
        raise AssertionError(
            "Perception benchmark exceeded 15 mm max position error"
        )
    if metrics["perception_size_max_error_mm"]["max"] >= 12.0:
        raise AssertionError(
            "Perception benchmark exceeded 12 mm max size error"
        )

    print("[4] Four full perception-driven Pick + Place trials...")
    successes = 0
    target_errors = []
    metrics.update(manipulation_total=len(MANIPULATION_TRIALS), manipulation_completed=0,
                   manipulation_successes=0, manipulation_success_rate=0.0)

    for index, (start, target) in enumerate(
        MANIPULATION_TRIALS,
        1,
    ):
        stage(f'trial_{index}:home')
        home = run_kit_coroutine(bundle.runtime.execute_async("home"), simulation_app)
        if not home.ok:
            raise RuntimeError(
                f"Home failed before trial {index}: {home.message}"
            )

        stage(f'trial_{index}:setup')
        truth_start = set_cube_pose(
            cube,
            start,
            bundle.world,
            settle_steps=60,
        )
        tracked = bundle.world_model.get(object_id)
        if tracked is None or not tracked.visible:
            metrics["failed_setup"] = {
                "trial": index, "world_state": trial_snapshot(bundle, object_id),
                "physical_state": physical_snapshot(bundle, cube),
            }
            raise RuntimeError(
                f"Object not visible before manipulation trial {index}"
            )

        async def pick_and_place():
            # A single Kit task preserves a direct sequential handoff. Launching
            # Place in a second task adds a physics tick under the old command
            # before Place captures the TCP orientation for its release poses.
            before_pick = trial_snapshot(bundle, object_id)
            stage(f'trial_{index}:pick')
            pick = await bundle.runtime.execute_async(
                "pick",
                object_id=object_id,
            )
            after_pick = trial_snapshot(bundle, object_id)
            place = None
            if pick.ok:
                stage(f'trial_{index}:place')
                place = await bundle.runtime.execute_async(
                    "place",
                    target=Pose(target),
                    mode="stable",
                )
            after_place = trial_snapshot(bundle, object_id) if place is not None else None
            return before_pick, pick, after_pick, place, after_place

        command_trace = []
        with trace_backend_commands(bundle, command_trace, cube):
            before_pick, pick, after_pick, place, after_place = run_kit_coroutine(
                pick_and_place(), simulation_app,
            )
        final_true = np.asarray(
            cube.get_world_pose()[0],
            dtype=float,
        )
        target_error = float(
            np.linalg.norm(final_true - np.asarray(target))
        )
        target_errors.append(target_error * 1000)

        ok = bool(pick.ok and place is not None and place.ok)
        successes += int(ok)
        row = {
            "index": index,
            "start": list(start),
            "true_start": truth_start.tolist(),
            "target": list(target),
            "pick_status": pick.status.value,
            "pick_ok": bool(pick.ok),
            "pick_message": pick.message,
            "pick_failure_code": (
                None
                if pick.failure_code is None
                else pick.failure_code.value
            ),
            "pick_details": pick.details,
            "place_status": None if place is None else place.status.value,
            "place_ok": bool(place is not None and place.ok),
            "place_message": None if place is None else place.message,
            "place_failure_code": (
                None
                if place is None or place.failure_code is None
                else place.failure_code.value
            ),
            "place_details": None if place is None else place.details,
            "true_final": final_true.tolist(),
            "target_error_mm": target_error * 1000,
            "object_after": (after_place or after_pick)["object"],
            "before_pick": before_pick,
            "after_pick": after_pick,
            "after_place": after_place,
            "backend_commands": command_trace,
        }
        metrics["manipulation_trials"].append(row)
        metrics.update(manipulation_completed=len(metrics["manipulation_trials"]),
                       manipulation_successes=successes,
                       manipulation_success_rate=successes/len(MANIPULATION_TRIALS),
                       target_error_mm=stats(target_errors))
        print(
            f"  trial {index}: "
            f"pick={'PASS' if pick.ok else 'FAIL'} "
            f"place={'PASS' if place is not None and place.ok else 'FAIL'} "
            f"target_error={target_error * 1000:.2f} mm"
        )

        if not ok:
            original_failure = (
                f"Trial {index}: Pick {row['pick_status']} [{row['pick_failure_code']}] {pick.message}; "
                f"Place {row['place_status']} [{row['place_failure_code']}] {row['place_message']}"
            )
            # Stop at the original failure. Opening here can drop a held object
            # from the pre-place height; Home/another trial is not recovery.
            row["failure_handling"] = "stopped_without_cleanup_commands"
            print(original_failure, flush=True)
            write_result(result_path, status="FAIL", metrics=metrics, error=original_failure)
            raise AssertionError(f"{original_failure}; benchmark stopped without cleanup commands")

    metrics["manipulation_successes"] = successes
    metrics["manipulation_total"] = len(MANIPULATION_TRIALS)
    metrics["manipulation_success_rate"] = (
        successes / len(MANIPULATION_TRIALS)
    )
    metrics["target_error_mm"] = stats(target_errors)

    if successes != len(MANIPULATION_TRIALS):
        raise AssertionError(
            f"Only {successes}/{len(MANIPULATION_TRIALS)} "
            "benchmark manipulations succeeded"
        )
    if metrics["target_error_mm"]["max"] >= 25.0:
        raise AssertionError(
            "Benchmark placement exceeded 25 mm true target error"
        )

    print("\n=== PHASE A BENCHMARK ===")
    print(
        "Perception position error [mm]:",
        metrics["perception_position_error_mm"],
    )
    print(
        "Perception size max error [mm]:",
        metrics["perception_size_max_error_mm"],
    )
    print(
        "Manipulation success:",
        f"{successes}/{len(MANIPULATION_TRIALS)}",
    )
    print(
        "Target error [mm]:",
        metrics["target_error_mm"],
    )
    print("PASS")
    print("=========================")
    write_result(result_path, status="PASS", metrics=metrics)


def main(result_path=None, artifacts_dir=None, control_trace=False):
    metrics = {"perception_trials": [], "manipulation_trials": []}
    try:
        with ExitStack() as stack:
            journal = None
            if control_trace:
                out = Path(artifacts_dir or Path('outputs')/'phase_a_benchmark_tmp')
                out.mkdir(parents=True, exist_ok=True)
                path = out/'control_trace.jsonl'
                stream = stack.enter_context(path.open('w', encoding='utf-8'))
                metrics['control_trace_file'] = str(path)
                journal = TraceJournal(stream)
                journal.mark('startup')
                print('Control trace journal:', path, flush=True)
            _run_benchmark(metrics, result_path, artifacts_dir, journal)
    except BaseException as exc:
        write_result(result_path, status="FAIL", metrics=metrics,
                     error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result")
    parser.add_argument("--artifacts-dir")
    parser.add_argument('--control-trace', action='store_true',
                        help='Stream loaded joint/TCP/cube feedback to control_trace.jsonl for diagnosis')
    args = parser.parse_args()
    try:
        main(args.result, args.artifacts_dir, args.control_trace)
    except Exception as exc:
        traceback.print_exc()
        sys.exit(1)
    finally:
        simulation_app.close()
