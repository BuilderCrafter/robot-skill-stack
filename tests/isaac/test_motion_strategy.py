"""Explicit direct strategy through normal skills, with mapping and safe cancellation.

Reuse the validated V5 physical-cube fixture so Pick/held Home/Place/released
Home have the same scene, camera calibration and outcome assertions. Extra
checks observe the selected strategy and final arm actions; occupancy never
changes the commanded path. Run through run_isaac_python.sh, sequentially.
"""
# This fixture performs the NumPy guard and owns the one SimulationApp instance.
from tests.spatial_mapping import test_runtime_integration as fixture

import argparse
import asyncio
import json
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np

from robot_skill_stack.common.types import Pose
from robot_skill_stack.integrations.isaac.manipulation.motion.direct import DirectRmpflowStrategy

V5_EVALUATE = fixture.evaluate


def vector(value):
    if value is None:
        return None
    return np.asarray(value).copy().tolist()


async def evaluate(args, metrics, checks):
    original_build = fixture.build_runtime
    original_evaluate = V5_EVALUATE
    strategy_calls, arm_actions, hold_checks = [], [], []
    metrics['motion_strategy'] = dict(calls=strategy_calls, final_holds=hold_checks)

    async def build_direct(profile):
        path = Path(profile).with_name('runtime_profile_direct.toml')
        path.write_text(Path(profile).read_text()+'\n[motion]\nstrategy = "direct"\n')
        bundle = await original_build(path)
        backend = bundle.backend
        strategy = backend.motion_strategy
        checks['explicit_direct_selected'] = (bundle.config.motion.strategy == 'direct'
                                               and isinstance(strategy, DirectRmpflowStrategy))
        assert checks['explicit_direct_selected']
        provider = strategy.mapping_source_provider
        checks['mapping_provider_resolves_current_service'] = (callable(provider)
                                                               and provider() is bundle.spatial_mapping)
        assert checks['mapping_provider_resolves_current_service']
        metrics['motion_strategy']['startup_home'] = fixture.serial(backend.named_poses['home'])
        original_apply = backend.robot.apply_action
        original_move = strategy.move_to_pose_async
        original_execute = bundle.runtime.execute_async
        commands, active_call = 0, None
        extras_done = False

        def apply(action, *positional, **keywords):
            indices = getattr(action, 'joint_indices', None)
            # Partial finger actions belong to backend gripper ownership.
            is_arm = indices is None or any(int(index) < 7 for index in indices)
            if is_arm:
                arm_actions.append(dict(at=time.monotonic(), call=active_call,
                    positions=vector(getattr(action, 'joint_positions', None)),
                    velocities=vector(getattr(action, 'joint_velocities', None)),
                    indices=vector(indices)))
            return original_apply(action, *positional, **keywords)

        def final_hold(before, label):
            actions = arm_actions[before:]
            if not actions:
                return None
            last, previous = actions[-1], actions[-2] if len(actions) > 1 else None
            held = (previous is not None and last['positions'] == previous['positions']
                and last['indices'] == previous['indices'] and last['velocities'] is not None
                and all(value == 0 for value in last['velocities']))
            hold_checks.append(dict(call=label, actions=len(actions), zero_velocity_hold=held))
            return held

        async def move(target, position_tolerance=None, orientation_tolerance=None, grasp_check=None):
            nonlocal commands, active_call
            commands += 1
            label, before, started = commands, len(arm_actions), time.monotonic()
            active_call = label
            row = dict(call=label, target=fixture.serial(target),
                position_tolerance=position_tolerance, orientation_tolerance=orientation_tolerance,
                had_grasp_check=grasp_check is not None, grasp_checks=0)
            strategy_calls.append(row)

            def checked():
                row['grasp_checks'] += 1
                return grasp_check()

            try:
                result = await original_move(target, position_tolerance=position_tolerance,
                    orientation_tolerance=orientation_tolerance,
                    grasp_check=None if grasp_check is None else checked)
                row.update(outcome='result', result=fixture.serial(result))
                return result
            except asyncio.CancelledError:
                row['outcome'] = 'cancelled'
                raise
            finally:
                row['seconds'] = time.monotonic()-started
                row['final_hold'] = final_hold(before, label)
                active_call = None

        async def extra_checks():
            # Rejected requests must issue neither controller actions nor a hold.
            before = len(arm_actions)
            rejected_frame = await backend.move_to_pose_async(Pose([.45, 0., .25], frame='robot'))
            rejected_ik = await backend.move_to_pose_async(Pose([20., 20., 20.]))
            checks['unsupported_frame_and_ik_rejected_without_actions'] = (
                not rejected_frame.ok and 'Unsupported frame' in rejected_frame.message
                and not rejected_ik.ok and 'IK solution' in rejected_ik.message
                and len(arm_actions) == before)
            assert checks['unsupported_frame_and_ik_rejected_without_actions']

            # Start one ordinary direct move, then cancel after an actual action.
            # No world stepping or joint teleport is used to create this state.
            task = asyncio.create_task(backend.move_to_pose_async(
                Pose([.45, .08, .25], [0., 0., 1., 0.])))
            try:
                await fixture.until(lambda: len(arm_actions) > before)
                assert not task.done(), 'Motion finished before cancellation could be observed'
                competing_before = len(arm_actions)
                competing = await backend.move_to_pose_async(Pose([.45, 0., .25]))
                checks['competing_motion_rejected_without_actions'] = (
                    not competing.ok and 'already in progress' in competing.message
                    and len(arm_actions) == competing_before)
                assert checks['competing_motion_rejected_without_actions']
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    checks['cancellation_propagated'] = True
                else:
                    raise AssertionError('Cancelled direct motion did not propagate cancellation')
            finally:
                if not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
            checks['cancelled_motion_applied_final_hold'] = strategy_calls[-1]['final_hold'] is True
            checks['cancelled_motion_released_ownership'] = not backend._motion_active
            assert checks['cancelled_motion_applied_final_hold']
            assert checks['cancelled_motion_released_ownership']
            recovery = await original_execute('home')
            checks['home_after_cancellation_succeeded'] = recovery.ok
            assert recovery.ok, recovery.message
            await fixture.fresh(bundle)
            metrics['motion_strategy']['after_cancellation'] = fixture.state(bundle)

        async def execute(name, *positional, **keywords):
            nonlocal extras_done
            result = await original_execute(name, *positional, **keywords)
            # The fixture's first Home is held; the second follows successful
            # release. Run cancellation only after that manipulation sequence.
            if name == 'home' and bundle.world_model.held_object_id is None and not extras_done:
                extras_done = True
                await extra_checks()
            return result

        backend.robot.apply_action = apply
        strategy.move_to_pose_async = move
        bundle.runtime.execute_async = execute
        return bundle

    # Existing V5 acceptance assertions are reused unchanged. Its lifecycle
    # matrix is independently rerun by the required spatial_mapping group.
    with patch.object(fixture, 'build_runtime', build_direct):
        await original_evaluate(args, metrics, checks)
    checks['all_executed_motions_applied_final_hold'] = bool(hold_checks) and all(
        row['zero_velocity_hold'] for row in hold_checks)
    checks['held_motion_grasp_checks_preserved'] = any(
        row['had_grasp_check'] and row['grasp_checks'] > 0 for row in strategy_calls)
    checks['normal_home_routed_through_strategy'] = sum(
        row['target'] == metrics['motion_strategy']['startup_home'] for row in strategy_calls) >= 3
    # Keep full raw actions outside the compact result, while retaining timing
    # distributions and action/hold counts for review.
    gaps = [(b['at']-a['at'])*1000 for a,b in zip(arm_actions, arm_actions[1:])
            if a['call'] is not None and a['call'] == b['call']]
    metrics['motion_strategy']['arm_action_count'] = len(arm_actions)
    metrics['motion_strategy']['control_interval_ms'] = dict(count=len(gaps),
        **({} if not gaps else dict(p50=float(np.percentile(gaps,50)),
            p95=float(np.percentile(gaps,95)), p99=float(np.percentile(gaps,99)),
            worst=float(max(gaps)))))
    Path(args.artifacts_dir, 'motion_actions.json').write_text(json.dumps(arm_actions, indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result', required=True)
    parser.add_argument('--artifacts-dir', default='outputs/motion-planning-v1-native')
    args = parser.parse_args()
    args.lifecycle_only, args.skip_lifecycle = False, True
    try:
        with patch.object(fixture, 'evaluate', evaluate):
            passed = fixture.main(args)
    finally:
        fixture.simulation_app.close()
    raise SystemExit(0 if passed else 1)
