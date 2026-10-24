"""Kit-owned RMPflow execution with prepared, timestamped occupancy cuboids."""
from __future__ import annotations

import asyncio
import math
import time

import numpy as np
from isaacsim.core.utils.types import ArticulationAction

from robot_skill_stack.common.types import GripperState, Pose
from robot_skill_stack.manipulation.backend import BackendResult
from robot_skill_stack.manipulation.motion import MappingSourceProvider
from .obstacles import RmpflowObstacleRegistry


class _UnavailableWorld(RuntimeError):
    pass


class ReactiveRmpflowStrategy:
    """Borrow the backend controller; never wait for a map or scan its grid.

    Pick/Place require intended-contact information not present in the current
    motion request. They are explicitly unsupported, as is carried geometry.
    Unknown space stays unknown; successful motion is not certified clearance.
    """
    supports_contact_motion = False

    def __init__(self, backend, *, mapping_source_provider: MappingSourceProvider | None,
                 max_obstacles=64, registry=None, clock=time.monotonic):
        self._backend, self._mapping_source_provider = backend, mapping_source_provider
        self._clock, self._closed = clock, False
        self._registry = registry if registry is not None else RmpflowObstacleRegistry(
            backend.motion_controller.get_motion_policy(), max_obstacles=max_obstacles)
        self._state, self._last_error = 'ready', None
        self._used = dict(map_generation=None, map_revision=None, map_timestamp=None,
                          map_age_seconds=None)
        self._accepted_packets, self._source_failures = 0, 0
        self._last_packet_key = None
        self._last_update_seconds = 0.

    @property
    def mapping_source_provider(self):
        return self._mapping_source_provider

    @property
    def diagnostics(self):
        registry = self._registry.diagnostics
        return dict(self._used, state=self._state, last_error=self._last_error,
            closed=self._closed, proxy_count=registry['proxy_count'],
            accepted_packets=self._accepted_packets, source_failures=self._source_failures,
            last_registry_update_seconds=self._last_update_seconds, registry=registry)

    def _world(self, *, initial=False, generation=None):
        if self._closed:
            raise _UnavailableWorld('Reactive strategy is closed')
        try:
            source = (None if self._mapping_source_provider is None
                      else self._mapping_source_provider())
            if source is None:
                raise _UnavailableWorld('Spatial Mapping is unavailable')
            view = source.snapshot()
            packet, health = view.obstacles, view.health
            now, stamp, limit = float(self._clock()), view.timestamp, float(view.stale_after)
            if (not math.isfinite(now) or stamp is None or not math.isfinite(stamp)
                    or stamp > now+1e-9 or now-stamp >= limit):
                raise _UnavailableWorld('Spatial Mapping is waiting, stale or has an invalid timestamp')
            # A repeated-frame notice may reuse accepted evidence during an
            # active command, within the original budget. Errors/reset/closure
            # never become an obstacle-unaware fallback.
            if initial and not health.get('fresh', False):
                raise _UnavailableWorld(f"Spatial Mapping is not fresh: {health.get('status')}")
            if health.get('status') not in ('ok', 'no_frame'):
                raise _UnavailableWorld(f"Spatial Mapping source failed: {health.get('status')}")
            if packet is None:
                raise _UnavailableWorld('Spatial Mapping has no prepared obstacle product')
            if (packet.generation != view.generation or packet.revision != view.revision
                    or packet.timestamp != stamp):
                raise _UnavailableWorld('Obstacle product does not match the published map')
            if generation is not None and view.generation != generation:
                raise _UnavailableWorld('Spatial Mapping generation changed during motion')
            if (self._last_packet_key is not None and view.generation == self._last_packet_key[0]
                    and view.revision < self._last_packet_key[1]):
                raise _UnavailableWorld('Spatial Mapping revision regressed')
            return view, packet, now
        except _UnavailableWorld:
            raise
        except Exception as exc:
            raise _UnavailableWorld(f'Spatial Mapping access failed: {exc}') from exc

    def _sync(self, view, packet, now):
        started = time.perf_counter()
        self._registry.sync(packet, now=now)
        self._last_update_seconds = time.perf_counter()-started
        key = (view.generation, view.revision)
        if key != self._last_packet_key:
            self._accepted_packets += 1
            self._last_packet_key = key
        self._used = dict(map_generation=view.generation, map_revision=view.revision,
            map_timestamp=view.timestamp, map_age_seconds=max(0., now-view.timestamp))

    def _unavailable(self, error):
        self._source_failures += 1
        self._state, self._last_error = 'source_unavailable', str(error)
        return BackendResult(False, str(error), details=dict(self._used,
            world_state_unavailable=True, motion_strategy='reactive'))

    def _finish(self, last_action, primary):
        errors = []
        if last_action is not None:
            try:
                self._backend.robot.apply_action(ArticulationAction(
                    joint_positions=last_action.joint_positions,
                    joint_velocities=np.zeros_like(last_action.joint_positions, dtype=float),
                    joint_indices=last_action.joint_indices))
            except Exception as exc:
                errors.append(exc)
        try:
            self._registry.clear()
        except Exception as exc:
            errors.append(exc)
        if primary is not None:
            for error in errors:
                primary.add_note(f'Reactive cleanup also failed: {error}')
        elif errors:
            raise errors[0]

    async def move_to_pose_async(self, target: Pose, position_tolerance=None,
                                 orientation_tolerance=None, grasp_check=None):
        backend = self._backend
        if target.frame != 'world':
            return BackendResult(False, f'Unsupported frame: {target.frame}')
        if not backend.is_ready:
            return self._unavailable(_UnavailableWorld(
                'Robot state is unavailable; reinitialize the runtime after Stop/Play.'))
        if not backend.check_reachability(target):
            return BackendResult(False, 'Target pose has no IK solution.')
        pos_tol = backend.position_tolerance if position_tolerance is None else position_tolerance
        ori_tol = backend.orientation_tolerance if orientation_tolerance is None else orientation_tolerance
        try:
            view, packet, now = self._world(initial=True)
        except _UnavailableWorld as exc:
            return self._unavailable(exc)
        feedback = backend.gripper_feedback()
        if feedback.state != GripperState.FREE:
            return BackendResult(False,
                'Reactive carrying is unsupported: finger/carried-object collision geometry is not modeled.',
                details=dict(unsupported_motion=True, motion_strategy='reactive',
                             grasp_state=feedback.state.value))

        backend._reset_motion_controller()
        self._registry.reset_tracking()  # SDK reset already erased registrations.
        self._last_packet_key = None
        self._state, self._last_error = 'moving', None
        generation = view.generation
        last_action = primary = None
        last_pos_error = last_ori_error = None
        try:
            for step in range(backend.max_motion_steps):
                if not backend.world.is_playing():
                    raise _UnavailableWorld('Simulation stopped during reactive motion')
                view, packet, now = self._world(generation=generation)
                self._sync(view, packet, now)
                action = backend.motion_controller.forward(
                    target_end_effector_position=target.position,
                    target_end_effector_orientation=target.orientation)
                backend.robot.apply_action(action)
                last_action = action
                await backend._next_update()
                if not backend.world.is_playing():
                    raise _UnavailableWorld('Simulation stopped during reactive motion')
                if not backend.is_ready:
                    raise _UnavailableWorld('Robot articulation was invalidated during reactive motion')
                self._world(generation=generation)
                if grasp_check is not None:
                    failure = grasp_check()
                    if failure is not None:
                        self._state = 'grasp_failure'
                        return failure
                current = backend.get_end_effector_pose()
                last_pos_error = float(np.linalg.norm(current.position-target.position))
                if target.orientation is None:
                    last_ori_error, orientation_ok = None, True
                else:
                    last_ori_error = backend._quat_error(current.orientation, target.orientation)
                    orientation_ok = last_ori_error <= ori_tol
                if last_pos_error <= pos_tol and orientation_ok:
                    self._state = 'succeeded'
                    return BackendResult(True, 'End effector reached target pose.', details=dict(
                        self._used, motion_strategy='reactive', steps=step+1,
                        position_error_m=last_pos_error, orientation_error_rad=last_ori_error,
                        position_tolerance_m=pos_tol, orientation_tolerance_rad=ori_tol))
            self._state = 'timed_out'
            return BackendResult(False, 'Reactive motion timed out before reaching target pose.',
                timed_out=True, details=dict(self._used, motion_strategy='reactive',
                    max_steps=backend.max_motion_steps, position_error_m=last_pos_error,
                    orientation_error_rad=last_ori_error, position_tolerance_m=pos_tol,
                    orientation_tolerance_rad=ori_tol))
        except _UnavailableWorld as exc:
            return self._unavailable(exc)
        except BaseException as exc:
            primary = exc
            self._state = 'cancelled' if isinstance(exc, asyncio.CancelledError) else 'error'
            self._last_error = str(exc)
            raise
        finally:
            self._finish(last_action, primary)

    def close(self):
        self._closed, self._state = True, 'closed'
        # The registry keeps failed removals owned so a later close can retry.
        self._registry.close()
