"""Research-only fixed joint-goal selection for the V4.3 matched experiment.

The existing planner still owns the real RRT and every limit, full-pose,
collision, route and generated-trajectory check. This adapter replaces only
its candidate IK selection with a recorded joint target. It never inserts a
waypoint, alters RRT settings or changes a process-global SDK object. Install
its factory for the outward comparison only; normal Home uses the normal
production factory.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from types import SimpleNamespace

import numpy as np

from robot_skill_stack.common.types import Pose
from robot_skill_stack.integrations.isaac.manipulation.motion import lula_global
from robot_skill_stack.manipulation.motion.joint_route import PlanningFailure


ARM_NAMES = tuple(f'panda_joint{index}' for index in range(1, 8))


def _finite(values, shape, label):
    array = np.asarray(values, dtype=float)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f'{label} must have finite shape {shape}')
    return array


@dataclass(frozen=True)
class FixedJointTarget:
    """Immutable joint ordering, numeric candidate and its requested world pose.

    Creating this object does not certify the candidate. Original production
    FK, limits and collision checks validate it afresh for every planning
    request. The baseline artifact digest is provenance, not an exemption from
    validation. The experiment harness separately checks the full source chain,
    measured initial state, selected description and RRT settings.
    """
    joint_names: tuple
    q: tuple
    position: tuple
    orientation: tuple
    frame: str = 'world'
    end_effector_frame: str = 'right_gripper'
    baseline_artifact_sha256: str | None = None

    def __post_init__(self):
        names = tuple(self.joint_names)
        if len(names) != 7 or set(names) != set(ARM_NAMES):
            raise ValueError('Fixed target must name all seven distinct Panda arm joints')
        q = _finite(self.q, (7,), 'Fixed joint target')
        position = _finite(self.position, (3,), 'Fixed world position')
        orientation = _finite(self.orientation, (4,), 'Fixed world orientation')
        if np.linalg.norm(orientation) < 1e-12:
            raise ValueError('Fixed target orientation must be nonzero')
        if self.frame != 'world' or not isinstance(self.end_effector_frame, str) or not self.end_effector_frame:
            raise ValueError('Fixed target requires a world pose and a named end-effector frame')
        digest = self.baseline_artifact_sha256
        if digest is not None and (not isinstance(digest, str) or len(digest) != 64
                or any(character not in '0123456789abcdef' for character in digest)):
            raise ValueError('Baseline artifact SHA256 must be a lower-case hexadecimal digest')
        object.__setattr__(self, 'joint_names', names)
        object.__setattr__(self, 'q', tuple(map(float, q)))
        object.__setattr__(self, 'position', tuple(map(float, position)))
        object.__setattr__(self, 'orientation', tuple(map(float, orientation/np.linalg.norm(orientation))))

    @classmethod
    def from_dict(cls, value):
        """Read the joint/pose subset of the root harness's frozen manifest."""
        target = value['target']
        source = value.get('source_result') or {}
        return cls(joint_names=value['joint_names'], q=value['q'],
            position=target['position'], orientation=target['orientation'],
            frame=target.get('frame', 'world'),
            end_effector_frame=value.get('end_effector_frame', 'right_gripper'),
            baseline_artifact_sha256=value.get('baseline_artifact_sha256', source.get('sha256')))

    def as_dict(self):
        return dict(joint_names=list(self.joint_names), q=list(self.q),
            target=dict(position=list(self.position), orientation=list(self.orientation), frame=self.frame),
            end_effector_frame=self.end_effector_frame,
            baseline_artifact_sha256=self.baseline_artifact_sha256)

    @property
    def sha256(self):
        return hashlib.sha256(json.dumps(self.as_dict(), sort_keys=True,
            separators=(',', ':'), allow_nan=False).encode()).hexdigest()

    def pose(self):
        return Pose(self.position, self.orientation, self.frame)

    def assert_request(self, frame, position, orientation):
        """Fail on a different request; never silently use ordinary IK instead."""
        try:
            actual_position = _finite(position, (3,), 'Requested position')
            actual_orientation = _finite(orientation, (4,), 'Requested orientation')
            same = (frame == self.end_effector_frame
                and np.allclose(actual_position, self.position, atol=1e-12, rtol=0.)
                and np.allclose(lula_global.quaternion_matrix(actual_orientation),
                    lula_global.quaternion_matrix(self.orientation), atol=1e-12, rtol=0.))
        except (TypeError, ValueError):
            same = False
        if not same:
            raise PlanningFailure('research_fixed_target_mismatch',
                'The requested full pose/frame differs from the frozen joint-target comparison')


class _FixedCandidateKinematics:
    """Delegate real FK/limits/base/tuning; replace candidate selection only."""
    def __init__(self, delegate, fixed, calls):
        object.__setattr__(self, '_delegate', delegate)
        object.__setattr__(self, '_fixed', fixed)
        object.__setattr__(self, '_calls', calls)

    def __getattr__(self, name):
        return getattr(self._delegate, name)

    def __setattr__(self, name, value):
        # Production sets native solver parameters through this object. Do not
        # shadow those settings on the proxy or create another solver policy.
        setattr(self._delegate, name, value)

    def compute_inverse_kinematics(self, frame, position, orientation, **options):
        self._fixed.assert_request(frame, position, orientation)
        warm = options.get('warm_start')
        self._calls.append(dict(frame=frame, position=np.asarray(position).tolist(),
            orientation=np.asarray(orientation).tolist(),
            position_tolerance=options.get('position_tolerance'),
            orientation_tolerance=options.get('orientation_tolerance'),
            warm_start=None if warm is None else np.asarray(warm).tolist(), q=list(self._fixed.q)))
        # "True" denotes an available supplied candidate, not a native IK
        # computation or proof of validity. The original planner immediately
        # applies its existing limit, full-pose FK and collision validation.
        return np.asarray(self._fixed.q, dtype=float).copy(), True


class FixedJointTargetPlanner(lula_global.LulaGlobalPlanner):
    """Real production RRT pipeline with one explicit research joint goal."""
    def __init__(self, backend, *, fixed_target, seed=123456, max_iterations=10000,
                 collision_model='enhanced', _sdk=None):
        if not isinstance(fixed_target, FixedJointTarget):
            raise TypeError('fixed_target must be an immutable FixedJointTarget')
        if collision_model != 'enhanced':
            raise ValueError('V4.3 fixed-target comparisons require the enhanced collision model')
        if backend.end_effector_frame != fixed_target.end_effector_frame:
            raise ValueError('Backend and frozen end-effector frames differ')
        self.fixed_target = fixed_target
        self._fixed_calls, self._sdk_is_wrapped = [], False
        super().__init__(backend, seed=seed, max_iterations=max_iterations,
            collision_model=collision_model, _sdk=_sdk)

    def _fixed_sdk(self):
        if not self._sdk_is_wrapped:
            actual = self._sdk or lula_global._load_sdk()
            def kinematics(**config):
                return _FixedCandidateKinematics(actual.Kinematics(**config), self.fixed_target, self._fixed_calls)
            self._sdk = SimpleNamespace(lula=actual.lula, RRT=actual.RRT,
                Kinematics=kinematics, Generator=actual.Generator, load_config=actual.load_config)
            self._sdk_is_wrapped = True

    @property
    def joint_names(self):
        self._assert_owner()
        self._fixed_sdk()
        names = super().joint_names
        if names != self.fixed_target.joint_names:
            raise PlanningFailure('research_fixed_joint_order_mismatch',
                'Native RRT joint order differs from the frozen target ordering')
        return names

    def plan(self, target, start_q, snapshot, position_tolerance, orientation_tolerance):
        result = failure = None
        self._fixed_calls.clear()
        try:
            if target.frame != self.fixed_target.frame:
                raise PlanningFailure('research_fixed_target_mismatch', 'Frozen comparison requires a world target')
            self.fixed_target.assert_request(self.backend.end_effector_frame, target.position, target.orientation)
            result = super().plan(target, start_q, snapshot, position_tolerance, orientation_tolerance)
            if (result.diagnostics.get('already_at_goal')
                    and not np.array_equal(np.asarray(start_q), np.asarray(self.fixed_target.q))):
                raise PlanningFailure('research_fixed_target_bypassed',
                    'The original zero-motion pose shortcut did not reach the fixed joint configuration')
            selected = result.diagnostics.get('selected_ik_index')
            if selected is not None and not np.array_equal(
                    result.diagnostics['ik_attempts'][selected]['ik_q'], self.fixed_target.q):
                raise PlanningFailure('research_fixed_target_mismatch', 'Selected IK candidate changed')
            return result
        except PlanningFailure as error:
            failure = error
            raise
        finally:
            research = dict(self.fixed_target.as_dict(), frozen_candidate_sha256=self.fixed_target.sha256,
                candidate_source='research_fixed_joint_target', normal_ik_solver_called=False,
                candidate_calls=deepcopy(self._fixed_calls), same_goal_for_all_attempts=True,
                production_validation_preserved=True,
                scope='Only candidate IK selection replaced; real RRT, limits, full-pose FK, collision, route and spline validation are unchanged')
            self._last_diagnostics['research_fixed_joint_target'] = research
            if result is not None:
                result.diagnostics['research_fixed_joint_target'] = research
            if failure is not None:
                failure.details['research_fixed_joint_target'] = research


def fixed_target_planner_factory(fixed_target):
    """Factory for the strategy's existing planner injection seam, outward only."""
    if not isinstance(fixed_target, FixedJointTarget):
        raise TypeError('Expected FixedJointTarget')
    def create(backend, **options):
        return FixedJointTargetPlanner(backend, fixed_target=fixed_target, **options)
    return create
