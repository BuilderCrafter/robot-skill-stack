"""Project-owned global collision model and its measured operating contract.

This module imports no simulator SDK and never changes robot state. The fixed
base is checked once against the frozen world; moving-link spheres remain in
the same Lula description used for search and trajectory validation.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re

import numpy as np

from robot_skill_stack.common.rotations import quat_matrix
from robot_skill_stack.manipulation.motion.joint_route import PlanningFailure


MODEL_DIRECTORY = Path(__file__).resolve().parents[5] / 'config' / 'motion'
MODEL_NAME = 'franka_enhanced_v4_2'
FINGER_NAMES = ('panda_finger_joint1', 'panda_finger_joint2')
FINGER_FRAMES = ('panda_leftfinger', 'panda_rightfinger')


def validate_model_selection(value):
    if not isinstance(value, str) or value not in ('stock', 'enhanced'):
        raise ValueError("global_collision_model must be 'stock' or 'enhanced'")
    return value


def _finite(value, shape, label):
    array = np.asarray(value, dtype=float)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f'{label} must contain finite values with shape {shape}')
    return array


@dataclass(frozen=True)
class EnhancedCollisionModel:
    descriptor_path: Path
    descriptor_sha256: str
    finger_range_m: tuple
    base_center: tuple
    base_radius: float
    base_corners: tuple
    urdf_sha256: str
    metadata_sha256: str
    base_link: str = 'panda_link0'
    certified_self_collision_exclusions: frozenset = frozenset()

    def check_gripper(self, robot):
        """Read named physical joints, plus native latched targets if exposed.

        The installed SingleArticulation getter reads PhysX target buffers
        during play. USD drive attributes are deliberately not consulted.
        Position targets do not certify measured positions; both are checked.
        """
        details = dict(finger_joint_names=list(FINGER_NAMES), finger_range_m=list(self.finger_range_m))
        try:
            names = tuple(robot.dof_names)
            if len(names) != len(set(names)) or not set(FINGER_NAMES) <= set(names):
                raise ValueError('Missing or ambiguous named finger joints')
            indices = tuple(names.index(name) for name in FINGER_NAMES)
            all_positions = _finite(robot.get_joint_positions(), (len(names),), 'Articulation positions')
            measured = all_positions[list(indices)]
            details['measured_positions_m'] = measured.tolist()
            low, high = self.finger_range_m
            if np.any(measured < low) or np.any(measured > high):
                raise ValueError('Measured fingers leave the validated collision-model range')
            details.update(joint_names=list(FINGER_NAMES), measured_positions_m=measured.tolist(),
                validated_range_m=list(self.finger_range_m), native_position_targets_m=None,
                target_feedback='unavailable; measured positions checked')
            getter = getattr(robot, 'get_applied_action', None)
            applied = getter() if callable(getter) else None
            if applied is not None and getattr(applied, 'joint_positions', None) is not None:
                positions = np.asarray(applied.joint_positions, dtype=float)
                action_indices = getattr(applied, 'joint_indices', None)
                if action_indices is None:
                    targets = _finite(positions, (len(names),), 'Native position targets')[list(indices)]
                else:
                    action_indices = np.asarray(action_indices)
                    if (action_indices.ndim != 1 or action_indices.dtype.kind not in 'iu'
                            or len(set(action_indices.tolist())) != len(action_indices)):
                        raise ValueError('Ambiguous native target joint indices')
                    _finite(positions, (len(action_indices),), 'Native position targets')
                    mapping = dict(zip(action_indices.tolist(), positions.tolist()))
                    if not set(indices) <= set(mapping):
                        details['target_feedback'] = 'partial action omits fingers; measured positions checked'
                        return details
                    targets = np.array([mapping[index] for index in indices])
                details['native_position_targets_m'] = targets.tolist()
                if np.any(targets < low) or np.any(targets > high):
                    raise ValueError('Native finger targets leave the validated collision-model range')
                details.update(native_position_targets_m=targets.tolist(),
                    target_feedback='SingleArticulation.get_applied_action native PhysX position targets during play')
            return details
        except Exception as error:
            raise PlanningFailure('unsupported_gripper_configuration', str(error), details) from error

    def preflight_base(self, snapshot, frame_position_m, frame_rotation):
        """Reject possible base collisions before RRT using an enclosing ball.

        An enclosing ball can falsely obstruct a finite nearby obstacle. For a
        certified plane only, the cooked-solid AABB corners prove half-space
        nonpenetration without falsely rejecting the mounted base's tangency.
        """
        position = _finite(frame_position_m, (3,), 'Base-frame position')
        rotation = _finite(frame_rotation, (3, 3), 'Base-frame rotation')
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-9, rtol=0):
            raise PlanningFailure('collision_model_invalid', 'Base-frame rotation is not orthonormal')
        center = rotation @ np.asarray(self.base_center) + position
        corners = np.asarray(self.base_corners) @ rotation.T + position
        rows = []
        for obstacle in snapshot.obstacles:
            if (obstacle.source_kind == 'Plane'
                    and obstacle.approximation == 'plane_halfspace_clipped_to_certified_domain'):
                matches = [item for item in snapshot.plane_domains if item.path == obstacle.path]
                if len(matches) != 1:
                    raise PlanningFailure('unsupported_geometry', 'Mounted-base plane has no unique certified domain')
                domain = matches[0]
                plane_rotation = quat_matrix(domain.orientation)
                expected = np.asarray(domain.position)-plane_rotation[:, 2]*domain.depth_m/2
                if (obstacle.maximum_surface_error_m != 0.
                        or not np.allclose(quat_matrix(obstacle.orientation), plane_rotation, atol=1e-10, rtol=0)
                        or not np.allclose(obstacle.position, expected, atol=1e-9, rtol=0)
                        or not np.allclose(obstacle.dimensions,
                            [2*domain.half_width_m, 2*domain.half_width_m, domain.depth_m], atol=1e-9, rtol=0)):
                    raise PlanningFailure('unsupported_geometry', 'Mounted-base plane and certified slab disagree')
                try:
                    domain.assert_contains(corners, np.zeros(len(corners)))
                except ValueError as error:
                    raise PlanningFailure('unsupported_geometry', str(error)) from error
                clearance = float(np.min((corners-np.asarray(domain.position)) @ plane_rotation[:, 2]))
                method = 'cooked-base AABB corners versus certified plane halfspace'
            else:
                offset = (center-np.asarray(obstacle.position)) @ quat_matrix(obstacle.orientation)
                if obstacle.kind == 'cuboid':
                    distance = np.linalg.norm(np.maximum(np.abs(offset)-np.asarray(obstacle.dimensions)/2, 0.))
                elif obstacle.kind == 'sphere':
                    distance = np.linalg.norm(offset)-obstacle.radius
                elif obstacle.kind == 'capsule':
                    nearest = np.array([0., 0., np.clip(offset[2], -obstacle.height/2, obstacle.height/2)])
                    distance = np.linalg.norm(offset-nearest)-obstacle.radius
                else:
                    raise PlanningFailure('unsupported_geometry', 'Unsupported fixed-base obstacle primitive')
                clearance = float(distance-self.base_radius)
                method = 'cooked-base enclosing sphere versus finite obstacle'
            rows.append(dict(path=obstacle.path, clearance_lower_bound_m=clearance, method=method))
            if clearance < 0. or (method.endswith('finite obstacle') and clearance == 0.):
                raise PlanningFailure('base_geometry_rejected',
                    'Fixed-base enclosure cannot establish clearance from the frozen obstacle',
                    dict(base_preflight=rows, conservative_rejection=True,
                        physical_collision_proven=False))
        return dict(checks=rows, fixed_base_frame=self.base_link,
            all_configurations=True, conservative_finite_obstacle_enclosure=True)


def load_enhanced_collision_model(directory=None):
    """Load immutable project artifacts; no temporary files or SDK resources."""
    folder = MODEL_DIRECTORY if directory is None else Path(directory)
    descriptor = folder / (MODEL_NAME+'.yaml')
    try:
        metadata_bytes = (folder / (MODEL_NAME+'.json')).read_bytes()
        metadata = json.loads(metadata_bytes)
        descriptor_bytes = descriptor.read_bytes()
        digest = hashlib.sha256(descriptor_bytes).hexdigest()
        if metadata['descriptor_sha256'] != digest:
            raise ValueError('Enhanced descriptor and geometry metadata hashes disagree')
        if (metadata['schema_version'] != 1 or metadata['model'] != MODEL_NAME
                or metadata['sphere_count'] != 102 or metadata['stock_sphere_count'] != 55
                or metadata['added_sphere_count'] != 47):
            raise ValueError('Unsupported frozen enhanced collision-model version')
        urdf_digest = metadata['urdf_sha256']
        if not isinstance(urdf_digest, str) or len(urdf_digest) != 64 or any(
                value not in '0123456789abcdef' for value in urdf_digest):
            raise ValueError('Enhanced-model URDF hash is invalid')
        if tuple(metadata['finger_joint_names']) != FINGER_NAMES or metadata['base_link'] != 'panda_link0':
            raise ValueError('Unsupported enhanced-model finger names or base frame')
        limits = _finite(metadata['finger_range_m'], (2,), 'Finger range')
        fixed = float(metadata['descriptor_fixed_finger_m'])
        if (tuple(limits) != (.001, .005) or fixed != .003
                or fixed != float(np.mean(limits))
                or metadata['finger_sweep_padding_m'] != float((limits[1]-limits[0])/2)):
            raise ValueError('Invalid enhanced-model finger operating range')
        # Accept only the generator's frozen two-rule section. This verifies
        # its actual fixed joints without introducing a runtime YAML parser.
        rules = re.findall(r'^cspace_to_urdf_rules:\n(.*?)^collision_spheres:',
            descriptor_bytes.decode(), flags=re.MULTILINE | re.DOTALL)
        expected = ''.join(f'- name: {name}\n  rule: fixed\n  value: {fixed}\n' for name in FINGER_NAMES)
        if rules != [expected]:
            raise ValueError('Descriptor fixed-finger rules differ from the validated contract')
        sphere = metadata['base_enclosing_sphere']
        center = _finite(sphere['center'], (3,), 'Base sphere center')
        radius = float(sphere['radius'])
        corners = _finite(metadata['base_aabb_corners_m'], (8, 3), 'Base AABB corners')
        if not np.isfinite(radius) or radius <= 0:
            raise ValueError('Base sphere radius must be positive and finite')
        low, high = corners.min(axis=0), corners.max(axis=0)
        if (np.any(low >= high) or not np.all((corners == low) | (corners == high))
                or len({tuple(row) for row in corners}) != 8):
            raise ValueError('Base AABB must contain its eight distinct corners')
        exclusions = metadata['certified_self_collision_exclusions']
        if len(exclusions) != 1:
            raise ValueError('Only one analytically certified finger-pair exclusion is supported')
        certificate = exclusions[0]
        projection = float(certificate['minimum_separating_projection_m'])
        tolerance = float(certificate['numerical_tolerance_m'])
        intervals = metadata['finger_hand_y_intervals_m']
        left = _finite(intervals[FINGER_FRAMES[0]], (2,), 'Left finger projection')
        right = _finite(intervals[FINGER_FRAMES[1]], (2,), 'Right finger projection')
        hashes = certificate['cooked_finger_vertices_sha256']
        if (tuple(certificate['frames']) != FINGER_FRAMES
                or tuple(certificate['finger_range_m']) != tuple(limits)
                or certificate['urdf_sha256'] != urdf_digest
                or not np.isfinite(projection) or not np.isfinite(tolerance)
                or not 0 <= tolerance <= 1e-9 or projection <= tolerance
                or left[0] > left[1] or right[0] > right[1]
                or abs(float(left[0]-right[1])-projection) > 1e-12
                or set(hashes) != set(FINGER_FRAMES)
                or any(not isinstance(value, str) or len(value) != 64
                    or any(character not in '0123456789abcdef' for character in value)
                    for value in hashes.values())):
            raise ValueError('Finger-pair exclusion lacks its matching full-range separating certificate')
        return EnhancedCollisionModel(descriptor, digest, tuple(limits), tuple(center), radius,
            tuple(tuple(row) for row in corners), urdf_digest, hashlib.sha256(metadata_bytes).hexdigest(),
            certified_self_collision_exclusions=frozenset((frozenset(FINGER_FRAMES),)))
    except Exception as error:
        raise PlanningFailure('collision_model_invalid', f'Enhanced collision model could not be loaded: {error}') from error
