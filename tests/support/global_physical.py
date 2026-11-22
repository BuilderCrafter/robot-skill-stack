"""Independent native regression reference; extracted from preserved source cdd2be9.

Numeric algorithms and safety allowances are retained unchanged; no planning input.
"""
from dataclasses import dataclass
import hashlib,itertools,time
from pathlib import Path
import numpy as np
from robot_skill_stack.common.rotations import quat_matrix
from tests.support.physical_fk import IndependentUrdfFK,candidate_world_poses,intersection_ball,body_mesh_vertices
from tests.support.convex_distance import convex_distance_details
ARM_NAMES=tuple(f'panda_joint{i}' for i in range(1,8))
FINGER_NAMES=('panda_finger_joint1','panda_finger_joint2')
BODY_NAMES=frozenset((*(f'panda_link{i}' for i in range(8)),'panda_hand','panda_leftfinger','panda_rightfinger'))
NUMERIC_FRAME_ALLOWANCE_M=1e-5


def immutable(value, *, dtype=float):
    array = np.ascontiguousarray(value, dtype=dtype)
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def body_name(value):
    return str(value).rstrip('/').rsplit('/', 1)[-1]


def _points(value, *, minimum=1):
    result = np.asarray(value, float)
    if result.ndim != 2 or result.shape[1:] != (3,) or len(result) < minimum or not np.isfinite(result).all():
        raise ValueError('Expected finite Nx3 physical vertices/points')
    return result


def _transform(value):
    result = np.asarray(value, float)
    if (result.shape != (4, 4) or not np.isfinite(result).all()
            or not np.allclose(result[3], [0., 0., 0., 1.], atol=1e-12, rtol=0.)
            or not np.allclose(result[:3, :3].T@result[:3, :3], np.eye(3), atol=1e-6, rtol=0.)
            or np.linalg.det(result[:3, :3]) <= 0):
        raise ValueError('Expected a finite rigid world/body transform in meters')
    return result


@dataclass(frozen=True)
class CookedRobotPart:
    """One copied convex cooked hull, already converted into its body frame."""
    body: str
    vertices: np.ndarray
    triangles: np.ndarray | None = None
    part_id: str = ''

    def __post_init__(self):
        name = body_name(self.body)
        vertices = _points(self.vertices, minimum=4)
        faces = self.triangles
        if faces is not None:
            faces = np.asarray(faces)
            if (faces.ndim != 2 or faces.shape[1:] != (3,) or faces.dtype.kind not in 'iu'
                    or np.any(faces < 0) or np.any(faces >= len(vertices))):
                raise ValueError('Copied cooked triangle indices are invalid')
            faces = immutable(faces, dtype=np.int64)
        object.__setattr__(self, 'body', name)
        object.__setattr__(self, 'vertices', immutable(vertices))
        object.__setattr__(self, 'triangles', faces)
        object.__setattr__(self, 'part_id', self.part_id or name)


@dataclass(frozen=True)
class PhysicalObstacle:
    """Independent world-space cooked hull or analytic physical support plane."""
    path: str
    kind: str
    vertices: np.ndarray | None = None
    point: tuple | None = None
    normal: tuple | None = None
    known_support: bool = False

    def __post_init__(self):
        if not isinstance(self.path, str) or not self.path:
            raise ValueError('Physical evaluation obstacles require stable paths')
        if self.kind == 'convex':
            object.__setattr__(self, 'vertices', immutable(_points(self.vertices, minimum=4)))
        elif self.kind == 'plane':
            point, normal = np.asarray(self.point, float), np.asarray(self.normal, float)
            if (point.shape != (3,) or normal.shape != (3,) or not np.isfinite(point).all()
                    or not np.isfinite(normal).all() or np.linalg.norm(normal) <= 0.):
                raise ValueError('Physical planes require a point and nonzero world normal')
            object.__setattr__(self, 'point', tuple(map(float, point)))
            object.__setattr__(self, 'normal', tuple(map(float, normal/np.linalg.norm(normal))))
        else:
            raise ValueError('Only actual cooked convex hulls and analytic physical planes are evaluated')

    @classmethod
    def convex(cls, path, world_vertices):
        return cls(path=path, kind='convex', vertices=world_vertices)

    @classmethod
    def plane(cls, path, point, normal, *, known_support=True):
        return cls(path=path, kind='plane', point=tuple(point), normal=tuple(normal), known_support=known_support)


def robot_parts_from_capture(meshes):
    """Copy numeric records on the owner; retain no SDK/mesh object handles."""
    result = []
    for index, part in enumerate(meshes):
        if not str(part.approximation).startswith('physx_cooked:'):
            raise ValueError('Research physical reference requires actual PhysX-cooked robot geometry')
        result.append(CookedRobotPart(part.body_path,
            body_mesh_vertices(part.mesh.vertices, part.body_from_mesh),
            part.mesh.triangles, part_id=f'{getattr(part, "path", part.body_path)}:{index}'))
    if {part.body for part in result} != BODY_NAMES:
        raise ValueError('Physical reference must include all eleven Franka bodies')
    return tuple(result)


def physical_obstacles_from_static(snapshot):
    """Evaluation convenience for exact boxes/planes only; never a source adapter.

    Native cylinders and meshes must use copied cooked hulls through
    PhysicalObstacle.convex rather than an enclosing primitive approximation.
    """
    result = []
    for item in snapshot.obstacles:
        rotation = quat_matrix(item.orientation)
        if item.source_kind == 'Plane':
            point = np.asarray(item.position)+rotation[:, 2]*item.dimensions[2]/2
            result.append(PhysicalObstacle.plane(item.path, point, rotation[:, 2]))
        elif item.kind == 'cuboid' and item.approximation == 'exact':
            half = np.asarray(item.dimensions)/2
            vertices = np.array(list(itertools.product(*[(-x, x) for x in half])))
            result.append(PhysicalObstacle.convex(item.path, vertices@rotation.T+item.position))
        else:
            raise ValueError('Non-box fixture evaluation requires its actual copied cooked convex hulls')
    return tuple(result)


def densify_joint_route(route, *, max_joint_step_rad=.01, max_samples=12000):
    values = np.asarray(route, float)
    if (values.ndim != 2 or values.shape[1:] != (7,) or not len(values)
            or not np.isfinite(values).all() or not 0. < max_joint_step_rad <= .01):
        raise ValueError('Physical route sampling requires finite Nx7 q and the original <=.01rad spacing')
    counts = np.maximum(1, np.ceil(np.max(abs(np.diff(values, axis=0)), axis=1)/max_joint_step_rad).astype(int))
    if int(counts.sum())+1 > max_samples:
        raise ValueError('Research physical route sample budget exceeded')
    return np.concatenate([*[np.linspace(a, b, int(n)+1)[:-1]
        for a, b, n in zip(values[:-1], values[1:], counts)], values[-1:]])


def _sampled_motion(payload, max_samples):
    names = tuple(payload['joint_names'])
    order = tuple(payload.get('route_joint_names', ARM_NAMES))
    if (len(names) != 9 or len(set(names)) != 9 or set(names) != set((*ARM_NAMES, *FINGER_NAMES))
            or len(order) != 7 or len(set(order)) != 7 or set(order) != set(ARM_NAMES)):
        raise ValueError('Explicit unambiguous seven-arm/nine-DOF ordering is required')
    route = densify_joint_route(payload['route'], max_samples=max_samples)
    spline, times = np.asarray(payload['trajectory_q'], float), np.asarray(payload['trajectory_times'], float)
    if (spline.ndim != 2 or spline.shape[1:] != (7,) or not len(spline)
            or times.shape != (len(spline),) or not np.isfinite(spline).all() or not np.isfinite(times).all()
            or np.any(np.diff(times) < 0.) or np.any(np.diff(times) > 1/240+1e-10)
            or np.any(np.max(abs(np.diff(spline, axis=0)), axis=1) > .01001)):
        raise ValueError('Physical spline samples must retain original <=1/240s and <=.01001rad spacing')
    if len(route)+len(spline) > max_samples:
        raise ValueError('Research physical route+spline sample budget exceeded')
    if (np.max(abs(route[0]-spline[0])) > 1e-6 or np.max(abs(route[-1]-spline[-1])) > 1e-6):
        raise ValueError('Physical route and generated spline endpoints disagree')
    finger = np.asarray(payload['finger_positions'], float)
    if finger.shape != (2,) or not np.isfinite(finger).all() or np.any(finger < .001) or np.any(finger > .005):
        raise ValueError('Physical evaluation requires measured fingers in the validated [1,5]mm range')
    arm = np.concatenate((route, spline))
    q = np.zeros((len(arm), 9))
    for column, name in enumerate(order):
        q[:, names.index(name)] = arm[:, column]
    for value, name in zip(finger, FINGER_NAMES):
        q[:, names.index(name)] = value
    return names, q, ['route']*len(route)+['trajectory']*len(spline), [None]*len(route)+times.tolist()


def _aabb_bound(first, last):
    return float(np.linalg.norm(np.maximum(np.maximum(first.min(0)-last.max(0), last.min(0)-first.max(0)), 0.)))


def _convex_relation(first, last, allowance):
    bound = _aabb_bound(first, last)
    if bound > allowance:
        return dict(lower_bound_m=bound, upper_bound_m=None, method='CONSERVATIVE_AABB_SEPARATION',
            collision_proven=False, unresolved=False, intersection_ball=None)
    detail = convex_distance_details(first, last)
    ball = intersection_ball([first, last]) if detail['lower_bound_m'] <= allowance else None
    collision = bool(ball and ball['status'] == 'VERIFIED_POSITIVE_INTERSECTION_BALL' and ball['radius_m'] > 0.)
    return dict(detail, method='COOKED_VERTEX_HULL_GJK', collision_proven=collision,
        unresolved=not collision and detail['lower_bound_m'] <= allowance, intersection_ball=ball)


class PhysicalMotionEvaluator:
    """Callable for the existing BoundedWorker; contains no simulator handles.

    Evaluation may conservatively veto an unresolved query. Such a veto is not
    evidence that the RGB-D planner avoided the hazard, nor proof of collision.
    Initial FK is checked against actual body poses independently of Lula.
    """
    def __init__(self, robot_parts, obstacles, urdf_path, *, include_self=False,
                 max_queries=100000, max_samples=12000,
                 frame_allowance_m=NUMERIC_FRAME_ALLOWANCE_M, require_all_robot_bodies=True):
        self.parts = tuple(part if isinstance(part, CookedRobotPart) else CookedRobotPart(**part)
                           for part in robot_parts)
        self.obstacles = tuple(item if isinstance(item, PhysicalObstacle) else PhysicalObstacle(**item)
                               for item in obstacles)
        if not self.obstacles:
            raise ValueError('Physical evaluation cannot silently use an empty ground-truth reference world')
        if require_all_robot_bodies and {part.body for part in self.parts} != BODY_NAMES:
            raise ValueError('Physical veto must retain all eleven cooked robot bodies')
        if not self.parts or len({part.part_id for part in self.parts}) != len(self.parts):
            raise ValueError('Physical parts require unique IDs and nonempty geometry')
        if type(max_queries) is not int or max_queries < 1 or type(max_samples) is not int or max_samples < 2:
            raise ValueError('Physical evaluation budgets must be positive integers')
        if not np.isfinite(frame_allowance_m) or not 0. <= frame_allowance_m <= NUMERIC_FRAME_ALLOWANCE_M:
            raise ValueError('Physical frame allowance cannot exceed the source-bound 10micrometre allowance')
        self.urdf_path, self.fk = str(urdf_path), IndependentUrdfFK(urdf_path)
        self.urdf_sha256 = hashlib.sha256(Path(urdf_path).read_bytes()).hexdigest()
        self.include_self, self.max_queries, self.max_samples = bool(include_self), max_queries, max_samples
        self.allowance = float(frame_allowance_m)
        from robot_skill_stack.integrations.isaac.manipulation.motion.lula_global import adjacent_self_collision_pairs
        self.self_exclusions = adjacent_self_collision_pairs(urdf_path)

    def _actual_alignment(self, payload, names, base):
        q = np.asarray(payload['actual_start_q'], float)
        if q.shape != (9,) or not np.isfinite(q).all():
            raise ValueError('Actual initial named nine-DOF feedback is required')
        actual = {body_name(k):_transform(v) for k, v in payload['actual_start_body_poses'].items()}
        expected = candidate_world_poses(self.fk, names, q, base)
        if not {part.body for part in self.parts} <= set(actual):
            raise ValueError('Actual alignment observations omit a cooked robot body')
        translation = max(float(np.linalg.norm(expected[name][:3, 3]-pose[:3, 3])) for name, pose in actual.items()
                          if name in {part.body for part in self.parts})
        rotation = max(float(np.max(abs(expected[name][:3, :3]-pose[:3, :3]))) for name, pose in actual.items()
                       if name in {part.body for part in self.parts})
        return dict(maximum_translation_error_m=translation, maximum_rotation_matrix_error=rotation,
            passed=translation <= self.allowance and rotation <= 1e-5,
            frame_allowance_m=self.allowance, rotation_matrix_allowance=1e-5)

    def __call__(self, payload):
        began = time.perf_counter()
        names, positions, phases, times = _sampled_motion(payload, self.max_samples)
        actual_start = np.asarray(payload['actual_start_q'], float)
        if actual_start.shape != (9,) or np.max(abs(positions[0]-actual_start)) > 1e-6:
            raise ValueError('Planned physical route does not begin at the measured source configuration')
        for name in FINGER_NAMES:
            if abs(positions[0, names.index(name)]-actual_start[names.index(name)]) > 1e-12:
                raise ValueError('Physical route finger coordinates differ from actual measured fingers')
        base = _transform(payload['base_transform'])
        alignment = self._actual_alignment(payload, names, base)
        result = dict(schema_version=1, source='GROUND_TRUTH_RESEARCH_EVALUATION_ONLY',
            planner_world_modified=False, reference_urdf_sha256=self.urdf_sha256,
            alignment=alignment, total_samples=len(positions), evaluated_samples=0,
            phases_requested={name:phases.count(name) for name in ('route','trajectory')},
            phases_evaluated={'route':0,'trajectory':0}, queries=0, broadphase_separations=0,
            minimum_external_clearance_lower_bound_m=None, minimum_nonmount_clearance_lower_bound_m=None,
            closest_bound_pair=None, collision_witnesses=[], unresolved=[], per_sample=[],
            self_evaluated=self.include_self,
            finger_scope='MEASURED_FIXED_FINGERS; future joint-range and tracking variations are not a swept-volume certificate',
            qualification='Source-bound independent URDF FK and copied cooked vertex hulls at declared route/spline samples. AABB separations are lower bounds, not exact closest distances. No continuous swept-volume, contact-offset or real-robot safety claim. GT veto never counts as RGBD avoidance.')
        if not alignment['passed']:
            result.update(status='VETO_UNVERIFIED_FK_ALIGNMENT', execution_allowed=False,
                elapsed_seconds=time.perf_counter()-began, samples_complete=False)
            return result
        closest = np.inf
        nonmount = np.inf
        stop = bool(payload.get('stop_on_collision', False))
        budget_exhausted = False
        for index, (q, phase, stamp) in enumerate(zip(positions, phases, times)):
            poses = candidate_world_poses(self.fk, names, q, base)
            world = [(part, part.vertices@poses[part.body][:3, :3].T+poses[part.body][:3, 3]) for part in self.parts]
            minimum = np.inf
            sample_nonmount = np.inf
            sample_collision = False
            relations = []
            for part, vertices in world:
                for obstacle in self.obstacles:
                    relations.append((part, vertices, obstacle))
            for part, vertices, obstacle in relations:
                if result['queries'] >= self.max_queries:
                    budget_exhausted = True
                    break
                result['queries'] += 1
                mount = part.body == 'panda_link0' and obstacle.kind == 'plane' and obstacle.known_support
                if obstacle.kind == 'plane':
                    normal, point = np.asarray(obstacle.normal), np.asarray(obstacle.point)
                    distance = float(np.min((vertices-point)@normal))
                    ball = intersection_ball([vertices], [np.r_[normal, -normal@point]]) if distance < 0. else None
                    relation = dict(lower_bound_m=distance, upper_bound_m=distance, method='ANALYTIC_PHYSICAL_PLANE',
                        collision_proven=bool(ball and ball['status']=='VERIFIED_POSITIVE_INTERSECTION_BALL'),
                        unresolved=distance < 0. if mount else distance <= self.allowance, intersection_ball=ball)
                else:
                    relation = _convex_relation(vertices, obstacle.vertices, self.allowance)
                result['broadphase_separations'] += int(relation['method']=='CONSERVATIVE_AABB_SEPARATION')
                lower = relation['lower_bound_m']
                minimum = min(minimum, lower)
                if not mount:
                    nonmount = min(nonmount, lower)
                    sample_nonmount = min(sample_nonmount, lower)
                if lower < closest:
                    closest = lower
                    result['closest_bound_pair'] = dict(sample=index, phase=phase, time=stamp,
                        robot_body=part.body, robot_part=part.part_id, obstacle=obstacle.path,
                        clearance_lower_bound_m=lower, qualified_mount=mount)
                row = dict(sample=index,phase=phase,time=stamp,robot_body=part.body,robot_part=part.part_id,
                    obstacle=obstacle.path,qualified_mount=mount,**relation)
                if relation['collision_proven']:
                    sample_collision = True
                    result['collision_witnesses'].append(row)
                elif relation['unresolved']:
                    result['unresolved'].append(row)
            if not budget_exhausted and self.include_self:
                for (first, left), (last, right) in itertools.combinations(world, 2):
                    if first.body == last.body or frozenset((first.body, last.body)) in self.self_exclusions:
                        continue
                    if result['queries'] >= self.max_queries:
                        budget_exhausted = True
                        break
                    result['queries'] += 1
                    relation = _convex_relation(left, right, self.allowance)
                    row = dict(sample=index,phase=phase,time=stamp,robot_bodies=[first.body,last.body],**relation)
                    if relation['collision_proven']:
                        sample_collision = True
                        result['collision_witnesses'].append(row)
                    elif relation['unresolved']:
                        result['unresolved'].append(row)
            if not budget_exhausted:
                result['evaluated_samples'] += 1
                result['phases_evaluated'][phase] += 1
            result['per_sample'].append(dict(sample=index,phase=phase,time=stamp,
                complete=not budget_exhausted,
                minimum_nonmount_clearance_lower_bound_m=float(sample_nonmount) if np.isfinite(sample_nonmount) else None,
                minimum_external_clearance_lower_bound_m=float(minimum) if np.isfinite(minimum) else None))
            if budget_exhausted or (stop and sample_collision):
                break
        result['minimum_external_clearance_lower_bound_m'] = float(closest) if np.isfinite(closest) else None
        result['minimum_nonmount_clearance_lower_bound_m'] = float(nonmount) if np.isfinite(nonmount) else None
        result['samples_complete'] = result['evaluated_samples'] == len(positions) and not budget_exhausted
        status = ('VETO_PREDICTED_PHYSICAL_COLLISION' if result['collision_witnesses'] else
                  'VETO_QUERY_BUDGET' if budget_exhausted else
                  'VETO_UNRESOLVED_CLEARANCE' if result['unresolved'] else
                  'ALLOW_SAMPLED_NUMERIC_CLEARANCE' if result['samples_complete'] else 'VETO_INCOMPLETE_EVALUATION')
        result.update(status=status, execution_allowed=status=='ALLOW_SAMPLED_NUMERIC_CLEARANCE',
            elapsed_seconds=time.perf_counter()-began)
        return result
