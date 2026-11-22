"""Independent native regression reference; extracted from preserved source cdd2be9.

Numeric algorithms and safety allowances are retained unchanged; no planning input.
"""
from dataclasses import dataclass
import xml.etree.ElementTree as ET
import numpy as np


def vector(text):
    value = np.fromstring(text, sep=' ')
    if value.shape != (3,) or not np.isfinite(value).all():
        raise ValueError('Finite URDF xyz/rpy/axis triples are required')
    return value


def axis_rotation(axis, angle):
    axis = np.asarray(axis, float)
    norm = np.linalg.norm(axis)
    if axis.shape != (3,) or not np.isfinite(axis).all() or norm < 1e-15 or not np.isfinite(angle):
        raise ValueError('Finite nonzero URDF axis and angle required')
    x, y, z = axis/norm
    skew = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
    return np.eye(3)+np.sin(angle)*skew+(1-np.cos(angle))*(skew@skew)


@dataclass
class Joint:
    name: str
    kind: str
    parent: str
    child: str
    origin: np.ndarray
    axis: np.ndarray


class IndependentUrdfFK:
    """URDF origin Rz(yaw)Ry(pitch)Rx(roll), followed by joint motion."""
    def __init__(self, path):
        document = ET.parse(path).getroot()
        self.joints = []
        children = set()
        for element in document.findall('joint'):
            origin = element.find('origin')
            xyz = vector('0 0 0' if origin is None else origin.get('xyz', '0 0 0'))
            rpy = vector('0 0 0' if origin is None else origin.get('rpy', '0 0 0'))
            transform = np.eye(4)
            transform[:3, :3] = (axis_rotation([0, 0, 1], rpy[2])@
                axis_rotation([0, 1, 0], rpy[1])@axis_rotation([1, 0, 0], rpy[0]))
            transform[:3, 3] = xyz
            axis_element = element.find('axis')
            axis = vector('0 0 1' if axis_element is None else axis_element.get('xyz', '0 0 1'))
            kind = element.get('type')
            if kind not in ('fixed', 'revolute', 'continuous', 'prismatic'):
                raise ValueError('Unsupported URDF joint type')
            if kind != 'fixed' and not np.isclose(np.linalg.norm(axis), 1., atol=1e-12, rtol=0):
                raise ValueError('The frozen physical model requires unit joint axes')
            child = element.find('child').get('link')
            children.add(child)
            self.joints.append(Joint(element.get('name'), kind, element.find('parent').get('link'),
                                     child, transform, axis))
        roots = {entry.get('name') for entry in document.findall('link')}-children
        if len(roots) != 1:
            raise ValueError('A connected URDF with one root is required')
        self.root, = roots

    def poses(self, names, positions):
        values = np.asarray(positions, float)
        if len(names) != len(set(names)) or values.shape != (len(names),) or not np.isfinite(values).all():
            raise ValueError('Unique names and finite measured joint positions are required')
        coordinates = dict(zip(names, values))
        result, remaining = {self.root: np.eye(4)}, list(self.joints)
        while remaining:
            progress = False
            for joint in remaining[:]:
                if joint.parent not in result:
                    continue
                motion = np.eye(4)
                if joint.kind != 'fixed' and joint.name not in coordinates:
                    raise ValueError('Missing independent physical joint coordinate: '+joint.name)
                value = coordinates.get(joint.name, 0.)
                if joint.kind in ('revolute', 'continuous'):
                    motion[:3, :3] = axis_rotation(joint.axis, value)
                elif joint.kind == 'prismatic':
                    motion[:3, 3] = joint.axis*value
                # Both captured finger coordinates are independent physical
                # observations; never replace finger2 by its URDF mimic rule.
                result[joint.child] = result[joint.parent]@joint.origin@motion
                remaining.remove(joint)
                progress = True
            if not progress:
                raise ValueError('Disconnected/cyclic URDF joint graph')
        return result


def intersection_ball(vertex_sets, extra_halfspaces=()):
    """Verify a common positive-radius ball; zero distance alone is insufficient."""
    from scipy.optimize import linprog
    from scipy.spatial import ConvexHull
    equations = [ConvexHull(np.asarray(vertices, float)).equations for vertices in vertex_sets]
    if len(extra_halfspaces):
        equations.append(np.asarray(extra_halfspaces, float))
    planes = np.concatenate(equations)
    normals, offsets = planes[:, :3], planes[:, 3]
    norms = np.linalg.norm(normals, axis=1)
    if not np.isfinite(planes).all() or np.any(norms <= 0):
        raise ValueError('Finite nondegenerate halfspaces are required')
    normals, offsets = normals/norms[:, None], offsets/norms
    matrix = np.c_[normals, np.ones(len(normals))]
    answer = linprog([0., 0., 0., -1.], A_ub=matrix, b_ub=-offsets,
        bounds=[(None, None)]*3+[(0., None)], method='highs')
    if not answer.success:
        return dict(status='NO_VERIFIED_INTERSECTION_BALL', solver_status=int(answer.status))
    center, raw_radius = answer.x[:3], float(answer.x[3])
    residual = float(np.max(normals@center+offsets+raw_radius))
    radius = max(0., raw_radius-max(0., residual)-1e-9)
    verified_residual = float(np.max(normals@center+offsets+radius))
    positive = radius > 0. and verified_residual <= 0.
    return dict(status='VERIFIED_POSITIVE_INTERSECTION_BALL' if positive else 'NO_POSITIVE_RADIUS_PROVEN',
        center_m=center.tolist(), radius_m=radius, raw_radius_m=raw_radius,
        maximum_constraint_residual_m=verified_residual,
        scope='Floating-point verified halfspace witness of numeric solid intersection, not observed native contact.')


def candidate_world_poses(fk, names, positions, base):
    relative = fk.poses(names, positions)
    root = base@np.linalg.inv(relative['panda_link0'])
    return {name: root@pose for name, pose in relative.items()}


def body_mesh_vertices(vertices, body_from_mesh):
    points, transform = np.asarray(vertices, float), np.asarray(body_from_mesh, float)
    if (points.ndim != 2 or points.shape[1:] != (3,) or transform.shape != (4, 4)
            or not np.isfinite(points).all() or not np.isfinite(transform).all()
            or not np.allclose(transform[3], [0, 0, 0, 1])
            or abs(np.linalg.det(transform[:3, :3])) < 1e-15):
        raise ValueError("Finite nondegenerate physical mesh-to-body transform required")
    return points@transform[:3, :3].T+transform[:3, 3]
