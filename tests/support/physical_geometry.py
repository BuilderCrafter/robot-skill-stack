"""Independent USD-mesh/cuboid clearance reduction, outside measured control.

The reference is copied PhysX cooked convex geometry, with any fallback to
authored CollisionAPI triangles explicitly identified. It does not use
occupancy or RMPflow's collision spheres. Native contact reports are recorded
separately. Distances are unsigned and sampled at captured body poses.
"""
from dataclasses import dataclass
import heapq

import numpy as np

from robot_skill_stack.world.spatial_mapping.robot_filter import TriangleMesh


def convex_polygon_triangles(vertices,counts,indices):
    """Fan triangulate SDK-guaranteed ordered convex faces without recooking.

    PhysX cooked float32 face vertices can have tiny nonplanarity after storage.
    Their SDK polygon topology is authoritative; the stricter authored-USD mesh
    validator is intentionally not applied to these copied SDK faces.
    """
    vertices,counts,indices = np.asarray(vertices,float),np.asarray(counts),np.asarray(indices)
    if (vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all()
            or counts.ndim != 1 or counts.dtype.kind not in 'iu' or np.any(counts<3)
            or indices.ndim != 1 or indices.dtype.kind not in 'iu'
            or int(counts.sum()) != len(indices) or np.any(indices<0) or np.any(indices>=len(vertices))):
        raise ValueError('Invalid copied PhysX convex polygon topology')
    triangles,cursor = [],0
    for count in counts:
        face = indices[cursor:cursor+int(count)]
        triangles.extend((int(face[0]),int(face[i]),int(face[i+1])) for i in range(1,int(count)-1))
        cursor += int(count)
    return np.asarray(triangles,dtype=np.int64).reshape(-1,3)


def _point_segments_squared(points, starts, ends):
    direction = ends-starts
    denominator = np.sum(direction*direction, axis=-1)
    numerator = np.sum((points-starts)*direction, axis=-1)
    fraction = np.divide(numerator, denominator,
        out=np.zeros_like(numerator), where=denominator > 1e-30)
    delta = points-(starts+np.clip(fraction, 0., 1.)[..., None]*direction)
    return np.sum(delta*delta, axis=-1)


def _points_triangles_squared(points, triangles):
    # One point/triangle pair per broadcasted position. Boundary and degenerate
    # cases use exact point/segment distance; interior projections use the face.
    a, b, c = triangles[..., 0, :], triangles[..., 1, :], triangles[..., 2, :]
    normal = np.cross(b-a, c-a)
    normal_squared = np.sum(normal*normal, axis=-1)
    plane = np.sum((points-a)*normal, axis=-1)
    projection = points-np.divide(plane, normal_squared,
        out=np.zeros_like(plane), where=normal_squared > 1e-30)[..., None]*normal
    sides = [np.sum(np.cross(end-start, projection-start)*normal, axis=-1)
             for start,end in ((a,b),(b,c),(c,a))]
    inside = ((normal_squared > 1e-30) & np.all(np.stack(sides, axis=-1)
        >= -1e-12*normal_squared[...,None], axis=-1))
    edge = np.minimum.reduce([_point_segments_squared(points, start, end)
                              for start,end in ((a,b),(b,c),(c,a))])
    face = np.divide(plane*plane, normal_squared, out=np.full_like(plane, np.inf),
                     where=normal_squared > 1e-30)
    return np.where(inside, np.minimum(edge, face), edge)


def _segments_squared(a, b, c, d):
    u, v, w = b-a, d-c, a-c
    aa, bb, cc = np.sum(u*u, -1), np.sum(u*v, -1), np.sum(v*v, -1)
    dd, ee = np.sum(u*w, -1), np.sum(v*w, -1)
    determinant = aa*cc-bb*bb
    first = np.divide(bb*ee-cc*dd, determinant,
        out=np.zeros_like(determinant), where=determinant > 1e-30)
    second = np.divide(aa*ee-bb*dd, determinant,
        out=np.zeros_like(determinant), where=determinant > 1e-30)
    interior = ((determinant > 1e-30) & (first >= 0) & (first <= 1)
                & (second >= 0) & (second <= 1))
    delta = w+first[..., None]*u-second[..., None]*v
    bounded = np.minimum.reduce([_point_segments_squared(a,c,d),
        _point_segments_squared(b,c,d), _point_segments_squared(c,a,b),
        _point_segments_squared(d,a,b)])
    return np.where(interior, np.minimum(bounded,np.sum(delta*delta,-1)), bounded)


def _box_vertices(half):
    return np.array([[x,y,z] for x in (-half[0],half[0])
        for y in (-half[1],half[1]) for z in (-half[2],half[2])])


def triangle_box_squared(triangles, half):
    """Exact triangle/AABB surface distance using SAT and closest features."""
    triangles, half = np.asarray(triangles,float), np.asarray(half,float)
    edge = np.roll(triangles,-1,axis=1)-triangles
    axes = [np.broadcast_to(axis,(len(triangles),3)) for axis in np.eye(3)]
    axes += [np.cross(edge[:,0],edge[:,1])]
    axes += [np.cross(edge[:,i],axis) for i in range(3) for axis in np.eye(3)]
    separated = np.zeros(len(triangles),bool)
    for axis in axes:
        length = np.linalg.norm(axis,axis=1)
        axis = np.divide(axis,length[:,None],out=np.zeros_like(axis),where=length[:,None] > 1e-30)
        projection = np.einsum('nti,ni->nt',triangles,axis)
        extent = np.abs(axis)@half
        separated |= ((projection.min(1) > extent+1e-13)
                      | (projection.max(1) < -extent-1e-13))
    answer = np.zeros(len(triangles))
    selected = np.flatnonzero(separated)
    if not len(selected):
        return answer
    tri = triangles[selected]
    delta = tri-np.clip(tri,-half,half)
    best = np.sum(delta*delta,-1).min(1)
    vertices = _box_vertices(half)
    best = np.minimum(best, _points_triangles_squared(
        vertices[None],tri[:,None]).min(1))
    pairs = [(i,j) for i in range(8) for j in range(i+1,8)
             if np.count_nonzero(vertices[i] != vertices[j]) == 1]
    starts, ends = vertices[[p[0] for p in pairs]],vertices[[p[1] for p in pairs]]
    tri_starts, tri_ends = tri,np.roll(tri,-1,axis=1)
    distance = _segments_squared(tri_starts[:,:,None],tri_ends[:,:,None],
                                  starts[None,None],ends[None,None])
    best = np.minimum(best,distance.min(axis=(1,2)))
    answer[selected] = np.maximum(best,0.)
    return answer


def _inside_closed_mesh(triangles, point):
    direction = np.array([1., .3713906763541037, .5291732910332951])
    a = triangles[:,0]
    first, second = triangles[:,1]-a,triangles[:,2]-a
    cross = np.cross(direction,second)
    determinant = np.sum(first*cross,axis=1)
    usable = np.abs(determinant) > 1e-15
    inverse = np.divide(1.,determinant,out=np.zeros_like(determinant),where=usable)
    offset = point-a
    u = np.sum(offset*cross,axis=1)*inverse
    q = np.cross(offset,first)
    v = (q@direction)*inverse
    depth = np.sum(second*q,axis=1)*inverse
    hits = np.sort(depth[usable & (u >= -1e-12) & (v >= -1e-12)
        & (u+v <= 1+1e-12) & (depth > 1e-12)])
    return bool(len(hits) and (1+np.count_nonzero(np.diff(hits) > 1e-10)) % 2)


def mesh_box_distance(mesh, box_from_mesh, half, *, upper_bound=np.inf):
    """Unsigned solid distance, pruning the cached triangle BVH in box space."""
    transform, half = np.asarray(box_from_mesh,float),np.asarray(half,float)
    linear, offset = transform[:3,:3],transform[:3,3]

    def lower(node):
        lo, hi = mesh._lower[node],mesh._upper[node]
        center, extent = linear@((lo+hi)/2)+offset,np.abs(linear)@((hi-lo)/2)
        delta = np.maximum(np.maximum(center-extent-half,-half-center-extent),0.)
        return float(delta@delta)

    # The caller reduces a global minimum. If even the root AABB cannot beat
    # its known witness, no triangle or vertex work is needed for this pose.
    if lower(0) >= float(upper_bound)**2:
        return float(upper_bound)
    vertices = mesh.vertices@linear.T+offset
    best = min(float(upper_bound)**2,
        float(np.sum((vertices-np.clip(vertices,-half,half))**2,axis=1).min()))
    stack = [(lower(0),0)]
    while stack:
        bound,node = heapq.heappop(stack)
        if bound > best:
            continue
        left,right = mesh._children[node]
        if left >= 0:
            for child in (int(left),int(right)):
                bound = lower(child)
                if bound <= best:
                    heapq.heappush(stack,(bound,child))
        else:
            start,stop = mesh._ranges[node]
            tri = mesh.triangles[mesh._permutation[start:stop]]
            best = min(best,float(triangle_box_squared(vertices[tri],half).min()))
            if best <= 1e-24:
                return 0.
    # A small box completely inside a closed robot solid has no intersecting
    # surface triangles. Report solid overlap rather than positive clearance.
    if np.all(vertices.min(0) <= 0) and np.all(vertices.max(0) >= 0):
        if _inside_closed_mesh(vertices[mesh.triangles],np.zeros(3)):
            return 0.
    return float(np.sqrt(best))


@dataclass(frozen=True)
class CollisionMeshReference:
    path: str
    body_path: str
    group: str
    mesh: TriangleMesh
    body_from_mesh: np.ndarray
    approximation: str = 'none'


def link_group(path):
    name = path.rsplit('/',1)[-1].lower()
    if 'finger' in name or 'hand' in name or 'gripper' in name:
        return 'gripper'
    if name.endswith(('link6','link7')):
        return 'wrist'
    if name.endswith('link5'):
        return 'forearm'
    if name.endswith(('link3','link4')):
        return 'elbow'
    return 'upper_arm_or_base'


def reduce_clearances(meshes, body_paths, body_transforms, obstacles):
    """Reduce all captured poses after timing; obstacles are (pose, dimensions)."""
    indices = {path:i for i,path in enumerate(body_paths)}
    groups = {part.group:np.inf for part in meshes}
    per_link = {part.body_path:np.inf for part in meshes}
    closest = None
    minimum = np.inf
    evaluated = 0
    seen = {}
    duplicate_skips = zero_skips = 0
    for sample,(poses,boxes) in enumerate(zip(body_transforms,obstacles)):
        prepared = [(path,np.linalg.inv(transform),np.asarray(dimensions)/2)
                    for path,transform,dimensions in boxes]
        for part in meshes:
            if per_link[part.body_path] == 0.:
                zero_skips += len(prepared)
                continue
            world_from_mesh = poses[indices[part.body_path]]@part.body_from_mesh
            for box_path,box_from_world,half in prepared:
                transform = box_from_world@world_from_mesh
                key = (part.path,box_path)
                query = transform.tobytes()+half.tobytes()
                previous = seen.setdefault(key,set())
                if query in previous:
                    duplicate_skips += 1
                    continue
                previous.add(query)
                distance = mesh_box_distance(part.mesh,transform,half,
                    upper_bound=per_link[part.body_path])
                per_link[part.body_path] = min(per_link[part.body_path],distance)
                groups[part.group] = min(groups[part.group],distance)
                evaluated += 1
                if distance < minimum:
                    minimum = distance
                    closest = dict(sample=sample,robot_link=part.body_path,mesh=part.path,
                                   obstacle=box_path,clearance_m=distance)
    finite = lambda value: None if not np.isfinite(value) else float(value)
    return dict(minimum_m=finite(minimum),by_group_m={k:finite(v) for k,v in groups.items()},
        by_link_m={k:finite(v) for k,v in per_link.items()},closest_pair=closest,
        mesh_box_evaluations=evaluated,sampled_poses=len(body_transforms),
        exact_duplicate_pose_skips=duplicate_skips,zero_minimum_skips=zero_skips,
        reference=('PhysX cooked convex hulls' if meshes and all(part.approximation.startswith('physx_cooked:')
            for part in meshes) else 'USD-authored or mixed collision triangles; explicit cooking metadata'),
        limitation='Sampled unsigned surface distance excludes contact/rest offsets and cannot certify continuous clearance.')
