"""Research-only closed triangle meshes for controlled rendering ablations.

Coordinates come from fixture truth for test construction. Production perception
never imports this module or receives its parameters, meshes or identities.
"""
import numpy as np


def mesh_surface(obj, segments):
    """Return world vertices/triangle indices at the requested radial resolution."""
    if type(segments) is not int or segments < 8 or segments % 2:
        raise ValueError('segments must be an even integer of at least eight')
    vertices, faces = [], []
    if obj.shape == 'sphere':
        rings = segments//2
        vertices = [[0., 0., obj.radius]]
        for ring in range(1, rings):
            polar = np.pi*ring/rings
            for segment in range(segments):
                angle = 2*np.pi*segment/segments
                vertices.append([obj.radius*np.sin(polar)*np.cos(angle),
                                 obj.radius*np.sin(polar)*np.sin(angle), obj.radius*np.cos(polar)])
        bottom = len(vertices)
        vertices.append([0., 0., -obj.radius])
        for segment in range(segments):
            following = (segment+1) % segments
            faces.append([0, 1+segment, 1+following])
            for ring in range(rings-2):
                a, b = 1+ring*segments+segment, 1+ring*segments+following
                faces.extend(([a, a+segments, b+segments], [a, b+segments, b]))
            faces.append([bottom, 1+(rings-2)*segments+following, 1+(rings-2)*segments+segment])
    elif obj.shape == 'cylinder':
        axis = np.asarray(obj.axis, float)
        if not np.isfinite(axis).all() or np.linalg.norm(axis) == 0:
            raise ValueError('Cylinder fixture axis must be finite and nonzero')
        axis = axis/np.linalg.norm(axis)
        reference = np.eye(3)[np.argmin(abs(axis))]
        side = np.cross(axis, reference)
        side /= np.linalg.norm(side)
        other = np.cross(axis, side)
        for end in (-1., 1.):
            for segment in range(segments):
                angle = 2*np.pi*segment/segments
                point = end*obj.length/2*axis+obj.radius*(np.cos(angle)*side+np.sin(angle)*other)
                vertices.append(point.tolist())
        vertices.extend((-obj.length/2*axis, obj.length/2*axis))
        for segment in range(segments):
            following = (segment+1) % segments
            faces.extend(([segment, following, segments+following],
                          [segment, segments+following, segments+segment],
                          [2*segments, following, segment],
                          [2*segments+1, segments+segment, segments+following]))
    else:
        raise ValueError('Controlled meshes support spheres and cylinders')
    return np.asarray(vertices)+obj.center, np.asarray(faces, int)
