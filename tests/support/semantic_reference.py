"""Legacy numerical-oracle formatting only; no synchronous capture interface."""
def describe(candidate, geometry):
    return {
        'position': candidate.position.tolist(), 'size': candidate.size.tolist(),
        'pixels': int(candidate.mask.sum()), 'spawnable': candidate.spawnable,
        'shape': None if geometry is None else geometry.shape.value,
        'scores': {} if geometry is None else geometry.scores,
        'confidence': None if geometry is None else geometry.confidence,
        'yaw': None if geometry is None else geometry.yaw,
        'axis': None if geometry is None or geometry.axis is None else geometry.axis.tolist(),
        'radius': None if geometry is None else geometry.radius,
        'length': None if geometry is None else geometry.length,
        'reason': None if geometry is None else geometry.metadata.get('reason'),
    }
