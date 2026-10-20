"""Controlled curved-surface parameter alternatives on identical saved clouds.

This is isolated oracle geometry research. Failed/unknown classifications remain
in every alternative. No scene truth enters a production fitter. Numerical
replay provides no native throughput, freshness, IK or motion evidence.
"""
import argparse
from copy import deepcopy
import importlib
import json
from pathlib import Path
import sys

import numpy as np

from robot_skill_stack.common.rotations import quat_matrix
from robot_skill_stack.common.types import Pose
from robot_skill_stack.manipulation.grasping.primitive import SimplePrimitiveGraspPlanner
from robot_skill_stack.world.model.entities import WorldObject
from robot_skill_stack.world.model.primitives import PrimitiveGeometry
from robot_skill_stack.world.perception_multiview.fitting import PrimitiveFit, SurfaceFitter, _radius_evidence
from scripts.benchmark_multiview_grasp_readiness import plain_values


def read_ply(path):
    with path.open() as stream:
        count = None
        for line in stream:
            if line.startswith('element vertex '):
                count = int(line.split()[-1])
            if line.strip() == 'end_header':
                break
        return np.empty((0, 3)) if count == 0 else np.loadtxt(stream).reshape(-1, 3)


def plan_fit(fit, *, graspable=True):
    obj = WorldObject('object', class_name=fit.geometry.shape.value, pose=Pose(fit.center),
                      size=fit.size, geometry=fit.geometry, graspable=graspable)
    return SimplePrimitiveGraspPlanner(support_plane_z=0.).plan(obj)


def truth_fit(obj):
    sphere = obj['shape'] == 'sphere'
    geometry = PrimitiveGeometry(obj['shape'], 1., radius=obj['radius'],
        length=None if sphere else obj['length'], axis=None if sphere else obj['axis'])
    return PrimitiveFit(geometry, np.asarray(obj['center']), np.asarray(obj['size']))


def parameter_candidates(fit, points, finalizer, include_alternatives):
    yield 'baseline', fit, fit.geometry.shape.value != 'unknown'
    finalized = finalizer.finalize(fit, points)
    yield 'finalized', finalized, finalized.geometry.metadata['parameter_observable']
    if not include_alternatives:
        return
    supported = fit.geometry.shape.value in ('sphere', 'cylinder')
    samples = _radius_evidence(points, fit, 0.) if supported else None
    for family, quantiles in (('contact', (50, 90, 95, 99)), ('reflected', (90, 95, 99))):
        for quantile in quantiles:
            result = PrimitiveFit(deepcopy(fit.geometry), fit.center.copy(), fit.size.copy())
            if samples is None:
                yield f'{family}_q{quantile}', result, False
                continue
            radius = float(np.percentile(samples, quantile))
            if family == 'reflected':
                radius = 2*radius-float(np.median(samples))
            result.geometry.radius = radius
            g = result.geometry
            if g.shape.value == 'sphere' or abs(g.axis[2]) < .001:
                result.center[2] = radius
            result.size = (np.full(3, 2*radius) if g.shape.value == 'sphere' else
                           abs(g.axis)*g.length+2*radius*np.sqrt(np.maximum(1-g.axis*g.axis, 0)))
            yield f'{family}_q{quantile}', result, True


def measurement_row(case, view, condition, points, mode, fit, observable, truth, truth_plan):
    obj, g = case['object'], fit.geometry
    correct = g.shape.value == obj['shape']
    decision = plan_fit(fit, graspable=observable)
    checks = None
    if decision.ok:
        grasp = decision.plan.grasp
        closing = quat_matrix(grasp.orientation)[:, 1]
        if obj['shape'] == 'sphere' or abs(obj['axis'][2]) > .999:
            width = 2*obj['radius']
        else:
            dot = float(abs(closing @ np.asarray(obj['axis'])))
            width = dot*obj['length']+2*obj['radius']*np.sqrt(max(0., 1-dot*dot))
        top = truth.center[2]+truth.size[2]/2
        palm, support = grasp.position[2]+.030-top, grasp.position[2]-.012
        checks = dict(width_m=width, palm_margin_m=palm, support_margin_m=support,
            clearance_violation=bool(min(palm, support) < .005-1e-9),
            width_violation=bool(not (.005 <= width <= .075 and width+.010 <= .080+1e-9)))
    radius_error = (g.radius-obj['radius'])*1000 if correct else None
    dimensions = None if not correct else ([2*g.radius]*3 if g.shape.value == 'sphere' else
                                          [2*g.radius, 2*g.radius, g.length])
    truth_dimensions = ([2*obj['radius']]*3 if obj['shape'] == 'sphere' else
                        [2*obj['radius'], 2*obj['radius'], obj['length']])
    return dict(scenario=case['scenario'], segments=case['segments'], view=view, condition=condition,
        mode=mode, points=len(points), predicted=g.shape.value, correct=correct,
        parameter_observable=observable, radius_error_mm=radius_error,
        radius_covered=bool(correct and g.radius >= obj['radius']-1e-9),
        center_error_mm=float(np.linalg.norm(fit.center-truth.center)*1000),
        dimension_error_mm=None if dimensions is None else float(np.max(abs(np.asarray(dimensions)-truth_dimensions))*1000),
        truth_feasible=truth_plan.ok, estimated_feasible=decision.ok,
        false_feasible=decision.ok and not truth_plan.ok, false_infeasible=truth_plan.ok and not decision.ok,
        proposed_truth_checks=checks, metadata=g.metadata)


def main(args):
    if np.__version__ != '1.26.4':
        raise RuntimeError('Use the established Isaac Python wrapper with NumPy 1.26.4')
    sys.path.insert(0, str(args.baseline_pythonpath.resolve()))
    baseline = importlib.import_module('baseline_multiview.fitting').SurfaceFitter()
    finalizer = SurfaceFitter()
    cases = json.loads((args.capture/'manifest.json').read_text())
    rng = np.random.default_rng(args.seed)
    rows = []
    args.output.mkdir(parents=True, exist_ok=True)
    for case in cases:
        truth = truth_fit(case['object'])
        truth_plan = plan_fit(truth)
        for view in ('camera_0', 'camera_1', 'fused'):
            points = read_ply(args.capture/case['directory']/f'{view}.ply')
            variants = dict(clean=points, noise03mm=points+rng.normal(0., .0003, points.shape),
                sparse4=points[::4], partial50=points[points[:, 0] >= np.median(points[:, 0])] if len(points) else points)
            for condition, cloud in variants.items():
                fit = baseline.fit(cloud)
                for mode, candidate, observable in parameter_candidates(fit, cloud, finalizer, args.alternatives):
                    rows.append(measurement_row(case, view, condition, cloud, mode, candidate,
                                                observable, truth, truth_plan))
        (args.output/'rows.json').write_text(json.dumps(plain_values(rows), indent=2))
    summary = []
    for condition in dict.fromkeys(row['condition'] for row in rows):
        for mode in dict.fromkeys(row['mode'] for row in rows):
            selected = [r for r in rows if r['condition'] == condition and r['mode'] == mode]
            valid = [r for r in selected if r['radius_error_mm'] is not None]
            summary.append(dict(condition=condition, mode=mode, rows=len(selected), correct=sum(r['correct'] for r in selected),
                observable=sum(r['parameter_observable'] for r in selected),
                observable_radius_covered=sum(r['radius_covered'] and r['parameter_observable'] for r in selected),
                radius_mae_mm=float(np.mean([abs(r['radius_error_mm']) for r in valid])) if valid else None,
                dimension_mae_mm=float(np.mean([r['dimension_error_mm'] for r in valid])) if valid else None,
                center_mean_mm=float(np.mean([r['center_error_mm'] for r in selected])),
                false_feasible=sum(r['false_feasible'] for r in selected),
                false_infeasible=sum(r['false_infeasible'] for r in selected),
                clearance_violations=sum(bool(r['proposed_truth_checks'] and r['proposed_truth_checks']['clearance_violation']) for r in selected),
                width_violations=sum(bool(r['proposed_truth_checks'] and r['proposed_truth_checks']['width_violation']) for r in selected)))
    result = dict(kind='controlled_oracle_geometry_replay', seed=args.seed, scenes=len(cases),
        uncertainty_note='Empirical envelopes, not calibrated confidence intervals or global bounds',
        timings_note='Functional replay only; no native throughput or source freshness claim', summary=summary)
    (args.output/'result.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--baseline-pythonpath', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=20261109)
    parser.add_argument('--alternatives', action='store_true')
    main(parser.parse_args())
