#!/usr/bin/env python3
"""Measure primitive-count sensitivity through the installed public Lula API.

Run with ./run_isaac_python.sh scripts/probe_rmpflow_obstacle_composition.py
--result outputs/obstacle-composition/result.json. This uses CPU policy queries
only: no SimulationApp, Kit stage, physical world, articulation or motion.

The recorded joint fixture is embedded so reproduction needs no ignored files.
Coincident copies isolate identical geometric unions; adjacent post slices
demonstrate representation sensitivity without overlapping positive volumes.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


RECORDED_Q = [-0.17791271209716797, -0.16587994992733002,
    -0.30747929215431213, -2.2318549156188965, -0.06004467234015465,
    2.0644288063049316, 0.34545713663101196]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result', type=Path, required=True)
    args = parser.parse_args()

    # The installed binary binding must never be imported with NumPy2.
    import numpy as np
    if np.__version__ != '1.26.4':
        raise RuntimeError('This installed Lula probe requires NumPy1.26.4 before SDK import; '
                           'launch through ./run_isaac_python.sh')
    sdk = Path(os.environ.get('ISAAC_SIM', str(Path.home()/'isaacsim'))).expanduser().resolve()
    bindings = sdk/'exts/isaacsim.robot_motion.lula/pip_prebundle'
    config_dir = sdk/'exts/isaacsim.robot_motion.motion_generation/motion_policy_configs/franka'
    files = dict(robot_descriptor=config_dir/'rmpflow/robot_descriptor.yaml',
        robot_urdf=config_dir/'lula_franka_gen.urdf',
        policy_config=config_dir/'rmpflow/franka_rmpflow_common.yaml',
        integration_config=config_dir/'rmpflow/config.json')
    if not bindings.is_dir() or any(not path.is_file() for path in files.values()):
        raise FileNotFoundError(f'Installed Isaac/Lula Franka configuration not found under {sdk}')
    sys.path.insert(0, str(bindings))
    import lula

    q = np.asarray(RECORDED_Q, dtype=np.float64)
    velocity = np.zeros_like(q)
    target, orientation = np.array([.45, .25, .35]), np.array([0., 0., 1., 0.])
    center, dimensions = np.array([.45, 0., .35]), np.array([.09, .12, .26])
    integration = json.loads(files['integration_config'].read_text())
    frame = integration['end_effector_frame_name']
    robot = lula.load_robot(str(files['robot_descriptor']), str(files['robot_urdf']))

    def evaluate(boxes):
        world = lula.create_world()
        view = world.add_world_view()
        config = lula.create_rmpflow_config(str(files['policy_config']), robot, frame, view)
        policy = lula.create_rmpflow(config)
        policy.set_end_effector_position_attractor(target)
        policy.set_end_effector_orientation_attractor(lula.Rotation3(*orientation))
        for point, size in boxes:
            obstacle = lula.create_obstacle(lula.Obstacle.Type.CUBE)
            obstacle.set_attribute(lula.Obstacle.Attribute.SIDE_LENGTHS, size)
            world.add_obstacle(obstacle, lula.Pose3(lula.Rotation3(np.eye(3)), point))
        policy.update_world_view()
        force, metric = policy.eval_force_and_metric(q, velocity)
        acceleration = np.empty_like(q)
        policy.eval_accel(q, velocity, acceleration)
        if not all(np.isfinite(value).all() for value in (force, metric, acceleration)):
            raise RuntimeError('Lula returned a nonfinite policy evaluation')
        return dict(force=np.asarray(force), metric=np.asarray(metric),
                    acceleration=acceleration, enabled_obstacles=view.num_enabled_obstacles())

    empty = evaluate([])
    rows = []

    def record(kind, count, boxes):
        result = evaluate(boxes)
        delta_force, delta_metric = result['force']-empty['force'], result['metric']-empty['metric']
        rows.append(dict(kind=kind, count=count, enabled_obstacles=result['enabled_obstacles'],
            natural_policy_force=result['force'].tolist(), policy_metric=result['metric'].tolist(),
            canonical_acceleration_rad_s2=result['acceleration'].tolist(),
            collision_force_delta=delta_force.tolist(), collision_metric_delta=delta_metric.tolist(),
            collision_force_norm=float(np.linalg.norm(delta_force)),
            collision_metric_norm=float(np.linalg.norm(delta_metric)),
            collision_metric_trace=float(np.trace(delta_metric))))

    for count in (1, 2, 4, 8, 16, 64):
        record('coincident_equal_union', count, [(center, dimensions)]*count)
    for count in (2, 4, 8, 16):
        size = dimensions.copy()
        size[2] /= count
        boxes = []
        for index in range(count):
            point = center.copy()
            point[2] += -.5*dimensions[2]+(index+.5)*size[2]
            boxes.append((point, size))
        record('adjacent_z_slices_equal_union', count, boxes)
    one = rows[0]
    checks = []
    for row in rows:
        count = row['count']
        force_error = np.linalg.norm(np.asarray(row['collision_force_delta'])-
                                     count*np.asarray(one['collision_force_delta']))
        metric_error = np.linalg.norm(np.asarray(row['collision_metric_delta'])-
                                      count*np.asarray(one['collision_metric_delta']))
        row.update(force_linear_duplicate_residual_norm=float(force_error),
            metric_linear_duplicate_residual_norm=float(metric_error),
            collision_metric_ratio_to_single=row['collision_metric_norm']/one['collision_metric_norm'])
        if row['kind'] == 'coincident_equal_union':
            checks.append(bool(np.allclose(row['collision_force_delta'],
                count*np.asarray(one['collision_force_delta']), rtol=1e-12, atol=1e-8) and
                np.allclose(row['collision_metric_delta'],
                count*np.asarray(one['collision_metric_delta']), rtol=1e-12, atol=1e-8)))

    files.update(lula_binding=Path(lula.__file__),
        lula_rmpflow=bindings/'_lula_libs/liblula_rmpflow.so', probe=Path(__file__).resolve())
    report = dict(schema_version=1, status='PASS' if all(checks) else 'FAIL',
        scope='Static public Lula CPU policy evaluation; no Kit/GPU/stage/physics/articulation/motion.',
        numpy_version=np.__version__, isaac_sim=str(sdk), end_effector_frame=frame,
        q=q.tolist(), qdot=velocity.tolist(), position_target=target.tolist(),
        orientation_wxyz=orientation.tolist(),
        recorded_fixture_source=dict(path='outputs/motion-planning-v2b-pilot-rgbd-static-reactive/result.json',
            sha256='9a7ef0578e5709dc697c46be0ae5033831cbfec339ada556cdf3181e34e02202',
            meaning='Initial arm joint positions are embedded; this retained artifact is not required to run the probe.'),
        post=dict(center=center.tolist(), dimensions=dimensions.tolist()),
        baseline_natural_policy_force=empty['force'].tolist(),
        baseline_policy_metric=empty['metric'].tolist(),
        baseline_acceleration_rad_s2=empty['acceleration'].tolist(), rows=rows,
        linear_duplicate_checks=checks,
        supported_public_primitive_types=sorted(lula.Obstacle.Type.__members__),
        source_hashes={name:dict(path=str(path), sha256=sha256(path)) for name, path in files.items()},
        limitations=[
            'Natural RMP force and metric are policy terms, not measured contact force, torque, Newtons or physical inertia.',
            'Linear collision-only natural force/metric increments do not imply linear canonical acceleration; other RMPs and canonical limits participate in resolution.',
            'Coincident duplicates are diagnostic only; the production extractor never emits overlapping duplicate geometry.',
            'Adjacent slices preserve exactly the same post union but are not the original sparse RGBD observations.',
            'This static zero-velocity experiment establishes representation sensitivity, not the complete dynamic pilot stall cause.',
            'Installed robot geometry, controller configuration and RMPflow parameters are read without modification.',
            'PASS verifies the declared numeric composition checks and is not an avoidance or physical-robot safety result.'
        ])
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(status=report['status'], result=str(args.result), cases=len(rows),
        same_union_slice_metric_ratios=[row['collision_metric_ratio_to_single'] for row in rows
                                      if row['kind'] == 'adjacent_z_slices_equal_union'])))
    if report['status'] != 'PASS':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
