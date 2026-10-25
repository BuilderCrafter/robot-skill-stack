import asyncio
import hashlib
import json
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from scripts import diagnose_global_solver_velocity_v4_1 as probe


class SolverVelocityProbeTests(unittest.TestCase):
    def test_probe_requires_the_entire_unchanged_reference_source_manifest(self):
        original = {'global_rrt.py':'before', 'backend.py':'same'}
        probe.require_matching_probe_baseline(dict(hashes=original), dict(original))
        for current in ({**original, 'global_rrt.py':'production-fix'},
                {**original, 'global_solver.py':'new-production-scope'},
                {'backend.py':'same'}, {}):
            with self.subTest(current=current), self.assertRaisesRegex(ValueError, 'unchanged source baseline'):
                probe.require_matching_probe_baseline(dict(hashes=original), current)
        with self.assertRaisesRegex(ValueError, 'validate_global_stabilization_v4_1.py'):
            probe.require_matching_probe_baseline({}, original)

    def test_zero_velocity_requires_all_arm_targets_and_accepts_full_dof_buffers(self):
        self.assertTrue(probe.zero_arm_velocity(dict(joint_velocities=[0.]*9, joint_indices=None), range(7)))
        self.assertTrue(probe.zero_arm_velocity(dict(joint_velocities=[0.]*7, joint_indices=list(range(7))), range(7)))
        self.assertFalse(probe.zero_arm_velocity(dict(joint_velocities=[0.]*6, joint_indices=list(range(6))), range(7)))
        self.assertFalse(probe.zero_arm_velocity(dict(joint_velocities=[.1]+[0.]*8, joint_indices=None), range(7)))

    def test_post_hold_advances_only_updates_and_keeps_home_outcome_separate(self):
        home = dict(label='home', target=dict(position=[0., 0., 0.], orientation=[1., 0., 0., 0.]),
            returned_success=True, duration=1.05)
        world = SimpleNamespace(current_time=5.)
        class Observer:
            def __init__(self):
                self.operations, self.samples, self.actions = [home], [], []
                self.metadata, self.indices = {}, tuple(range(7))
                self.native_count = 32
                self.bundle = SimpleNamespace(world=world)
                self.backend = SimpleNamespace(_next_update=self.update, _quat_error=lambda a,b:0.)
            def sample(self):
                self.samples.append(dict(simulation_time=world.current_time,
                    tcp_pose=dict(position=[0., 0., 0.], orientation=[1., 0., 0., 0.]),
                    actual_minus_applied_target_arm_rad=[.01]*7,
                    requested_action=dict(joint_velocities=[0.]*7, joint_indices=list(range(7))),
                    feedback=dict(velocities=[.001]*9, action=dict(joint_velocities=[0.]*9, joint_indices=None)),
                    native_actor_solver_properties=dict(available=True, position=self.native_count, velocity=1)))
            async def update(self):
                world.current_time += .25
                self.sample()
            def begin_operation(self, label, target, *, kind):
                row = dict(label=label, target=target, kind=kind, first_sample=len(self.samples),
                    first_action=len(self.actions), before=dict(simulation_time=world.current_time))
                self.operations.append(row)
                self.sample()
                return row
            def end_operation(self, row):
                self.sample()
                row.update(last_sample=len(self.samples), last_action=len(self.actions),
                    after=dict(simulation_time=world.current_time))
        observer = Observer()
        asyncio.run(probe.observe_post_home_hold(observer, 1.))
        report = observer.operations[-1]['post_home_hold']
        self.assertEqual(world.current_time, 6.)
        self.assertEqual(report['additional_arm_commands'], 0)
        self.assertTrue(report['kinematic_stability_passed'])
        self.assertEqual(home['duration'], 1.05)
        self.assertTrue(home['returned_success'])
        self.assertEqual(observer.operations[-1]['kind'], 'post_hold')
        observer.native_count = 255
        asyncio.run(probe.observe_post_home_hold(observer, 1., label='retained_home_hold', expected_position_count=255))
        retained = observer.operations[-1]['retained_home_hold']
        self.assertTrue(retained['kinematic_stability_passed'])
        self.assertTrue(retained['all_native_actor_counts_expected'])
        self.assertFalse(retained['all_native_actor_counts_nominal'])
        self.assertEqual(retained['final_actual_arm_velocity_rad_s'], [.001]*7)

    def test_retained_home_defers_restore_until_explicit_cleanup(self):
        class Base:
            def __init__(self): pass
            def end_operation(self, *args, **kwargs): pass
            def restore(self): pass
        with patch.object(probe.controlled, 'Observer', Base):
            observer = probe.probe_observer_class({}, position_iterations=255, retain_home_solver=True)()
        observer._restore_solver = Mock()
        with patch.object(probe, 'compare_home', return_value={}):
            observer.end_operation(dict(label='home', solver_velocity_probe={}))
        observer._restore_solver.assert_not_called()
        observer.restore()
        observer._restore_solver.assert_called_once()

    def test_comparison_checks_actual_start_and_unchanged_common_commands(self):
        before = dict(feedback=dict(positions=[0.]*9, velocities=[0.]*9),
            simulation_time=3.5, physics_step=210)
        commands = [dict(sim_time=3.5, q=[0.]*7, qd=[0.]*7),
            dict(sim_time=3.5+1/60, q=[.01]*7, qd=[.1]*7)]
        plan = dict(route=[[0.]*7, [.1]*7], trajectory=[dict(time=0., q=[0.]*7, qd=[0.]*7)],
            commanded=commands)
        old = dict(label='home', before=before, plan_trace=plan)
        current = dict(label='home', before=before, plan_trace={**plan, 'commanded':commands[:1]})
        result = probe.compare_home(dict(operations=[old]), current)
        self.assertEqual(result['before_home_joint_position_max_difference_rad'], 0.)
        self.assertEqual(result['route_max_difference_rad'], 0.)
        self.assertTrue(result['trajectory_samples_equal'])
        self.assertEqual(result['common_command_prefix_count'], 1)
        self.assertEqual(result['command_prefix_q_max_difference_rad'], 0.)
        self.assertEqual(result['command_counts'], [2, 1])

    def test_non_home_never_requests_a_solver_change(self):
        class Base:
            def __init__(self): pass
            def begin_operation(self, *args, **kwargs): return dict(label=args[0])
        with patch.object(probe.controlled, 'Observer', Base):
            instance = probe.probe_observer_class({})()
        self.assertEqual(instance.begin_operation('move_to_goal', None)['label'], 'move_to_goal')
        self.assertIsNone(instance._lease)

    def test_home_exception_still_restores_and_hook_cleanup_is_unconditional(self):
        class Base:
            def __init__(self): self.restored = False
            def end_operation(self, *args, **kwargs): raise RuntimeError('observation failed')
            def restore(self): self.restored = True
        with patch.object(probe.controlled, 'Observer', Base):
            instance = probe.probe_observer_class({})()
        instance._restore_solver = Mock()
        with self.assertRaisesRegex(RuntimeError, 'observation failed'):
            instance.end_operation(dict(label='home'))
        instance._restore_solver.assert_called_once()
        instance._restore_solver.side_effect = RuntimeError('restore request failed')
        with self.assertRaisesRegex(RuntimeError, 'restore request failed'):
            instance.restore()
        self.assertTrue(instance.restored)

    def test_restore_removes_only_new_session_opinion_and_restores_requested_count(self):
        properties = {probe.VELOCITY: 0}
        class Context:
            def __init__(self, *args): pass
            def __enter__(self): return self
            def __exit__(self, *args): pass
        layer = SimpleNamespace(ExportToString=lambda:json.dumps(properties, sort_keys=True),
            GetAttributeAtPath=lambda path:None if path.split('.')[-1] not in properties else SimpleNamespace(
                HasInfo=lambda name:name == 'default', GetInfo=lambda name:properties[path.split('.')[-1]]))
        def set_counts(values): properties[probe.VELOCITY] = int(values[0])
        view = SimpleNamespace(set_solver_velocity_iteration_counts=Mock(side_effect=set_counts),
            get_solver_velocity_iteration_counts=lambda:np.array([properties.get(probe.VELOCITY, 1)]),
            get_solver_position_iteration_counts=lambda:np.array([32]))
        prim = SimpleNamespace(GetAttribute=lambda name:None,
            RemoveProperty=lambda name:properties.pop(name))
        stage = SimpleNamespace(GetPrimAtPath=lambda path:prim)
        class Base:
            def __init__(self): self.robot = SimpleNamespace(_articulation_view=view)
        with patch.object(probe.controlled, 'Observer', Base):
            instance = probe.probe_observer_class({})()
        record = dict(session_layer_sha256_before=hashlib.sha256(b'{}').hexdigest())
        instance._lease = dict(stage=stage, session=layer, old=np.array([1]),
            opinions=[dict(path='/Franka', existed=False, default_present=False, default=None)],
            owned_empty_ancestor_candidates=[], record=record)
        with patch.dict('sys.modules', {'pxr':SimpleNamespace(Usd=SimpleNamespace(EditContext=Context))}), \
                patch.object(probe, 'native_properties', return_value=dict(available=False)), \
                patch.object(probe, 'remove_owned_empty_overs', return_value=[]):
            instance._restore_solver()
        view.set_solver_velocity_iteration_counts.assert_called_once()
        self.assertNotIn(probe.VELOCITY, properties)
        self.assertEqual(record['restored_usd_requested_velocity_counts'], [1])
        self.assertEqual(record['restored_usd_requested_position_counts'], [32])
        self.assertTrue(record['original_session_opinion_restored'])
        self.assertTrue(record['entire_session_layer_equal'])
        self.assertIsNone(instance._lease)

    def test_cleanup_removes_only_owned_empty_over_and_preserves_other_opinions(self):
        def spec(*, fields=('specifier',), children=(), properties=(), kind='over'):
            return SimpleNamespace(specifier=kind, properties=properties, nameChildren=children,
                ListInfoKeys=lambda:list(fields))
        specs = {'/New/Empty':spec(), '/New':spec(children=('Preserve',)),
            '/Preexisting':spec(), '/WithMetadata':spec(fields=('specifier', 'active')),
            '/WithAttribute':spec(properties=('x',)), '/Def':spec(kind='def')}
        stage = SimpleNamespace(RemovePrim=Mock(side_effect=lambda path:specs.pop(path) is not None))
        session = SimpleNamespace(GetPrimAtPath=lambda path:specs.get(path))
        owned = ['/New/Empty', '/New', '/WithMetadata', '/WithAttribute', '/Def']
        with patch.dict('sys.modules', {'pxr':SimpleNamespace(Sdf=SimpleNamespace(SpecifierOver='over'))}):
            removed = probe.remove_owned_empty_overs(stage, session, owned)
        self.assertEqual(removed, ['/New/Empty'])
        stage.RemovePrim.assert_called_once_with('/New/Empty')
        self.assertIn('/Preexisting', specs)
        self.assertIn('/WithMetadata', specs)

    def test_missing_ancestors_excludes_preexisting_session_prim(self):
        session = SimpleNamespace(GetPrimAtPath=lambda path:object() if path == '/Existing' else None)
        self.assertEqual(probe.missing_session_ancestors(session, ['/Existing/Robot', '/New/Robot']),
            ['/New/Robot', '/Existing/Robot', '/New'])

    def test_ceiling_position_probe_changes_only_position_count_and_restores(self):
        counts = dict(position=np.array([32]), velocity=np.array([1]))
        specs = {}
        class Attribute:
            def __init__(self, key): self.key = key
            def HasInfo(self, name): return name == 'default'
            def GetInfo(self, name): return specs[self.key]
        class Context:
            def __init__(self, *args): pass
            def __enter__(self): return self
            def __exit__(self, *args): pass
        layer = SimpleNamespace(identifier='anon:session',
            GetPrimAtPath=lambda path:None,
            GetAttributeAtPath=lambda path:Attribute(path) if path in specs else None,
            ExportToString=lambda:json.dumps(specs, sort_keys=True))
        def set_count(axis, value):
            counts[axis] = np.asarray(value).copy()
            name = probe.POSITION if axis == 'position' else probe.VELOCITY
            specs['/Franka.'+name] = int(value[0])
        view = SimpleNamespace(prim_paths=['/Franka'],
            get_solver_position_iteration_counts=lambda:counts['position'],
            get_solver_velocity_iteration_counts=lambda:counts['velocity'],
            set_solver_position_iteration_counts=Mock(side_effect=lambda value:set_count('position', value)),
            set_solver_velocity_iteration_counts=Mock(side_effect=lambda value:set_count('velocity', value)))
        prim = SimpleNamespace(RemoveProperty=lambda name:specs.pop('/Franka.'+name))
        stage = SimpleNamespace(GetSessionLayer=lambda:layer, GetPrimAtPath=lambda path:prim,
            GetEditTarget=lambda:SimpleNamespace(GetLayer=lambda:layer))
        state = dict(physics_step=10, simulation_time=.5, feedback=dict(positions=[0.]*9))
        class Base:
            def __init__(self):
                self.robot = SimpleNamespace(_articulation_view=view)
                self.bundle = SimpleNamespace(world=SimpleNamespace(
                    get_physics_context=lambda:SimpleNamespace(get_solver_type=lambda:'TGS')))
                self.metadata = {}
            def state(self): return state
            def begin_operation(self, label, *args, **kwargs): return dict(label=label)
        omni = SimpleNamespace(usd=SimpleNamespace(get_context=lambda:SimpleNamespace(get_stage=lambda:stage)))
        modules = {'omni':omni, 'omni.usd':omni.usd,
            'pxr':SimpleNamespace(Usd=SimpleNamespace(EditContext=Context))}
        with patch.object(probe.controlled, 'Observer', Base):
            instance = probe.probe_observer_class({}, position_iterations=255)()
        with patch.dict('sys.modules', modules), \
                patch.object(probe, 'scene_configuration', return_value=dict(scene_minimum_allows_zero=True)), \
                patch.object(probe, 'native_properties', return_value=dict(available=False)), \
                patch.object(probe, 'remove_owned_empty_overs', return_value=['/Franka']):
            row = instance.begin_operation('home', None)
            self.assertEqual(counts['position'].tolist(), [255])
            self.assertEqual(counts['velocity'].tolist(), [1])
            instance._restore_solver()
        self.assertEqual(counts['position'].tolist(), [32])
        view.set_solver_velocity_iteration_counts.assert_not_called()
        self.assertEqual(row['solver_velocity_probe']['changed_count'], 'position')
        self.assertTrue(row['solver_velocity_probe']['original_session_opinion_restored'])


if __name__ == '__main__':
    unittest.main()
