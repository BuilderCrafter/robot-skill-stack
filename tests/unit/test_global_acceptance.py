"""Retained independent reference integrity and test-only execution-veto boundary."""
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
import tempfile
from unittest.mock import Mock,patch

import numpy as np

from tests.isaac import test_global_planning as native
from robot_skill_stack.manipulation.motion.joint_route import PlanningFailure

ROOT=Path(__file__).resolve().parents[2]


class ReferenceIntegrityTests(unittest.TestCase):
    def test_both_native_profiles_validate_before_sdk_construction(self):
        from robot_skill_stack.integrations.isaac.config import load_scene_config
        original=(ROOT/'tests/fixtures/global_rgbd_partial.toml').read_text()
        with tempfile.TemporaryDirectory() as directory:
            for source in ('static','frozen_rgbd'):
                path=Path(directory)/(source+'.toml');path.write_text(native.profile_text(source,original))
                cfg=load_scene_config(path)
                self.assertEqual(cfg.motion.global_obstacle_source,source)
                self.assertEqual(cfg.spatial_mapping.enabled,source=='frozen_rgbd')
                if source=='static':self.assertEqual(cfg.ground_truth.objects_root,'/World/ReactiveEvaluation')
                else:self.assertEqual(path.read_text(),original)

    def test_extracted_reference_definitions_and_fixture_bytes_are_unchanged(self):
        records=json.loads((ROOT/'tests/fixtures/global_acceptance_provenance.json').read_text())['definitions']
        for key,record in records.items():
            if key.endswith(('.toml','.json')):
                data=(ROOT/'tests/fixtures'/key).read_bytes();expected=record['sha256']
            else:
                filename,name=key.split(':') if ':' in key else ('global_native.py',key)
                text=(ROOT/'tests/support'/filename).read_text();tree=ast.parse(text)
                node=next(n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==name)
                first=min([node.lineno]+[d.lineno for d in node.decorator_list])-1
                data=('\n'.join(text.splitlines()[first:node.end_lineno])+'\n').encode()
                expected=record.get('retained_sha256',record.get('sha256'))
            with self.subTest(definition=key):self.assertEqual(hashlib.sha256(data).hexdigest(),expected)

    def test_original_physical_and_sampling_contracts_are_retained(self):
        from tests.support.global_physical import BODY_NAMES,NUMERIC_FRAME_ALLOWANCE_M
        from robot_skill_stack.integrations.isaac.config import load_scene_config
        self.assertEqual(len(BODY_NAMES),11)
        self.assertTrue({'panda_link0','panda_leftfinger','panda_rightfinger'}<=BODY_NAMES)
        self.assertEqual(NUMERIC_FRAME_ALLOWANCE_M,1e-5)
        cfg=load_scene_config(ROOT/'tests/fixtures/global_rgbd_partial.toml')
        self.assertEqual(cfg.motion.global_rgbd_representation,'reconstructed')
        self.assertEqual((cfg.motion.max_obstacles,cfg.motion.rrt_seed,cfg.motion.rrt_max_iterations),(64,123456,10000))
        self.assertEqual(tuple(cfg.multi_view.resolution),(640,480))
        self.assertEqual(len(cfg.multi_view.camera_prim_paths),2)
        self.assertEqual((cfg.spatial_mapping.max_rays,cfg.spatial_mapping.voxel_size),(1024,.01))
        self.assertEqual((cfg.spatial_mapping.stale_after,cfg.spatial_mapping.max_frame_age),(1.,1.))


class VetoBoundaryTests(unittest.TestCase):
    def make(self,allowed=True):
        primary=Mock();primary._config={'urdf_path':'physical-reference-only.urdf'}
        primary.plan.return_value=NS(route=np.zeros((2,7)),joint_names=tuple(f'panda_joint{i}' for i in range(1,8)))
        backend=NS(world=NS(current_time=1.,current_time_step_index=60),robot=Mock())
        observer=NS(context=lambda:{'actual_start_q':[0.]*9},arm_commands_allowed=False)
        checked=dict(execution_allowed=allowed,status='ALLOW_SAMPLED_NUMERIC_CLEARANCE' if allowed else 'VETO_PREDICTED_PHYSICAL_COLLISION')
        helper=NS(sample_trajectory=Mock(return_value=(np.array([0.,.001]),np.zeros((2,7)))),bounded_evaluate=Mock(return_value=checked))
        evaluator=Mock();state={}
        planner=native.VetoPlanner(primary,backend,observer,'physical-parts','physical-obstacles',state,False)
        return planner,primary,backend,helper,evaluator,state

    def test_reference_geometry_never_enters_selected_planner(self):
        planner,primary,backend,helper,evaluator,state=self.make()
        target=object();start=np.zeros(7);world=object()
        with patch.multiple(native,fixture=helper,PhysicalMotionEvaluator=evaluator,create=True):
            result=planner.plan(target,start,world)
        primary.plan.assert_called_once_with(target,start,world)
        evaluator.assert_called_once_with('physical-parts','physical-obstacles','physical-reference-only.urdf',max_queries=500000,max_samples=12000)
        self.assertIs(result,primary.plan.return_value);self.assertTrue(state['physical_evaluation']['execution_allowed'])
        self.assertTrue(planner.observer.arm_commands_allowed)
        backend.robot.apply_action.assert_not_called();planner.close();primary.close.assert_called_once()
        self.assertTrue(state['planner_closed'])

    def test_refusal_prevents_returning_a_plan_and_preserves_diagnostics(self):
        planner,primary,backend,helper,evaluator,state=self.make(False)
        with patch.multiple(native,fixture=helper,PhysicalMotionEvaluator=evaluator,create=True),self.assertRaisesRegex(PlanningFailure,'Independent physical veto'):
            planner.plan(object(),np.zeros(7),object())
        backend.robot.apply_action.assert_not_called();self.assertFalse(state['physical_evaluation']['execution_allowed'])
        self.assertFalse(planner.observer.arm_commands_allowed)

    def test_physics_cannot_advance_during_plan_and_veto(self):
        planner,primary,backend,helper,evaluator,state=self.make()
        def advanced(*args,**kwargs):
            backend.world.current_time_step_index+=1
            return dict(execution_allowed=True,status='ALLOW_SAMPLED_NUMERIC_CLEARANCE')
        helper.bounded_evaluate.side_effect=advanced
        with patch.multiple(native,fixture=helper,PhysicalMotionEvaluator=evaluator,create=True),self.assertRaisesRegex(PlanningFailure,'Physics advanced'):
            planner.plan(object(),np.zeros(7),object())
        backend.robot.apply_action.assert_not_called()
