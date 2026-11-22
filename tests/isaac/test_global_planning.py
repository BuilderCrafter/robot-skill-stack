"""Bounded GT or nominal partial RGB-D acceptance with an independent motion veto.

Run one source per fresh Isaac process. Physical reference geometry is used only
by the test veto; the selected planner owns its normal IK and collision source.
"""
import argparse
from collections import deque
import io
import json
from pathlib import Path
import pickle
import time
import traceback

import numpy as np



class Observer:
    def __init__(self,bundle):
        from pxr import Usd,UsdPhysics,UsdGeom
        import omni.usd
        from isaacsim.core.simulation_manager import SimulationManager
        self.bundle=bundle;self.backend=bundle.backend;self.robot=self.backend.robot
        stage=omni.usd.get_context().get_stage();self.units=float(UsdGeom.GetStageMetersPerUnit(stage))
        robot_root=stage.GetPrimAtPath('/Franka')
        paths=[str(p.GetPath()) for p in Usd.PrimRange(robot_root,
            Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate)) if p.HasAPI(UsdPhysics.RigidBodyAPI)]
        external=[]
        for prim in stage.Traverse():
            path=str(prim.GetPath())
            if path.startswith('/Franka') or not prim.HasAPI(UsdPhysics.CollisionAPI):continue
            if UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is False:continue
            owner=prim
            while owner and not owner.IsPseudoRoot() and not owner.HasAPI(UsdPhysics.RigidBodyAPI):owner=owner.GetParent()
            external.append(str(owner.GetPath()) if owner and not owner.IsPseudoRoot() else path)
        filters=list(dict.fromkeys([*paths,*external]));sim=SimulationManager.get_physics_sim_view()
        self.contacts=sim.create_rigid_contact_view(paths,filter_patterns=[filters for _ in paths],max_contact_data_count=4096)
        self.bodies=sim.create_rigid_body_view(paths)
        if (not set(paths)<=set(self.bodies.prim_paths) or not set(paths)<=set(self.contacts.sensor_paths)
                or any(not set(filters)<=set(group) for group in self.contacts.filter_paths)):
            raise RuntimeError('Physical observation omitted a body, contact sensor or filter')
        self.samples=[];self.actions=[];self.original_apply=self.robot.apply_action
        def apply(action,*values,**options):
            indices=getattr(action,'joint_indices',None)
            if indices is None or any(int(i)<7 for i in indices):
                self.actions.append(dict(at=time.monotonic(),positions=fixture.serial(getattr(action,'joint_positions',None))))
            return self.original_apply(action,*values,**options)
        self.robot.apply_action=apply
        bundle.world.add_physics_callback('global_acceptance_observer',lambda dt:self.sample())

    def sample(self):
        if len(self.samples)>=20000:raise RuntimeError('Observation sample budget exceeded')
        contacts,saturated=fixture.contact_rows(self.contacts,self.units)
        self.samples.append(dict(step=int(self.bundle.world.current_time_step_index),at=time.monotonic(),
            q=np.asarray(self.robot.get_joint_positions()).copy(),contacts=contacts,saturated=saturated))

    def context(self):
        from robot_skill_stack.integrations.isaac.spatial_mapping.robot_filter import _pose_matrices
        poses=_pose_matrices(np.asarray(self.bodies.get_transforms()).copy(),self.units)
        q=np.asarray(self.robot.get_joint_positions()).copy();names=list(self.robot.dof_names)
        body_poses=dict(zip(self.bodies.prim_paths,poses))
        return dict(joint_names=names,actual_start_q=q.tolist(),
            actual_start_body_poses={k:v.tolist() for k,v in body_poses.items()},base_transform=body_poses['/Franka/panda_link0'].tolist(),
            finger_positions=[float(q[names.index(k)]) for k in ('panda_finger_joint1','panda_finger_joint2')])

    def restore(self):
        self.bundle.world.remove_physics_callback('global_acceptance_observer');self.robot.apply_action=self.original_apply


class VetoPlanner:
    """Test-only planner observer: normal production plan followed by exact veto."""
    def __init__(self,primary,backend,observer,parts,obstacles,state,require_detour):
        self.primary=primary;self.backend=backend;self.observer=observer
        self.parts=parts;self.obstacles=obstacles;self.state=state;self.require_detour=require_detour

    def __getattr__(self,name):return getattr(self.primary,name)

    def plan(self,target,start_q,world,**tolerances):
        from robot_skill_stack.manipulation.motion.joint_route import PlanningFailure
        instant=(self.backend.world.current_time,self.backend.world.current_time_step_index)
        context=self.observer.context()
        plan=self.primary.plan(target,start_q,world,**tolerances)
        times,q=fixture.sample_trajectory(plan)
        payload=dict(context,route_joint_names=list(plan.joint_names),route=plan.route.tolist(),
            trajectory_times=times.tolist(),trajectory_q=q.tolist(),stop_on_collision=True)
        evaluator=PhysicalMotionEvaluator(self.parts,self.obstacles,self.primary._config['urdf_path'],max_queries=500000,max_samples=12000)
        checked=fixture.bounded_evaluate(evaluator,payload)
        self.state.update(plan=payload,physical_evaluation=checked)
        if checked.get('execution_allowed') is not True:
            raise PlanningFailure('acceptance_physical_veto','Independent physical veto refused execution',checked)
        if self.require_detour:
            a,b=plan.route[0],plan.route[-1];direction=b-a
            fraction=np.clip((plan.route-a)@direction/(direction@direction),0.,1.)
            deviation=float(np.linalg.norm(plan.route-(a+fraction[:,None]*direction),axis=1).max())
            count=max(2,int(np.ceil(np.max(abs(direction))/.01))+1)
            straight=np.linspace(a,b,count)
            control=dict(payload,route=[a.tolist(),b.tolist()],trajectory_times=(np.arange(count)/240.).tolist(),trajectory_q=straight.tolist())
            veto=fixture.bounded_evaluate(evaluator,control)
            self.state.update(detour_deviation_rad=deviation,straight_control_evaluation=veto)
            if len(plan.route)<3 or deviation<=.05 or veto.get('status')!='VETO_PREDICTED_PHYSICAL_COLLISION':
                raise PlanningFailure('acceptance_detour_missing','Nominal obstacle-induced detour was not established')
        if instant!=(self.backend.world.current_time,self.backend.world.current_time_step_index):
            raise PlanningFailure('acceptance_physics_advanced','Physics advanced during independent validation')
        return plan

    def close(self):
        try:self.primary.close()
        finally:self.state['planner_closed']=True


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',choices=('static','frozen_rgbd'),required=True)
    parser.add_argument('--result',type=Path,required=True)
    parser.add_argument('--artifacts-dir',type=Path,required=True)
    args=parser.parse_args();args.artifacts_dir.mkdir(parents=True,exist_ok=False)
    if np.__version__!='1.26.4':raise RuntimeError('Use unchanged launcher and NumPy1.26.4')
    print(f'Test NumPy: {np.__version__}',flush=True)
    from isaacsim import SimulationApp
    app=SimulationApp({'headless':True});state={};checks={};error=None
    global fixture,PhysicalMotionEvaluator,robot_parts_from_capture,physical_obstacles_from_static
    from tests.support import global_native as fixture
    from tests.support.global_physical import PhysicalMotionEvaluator,robot_parts_from_capture,physical_obstacles_from_static
    from tests.support.isaac_async import run_kit_coroutine
    async def evaluate():
        import omni.kit.app
        import omni.usd
        from pxr import Gf,UsdGeom,UsdPhysics,PhysxSchema
        from isaacsim.core.simulation_manager import SimulationManager
        from robot_skill_stack.common.types import Pose
        from robot_skill_stack.integrations.isaac.bootstrap import build_runtime
        from robot_skill_stack.integrations.isaac.manipulation.motion.direct import DirectRmpflowStrategy
        from robot_skill_stack.integrations.isaac.manipulation.motion.static_world import capture_static_world
        bundle=observer=None;original_factory=None;original_submit=None;inputs=deque(maxlen=4)
        rgbd=args.source=='frozen_rgbd'
        objects=json.loads((fixture.ROOT/'tests/fixtures/global_partial.json').read_text()) if rgbd else []
        stage=fixture.prepare_stage(app)
        for index,item in enumerate(objects):
            cube=UsdGeom.Cube.Define(stage,item['path']);cube.GetSizeAttr().Set(1.)
            cube.AddTranslateOp().Set(Gf.Vec3d(3.+index,3.,3.));cube.AddScaleOp().Set(Gf.Vec3f(*item['dimensions']))
            cube.GetDisplayColorAttr().Set([Gf.Vec3f(.75,.18,.08)])
            UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
            UsdPhysics.RigidBodyAPI.Apply(cube.GetPrim()).CreateKinematicEnabledAttr().Set(True)
            PhysxSchema.PhysxRigidBodyAPI.Apply(cube.GetPrim()).CreateDisableGravityAttr().Set(True)
            PhysxSchema.PhysxContactReportAPI.Apply(cube.GetPrim()).CreateThresholdAttr().Set(0.)
        text=(fixture.ROOT/'tests/fixtures/global_rgbd_partial.toml').read_text()
        if not rgbd:
            for old,new in [('provider = "multi_view"','provider = "ground_truth"'),
                ('global_obstacle_source = "frozen_rgbd"','global_obstacle_source = "static"'),
                ('global_rgbd_representation = "reconstructed"','global_rgbd_representation = "occupied"'),
                ('enabled = true','enabled = false')]:
                if text.count(old)!=1:raise RuntimeError('Frozen profile field changed: '+old)
                text=text.replace(old,new)
        profile=args.artifacts_dir/'runtime_profile.toml';profile.write_text(text)
        try:
            bundle=await build_runtime(profile);backend=bundle.backend;strategy=backend.motion_strategy
            backend._motion_strategy=DirectRmpflowStrategy(backend)
            try:
                setup=await bundle.runtime.execute_async('move_to_pose',target=Pose(fixture.START,fixture.ORIENTATION),position_tolerance=.025,orientation_tolerance=.10)
                state['setup']=setup
                if not setup.ok:raise RuntimeError('Ordinary direct START setup failed')
            finally:backend._motion_strategy=strategy
            if objects:
                view=SimulationManager.get_physics_sim_view().create_rigid_body_view([o['path'] for o in objects]);ordered={o['path']:o for o in objects}
                poses=np.array([[*ordered[p]['center'],0.,0.,0.,1.] for p in view.prim_paths],np.float32)
                for item in objects:stage.GetPrimAtPath(item['path']).GetAttribute('xformOp:translate').Set(Gf.Vec3d(*item['center']))
                view.set_transforms(poses,np.arange(len(poses),dtype=np.int32));before=np.asarray(view.get_transforms()).copy()
            if rgbd:
                service=bundle.spatial_mapping;original_submit=service.submit
                def retained(payload):
                    accepted=original_submit(payload)
                    if accepted:inputs.append(payload)
                    return accepted
                service.submit=retained;started=time.monotonic();published=service.health['published']
                while time.monotonic()-started<30.:
                    await omni.kit.app.get_app().next_update_async()
                    if time.monotonic()-started>=8. and service.health['published']-published>=8 and service.health['fresh']:break
                state['warmup_health']=dict(service.health)
            else:checks['gt_independent_of_mapping']=bundle.spatial_mapping is None
            observer=Observer(bundle)
            paths=tuple(str(p.GetPath()) for p in stage.Traverse())
            meshes,cooking=fixture._capture_robot_meshes(stage,stage.GetPrimAtPath('/Franka'))
            checks['capture_no_usd_prim_side_effects']=paths==tuple(str(p.GetPath()) for p in stage.Traverse())
            physical=capture_static_world(stage,'/Franka')
            parts=robot_parts_from_capture(meshes);obstacles=physical_obstacles_from_static(physical)
            checks['all_eleven_physical_robot_bodies']=len({p.body for p in parts})==11
            original_factory=strategy._planner_factory
            strategy._planner_factory=lambda backend,**options:VetoPlanner(original_factory(backend,**options),backend,observer,parts,obstacles,state,rgbd)
            observer.sample();initial=np.asarray(backend.robot.get_joint_positions()).copy()
            result=await bundle.runtime.execute_async('move_to_pose',target=Pose(fixture.GOAL,fixture.ORIENTATION),position_tolerance=.025,orientation_tolerance=.10)
            state['motion']=result;state['strategy_after_motion']=dict(strategy.diagnostics)
            pose=backend.get_end_effector_pose();state['position_error_m']=float(np.linalg.norm(pose.position-np.asarray(fixture.GOAL)))
            checks.update(actual_motion_succeeded=result.ok,endpoint_position_within_tolerance=state['position_error_m']<=.025,
                physical_veto_allowed=state.get('physical_evaluation',{}).get('execution_allowed') is True,
                genuine_joint_motion=bool(observer.samples) and max(float(np.max(abs(row['q'][:7]-initial[:7]))) for row in observer.samples)>.05,
                zero_reported_contacts=not any(row['contacts'] for row in observer.samples),
                contact_buffer_not_saturated=not any(row['saturated'] for row in observer.samples),
                arm_actions_observed=bool(observer.actions),planner_closed=state.get('planner_closed') is True)
            if objects:checks['physical_fixtures_unchanged']=np.array_equal(before,np.asarray(view.get_transforms()))
            if rgbd:
                frozen=strategy.last_plan_trace.get('frozen_rgbd')
                checks['original_timestamp_and_source_preserved']=bool(frozen and frozen.timestamp==frozen.view.inferred.timestamp)
                checks['qualified_reconstructed_world']=bool(frozen and frozen.view.inferred.status=='qualified' and len(frozen.view.inferred.models)==2)
                coverage=result.details.get('planning_source',{}).get('occupied_coverage',{})
                checks['original_selected_union_and_budget_preserved']=coverage.get('occupied_union_preserved') is True and coverage.get('combined_camera_proxies',65)<=64
                if frozen:
                    state['inferred']=fixture.serial(frozen.view.inferred)
                    selected=next((p for p in inputs if p.generation==frozen.generation and min(f.acquired_at for f in p.frames)==frozen.timestamp),None)
                    checks['original_camera_input_retained']=selected is not None
                    if selected is not None:
                        from robot_skill_stack.runtime.process_worker import _TransportPickler
                        stream=io.BytesIO();_TransportPickler(stream,protocol=5).dump(selected)
                        (args.artifacts_dir/'mapping_input.pickle').write_bytes(stream.getvalue())
                before_rejected=len(observer.actions);service.close()
                refused=await bundle.runtime.execute_async('move_to_pose',target=Pose(fixture.START,fixture.ORIENTATION),position_tolerance=.025,orientation_tolerance=.10)
                state['closed_source_result']=refused
                checks['closed_source_refused_without_fallback']=not refused.ok and len(observer.actions)==before_rejected and str(refused.details.get('failure_code','')).startswith('planning_source_')
            steps=sorted({s['step'] for s in observer.samples});checks['complete_sampled_physics_coverage']=bool(steps) and steps==list(range(steps[0],steps[-1]+1))
            state['observed_actions']=len(observer.actions);state['physics_steps']=len(steps)
            (args.artifacts_dir/'motion_trace.json').write_text(json.dumps(fixture.serial(dict(samples=observer.samples,actions=observer.actions)),indent=2)+'\n')
        finally:
            if bundle is not None:
                if original_submit is not None:bundle.spatial_mapping.submit=original_submit
                if original_factory is not None:bundle.backend.motion_strategy._planner_factory=original_factory
                if observer is not None:observer.restore()
                bundle.close();checks['runtime_closed']=bundle._closed
                checks['strategy_closed']=bundle.backend.motion_strategy.diagnostics.get('closed') is True
    try:run_kit_coroutine(evaluate(),app)
    except BaseException:error=traceback.format_exc()
    finally:
        result=dict(status='PASS' if error is None and checks and all(checks.values()) else 'FAIL',source=args.source,checks=checks,metrics=fixture.serial(state),error=error,
            limitation='Independent sampled cooked-geometry veto and native contact observations do not certify continuous collision safety.')
        args.result.parent.mkdir(parents=True,exist_ok=True);args.result.write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(dict(status=result['status'],source=args.source,checks=checks,error=error)),flush=True)
        app.close()
    return 0 if result['status']=='PASS' else 1


if __name__=='__main__':raise SystemExit(main())
