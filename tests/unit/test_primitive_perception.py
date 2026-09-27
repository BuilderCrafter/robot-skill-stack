from __future__ import annotations
import numpy as np
from types import SimpleNamespace
from robot_skill_stack.common.types import Pose
from robot_skill_stack.world.model.primitives import PrimitiveGeometry,PrimitiveShape
from robot_skill_stack.world.model.entities import WorldObject
from robot_skill_stack.world.model.world_model import WorldModel
from robot_skill_stack.world.model.context import ObservationContext,AssociationHint
from robot_skill_stack.world.model.observations import ObjectObservation
from robot_skill_stack.world.perception.primitives import PrimitiveEstimator
from robot_skill_stack.world.perception.tracker import ObjectTracker
from robot_skill_stack.world.perception.semantics import SemanticPrediction
from robot_skill_stack.manipulation.grasping import SimplePrimitiveGraspPlanner,ParallelJawGripperSpec,GraspFailureReason
rng=np.random.default_rng(7)
def C(p): return SimpleNamespace(position=np.mean(p,0),size=np.ptp(p,0),mask=np.ones((2,2),bool),confidence=1.,spawnable=True,metadata={'points':p})
def cube(y=.52,n=4000):
 p=rng.uniform(-.025,.025,(n,3)); f=rng.integers(0,6,n); a=f//2;p[np.arange(n),a]=np.where(f%2,.025,-.025); R=np.array([[np.cos(y),-np.sin(y),0],[np.sin(y),np.cos(y),0],[0,0,1]]);return p@R.T+[.45,0,.025]
def sphere(n=4000):
 v=rng.normal(size=(n,3));v/=np.linalg.norm(v,axis=1)[:,None];return .025*v+[.45,0,.025]
def cyl(horizontal=False,n=4000):
 t=rng.uniform(0,2*np.pi,n);z=rng.uniform(-.05,.05,n);p=np.c_[.02*np.cos(t),.02*np.sin(t),z];p=p[:,[2,1,0]] if horizontal else p;return p+[.45,0,.02 if horizontal else .05]
def obj(g,size=(.05,.05,.05)):return WorldObject('o','cube' if g.shape==PrimitiveShape.CUBE else g.shape.value,Pose([.45,0,.03]),np.array(size),geometry=g,visible=True)
def main():
 e=PrimitiveEstimator(); gc=e.estimate(C(cube())); gs=e.estimate(C(sphere())); gu=e.estimate(C(cyl())); gl=e.estimate(C(cyl(True)))
 assert gc.shape==PrimitiveShape.CUBE and min(abs(gc.yaw-.52),abs(gc.yaw-.52+np.pi/2),abs(gc.yaw-.52-np.pi/2))<.03
 assert gs.shape==PrimitiveShape.SPHERE and gs.yaw is None and gs.axis is None
 assert gu.shape==PrimitiveShape.CYLINDER and abs(abs(gu.axis[2])-1)<.05
 assert gl.shape==PrimitiveShape.CYLINDER and abs(gl.axis[2])<.1 and abs(abs(gl.axis[0])-1)<.08
 weird=rng.uniform(-1,1,(2000,3))*np.array([.035,.025,.0175])+[.45,0,.04]; assert e.estimate(C(weird)).shape==PrimitiveShape.UNKNOWN
 tr=ObjectTracker(min_confirm_hits=3,semantic_min_samples=1); pred=SemanticPrediction('cube',1); c=C(cube())
 tr.update([c],[pred],frame_id=1,timestamp=1); assert not tr.tracks()[0].confirmed
 tr.update([c],[pred],frame_id=2,timestamp=2); assert not tr.tracks()[0].confirmed
 tr.update([c],[pred],frame_id=3,timestamp=3); assert tr.tracks()[0].confirmed
 m=WorldModel(stale_object_ttl=2); m.apply_observations([ObjectObservation('x',pose=Pose([0,0,0]),visible=True,timestamp=1)]); m.apply_observations([],mark_missing_invisible=True); assert m.expire_stale(4)==('x',)
 for protected in ('held','hint'):
  m=WorldModel(stale_object_ttl=1);m.apply_observations([ObjectObservation(protected,pose=Pose([0,0,0]),visible=True,timestamp=1)]);m.apply_observations([],mark_missing_invisible=True)
  if protected=='held':m.set_held(protected)
  else:m._association_hints[protected]=AssociationHint(protected,np.zeros(3),.1,9999999999.,'place')
  assert not m.expire_stale(5)
 m=WorldModel();m.apply_observations([ObjectObservation('lost',visible=False,timestamp=1)]); forgotten=[];assert m.clear_lost(forgotten.append)==('lost',) and forgotten==['lost']
 p=SimplePrimitiveGraspPlanner(gripper=ParallelJawGripperSpec(.075));
 for g in (gc,gs,gu,gl): assert p.plan(obj(g)).ok
 assert p.plan(obj(PrimitiveGeometry(PrimitiveShape.SPHERE,1,radius=.05))).failure_reason==GraspFailureReason.OBJECT_TOO_LARGE
 assert p.plan(obj(PrimitiveGeometry(PrimitiveShape.UNKNOWN))).failure_reason==GraspFailureReason.NO_FEASIBLE_GRASP
 # lying cylinder uses diameter, not length; local-Y closing axis is perpendicular after wrist yaw mapping
 r=p.plan(obj(PrimitiveGeometry(PrimitiveShape.CYLINDER,1,axis=[1,0,0],radius=.02,length=.10),(.10,.04,.04)));assert r.ok and abs(r.details['required_width_m']-.04)<1e-9
 print('PASS primitive perception/grasp suite')
if __name__=='__main__':main()
