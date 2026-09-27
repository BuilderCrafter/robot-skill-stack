from __future__ import annotations
import numpy as np
from robot_skill_stack.common.types import Pose
from robot_skill_stack.world.model.primitives import PrimitiveShape
from robot_skill_stack.manipulation.grasping.types import GraspFailureReason,GraspPlan,GraspResult,ParallelJawGripperSpec

def _qmul(a,b):
 w,x,y,z=a; W,X,Y,Z=b
 return np.array([w*W-x*X-y*Y-z*Z,w*X+x*W+y*Z-z*Y,w*Y-x*Z+y*W+z*X,w*Z+x*Y-y*X+z*W])
def _top_q(yaw): return _qmul(np.array([np.cos(yaw/2),0,0,np.sin(yaw/2)]),np.array([0.,0.,1.,0.]))

class SimplePrimitiveGraspPlanner:
    def __init__(self,*,approach_height=.10,default_lift_height=.12,grasp_z_offset=0.,gripper=None,grasp_orientation=None):
        self.approach_height=float(approach_height); self.default_lift_height=float(default_lift_height); self.grasp_z_offset=float(grasp_z_offset); self.gripper=gripper or ParallelJawGripperSpec(.075)
    def _fail(self,r,m,**d): return GraspResult.failure(r,m,**d)
    def plan(self,obj,*,lift_height=None,grasp_hint=None):
        if obj.pose is None:return self._fail(GraspFailureReason.OBJECT_POSE_UNKNOWN,'Object pose is unknown.')
        if not obj.graspable:return self._fail(GraspFailureReason.OBJECT_NOT_GRASPABLE,'Object is marked non-graspable.')
        if grasp_hint not in (None,'top','top_down'):return self._fail(GraspFailureReason.UNSUPPORTED_HINT,f"Unsupported grasp hint '{grasp_hint}'.")
        g=obj.geometry
        if g is None:
            # Backward-compatible cube behavior for legacy/ground-truth objects.
            if obj.class_name=='cube' and obj.size is not None:
                from robot_skill_stack.world.model.primitives import PrimitiveGeometry
                g=PrimitiveGeometry(PrimitiveShape.CUBE,1.,box_size=obj.size,yaw=0.)
            else:return self._fail(GraspFailureReason.NO_FEASIBLE_GRASP,'Primitive geometry is unknown.')
        if g.shape==PrimitiveShape.UNKNOWN:return self._fail(GraspFailureReason.NO_FEASIBLE_GRASP,'Unknown primitive: refusing to guess a grasp.')
        yaw=0.; width=None; strategy=g.shape.value
        if g.shape==PrimitiveShape.CUBE:
            dims=np.asarray(g.box_size if g.box_size is not None else obj.size,float); yaw=float(g.yaw or 0.)
            # Closing axis is hand local Y. Choose the narrower feasible face pair.
            options=[(dims[1],yaw),(dims[0],yaw-np.pi/2)]; feasible=[x for x in options if x[0]<=self.gripper.max_width]
            if feasible: width,yaw=min(feasible,key=lambda x:x[0])
            else: width=min(x[0] for x in options)
        elif g.shape==PrimitiveShape.SPHERE: width=2*float(g.radius)
        else:
            width=2*float(g.radius); axis=np.asarray(g.axis,float)
            if abs(axis[2])<.7:
                theta=float(np.arctan2(axis[1],axis[0])); yaw=theta # local Y becomes perpendicular to cylinder axis
                strategy='lying_cylinder'
            else: strategy='upright_cylinder'
        if width is None:return self._fail(GraspFailureReason.SIZE_UNKNOWN,'Primitive size is unknown.')
        if width>self.gripper.max_width:return self._fail(GraspFailureReason.OBJECT_TOO_LARGE,f'Primitive diameter/width {width:.3f} m exceeds gripper max width {self.gripper.max_width:.3f} m.',required_width_m=width,gripper_max_width_m=self.gripper.max_width)
        lift=self.default_lift_height if lift_height is None else float(lift_height); pos=obj.pose.position.copy(); pos[2]+=self.grasp_z_offset; grasp=Pose(pos,_top_q(yaw),obj.pose.frame)
        return GraspResult.success(GraspPlan(grasp.translated([0,0,self.approach_height]),grasp,grasp.translated([0,0,lift])),strategy=strategy,required_width_m=width,gripper_max_width_m=self.gripper.max_width,feasibility_checked=True,wrist_yaw_rad=yaw)
