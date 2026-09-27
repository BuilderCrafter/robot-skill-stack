from __future__ import annotations

import re

import numpy as np
import omni.usd
from pxr import Usd, UsdGeom

from robot_skill_stack.common.types import Pose
from robot_skill_stack.world.model.observations import ObjectObservation
from robot_skill_stack.world.model.primitives import PrimitiveGeometry, PrimitiveShape


class IsaacGroundTruthProvider:
    def __init__(self, objects_root: str, object_configs: dict | None = None):
        self.objects_root = objects_root
        self.object_configs = object_configs or {}
        self.bbox_cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            [UsdGeom.Tokens.default_],
            useExtentsHint=True,
        )

    def _objects(self):
        stage = omni.usd.get_context().get_stage()
        root = stage.GetPrimAtPath(self.objects_root)

        if not root.IsValid():
            raise RuntimeError(f"Objects root not found: {self.objects_root}")

        return [
            prim
            for prim in root.GetChildren()
            if prim.IsValid() and prim.IsActive()
        ]

    @staticmethod
    def _object_id(prim):
        custom = prim.GetCustomData()
        return str(custom.get("object_id", prim.GetName())).lower()

    @staticmethod
    def _class_name(prim, object_id):
        custom = prim.GetCustomData()
        if "class_name" in custom:
            return str(custom["class_name"])
        return re.sub(r"_\d+$", "", object_id)

    @staticmethod
    def _pose(prim):
        transform = UsdGeom.Xformable(
            prim
        ).ComputeLocalToWorldTransform(Usd.TimeCode.Default())

        p = transform.ExtractTranslation()
        q = transform.ExtractRotationQuat()
        imag = q.GetImaginary()

        return Pose(
            position=np.array([p[0], p[1], p[2]], dtype=float),
            orientation=np.array(
                [q.GetReal(), imag[0], imag[1], imag[2]],
                dtype=float,
            ),
            frame="world",
        )

    def _size(self, prim, cfg):
        if cfg is not None and cfg.size is not None:
            return cfg.size.copy()

        self.bbox_cache.Clear()
        bounds = self.bbox_cache.ComputeLocalBound(
            prim
        ).ComputeAlignedRange()
        size = np.asarray(bounds.GetSize(), dtype=float)

        return size if np.all(np.isfinite(size)) else None

    @staticmethod
    def _geometry(class_name, pose, size):
        name=(class_name or "").lower()
        if size is None: return None
        if name in ("cube","box"):
            q=pose.orientation; yaw=np.arctan2(2*(q[0]*q[3]+q[1]*q[2]),1-2*(q[2]**2+q[3]**2)) if q is not None else 0.
            return PrimitiveGeometry(PrimitiveShape.CUBE,1.,box_size=size,yaw=float(yaw%(np.pi/2)))
        if name=="sphere": return PrimitiveGeometry(PrimitiveShape.SPHERE,1.,radius=float(np.mean(size)/2))
        if name=="cylinder":
            q=pose.orientation
            if q is None: axis=np.array([0.,0.,1.])
            else:
                w,x,y,z=q; axis=np.array([2*(x*z+w*y),2*(y*z-w*x),1-2*(x*x+y*y)])
            axis=axis/max(np.linalg.norm(axis),1e-9); length=float(size[np.argmax(np.abs(axis))]); radius=float(np.median(np.delete(size,np.argmax(np.abs(axis))))/2)
            return PrimitiveGeometry(PrimitiveShape.CYLINDER,1.,axis=axis,radius=radius,length=length)
        return PrimitiveGeometry(PrimitiveShape.UNKNOWN,1.)

    @staticmethod
    def _graspable(prim, cfg):
        if cfg is not None:
            return cfg.graspable
        return bool(prim.GetCustomData().get("graspable", True))

    def observe(self, context=None) -> list[ObjectObservation]:
        observations = []

        for prim in self._objects():
            object_id = self._object_id(prim)
            cfg = self.object_configs.get(object_id)

            class_name=self._class_name(prim, object_id); pose=self._pose(prim); size=self._size(prim, cfg)
            observations.append(
                ObjectObservation(
                    object_id=object_id,
                    class_name=class_name,
                    pose=pose,
                    size=size,
                    geometry=self._geometry(class_name, pose, size),
                    graspable=self._graspable(prim, cfg),
                    visible=True,
                    confidence=1.0,
                    source="ground_truth",
                    metadata={
                        "prim_path": prim.GetPath().pathString,
                    },
                )
            )

        return observations
