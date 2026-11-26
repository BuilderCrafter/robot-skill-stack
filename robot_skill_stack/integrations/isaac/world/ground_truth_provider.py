from __future__ import annotations

import re
import time
from typing import TYPE_CHECKING

import numpy as np
import omni.usd
from pxr import Usd, UsdGeom

from robot_skill_stack.common.types import Pose
from robot_skill_stack.world.model.observations import ObjectObservation
from robot_skill_stack.world.model.provider import ObservationBatch
from robot_skill_stack.world.model.primitives import PrimitiveGeometry, PrimitiveShape

if TYPE_CHECKING:
    from robot_skill_stack.integrations.isaac.config import ObjectConfig
    from robot_skill_stack.world.model.context import ObservationContext


class IsaacGroundTruthProvider:
    """Publish current active USD children without owning cameras or subscriptions."""

    def __init__(self, objects_root: str, object_configs: dict[str, ObjectConfig] | None = None) -> None:
        self.objects_root = objects_root
        self.object_configs = object_configs or {}
        self.bbox_cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            [UsdGeom.Tokens.default_],
            useExtentsHint=True,
        )

    def _objects(self) -> list[Usd.Prim]:
        """Enumerate live children on each sample instead of caching scene identities."""
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
    def _object_id(prim: Usd.Prim) -> str:
        custom = prim.GetCustomData()
        return str(custom.get("object_id", prim.GetName())).lower()

    @staticmethod
    def _class_name(prim: Usd.Prim, object_id: str) -> str:
        custom = prim.GetCustomData()
        if "class_name" in custom:
            return str(custom["class_name"])
        return re.sub(r"_\d+$", "", object_id)

    @staticmethod
    def _pose(prim: Usd.Prim) -> Pose:
        """Read the USD world transform and preserve scalar-first quaternion ordering."""
        transform = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())

        position = transform.ExtractTranslation()
        rotation = transform.ExtractRotationQuat()
        imaginary = rotation.GetImaginary()

        return Pose(
            position=np.array([position[0], position[1], position[2]], dtype=float),
            orientation=np.array([rotation.GetReal(), imaginary[0], imaginary[1], imaginary[2]], dtype=float),
            frame="world",
        )

    def _size(self, prim: Usd.Prim, cfg: ObjectConfig | None) -> np.ndarray | None:
        """Prefer configured size, otherwise use USD's aligned local-bound extents."""
        if cfg is not None and cfg.size is not None:
            return cfg.size.copy()

        self.bbox_cache.Clear()
        bounds = self.bbox_cache.ComputeLocalBound(prim).ComputeAlignedRange()
        size = np.asarray(bounds.GetSize(), dtype=float)

        return size if np.all(np.isfinite(size)) else None

    @staticmethod
    def _geometry(class_name: str | None, pose: Pose, size: np.ndarray | None) -> PrimitiveGeometry | None:
        """Build primitive geometry using world orientation and the existing size convention."""
        name = (class_name or "").lower()
        if size is None:
            return None
        if name in ("cube", "box"):
            quaternion = pose.orientation
            # World-Z yaw for [w, x, y, z], normalized to the existing box symmetry.
            yaw = (np.arctan2(2*(quaternion[0]*quaternion[3]+quaternion[1]*quaternion[2]),
                             1-2*(quaternion[2]**2+quaternion[3]**2)) if quaternion is not None else 0.)
            return PrimitiveGeometry(PrimitiveShape.CUBE, 1., box_size=size, yaw=float(yaw % (np.pi/2)))
        if name == "sphere":
            return PrimitiveGeometry(PrimitiveShape.SPHERE, 1., radius=float(np.mean(size)/2))
        if name == "cylinder":
            quaternion = pose.orientation
            if quaternion is None:
                axis = np.array([0., 0., 1.])
            else:
                w, x, y, z = quaternion
                # The transformed local +Z axis is the cylinder axis in world coordinates.
                axis = np.array([2*(x*z+w*y), 2*(y*z-w*x), 1-2*(x*x+y*y)])
            axis = axis/max(np.linalg.norm(axis), 1e-9)
            axial_index = np.argmax(np.abs(axis))
            length = float(size[axial_index])
            radius = float(np.median(np.delete(size, axial_index))/2)
            return PrimitiveGeometry(PrimitiveShape.CYLINDER, 1., axis=axis, radius=radius, length=length)
        return PrimitiveGeometry(PrimitiveShape.UNKNOWN, 1.)

    @staticmethod
    def _graspable(prim: Usd.Prim, cfg: ObjectConfig | None) -> bool:
        if cfg is not None:
            return cfg.graspable
        return bool(prim.GetCustomData().get("graspable", True))

    def observe(self, context: ObservationContext | None = None) -> ObservationBatch:
        observations = []

        for prim in self._objects():
            object_id = self._object_id(prim)
            cfg = self.object_configs.get(object_id)

            class_name = self._class_name(prim, object_id)
            pose = self._pose(prim)
            size = self._size(prim, cfg)
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

        return ObservationBatch(observations, time.monotonic())

    def forget(self, object_id: str) -> None:
        pass  # Ground truth enumerates live scene objects on every sample.

    def close(self) -> None:
        pass  # This adapter owns no camera or subscriptions.
