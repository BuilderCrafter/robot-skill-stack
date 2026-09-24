from __future__ import annotations

import re

import numpy as np
import omni.usd
from pxr import Usd, UsdGeom

from core.types import Pose
from world_model.observations import ObjectObservation


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
    def _graspable(prim, cfg):
        if cfg is not None:
            return cfg.graspable

        return bool(prim.GetCustomData().get("graspable", True))

    def observe(self) -> list[ObjectObservation]:
        observations = []

        for prim in self._objects():
            object_id = self._object_id(prim)
            cfg = self.object_configs.get(object_id)

            observations.append(
                ObjectObservation(
                    object_id=object_id,
                    class_name=self._class_name(prim, object_id),
                    pose=self._pose(prim),
                    size=self._size(prim, cfg),
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