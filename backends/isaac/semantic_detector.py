from __future__ import annotations

import re

import numpy as np
import omni.usd
from isaacsim.core.utils.semantics import add_labels

from perception.detector import Detection


class IsaacSemanticDetector:
    def __init__(
        self,
        camera,
        *,
        objects_root="/World/Objects",
    ):
        self.camera = camera
        self.objects_root = objects_root
        self._labelled_paths = set()
        self._classes: dict[str, str] = {}

        self._ensure_labels()

    @staticmethod
    def _identity(prim):
        custom = prim.GetCustomData()
        object_id = str(
            custom.get("object_id", prim.GetName())
        ).lower()

        class_name = str(
            custom.get(
                "class_name",
                re.sub(r"_\d+$", "", object_id),
            )
        )

        return object_id, class_name

    def _ensure_labels(self):
        stage = omni.usd.get_context().get_stage()
        root = stage.GetPrimAtPath(self.objects_root)

        if not root.IsValid():
            raise RuntimeError(
                f"Objects root not found: {self.objects_root}"
            )

        current_paths = set()

        for prim in root.GetChildren():
            if not prim.IsValid() or not prim.IsActive():
                continue

            path = prim.GetPath().pathString
            current_paths.add(path)

            object_id, class_name = self._identity(prim)
            self._classes[object_id] = class_name

            if path not in self._labelled_paths:
                add_labels(
                    prim,
                    labels=[object_id],
                    instance_name="class",
                )
                self._labelled_paths.add(path)

        self._labelled_paths.intersection_update(current_paths)

    def detect_all(self) -> list[Detection]:
        self._ensure_labels()

        seg, info = self.camera.get_semantic_segmentation()
        if seg is None:
            return []

        detections = []

        for value, labels in info.get("idToLabels", {}).items():
            object_id = labels.get("class")

            if not object_id or object_id in (
                "BACKGROUND",
                "UNLABELLED",
            ):
                continue

            mask = seg == int(value)
            if not mask.any():
                continue

            detections.append(
                Detection(
                    object_id=object_id,
                    class_name=self._classes.get(
                        object_id,
                        object_id,
                    ),
                    mask=mask,
                    confidence=1.0,
                    metadata={
                        "detector": "isaac_semantic",
                    },
                )
            )

        return detections

    def detect(self, object_id: str) -> Detection | None:
        return next(
            (
                detection
                for detection in self.detect_all()
                if detection.object_id == object_id
            ),
            None,
        )