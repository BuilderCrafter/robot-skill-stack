from __future__ import annotations

import numpy as np
import omni.usd
from isaacsim.core.utils.semantics import add_labels

from perception.detector import Detection


class IsaacSemanticDetector:
    def __init__(self, camera, object_prim_paths: dict[str, str]):
        self.camera = camera
        stage = omni.usd.get_context().get_stage()

        for object_id, prim_path in object_prim_paths.items():
            prim = stage.GetPrimAtPath(prim_path)
            if not prim.IsValid():
                raise ValueError(f"Object prim not found: {prim_path}")
            add_labels(prim, labels=[object_id], instance_name="class")

    def detect(self, object_id: str) -> Detection | None:
        seg, info = self.camera.get_semantic_segmentation()
        if seg is None:
            return None

        ids = [
            int(value)
            for value, labels in info.get("idToLabels", {}).items()
            if labels.get("class") == object_id
        ]

        if not ids:
            return None

        mask = np.isin(seg, ids)
        if not mask.any():
            return None

        return Detection(object_id, mask, 1.0)