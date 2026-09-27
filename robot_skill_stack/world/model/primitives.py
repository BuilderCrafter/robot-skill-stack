from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any
import numpy as np

class PrimitiveShape(str, Enum):
    CUBE="cube"; SPHERE="sphere"; CYLINDER="cylinder"; UNKNOWN="unknown"

@dataclass
class PrimitiveGeometry:
    shape: PrimitiveShape
    confidence: float=0.0
    box_size: np.ndarray|None=None
    yaw: float|None=None
    radius: float|None=None
    axis: np.ndarray|None=None
    length: float|None=None
    scores: dict[str,float]=field(default_factory=dict)
    metadata: dict[str,Any]=field(default_factory=dict)
    def __post_init__(self):
        self.shape=PrimitiveShape(self.shape); self.confidence=float(np.clip(self.confidence,0,1))
        if self.box_size is not None: self.box_size=np.asarray(self.box_size,dtype=float)
        if self.axis is not None:
            self.axis=np.asarray(self.axis,dtype=float); n=np.linalg.norm(self.axis)
            if self.axis.shape!=(3,) or n==0: raise ValueError("axis must be nonzero shape (3,)")
            self.axis=self.axis/n
