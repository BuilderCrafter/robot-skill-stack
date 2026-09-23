from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Pose:
    """
    Generic robot pose.

    Position:
        [x, y, z] in meters

    Orientation:
        quaternion [w, x, y, z]

    Frame:
        coordinate frame name, currently normally "world"
    """

    position: np.ndarray
    orientation: np.ndarray | None = None
    frame: str = "world"

    def __post_init__(self) -> None:
        self.position = np.asarray(
            self.position,
            dtype=float,
        )

        if self.position.shape != (3,):
            raise ValueError(
                "Pose.position must have shape (3,)"
            )

        if self.orientation is not None:
            self.orientation = np.asarray(
                self.orientation,
                dtype=float,
            )

            if self.orientation.shape != (4,):
                raise ValueError(
                    "Pose.orientation must have shape (4,) "
                    "in [w, x, y, z] order"
                )

            norm = np.linalg.norm(self.orientation)

            if norm == 0.0:
                raise ValueError(
                    "Pose.orientation cannot be a zero quaternion"
                )

            # Always store a normalized quaternion.
            self.orientation = (
                self.orientation / norm
            )


    def translated(
        self,
        delta,
    ) -> "Pose":
        """
        Return a copy of this pose translated by delta.
        """

        return Pose(
            position=(
                self.position
                + np.asarray(delta, dtype=float)
            ),
            orientation=(
                None
                if self.orientation is None
                else self.orientation.copy()
            ),
            frame=self.frame,
        )