"""Small NumPy ownership operations shared by SDK-free consumers."""
from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, DTypeLike


def readonly_copy(values: ArrayLike, dtype: DTypeLike | None = None) -> np.ndarray:
    """Own a C-contiguous, bytes-backed copy whose writes cannot be re-enabled."""
    array = np.asarray(values, dtype=dtype)
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)
