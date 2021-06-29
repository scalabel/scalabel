"""Common types used in the project."""

from typing import Any, Dict

import numpy as np
import numpy.typing as npt

DictStrAny = Dict[str, Any]  # type: ignore[misc]

FloatArray = npt.NDArray[np.float32]
IntArray = npt.NDArray[np.int32]
UintArray = npt.NDArray[np.uint8]
