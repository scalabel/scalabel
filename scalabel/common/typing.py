"""Common types used in the project."""

from typing import Any, Dict

import numpy as np
import numpy.typing as npt

DictStrAny = Dict[str, Any]  # type: ignore[misc]

NDArrayF64 = npt.NDArray[np.float64]
NDArrayI32 = npt.NDArray[np.int32]
NDArrayU8 = npt.NDArray[np.uint8]
