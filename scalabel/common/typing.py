"""Common types used in the project."""

from typing import Any, Dict

import numpy as np
import numpy.typing as npt

DictStrAny = Dict[str, Any]  # type: ignore[misc]

NDArray64 = npt.NDArray[np.float64]
