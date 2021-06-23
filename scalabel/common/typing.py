"""Common types used in the project."""

from typing import Any, Dict

import numpy as np

DictStrAny = Dict[str, Any]  # type: ignore[misc]

NDArray64 = np.typing.NDArray[np.float64]
