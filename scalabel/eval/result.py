"""Definition of the base evaluation result class."""

from typing import Callable, Dict, List, Optional

import pandas as pd
from mypy_extensions import KwArg, VarArg

FORMATTER = Callable[[VarArg(object), KwArg(object)], str]


class EvalResult:
    """The base class for the evaluation result."""

    def __init__(
        self,
        res_dict: Dict[str, float],
        data_frame: pd.DataFrame,
        formatters: Optional[Dict[str, FORMATTER]] = None,
        row_breaks: Optional[List[int]] = None,
    ) -> None:
        """Initialization of the class."""
        self.res_dict = res_dict
        self.data_frame = data_frame
        if formatters is not None:
            self.formatters = formatters
        else:
            self.formatters = {
                metric: "{:.1%}".format for metric in self.data_frame.columns
            }

        if row_breaks is not None:
            self.row_breaks = row_breaks
        else:
            self.row_breaks = []

    def __str__(self) -> str:
        """The base printing function."""
        summary = self.data_frame.to_string(formatters=self.formatters)
        strs = summary.split("\n")
        split_line = "-" * len(strs[0])
        for row_ind in self.row_breaks:
            strs.insert(row_ind, split_line)
        summary = "".join([f"{s}\n" for s in strs])
        summary = "\n" + summary
        return summary  # type: ignore
