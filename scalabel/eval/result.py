"""Definition of the base evaluation result class."""

from collections import defaultdict
from typing import AbstractSet, Callable, Dict, List, Optional, Union

from mypy_extensions import KwArg, VarArg
from pandas import DataFrame
from pydantic import BaseModel, PrivateAttr

FORMATTER = Callable[[VarArg(object), KwArg(object)], str]
FlattenDict = Dict[str, Union[int, float]]
NestedDict = Dict[str, Dict[str, Union[int, float]]]
AVERAGE = "AVERAGE"
OVERALL = "OVERALL"


class Result(BaseModel):
    """The base class for bdd100k evaluation results.

    Each data field corresponds to a evluation metric. The value for each
    metric is a dict that maps the category names to scores.

    Functions:
        dict() -> dict[str, dict[str, int | float]]:
            export all data to a nested dict.
        json() -> str:
            export the nested dict of `dict()` into a JSON string.
        pd_frame() -> pandas.DataFrame:
            export data fields to a formatted DataFrame.
        table() -> str:
            export data fields to a formatted table string.
        summary() -> dict[str, int | float]:
            export most important fields to a flattened dict.
        __str__() -> str:
            the same as `table()`.
    """

    _formatters: Dict[str, FORMATTER] = PrivateAttr(dict())
    _row_breaks: List[int] = PrivateAttr([])

    def __eq__(self, other: "Result") -> bool:  # type: ignore
        """Check whether two instances are equal."""
        if self._row_breaks != other._row_breaks:
            return False
        return super().__eq__(other)

    def pd_frame(
        self,
        include: Optional[AbstractSet[str]] = None,
        exclude: Optional[AbstractSet[str]] = None,
    ) -> DataFrame:
        """Convert data model into a data frame.

        Args:
            include (set[str]): Optional, the metrics to convert
            exclude (set[str]): Optional, the metrics not to convert
        Returns:
            data_frame (pandas.DataFrmae): the exported DataFrame
        """
        frame_dict: Dict[str, Dict[str, Union[int, float]]] = defaultdict(dict)
        for metric, scores in self.dict(
            include=include, exclude=exclude  # type: ignore
        ).items():
            if not isinstance(scores, dict):
                continue
            for cls_, score in scores.items():
                frame_dict[metric][cls_] = score
        return DataFrame.from_dict(frame_dict)

    def table(
        self,
        include: Optional[AbstractSet[str]] = None,
        exclude: Optional[AbstractSet[str]] = None,
    ) -> str:
        """Convert data model into a table for formatted printing.

        Args:
            include (set[str]): Optional, the metrics to convert
            exclude (set[str]): Optional, the metrics not to convert
        Returns:
            table (str): the exported table string
        """
        data_frame = self.pd_frame(include, exclude)
        if not self._formatters:
            formatters = {
                metric: "{:.1f}".format for metric in data_frame.columns
            }
        else:
            formatters = self._formatters

        summary = data_frame.to_string(formatters=formatters)
        strs = summary.split("\n")
        split_line = "-" * len(strs[0])

        for row_ind in self._row_breaks:
            strs.insert(row_ind, split_line)
        summary = "".join([f"{s}\n" for s in strs])
        summary = "\n" + summary
        return summary  # type: ignore

    def __str__(self) -> str:
        """Convert the data into a printable string."""
        return self.table()

    def summary(self) -> Dict[str, Union[int, float]]:
        """Convert the data into a flattened dict as the summary.

        This function is different to the `.dict()` function.
        As a comparison, `.dict()` will export all data fields as a nested
        dict, While `.summary()` only exports most important information,
        like the overall scores, as a flattened compact dict.
        """
        return dict(self)
