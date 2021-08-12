"""Definition of the base evaluation result class."""

from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Union

from mypy_extensions import KwArg, VarArg
from pandas import DataFrame
from pydantic import BaseModel, PrivateAttr

FORMATTER = Callable[[VarArg(object), KwArg(object)], str]
FlattenDict = Dict[str, Union[int, float]]
NestedDict = Dict[str, Dict[str, Union[int, float]]]
AVERAGE = "AVERAGE"
OVERALL = "OVERALL"


def result_to_flatten_dict(result: BaseModel) -> FlattenDict:
    """Convert a result to a faltten dict.."""
    flat_dict: FlattenDict = dict()
    for name, content in dict(result).items():
        if isinstance(content, list):
            flat_dict[name] = content[-1]
        elif isinstance(content, (int, float)):
            flat_dict[name] = content
    return flat_dict


def result_to_nested_dict(
    result: BaseModel, all_classes: List[str]
) -> NestedDict:
    """Convert a result to a nested dict.."""
    nest_dict: NestedDict = defaultdict(dict)
    for name, content in dict(result).items():
        if isinstance(content, list):
            nest_dict[name].update(dict(zip(all_classes, content)))
    return nest_dict


def nested_dict_to_data_frame(
    res_dict: NestedDict,
    primary_key: str = "metric",
    default_value: float = 0.0,
    default_values: Optional[FlattenDict] = None,
) -> DataFrame:
    """Fill the nested_dict as a full 2-d array."""
    assert primary_key in ["metric", "category"]

    if default_values is None:
        default_values = dict()

    keys_2rd = set()
    for simp_dict in res_dict.values():
        for key in simp_dict.keys():
            keys_2rd.add(key)
    key_2rd_list = sorted(list(keys_2rd))

    for simp_dict in res_dict.values():
        for key_2rd in key_2rd_list:
            if key_2rd not in simp_dict:
                simp_dict[key_2rd] = default_values.get(key_2rd, default_value)

    orient = {"metric": "columns", "category": "index"}[primary_key]
    return DataFrame.from_dict(res_dict, orient=orient)


def data_frame_to_str(
    data_frame: DataFrame,
    formatters: Optional[Dict[str, FORMATTER]] = None,
    row_breaks: Optional[List[int]] = None,
) -> str:
    """Convern a DataFrame to structured text."""
    if formatters is None:
        formatters = {metric: "{:.1%}".format for metric in data_frame.columns}
    summary = data_frame.to_string(formatters=formatters)
    strs = summary.split("\n")
    split_line = "-" * len(strs[0])

    if row_breaks is None:
        row_breaks = []
    for row_ind in row_breaks:
        strs.insert(row_ind, split_line)
    summary = "".join([f"{s}\n" for s in strs])
    summary = "\n" + summary
    return summary  # type: ignore


class BaseResult(BaseModel):
    """The base class for bdd100k evaluation results."""

    _all_classes: List[str] = PrivateAttr()
    _row_breaks: List[int] = PrivateAttr()
    _formatters: Dict[str, FORMATTER] = PrivateAttr()

    def __init__(  # type: ignore
        self,
        all_classes: List[str],
        row_breaks: Optional[List[int]] = None,
        **data: Any,
    ) -> None:
        """Set extram parameters."""
        for scores in data.values():
            if isinstance(scores, list):
                assert len(scores) == len(all_classes)
        super().__init__(**data)
        self._all_classes = all_classes
        if row_breaks is None:
            row_breaks = []
        self._row_breaks = row_breaks
        self._formatters = dict()

    def __eq__(self, other: "BaseResult") -> bool:  # type: ignore
        """Check whether two instances are equal."""
        if self._all_classes != other._all_classes:
            return False
        return super().__eq__(other)

    def __str__(self) -> str:
        """Convert data model into a structures string."""
        nested_dict = result_to_nested_dict(self, self._all_classes)
        data_frame = nested_dict_to_data_frame(nested_dict)
        return data_frame_to_str(
            data_frame, self._formatters, self._row_breaks
        )
