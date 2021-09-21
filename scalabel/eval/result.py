"""Definition of the base evaluation result class."""

from collections import defaultdict
from typing import AbstractSet, Dict, List, Optional, Union

import numpy as np
from pandas import DataFrame
from pydantic import BaseModel, PrivateAttr

Scores = Dict[str, Union[int, float]]
ScoresList = List[Scores]
AVERAGE = "AVERAGE"
OVERALL = "OVERALL"


class Result(BaseModel):
    """The base class for bdd100k evaluation results.

    Each data field corresponds to a evluation metric. The value for each
    metric is a list of dicts, each dict maps the category names to scores.
    There used to be two or three dicts in the list. The first one contains
    keys of basic categories, and the last one contains conclusion categories
    like 'OVERALL' and 'AVERAGE'. The middle one (optional), contains super
    classes for the two-level class hierarchy case.

    Functions:
        {} -> dict[str, dict[str, int | float]]:
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

    _row_breaks: List[int] = PrivateAttr([])

    def __init__(self, **data: Union[int, float, ScoresList]) -> None:
        """Check the input structure and initiliaze the model.

        All keys in the Scoreslist need to be the same set for the different
        evaluation metrics.
        """
        data_check: Dict[str, ScoresList] = {
            metric: cont
            for metric, cont in data.items()
            if isinstance(cont, list)
        }
        ref_scores_list = data_check[list(data_check.keys())[0]]
        for scores_list in data_check.values():
            assert len(scores_list) == len(ref_scores_list)
            for scores, ref_scores in zip(scores_list, ref_scores_list):
                assert scores.keys() == ref_scores.keys()
        super().__init__(**data)
        cur_index = 1
        self._row_breaks = [1]
        for scores in ref_scores_list[:-1]:
            cur_index += 1 + len(scores)
            self._row_breaks.append(cur_index)

    def __eq__(self, other: "Result") -> bool:  # type: ignore
        """Check whether two instances are equal."""
        if self._row_breaks != other._row_breaks:
            return False
        other_dict = dict(other)
        for metric, scores_list in self:
            other_scores_list = other_dict[metric]
            if not isinstance(scores_list, list):
                if scores_list != other_scores_list:
                    return False
                continue
            if len(scores_list) != len(other_scores_list):
                return False
            for scores, other_scores in zip(scores_list, other_scores_list):
                if set(scores.keys()) != set(other_scores.keys()):
                    return False
                for category, score in scores.items():
                    if not np.isclose(score, other_scores[category]):
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
        frame_dict: Dict[str, Scores] = defaultdict(dict)
        for metric, scores_list in self.dict(
            include=include, exclude=exclude  # type: ignore
        ).items():
            if not isinstance(scores_list, list):
                continue
            for scores in scores_list:
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
        summary = data_frame.to_string(float_format=lambda num: f"{num:.1f}")
        summary = summary.replace("NaN", " - ")
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

    def summary(
        self,
        include: Optional[AbstractSet[str]] = None,
        exclude: Optional[AbstractSet[str]] = None,
    ) -> Scores:
        """Convert the data into a flattened dict as the summary.

        This function is different to the `.dict()` function.
        As a comparison, `.dict()` will export all data fields as a nested
        dict, While `.summary()` only exports most important information,
        like the overall scores, as a flattened compact dict.

        Args:
            include (set[str]): Optional, the metrics to convert
            exclude (set[str]): Optional, the metrics not to convert
        Returns:
            dict[str, int | float]: returned summary of the result
        """
        summary_dict: Dict[str, Union[int, float]] = {}
        for metric, scores_list in self.dict(
            include=include, exclude=exclude  # type: ignore
        ).items():
            if not isinstance(scores_list, list):
                summary_dict[metric] = scores_list
            else:
                summary_dict[metric] = scores_list[-1].get(
                    OVERALL, scores_list[-1].get(AVERAGE)
                )
        return summary_dict
