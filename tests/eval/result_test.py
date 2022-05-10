"""Test cases for mot.py."""

import unittest
from typing import Any, Dict, List

import numpy as np

from .result import Result


class TestModel(Result):
    """A subclass of BaseModel for build up test cases."""

    list_str: List[Dict[str, float]]
    one_int: int
    one_float: float

    def __init__(self, **data: Any) -> None:  # type: ignore
        """Add a blank row_breaks."""
        super().__init__(**data)
        self._row_breaks = [1]


class TestBaseResult(unittest.TestCase):
    """Test cases for the base result class."""

    test_model = TestModel(
        list_str=[{"a": 1.0, "b": 2.0}, {"c": 3.0}], one_int=1, one_float=0.0
    )

    def test_frame(self) -> None:
        """Test case for the function frame()."""
        data_frame = self.test_model.pd_frame()
        self.assertTrue(
            np.isclose(
                data_frame.to_numpy()[:, 0], np.array([1.0, 2.0, 3.0])
            ).all()
        )
        self.assertListEqual(data_frame.columns.values.tolist(), ["list_str"])
        self.assertListEqual(data_frame.index.values.tolist(), ["a", "b", "c"])

    def test_table(self) -> None:
        """Test case for the function table()."""
        table = self.test_model.table()
        self.assertEqual(len(table.split("\n")), 7)
