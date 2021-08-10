"""Test cases for mot.py."""

import unittest
from typing import List

import numpy as np
from pydantic import BaseModel

from .result import (
    nested_dict_to_data_frame,
    result_to_flatten_dict,
    result_to_nested_dict,
)


class TestModel(BaseModel):
    """A subclass of BaseModel for build up test cases."""

    list_str: List[str]
    one_int: int
    one_float: float


class TestResultToDict(unittest.TestCase):
    """Test cases for the result_to_*_dict functions."""

    test_model = TestModel(list_str=["a", "b", "c"], one_int=1, one_float=0.0)

    def test_to_flatten_dict(self) -> None:
        """Test case for the result_to_flatten_dict function."""
        flat_dict = result_to_flatten_dict(self.test_model)
        self.assertDictEqual(
            flat_dict, dict(list_str="c", one_int=1, one_float=0.0)
        )

    def test_to_nested_dict(self) -> None:
        """Test case for the result_to_nested_dict function."""
        nest_dict = result_to_nested_dict(self.test_model, ["A", "B", "C"])
        self.assertDictEqual(
            nest_dict, dict(list_str=dict(A="a", B="b", C="c"))
        )


class TestNestedDictToDataFrame(unittest.TestCase):
    """Test cases for the nested_dict_to_dat_frame functions."""

    def test_general_case(self) -> None:
        """Test the general case."""
        nest_dict = dict(a=dict(A=1, B=2, C=3), b=dict(A=2, B=1, C=3))
        data_frame = nested_dict_to_data_frame(nest_dict)  # type: ignore
        self.assertListEqual(data_frame.columns.values.tolist(), ["a", "b"])
        self.assertListEqual(data_frame.index.values.tolist(), ["A", "B", "C"])
        self.assertTrue(
            (data_frame.to_numpy() == np.array([[1, 2], [2, 1], [3, 3]])).all()
        )

    def test_transposed_case(self) -> None:
        """Test the tranposed input case."""
        nest_dict = dict(A=dict(a=1, b=2), B=dict(a=2, b=1), C=dict(a=3, b=3))
        data_frame = nested_dict_to_data_frame(
            nest_dict, primary_key="category"  # type: ignore
        )
        self.assertListEqual(data_frame.columns.values.tolist(), ["a", "b"])
        self.assertListEqual(data_frame.index.values.tolist(), ["A", "B", "C"])
        self.assertTrue(
            (data_frame.to_numpy() == np.array([[1, 2], [2, 1], [3, 3]])).all()
        )

    def test_missing_case(self) -> None:
        """Test the case with missing entries & single default value."""
        nest_dict = dict(a=dict(A=1, B=2), b=dict(A=2, C=3))
        data_frame = nested_dict_to_data_frame(
            nest_dict, default_value=0  # type: ignore
        )
        self.assertListEqual(data_frame.columns.values.tolist(), ["a", "b"])
        self.assertListEqual(data_frame.index.values.tolist(), ["A", "B", "C"])
        self.assertTrue(
            (data_frame.to_numpy() == np.array([[1, 2], [2, 0], [0, 3]])).all()
        )

    def test_missing_case2(self) -> None:
        """Test the case with missing entries & multiple default values."""
        nest_dict = dict(a=dict(A=1, B=2), b=dict(A=2, C=3))
        data_frame = nested_dict_to_data_frame(
            nest_dict, default_values=dict(B=4, C=5)  # type: ignore
        )
        self.assertListEqual(data_frame.columns.values.tolist(), ["a", "b"])
        self.assertListEqual(data_frame.index.values.tolist(), ["A", "B", "C"])
        self.assertTrue(
            (data_frame.to_numpy() == np.array([[1, 2], [2, 4], [5, 3]])).all()
        )
