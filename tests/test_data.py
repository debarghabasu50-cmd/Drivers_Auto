"""
Tests for canonical hashing and contracts.
"""
import pytest
from shared.schema.contracts import fp, data_fp


def test_fp_canonical_json():
    # Key ordering should not affect hash
    d1 = {"b": 2, "a": 1, "c": [3, 4]}
    d2 = {"a": 1, "c": [3, 4], "b": 2}
    h1 = fp(d1)
    h2 = fp(d2)
    assert h1 == h2
    assert len(h1) == 16


def test_fp_different_contents():
    h1 = fp({"key": "val1"})
    h2 = fp({"key": "val2"})
    assert h1 != h2


def test_data_fp_reproducibility():
    data = [{"col1": 10, "col2": "a"}, {"col1": 20, "col2": "b"}]
    h1 = data_fp(data)
    h2 = data_fp(data)
    assert h1 == h2
    assert len(h1) == 16
