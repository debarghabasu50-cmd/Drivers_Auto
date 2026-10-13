"""
Tests for Stage-1 residual caching (.npz and metadata.json).
"""
import pytest
import shutil
from pathlib import Path
from shared.path_builder.storage import BlobStorageClient
from components.residual_caching.residual_caching import (
    compute_unit_key,
    check_cache_hits,
    save_cached_unit,
    load_cached_unit
)


@pytest.fixture
def temp_storage(tmp_path):
    """Provides a BlobStorageClient with temporary local mock storage."""
    client = BlobStorageClient(base_dir=str(tmp_path))
    return client


def test_residual_caching_hit_and_miss(temp_storage):
    target = "orders"
    train_window = "2021-01-01_2024-12-31"
    
    key1 = compute_unit_key(
        target=target,
        role="outcome",
        variable="orders",
        controls=["kpi_1", "kpi_2"],
        source_data_hash="abc1234567890def",
        train_window=train_window,
        test_window="2025-01-01_2025-03-31",
        inner_cv_splits=5,
        learner_spec={"model": "RF"},
        seed=42,
        cache_version=1
    )
    key2 = compute_unit_key(
        target=target,
        role="treatment",
        variable="kpi_1",
        controls=["kpi_2"],
        source_data_hash="abc1234567890def",
        train_window=train_window,
        test_window="2025-01-01_2025-03-31",
        inner_cv_splits=5,
        learner_spec={"model": "RF"},
        seed=42,
        cache_version=1
    )

    # Initially both are misses
    hits, misses = check_cache_hits(temp_storage, target, train_window, [key1, key2])
    assert len(hits) == 0
    assert misses == {key1, key2}

    # Save key1
    mock_arrays = {"train_res": [0.1, -0.2], "test_res": [0.05]}
    mock_meta = {"fingerprint": key1, "cache_version": 1, "driver_col": "orders"}
    saved = save_cached_unit(temp_storage, target, train_window, key1, mock_arrays, mock_meta)
    assert saved is True

    # Check again: key1 should be a hit, key2 a miss
    hits, misses = check_cache_hits(temp_storage, target, train_window, [key1, key2])
    assert hits == {key1}
    assert misses == {key2}

    # Verify loading
    loaded_arrays, loaded_meta = load_cached_unit(temp_storage, target, train_window, key1)
    assert loaded_meta["fingerprint"] == key1
