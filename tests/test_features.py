"""
Tests for canonical path builder and feature selection contracts.
"""
import pytest

from shared.path_builder.paths import (
    get_snapshot_path,
    get_manifest_path,
    get_master_config_path,
    get_feature_selection_parquet,
    get_lagged_features_path,
    get_residual_cache_prefix,
    get_published_result_path,
    get_published_latest_path,
    get_task_log_path,
)


def test_paths_generation():
    assert get_snapshot_path("snap_123") == "snapshots/snap_123/input.parquet"
    assert get_manifest_path("dra_run_1", div="EL", quarter="2026Q3") == "runs/EL/2026Q3/dra_run_1/run_manifest.json"
    assert get_master_config_path("dra_run_1", div="EL", quarter="2026Q3") == "runs/EL/2026Q3/dra_run_1/master_config.json"
    assert get_task_log_path("dra_run_1", "init_run", div="EL", quarter="2026Q3") == "runs/EL/2026Q3/dra_run_1/logs/init_run.log"
    assert get_feature_selection_parquet("dra_run_1", "orders", div="EL", quarter="2026Q3") == "runs/EL/2026Q3/dra_run_1/feature_selection/orders/selected_features.parquet"
    assert get_lagged_features_path("dra_run_1", "orders", div="EL", quarter="2026Q3") == "runs/EL/2026Q3/dra_run_1/feature_selection/orders/data_with_lagged_features.parquet"
    assert get_residual_cache_prefix("orders", "2021_2024", div="EL") == "cache/dml_stage1/EL/orders/2021_2024/"
    assert get_published_result_path("dra_run_1", div="EL", quarter="2026Q3") == "published/EL/2026Q3/runs/dra_run_1/consolidated_result.parquet"
    assert get_published_latest_path(div="EL", quarter="2026Q3") == "published/EL/2026Q3/latest/consolidated_result.parquet"


def test_task_logs_stored_as_python_log_files_not_json(tmp_path):
    import json
    from shared.path_builder.storage import BlobStorageClient
    import run_pipeline

    storage = BlobStorageClient(base_dir=str(tmp_path))
    run_id = "dra_test_log_001"
    run_pipeline.main(["--step", "init_run", "--run-id", run_id], storage=storage)

    log_path = get_task_log_path(run_id, "init_run", div="EL", quarter="2026Q3")
    assert storage.exists(log_path), f"Log file missing: {log_path}"

    log_text = storage.read_text(log_path)
    assert "[INFO]" in log_text
    assert "[init_run]" in log_text
    assert "event=" in log_text

    # Verify it is a standard Python text log and NOT a JSON document
    with pytest.raises(json.JSONDecodeError):
        json.loads(log_text)


def test_consumer_serving_layer_and_multiple_reruns(tmp_path):
    from run_local import run_all
    from shared.path_builder.storage import BlobStorageClient
    from shared.path_builder.paths import (
        get_published_run_info_path,
        get_published_latest_run_info_path,
    )

    storage = BlobStorageClient(base_dir=str(tmp_path))

    # Run 1
    run_1 = run_all(storage)
    pub_run1 = get_published_result_path(run_1, div="EL", quarter="2026Q3")
    pub_info1 = get_published_run_info_path(run_1, div="EL", quarter="2026Q3")
    pub_latest = get_published_latest_path(div="EL", quarter="2026Q3")
    pub_latest_info = get_published_latest_run_info_path(div="EL", quarter="2026Q3")

    assert storage.exists(pub_run1)
    assert storage.exists(pub_info1)
    assert storage.exists(pub_latest)
    assert storage.exists(pub_latest_info)
    assert storage.read_json(pub_latest_info)["run_id"] == run_1

    # Run 2 (Rerun for the same div & quarter)
    run_2 = run_all(storage)
    assert run_2 != run_1
    pub_run2 = get_published_result_path(run_2, div="EL", quarter="2026Q3")
    pub_info2 = get_published_run_info_path(run_2, div="EL", quarter="2026Q3")

    # Run 1 artifacts remain untouched (immutable history)
    assert storage.exists(pub_run1)
    assert storage.read_json(pub_info1)["run_id"] == run_1

    # Run 2 artifacts exist under its own run folder
    assert storage.exists(pub_run2)
    assert storage.read_json(pub_info2)["run_id"] == run_2

    # Serving layer 'latest' atomically points to run 2
    assert storage.exists(pub_latest)
    latest_meta = storage.read_json(pub_latest_info)
    assert latest_meta["run_id"] == run_2
    assert latest_meta["div"] == "EL"
    assert latest_meta["quarter"] == "2026Q3"


def test_feature_selection_emits_lagged_features_and_common_confounders(tmp_path):
    from shared.path_builder.storage import BlobStorageClient
    from shared.path_builder.paths import get_confounder_selection_path
    import run_pipeline

    storage = BlobStorageClient(base_dir=str(tmp_path))
    run_id = "dra_test_lag_001"

    # 1. Run init_run
    assert run_pipeline.main(["--step", "init_run", "--run-id", run_id], storage=storage) == 0

    # 2. Run feature_selection
    assert run_pipeline.main(["--step", "feature_selection", "--run-id", run_id, "--target", "orders"], storage=storage) == 0
    lagged_path = get_lagged_features_path(run_id, "orders", div="EL", quarter="2026Q3")
    assert storage.exists(lagged_path), "data_with_lagged_features.parquet was not emitted"

    lagged_df = storage.read_parquet(lagged_path)
    # Check that lagged columns are present in the dataset
    lag_cols = [c for c in lagged_df.columns if "_lag" in c]
    assert len(lag_cols) > 0, "No lagged feature columns found in dataset"

    # 3. Run confounder_selection
    assert run_pipeline.main(["--step", "confounder_selection", "--run-id", run_id, "--target", "orders"], storage=storage) == 0
    cs_path = get_confounder_selection_path(run_id, "orders", div="EL", quarter="2026Q3")
    assert storage.exists(cs_path)

    cs_data = storage.read_json(cs_path)
    assert "common_confounders" in cs_data
    assert "confounder_set" in cs_data
    # Verify that each treatment shares the common confounder set (minus itself)
    for t in cs_data["treatments"]:
        t_confounders = [c["kpi"] for c in t["confounders"]]
        assert t["treatment_kpi"] not in t_confounders


def test_simplified_cache_unit_key():
    from components.residual_caching.residual_caching import compute_unit_key

    # Call with basic params
    key_base = compute_unit_key(
        target="orders",
        role="outcome",
        variable="orders",
        controls=["price_index", "gdp_growth"],
        source_data_hash="abc_hash_123",
        train_window="2021_2024",
        test_window="2025_2025",
        cache_version=1
    )

    # Call with different inner_cv_splits, learner_spec, and seed
    key_with_modelling_params = compute_unit_key(
        target="orders",
        role="outcome",
        variable="orders",
        controls=["price_index", "gdp_growth"],
        source_data_hash="abc_hash_123",
        train_window="2021_2024",
        test_window="2025_2025",
        cache_version=1,
        inner_cv_splits=10,  # different CV
        learner_spec={"model": "LassoCV", "alpha": 0.5},  # different model
        seed=999  # different seed
    )

    # Simplified key must be invariant to modelling params and inner CV!
    assert key_base == key_with_modelling_params, "Cache key should not depend on modelling parameters or inner CV"

    # Key DOES change if cache_version or controls change
    key_bumped = compute_unit_key(
        target="orders",
        role="outcome",
        variable="orders",
        controls=["price_index", "gdp_growth"],
        source_data_hash="abc_hash_123",
        train_window="2021_2024",
        test_window="2025_2025",
        cache_version=2
    )
    assert key_base != key_bumped
