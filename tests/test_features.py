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
    from components.residual_caching.cache import compute_unit_key

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


def test_provenance_and_component_events_logged(tmp_path, capsys):
    import json
    import run_pipeline
    from shared.path_builder.storage import BlobStorageClient

    storage = BlobStorageClient(base_dir=str(tmp_path))
    run_id = "dra_test_prov_001"

    # Step 1: init_run
    assert run_pipeline.main(["--step", "init_run", "--run-id", run_id], storage=storage) == 0
    # Step 2: feature_selection
    assert run_pipeline.main(["--step", "feature_selection", "--run-id", run_id, "--target", "orders"], storage=storage) == 0

    records = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.strip().startswith("{")]
    
    # Check INPUT_LOADED carries provenance (source_run_id, path, target, fingerprint)
    input_loaded = [r for r in records if r.get("event") == "INPUT_LOADED"]
    assert len(input_loaded) > 0
    snap_loaded = [r for r in input_loaded if "snapshots" in str(r.get("path"))]
    assert len(snap_loaded) > 0
    assert snap_loaded[0]["source_run_id"] == run_id
    assert snap_loaded[0]["fingerprint"].startswith("snap_")

    # Check meaningful component events exist
    events = {r.get("event") for r in records}
    assert "COMPUTATION_STARTED" in events
    assert "COMPUTATION_COMPLETED" in events
    assert "OUTPUT_WRITTEN" in events


def test_preserve_logs_per_retry(tmp_path):
    import run_pipeline
    from shared.path_builder.paths import get_task_status_path
    from shared.path_builder.storage import BlobStorageClient

    storage = BlobStorageClient(base_dir=str(tmp_path))
    run_id = "dra_test_retry_log"

    # Run attempt 1
    run_pipeline.main(["--step", "init_run", "--run-id", run_id, "--attempt", "1"], storage=storage)
    log_canonical = get_task_log_path(run_id, "init_run", div="EL", quarter="2026Q3")
    log_attempt1 = get_task_log_path(run_id, "init_run", div="EL", quarter="2026Q3", attempt=1)

    assert storage.exists(log_canonical)
    assert storage.exists(log_attempt1)
    text1 = storage.read_text(log_attempt1)
    assert "[init_run]" in text1

    # Run attempt 2 (retry)
    run_pipeline.main(["--step", "init_run", "--run-id", run_id, "--attempt", "2"], storage=storage)
    log_attempt2 = get_task_log_path(run_id, "init_run", div="EL", quarter="2026Q3", attempt=2)

    assert storage.exists(log_attempt2)
    # Attempt 1 log file must still be preserved!
    assert storage.exists(log_attempt1)
    assert storage.read_text(log_attempt1) == text1

    # Status file remains as latest state
    status_path = get_task_status_path(run_id, "init_run", div="EL", quarter="2026Q3")
    st = storage.read_json(status_path)
    assert st["attempt"] == 2


def test_status_write_failure_is_unmistakable(tmp_path, monkeypatch, capsys):
    import json
    import run_pipeline
    from shared.path_builder.storage import BlobStorageClient

    storage = BlobStorageClient(base_dir=str(tmp_path))
    run_id = "dra_test_status_fail"

    orig_write_json = storage.write_json

    def flaky_write_json(path, data, **kwargs):
        if "status/" in path:
            raise OSError("disk quota exceeded or blob write timeout")
        return orig_write_json(path, data, **kwargs)

    monkeypatch.setattr(storage, "write_json", flaky_write_json)

    code = run_pipeline.main(["--step", "init_run", "--run-id", run_id], storage=storage)
    # Must return retryable exit code 1
    assert code == 1

    captured = capsys.readouterr()
    # Stderr must contain unmistakable critical message
    assert "CRITICAL: Failed to write" in captured.err
    # Stdout must contain structured STATUS_WRITE_FAILED event
    lines = [json.loads(l) for l in captured.out.splitlines() if l.strip().startswith("{")]
    status_fails = [l for l in lines if l.get("event") == "STATUS_WRITE_FAILED"]
    assert len(status_fails) > 0
    assert status_fails[0]["error_category"] == "INFRA_TRANSIENT"
    assert "status_path" in status_fails[0]


def test_consolidation_failure_sets_terminal_manifest_and_stopping_rule(tmp_path, monkeypatch):
    import run_pipeline
    from pipelines import run_status
    from run_local import run_all
    from shared.path_builder.storage import BlobStorageClient

    storage = BlobStorageClient(base_dir=str(tmp_path))
    run_id = run_all(storage)

    # Initially, completed run is terminal
    assert run_status.is_run_terminal(storage, run_id) is True

    # Now test a run where consolidation crashes unexpectedly during execution
    run_id2 = "dra_test_cons_fail"
    run_pipeline.main(["--step", "init_run", "--run-id", run_id2], storage=storage)

    # Midrun: manifest is RUNNING, not terminal
    assert run_status.is_run_terminal(storage, run_id2) is False

    # Simulate an unexpected error inside consolidation
    def bad_collect(*args, **kwargs):
        raise RuntimeError("simulated unexpected crash in consolidation")

    monkeypatch.setattr("pipelines.consolidate._collect_statuses", bad_collect)
    
    # Run consolidate (which triggers the crash)
    code = run_pipeline.main(["--step", "consolidate", "--run-id", run_id2], storage=storage)
    assert code != 0  # Failed

    # Manifest MUST have been updated with terminal status "FAILED"!
    manifest = storage.read_json(get_manifest_path(run_id2, "EL", "2026Q3"))
    assert manifest["status"] == "FAILED"
    assert manifest.get("end_time") is not None

    # run_status.is_run_terminal must now return True so --watch stops refreshing!
    assert run_status.is_run_terminal(storage, run_id2) is True


def test_lagged_data_fails_loudly_when_missing(tmp_path):
    import pytest
    import run_pipeline
    from pipelines.init_run import load_repo_config
    from shared.context import StepContext
    from shared.path_builder.storage import BlobStorageClient
    from shared.schema.contracts import ErrorCategory, PipelineError

    storage = BlobStorageClient(base_dir=str(tmp_path))
    config = load_repo_config()
    run_id = "dra_test_lag_loud"

    # Initialize run so manifest exists
    assert run_pipeline.main(["--step", "init_run", "--run-id", run_id], storage=storage) == 0

    ctx = StepContext(run_id=run_id, step="dml", config=config, storage=storage, attempt=1)

    # Calling without target raises CONFIG
    with pytest.raises(PipelineError) as exc_cfg:
        ctx.load_lagged_dataset()
    assert exc_cfg.value.category == ErrorCategory.CONFIG

    # Calling with target when lagged features parquet doesn't exist raises UPSTREAM_MISSING
    with pytest.raises(PipelineError) as exc_up:
        ctx.load_lagged_dataset(target="orders")
    assert exc_up.value.category == ErrorCategory.UPSTREAM_MISSING
    assert "required lagged dataset missing" in exc_up.value.message


def test_candidate_resolution_orders_by_ended_at_and_rejects_incomplete_artifacts(tmp_path):
    from pipelines.init_run import load_repo_config
    from shared.context import StepContext
    from shared.path_builder.paths import (
        get_feature_selection_summary,
        get_lagged_features_path,
        get_task_status_path,
    )
    from shared.path_builder.storage import BlobStorageClient
    import pandas as pd

    storage = BlobStorageClient(base_dir=str(tmp_path))
    config = load_repo_config()

    target = "orders"
    expected_fp = "expected_step_fp_123"

    # Context for a new run trying to resolve feature_selection
    current_run = "dra_20261014T1200_current"
    ctx = StepContext(run_id=current_run, step="dml", config=config, storage=storage, target=target)

    # Monkeypatch ctx.step_fp so expected matches
    ctx.step_fp = lambda step: expected_fp

    # Create run_001 (alphabetically earlier, but finished LATER at 14:00)
    # Create run_999 (alphabetically later, but finished EARLIER at 10:00)
    run_earlier_name = "dra_run_001"
    run_later_name = "dra_run_999"

    for r_id, ended_at in [(run_earlier_name, "2026-10-14T14:00:00+00:00"), (run_later_name, "2026-10-14T10:00:00+00:00")]:
        storage.write_json(
            get_feature_selection_summary(r_id, target, div="EL", quarter="2026Q3"),
            {"run_id": r_id, "target": target, "step_fp": expected_fp},
        )
        storage.write_json(
            get_task_status_path(r_id, f"feature_selection_{target}", div="EL", quarter="2026Q3"),
            {"status": "SUCCEEDED", "ended_at": ended_at},
        )
        storage.write_parquet(
            get_lagged_features_path(r_id, target, div="EL", quarter="2026Q3"),
            pd.DataFrame({"date": ["2026-01-01"], "orders": [10]}),
        )

    # Automatic candidate resolution must choose run_earlier_name because its ended_at (14:00) > run_later_name (10:00)
    resolved = ctx._latest_compatible_run("feature_selection", target, expected_fp)
    assert resolved == run_earlier_name

    # Incomplete artifact set: remove lagged features parquet from run_earlier_name
    lag_earlier = get_lagged_features_path(run_earlier_name, target, div="EL", quarter="2026Q3")
    storage._path(lag_earlier).unlink()

    # Now it must reject run_earlier_name and resolve run_later_name instead
    resolved_fallback = ctx._latest_compatible_run("feature_selection", target, expected_fp)
    assert resolved_fallback == run_later_name

    # If run_later_name status is not SUCCEEDED, it must reject both and return None
    storage.write_json(
        get_task_status_path(run_later_name, f"feature_selection_{target}", div="EL", quarter="2026Q3"),
        {"status": "FAILED", "ended_at": "2026-10-14T10:00:00+00:00"},
    )
    assert ctx._latest_compatible_run("feature_selection", target, expected_fp) is None


def test_reinitialization_guards(tmp_path):
    import pytest
    import run_pipeline
    from pipelines.init_run import load_repo_config
    from run_local import run_all
    from shared.context import StepContext
    from shared.path_builder.paths import get_manifest_path
    from shared.path_builder.storage import BlobStorageClient
    from shared.schema.contracts import ErrorCategory, PipelineError

    storage = BlobStorageClient(base_dir=str(tmp_path))
    run_id = "dra_test_guard_001"

    # Run full pipeline with explicit run_id
    run_all(storage, run_id=run_id)

    # Calling run_all with the same run_id must raise PipelineError(CONFIG)
    with pytest.raises(PipelineError) as exc_local:
        run_all(storage, run_id=run_id)
    assert exc_local.value.category == ErrorCategory.CONFIG
    assert "already exists and is initialized" in exc_local.value.message

    # Attempt 1 of init_run on existing run must raise CONFIG error
    config = load_repo_config()
    ctx1 = StepContext(run_id=run_id, step="init_run", config=config, storage=storage, attempt=1)
    with pytest.raises(PipelineError) as exc_init1:
        run_pipeline.STEPS["init_run"].fn(ctx1)
    assert exc_init1.value.category == ErrorCategory.CONFIG
    assert "already initialized" in exc_init1.value.message

    # Attempt > 1 of init_run on an already terminal run (SUCCEEDED) must raise CONFIG error
    ctx2 = StepContext(run_id=run_id, step="init_run", config=config, storage=storage, attempt=2)
    with pytest.raises(PipelineError) as exc_init2:
        run_pipeline.STEPS["init_run"].fn(ctx2)
    assert exc_init2.value.category == ErrorCategory.CONFIG
    assert "already completed" in exc_init2.value.message

    # But for a non-terminal run (e.g. RUNNING), attempt 2 retry of init_run is allowed
    incomplete_run = "dra_incomplete_002"
    run_pipeline.main(["--step", "init_run", "--run-id", incomplete_run, "--attempt", "1"], storage=storage)
    m = storage.read_json(get_manifest_path(incomplete_run, "EL", "2026Q3"))
    assert m["status"] == "RUNNING"
    code = run_pipeline.main(["--step", "init_run", "--run-id", incomplete_run, "--attempt", "2"], storage=storage)
    assert code == 0


def test_terminal_manifest_when_init_run_fails(tmp_path):
    import run_pipeline
    from pipelines import run_status
    from shared.path_builder.storage import BlobStorageClient

    storage = BlobStorageClient(base_dir=str(tmp_path))
    run_id = "dra_fail_init_001"

    # Run init_run with invalid config so it fails before writing a manifest
    code = run_pipeline.main(["--step", "init_run", "--run-id", run_id, "--config", "nonexistent_config.json"], storage=storage)
    assert code != 0

    # Ensure run_status detects terminal state so --watch does NOT loop indefinitely
    assert run_status.is_run_terminal(storage, run_id, "EL", "2026Q3") is True
    rendered = run_status.render(storage, run_id, "EL", "2026Q3")
    assert "status=FAILED" in rendered
    assert "init_run failed" in rendered


def test_terminal_manifest_when_consolidation_fails_early(tmp_path, monkeypatch):
    import run_pipeline
    from pipelines import run_status
    from shared.path_builder.paths import get_manifest_path
    from shared.path_builder.storage import BlobStorageClient

    storage = BlobStorageClient(base_dir=str(tmp_path))
    run_id = "dra_cons_early_crash"

    # Initialize run
    assert run_pipeline.main(["--step", "init_run", "--run-id", run_id], storage=storage) == 0

    # Simulate unexpected crash in _collect_statuses
    def mock_collect(*args, **kwargs):
        raise RuntimeError("simulated crash during status collection")

    monkeypatch.setattr("pipelines.consolidate._collect_statuses", mock_collect)

    code = run_pipeline.main(["--step", "consolidate", "--run-id", run_id], storage=storage)
    assert code != 0

    # Terminal status FAILED must be recorded on manifest
    m = storage.read_json(get_manifest_path(run_id, "EL", "2026Q3"))
    assert m["status"] == "FAILED"
    assert m.get("end_time") is not None
    assert run_status.is_run_terminal(storage, run_id, "EL", "2026Q3") is True


def test_terminal_manifest_on_partial_and_total_failure(tmp_path, monkeypatch):
    import run_pipeline
    from pipelines import run_status
    from run_local import run_all
    from shared.path_builder.paths import get_manifest_path
    from shared.path_builder.storage import BlobStorageClient

    storage = BlobStorageClient(base_dir=str(tmp_path))

    # Case 1: Partial failure (simulate one fold failure)
    real_fit = run_pipeline.STEPS["dml"].fn
    def flaky_fit(ctx):
        if ctx.fold_id == "f2":
            raise RuntimeError("flaky fold f2 failure")
        return real_fit(ctx)

    monkeypatch.setattr(run_pipeline.STEPS["dml"], "fn", flaky_fit)
    run_partial = run_all(storage)
    m_partial = storage.read_json(get_manifest_path(run_partial, "EL", "2026Q3"))
    assert m_partial["status"] == "PARTIAL"
    assert run_status.is_run_terminal(storage, run_partial, "EL", "2026Q3") is True

    # Case 2: Total failure (feature selection fails -> all subsequent steps fail)
    def fail_fs(ctx):
        raise RuntimeError("fs total failure")
    monkeypatch.setattr(run_pipeline.STEPS["feature_selection"], "fn", fail_fs)
    run_failed = run_all(storage)
    m_failed = storage.read_json(get_manifest_path(run_failed, "EL", "2026Q3"))
    assert m_failed["status"] == "FAILED"
    assert run_status.is_run_terminal(storage, run_failed, "EL", "2026Q3") is True


