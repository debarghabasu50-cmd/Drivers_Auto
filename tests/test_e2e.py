"""
End-to-end flow tests on mock storage: init_run -> FS -> CS -> DML folds -> consolidate.
Covers traceability, reproducibility, caching, failure isolation, toggles and log contract.
"""
import copy
import json

import pytest

import run_pipeline
from components.dml import dml_analysis
from pipelines import run_status
from pipelines.init_run import load_repo_config
from run_local import run_all
from shared.combinations import enumerate_combinations, expected_rows
from shared.path_builder.paths import (
    get_dml_result_path, get_manifest_path, get_residual_metadata_path, get_task_status_path, make_task_id,
    plan_tasks, get_stage1_refs_path, get_feature_selection_summary,
)
from shared.path_builder.storage import BlobStorageClient
from shared.schema.contracts import ErrorCategory, PipelineError


@pytest.fixture
def storage(tmp_path):
    return BlobStorageClient(base_dir=str(tmp_path))


def manifest(storage, run_id):
    return storage.read_json(get_manifest_path(run_id))


def status(storage, run_id, task_id):
    return storage.read_json(get_task_status_path(run_id, task_id))


def use_config(monkeypatch, edit):
    cfg = load_repo_config()
    edit(cfg)
    monkeypatch.setattr(run_pipeline, "load_repo_config", lambda path=None: copy.deepcopy(cfg))


def test_happy_path_is_traceable(storage):
    run_id = run_all(storage)
    m = manifest(storage, run_id)
    assert m["status"] == "SUCCEEDED"
    assert m["completeness"]["present_rows"] == m["completeness"]["expected_rows"] == 60  # 2 folds x C(5,3)=10 combos x 3 drivers
    assert m["snapshot"]["snapshot_id"].startswith("snap_") and m["config"]["config_version"].startswith("cfg_")
    # every planned task has a status file under its canonical id
    for t in plan_tasks(storage.read_json(m["config"]["uri"])):
        assert status(storage, run_id, t["task_id"])["status"] == "SUCCEEDED", t["task_id"]
    assert storage.exists(m["lineage"]["published"])
    assert storage.exists(f"runs/_index/{run_id}.json")
    # fold output is linked to the cache units it used
    refs = storage.read_parquet(get_stage1_refs_path(run_id, "orders", "f1"))
    assert set(refs["status"]) == {"FITTED"}


def test_rerun_is_reproducible_and_fully_cached(storage):
    r1, r2 = run_all(storage), run_all(storage)
    m1, m2 = manifest(storage, r1), manifest(storage, r2)
    assert m1["snapshot"]["snapshot_id"] == m2["snapshot"]["snapshot_id"]
    assert m1["config"]["config_version"] == m2["config"]["config_version"]
    c = status(storage, r2, "dml_fold_orders_f1")["counters"]
    assert c["units_fitted"] == 0 and c["cache_hits"] == c["units_total"]
    a = storage.read_parquet(get_dml_result_path(r1, "orders", "f2"))
    b = storage.read_parquet(get_dml_result_path(r2, "orders", "f2"))
    assert a["elasticity"].tolist() == b["elasticity"].tolist()


def test_cache_version_bump_invalidates(storage, monkeypatch):
    run_all(storage)
    use_config(monkeypatch, lambda c: c["dml"].update(cache_version=2))
    r2 = run_all(storage)
    assert status(storage, r2, "dml_fold_orders_f1")["counters"]["cache_hits"] == 0


def test_metadata_is_the_commit_marker(storage):
    r1 = run_all(storage)
    refs = storage.read_parquet(get_stage1_refs_path(r1, "orders", "f1"))
    victim = refs["unit_key"].iloc[0]
    storage._path(get_residual_metadata_path("orders", "2021-01-01_2024-12-31", victim)).unlink()  # simulated crash
    r2 = run_all(storage)
    c = status(storage, r2, "dml_fold_orders_f1")["counters"]
    assert c["units_fitted"] == 1 and c["cache_hits"] == c["units_total"] - 1


def test_transient_fold_failure_is_isolated_and_retry_uses_cache(storage, monkeypatch):
    real, calls = dml_analysis.fit_stage1_unit, {"n": 0}

    def flaky(unit, train, test, date_col, learner):
        if train[date_col].max() > __import__("pandas").Timestamp("2025-01-01"):  # fold f2 only
            calls["n"] += 1
            if calls["n"] == 5:
                raise PipelineError(ErrorCategory.INFRA_TRANSIENT, "node preempted")
        return real(unit, train, test, date_col, learner)

    monkeypatch.setattr(dml_analysis, "fit_stage1_unit", flaky)
    run_id = run_all(storage, retries=0)
    m = manifest(storage, run_id)
    assert m["status"] == "PARTIAL"
    assert status(storage, run_id, "dml_fold_orders_f2")["error"]["category"] == "INFRA_TRANSIENT"
    assert status(storage, run_id, "dml_fold_orders_f1")["status"] == "SUCCEEDED"
    assert m["completeness"]["failed_tasks"] == ["dml_fold_orders_f2"]
    assert m["completeness"]["present_rows"] == 30 and storage.exists(m["lineage"]["published"])

    # Airflow-style retry of just that fold: finished units are cache hits
    code = run_pipeline.main(["--step", "dml", "--run-id", run_id, "--target", "orders", "--fold-id", "f2",
                              "--attempt", "2"], storage=storage)
    s = status(storage, run_id, "dml_fold_orders_f2")
    assert code == 0 and s["status"] == "SUCCEEDED" and s["attempt"] == 2
    assert 0 < s["counters"]["cache_hits"] < s["counters"]["units_total"]


def test_unit_level_failure_marks_only_affected_rows(storage, monkeypatch):
    real = dml_analysis.fit_stage1_unit

    def bad(unit, train, test, date_col, learner):
        if unit["role"] == "treatment" and unit["variable"] == "price_index":
            raise PipelineError(ErrorCategory.MODEL_CONVERGENCE, "did not converge")
        return real(unit, train, test, date_col, learner)

    monkeypatch.setattr(dml_analysis, "fit_stage1_unit", bad)
    run_id = run_all(storage)
    s = status(storage, run_id, "dml_fold_orders_f1")
    assert s["status"] == "PARTIAL" and s["counters"]["rows_failed"] == 6  # combos containing price_index: C(4,2)
    res = storage.read_parquet(get_dml_result_path(run_id, "orders", "f1"))
    assert set(res[res["status"] == "FAILED"]["treatment_kpi"]) == {"price_index"}
    assert (res["status"] == "SUCCEEDED").sum() == 24
    assert manifest(storage, run_id)["status"] == "PARTIAL"


def test_missing_frozen_config_is_not_retryable(storage):
    code = run_pipeline.main(["--step", "dml", "--run-id", "dra_nope", "--target", "orders", "--fold-id", "f1"],
                             storage=storage)
    assert code == 2
    s = status(storage, "dra_nope", "dml_fold_orders_f1")
    assert s["status"] == "FAILED" and s["error"]["category"] == ErrorCategory.UPSTREAM_MISSING


def test_toggle_off_reads_upstream_from_source_run(storage, monkeypatch):
    first = run_all(storage)

    def edit(c):
        c["steps"]["feature_selection"] = "off"
        c["source_artifacts"]["feature_selection_run_id"] = first
    use_config(monkeypatch, edit)
    second = run_all(storage)
    assert status(storage, second, "feature_selection_orders")["status"] == "SKIPPED"
    assert not storage.exists(get_feature_selection_summary(second, "orders"))
    assert manifest(storage, second)["status"] == "SUCCEEDED"


def test_toggle_off_without_source_fails_with_config_error(storage, monkeypatch):
    use_config(monkeypatch, lambda c: c["steps"].update(feature_selection="off"))
    run_id = run_all(storage)
    assert status(storage, run_id, "feature_selection_orders")["error"]["category"] == "CONFIG"
    assert manifest(storage, run_id)["status"] == "FAILED"  # no results; consolidate still closed the run


def test_logs_follow_the_contract(storage, capsys):
    run_id = run_all(storage)
    records = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.strip()]
    assert all(r["run_id"] == run_id and r["task_id"] for r in records)
    summaries = [r["task_id"] for r in records if r["event"] == "TASK_SUMMARY"]
    assert sorted(summaries) == sorted(t["task_id"] for t in plan_tasks(load_repo_config()))
    assert all(r["event"] != "TASK_FAILED" for r in records)


def test_failure_log_carries_traceback_and_category(storage, capsys):
    run_pipeline.main(["--step", "dml", "--run-id", "dra_x", "--target", "orders", "--fold-id", "f1"], storage=storage)
    errors = [json.loads(l) for l in capsys.readouterr().out.splitlines() if '"TASK_FAILED"' in l]
    assert errors[0]["error_category"] == "UPSTREAM_MISSING" and "Traceback" in errors[0]["exc"]


def test_run_status_shows_pending_tasks_midrun(storage):
    run_id = "dra_mid_000001"
    assert run_pipeline.main(["--step", "init_run", "--run-id", run_id], storage=storage) == 0
    out = run_status.render(storage, run_id)
    assert "status=RUNNING" in out and out.count("PENDING") == len(plan_tasks(load_repo_config())) - 1
    assert make_task_id("dml", "orders", "f1") in out


# ---------------------------- combination enumeration ----------------------------
def test_combination_enumeration_rules():
    drivers = ["a", "b", "c", "d", "e"]
    cfg = {"drivers_per_combo": 3, "max_combinations": 100, "must_include": []}
    assert len(enumerate_combinations(drivers, cfg)) == 10
    assert enumerate_combinations(drivers, cfg) == enumerate_combinations(drivers[::-1], cfg)  # order independent
    must = enumerate_combinations(drivers, {**cfg, "must_include": ["c"]})
    assert len(must) == 6 and all("c" in c for c in must)
    capped = enumerate_combinations(drivers, {**cfg, "max_combinations": 4})
    assert len(capped) == 4 and capped == enumerate_combinations(drivers[::-1], {**cfg, "max_combinations": 4})
    assert expected_rows(5, cfg) == 30 and expected_rows(5, {**cfg, "max_combinations": 4}) == 12
    with pytest.raises(PipelineError):
        enumerate_combinations(drivers, {**cfg, "drivers_per_combo": 6})
    with pytest.raises(PipelineError):
        enumerate_combinations(drivers, {**cfg, "must_include": ["zzz"]})


def test_heavy_fold_enumerates_all_combinations_and_shares_units(storage, monkeypatch):
    def edit(c):
        c["kpis"]["extra"] = [f"k{i}" for i in range(8)]
        c["combinations"].update(drivers_per_combo=4, max_combinations=40)
    use_config(monkeypatch, edit)
    run_id = run_all(storage)
    m = manifest(storage, run_id)
    assert m["status"] == "SUCCEEDED" and m["completeness"]["expected_rows"] == 2 * 40 * 4
    res = storage.read_parquet(get_dml_result_path(run_id, "orders", "f1"))
    assert res["combination_id"].nunique() == 40 and len(res) == 160
    pub = storage.read_parquet(m["lineage"]["published"])
    assert pub["n_folds"].eq(2).all()  # each (combination, driver) aggregated over both folds


def test_parallel_fit_matches_sequential(storage, tmp_path, monkeypatch):
    seq = run_all(storage)
    use_config(monkeypatch, lambda c: c["execution"].update(n_jobs=2))
    par_storage = BlobStorageClient(base_dir=str(tmp_path / "par"))  # empty cache -> misses go through the pool
    par = run_all(par_storage)
    a = storage.read_parquet(get_dml_result_path(seq, "orders", "f1")).sort_values(["combination_id", "treatment_kpi"])
    b = par_storage.read_parquet(get_dml_result_path(par, "orders", "f1")).sort_values(["combination_id", "treatment_kpi"])
    assert status(par_storage, par, "dml_fold_orders_f1")["counters"]["units_fitted"] > 0
    assert a["elasticity"].tolist() == b["elasticity"].tolist()
