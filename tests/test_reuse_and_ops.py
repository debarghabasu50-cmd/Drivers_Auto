"""
Artifact reuse across runs, cache modes/self-healing, reports, live-storage semantics (fake Azure client),
AML submission (fake MLClient) and Airflow-side error mapping.
"""
import importlib.util
import io
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

import run_pipeline
from run_local import run_all
from pipelines.init_run import _synthetic_input, load_repo_config
from shared import aml
from shared.path_builder.paths import (
    get_published_result_path,
    get_residual_npz_path,
    get_stage1_refs_path,
    get_report_path,
    get_master_config_path,
)
from shared.path_builder.storage import BlobStorageClient
from shared.schema.contracts import new_run_id
from tests.test_e2e import manifest, status, storage, use_config  # noqa: F401  (fixtures/helpers)


# ------------------------------ artifact re-usability ------------------------------
def test_run_until_feature_selection_then_resume_without_rerunning_it(storage, monkeypatch):
    # Day 1: run up to confounder selection only; the run still closes cleanly.
    use_config(monkeypatch, lambda c: c["steps"].update(dml="off", consolidation="off"))
    day1 = run_all(storage)
    m1 = manifest(storage, day1)
    assert m1["status"] == "SUCCEEDED" and m1["lineage"]["published"] is None
    assert status(storage, day1, "dml_fold_orders_f1")["status"] == "SKIPPED"

    # Day 2: feature + confounder selection are "off"; no run id configured -> latest compatible run is found.
    use_config(monkeypatch, lambda c: c["steps"].update(feature_selection="off", confounder_selection="off"))
    day2 = run_all(storage)
    m2 = manifest(storage, day2)
    assert m2["status"] == "SUCCEEDED" and m2["completeness"]["present_rows"] == 60
    assert status(storage, day2, "feature_selection_orders")["status"] == "SKIPPED"
    assert status(storage, day2, "feature_selection_orders")["counters"]["reused_from"] == day1
    assert m2["lineage"]["reused_from"] == {"feature_selection_orders": day1, "confounder_selection_orders": day1}
    assert not storage.list_blobs(f"runs/EL/2026Q3/{day2}/feature_selection/")  # it really did not run again
    assert not storage.list_blobs(f"runs/EL/2026Q3/{day2}/confounder_selection/")


def test_explicit_source_run_is_used(storage, monkeypatch):
    first = run_all(storage)
    second = run_all(storage)  # a newer compatible run exists, but we pin the older one
    use_config(monkeypatch, lambda c: (c["steps"].update(feature_selection="off"),
                                       c["source_artifacts"].update(feature_selection_run_id=first)))
    third = run_all(storage)
    assert status(storage, third, "feature_selection_orders")["counters"]["reused_from"] == first
    assert second != first


def test_incompatible_artifacts_are_rejected_not_silently_reused(storage, monkeypatch):
    first = run_all(storage)

    def edit(c):
        c["steps"]["feature_selection"] = "off"
        c["kpis"]["exclude"] = ["price_index"]  # a config change that alters feature selection's output
    use_config(monkeypatch, edit)
    auto = run_all(storage)
    s = status(storage, auto, "feature_selection_orders")
    assert s["status"] == "FAILED" and s["error"]["category"] == "CONFIG" and "no compatible" in s["error"]["message"]

    def pinned(c):
        edit(c)
        c["source_artifacts"]["feature_selection_run_id"] = first
    use_config(monkeypatch, pinned)
    explicit = run_all(storage)
    assert "not compatible" in status(storage, explicit, "feature_selection_orders")["error"]["message"]
    assert manifest(storage, explicit)["status"] == "FAILED"


def test_consolidation_switch_off_still_closes_run_without_publishing(storage, monkeypatch):
    use_config(monkeypatch, lambda c: c["steps"].update(consolidation="off"))
    run_id = run_all(storage)
    m = manifest(storage, run_id)
    assert m["status"] == "SUCCEEDED" and m["lineage"]["published"] is None
    assert not storage.exists(get_published_result_path(run_id))


# ------------------------------------ cache modes ----------------------------------
def test_cache_mode_off_ignores_and_read_only_never_writes(storage, monkeypatch, tmp_path):
    run_all(storage)
    use_config(monkeypatch, lambda c: c["cache"].update(mode="off"))
    off = run_all(storage)
    assert status(storage, off, "dml_fold_orders_f1")["counters"]["cache_hits"] == 0

    use_config(monkeypatch, lambda c: c["cache"].update(mode="read_only"))
    fresh = BlobStorageClient(base_dir=str(tmp_path / "fresh"))
    ro = run_all(fresh)
    assert status(fresh, ro, "dml_fold_orders_f1")["counters"]["units_fitted"] > 0
    assert not fresh.list_blobs("cache/")


def test_corrupt_cache_entry_is_healed_not_fatal(storage):
    r1 = run_all(storage)
    refs = storage.read_parquet(get_stage1_refs_path(r1, "orders", "f1"))
    victim = refs["unit_key"].iloc[0]
    storage.write_bytes(get_residual_npz_path("orders", "2021-01-01_2024-12-31", victim), b"not an npz", overwrite=True)
    r2 = run_all(storage)
    c = status(storage, r2, "dml_fold_orders_f1")["counters"]
    assert manifest(storage, r2)["status"] == "SUCCEEDED" and c["units_fitted"] == 1
    r3 = run_all(storage)
    assert status(storage, r3, "dml_fold_orders_f1")["counters"]["units_fitted"] == 0  # repaired in place


# ------------------------------------- reports -------------------------------------
def test_excel_reports_are_real_workbooks(storage):
    run_id = run_all(storage)
    dml_rep = get_report_path("dml", run_id)
    fs_rep = get_report_path("feature_selection", run_id)
    for path in (dml_rep, fs_rep):
        sheets = pd.read_excel(io.BytesIO(storage.read_bytes(path)), sheet_name=None)
        assert sheets
    results = pd.read_excel(io.BytesIO(storage.read_bytes(dml_rep)), sheet_name="results")
    assert len(results) == 60


# ------------------------- live Azure storage semantics (fake) -------------------------
class FakeContainer:
    """Minimal stand-in for azure.storage.blob.ContainerClient."""
    def __init__(self):
        self.blobs = {}

    def get_blob_client(self, name):
        return SimpleNamespace(exists=lambda: name in self.blobs)

    def download_blob(self, name):
        return SimpleNamespace(readall=lambda: self.blobs[name])

    def upload_blob(self, name, data, overwrite):
        from azure.core.exceptions import ResourceExistsError
        if name in self.blobs and not overwrite:
            raise ResourceExistsError("exists")
        self.blobs[name] = data

    def list_blobs(self, name_starts_with):
        return [SimpleNamespace(name=n) for n in self.blobs if n.startswith(name_starts_with)]


def test_live_backend_matches_mock_semantics():
    live = BlobStorageClient(connection_string="UseDevelopmentStorage=true")
    live._live = FakeContainer()
    assert not live.is_mock
    assert live.write_json("a/b.json", {"x": 1}) and live.read_json("a/b.json") == {"x": 1}
    assert live.write_bytes("a/c.bin", b"1", overwrite=False) is True
    assert live.write_bytes("a/c.bin", b"2", overwrite=False) is False and live.read_bytes("a/c.bin") == b"1"
    assert live.exists("a/b.json") and not live.exists("nope")
    assert live.list_blobs("a/") == ["a/b.json", "a/c.bin"]
    live.write_parquet("p.parquet", pd.DataFrame({"v": [1, 2]}))
    assert live.read_parquet("p.parquet")["v"].tolist() == [1, 2]


def test_full_pipeline_runs_on_the_live_backend_code_path():
    live = BlobStorageClient(connection_string="UseDevelopmentStorage=true")
    live._live = FakeContainer()
    live.write_parquet(load_repo_config()["data"]["input_uri"], _synthetic_input(load_repo_config()))  # live needs real input
    run_id = run_all(live)
    assert manifest(live, run_id)["status"] == "SUCCEEDED"
    # without input the live backend refuses to invent data
    empty = BlobStorageClient(connection_string="UseDevelopmentStorage=true")
    empty._live = FakeContainer()
    assert status(empty, run_all(empty), "init_run")["error"]["category"] == "UPSTREAM_MISSING"


# ----------------------------------- AML submission -----------------------------------
class FakeMLClient:
    def __init__(self, states):
        self.states, self.cancelled, self.jobs = list(states), [], self
        self.submitted = None

    def create_or_update(self, job):
        self.submitted = job
        return SimpleNamespace(name="job-1", studio_url="https://ml.example/job-1")

    def get(self, name):
        return SimpleNamespace(status=self.states.pop(0) if len(self.states) > 1 else self.states[0])

    def begin_cancel(self, name):
        self.cancelled.append(name)


ENV = {"azure_ml": {"environment_name": "env:1", "compute_cluster": "cpu", "subscription_id": "s",
                    "resource_group": "r", "workspace_name": "w"}, "storage": {"container": "c1"}}


def test_aml_job_is_built_from_argv_and_waits_for_completion():
    client = FakeMLClient(["Running", "Running", "Completed"])
    argv = ["--step", "dml", "--run-id", "dra_1", "--target", "orders", "--fold-id", "f1"]
    code = aml.submit_and_wait(argv, {"AZURE_STORAGE_CONNECTION_STRING": "cs"}, {"task_id": "dml_fold_orders_f1"},
                               ml_client=client, env_cfg=ENV, sleep=lambda s: None)
    job = client.submitted
    assert code == 0
    assert job.command == "python run_pipeline.py --step dml --run-id dra_1 --target orders --fold-id f1"
    assert job.environment_variables["AZURE_STORAGE_CONNECTION_STRING"] == "cs"
    assert job.environment_variables["AZURE_STORAGE_CONTAINER"] == "c1" and job.tags["task_id"] == "dml_fold_orders_f1"


def test_aml_failed_job_returns_nonzero_and_timeout_cancels():
    assert aml.submit_and_wait([], {}, {}, ml_client=FakeMLClient(["Failed"]), env_cfg=ENV, sleep=lambda s: None) == 1
    stuck = FakeMLClient(["Running"])
    assert aml.submit_and_wait([], {}, {}, ml_client=stuck, env_cfg=ENV, timeout_s=-1, sleep=lambda s: None) == 1
    assert stuck.cancelled == ["job-1"]


def test_env_yaml_files_have_everything_the_submission_needs():
    for name in ("dev_env", "prod"):
        cfg = aml.load_env_config(name)
        assert {"subscription_id", "resource_group", "workspace_name", "compute_cluster", "environment_name"} <= set(cfg["azure_ml"])
        assert cfg["storage"]["container"]


# ----------------------------- Airflow wrapper behaviour -----------------------------
@pytest.fixture
def dag(monkeypatch, tmp_path):
    monkeypatch.setenv("MOCK_BLOB_DIR", str(tmp_path / "blob"))
    spec = importlib.util.spec_from_file_location("dag_under_test", Path(__file__).parents[1] / "airflow" / "dags" / "driver_analysis_dag.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_dag_does_not_retry_config_errors_but_retries_missing_status(dag, monkeypatch):
    with pytest.raises(dag.AirflowFailException):  # no frozen config -> UPSTREAM_MISSING, non-retryable
        dag.run_step("dml", "dra_x", target="orders", fold_id="f1")
    monkeypatch.setattr(dag, "submit_and_wait", lambda argv, tags: 1)  # job died, wrote nothing
    monkeypatch.setenv("AML_MODE", "aml")
    with pytest.raises(RuntimeError):  # no final status -> transient, Airflow retries
        dag.run_step("consolidate", "dra_never")


def test_dag_runs_the_full_flow_and_skips_switched_off_steps_without_a_cluster(dag, monkeypatch):
    run_id = new_run_id()
    ti = SimpleNamespace(try_number=1, pushed={}, xcom_push=lambda key, value: ti.pushed.__setitem__(key, value),
                         xcom_pull=lambda task_ids, key: ti.pushed[key])
    monkeypatch.setenv("AML_MODE", "aml")
    monkeypatch.setattr(aml, "submit_and_wait", lambda *a, **k: (_ for _ in ()).throw(AssertionError("job submitted")))
    # init_run itself must be submitted as a job in aml mode -> run it in-process to prepare the run
    monkeypatch.setenv("AML_MODE", "local")
    items = dag.task_init_run(ti=ti, run_id="manual__1", logical_date=None)
    assert items == [{"work_item": "orders|f1"}, {"work_item": "orders|f2"}] and ti.pushed["target_items"] == [{"target": "orders"}]
    rid = ti.pushed["run_id"]
    # switch dml off in the frozen config, then ask for it in aml mode: must be handled in-process
    storage = dag.get_storage_client()
    cfg_path = get_master_config_path(rid)
    cfg = storage.read_json(cfg_path)
    cfg["steps"]["dml"] = "off"
    storage.write_json(cfg_path, cfg)
    monkeypatch.setenv("AML_MODE", "aml")
    dag.run_step("dml", rid, target="orders", fold_id="f1")
    assert status(storage, rid, "dml_fold_orders_f1")["status"] == "SKIPPED"


def test_run_id_is_idempotent_for_a_given_dag_run():
    assert new_run_id(seed="manual__2026") [-6:] == new_run_id(seed="manual__2026")[-6:]
    assert new_run_id()[-6:] != new_run_id()[-6:]


def test_azure_sdk_errors_are_classified():
    def err(name, status_code=None):
        cls = type(name, (Exception,), {"__module__": "azure.core.exceptions"})
        e = cls("boom")
        e.status_code = status_code
        return e
    assert run_pipeline._classify(err("ResourceNotFoundError")) == "UPSTREAM_MISSING"
    assert run_pipeline._classify(err("HttpResponseError", 403)) == "CONFIG"
    assert run_pipeline._classify(err("ServiceRequestError")) == "INFRA_TRANSIENT"
