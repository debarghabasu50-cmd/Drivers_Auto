"""
Run the whole pipeline locally in DAG order (mock storage), mirroring the Airflow trigger rules:
steps run in order, a failed task does not stop the run, consolidate always runs last.

    python run_local.py [--run-id ID] [--resume] [--config path]   # prints the status table at the end (stderr)
    Logs (JSON lines) go to stdout; storage is .mock_blob/ (or $MOCK_BLOB_DIR). No Azure needed.
"""
import argparse
import sys

import run_pipeline
from pipelines import run_status
from shared.path_builder.paths import (
    get_manifest_path,
    get_master_config_path,
    get_task_status_path,
    plan_tasks,
    set_active_scope,
)
from shared.schema.contracts import ErrorCategory, PipelineError, new_run_id


def run_all(
    storage=None,
    run_id: str = None,
    retries: int = 0,
    config: str = None,
    resume: bool = False,
) -> str:
    """Returns the run_id. `retries` mimics Airflow retries for retryable (exit code 1) failures."""
    storage = storage or run_pipeline.get_storage_client()
    cfg = run_pipeline.load_repo_config(config)
    div = cfg.get("div") or cfg.get("pipeline", {}).get("div", "EL")
    quarter = cfg.get("quarter") or cfg.get("pipeline", {}).get("quarter", "2026Q3")
    set_active_scope(div, quarter)

    if run_id:
        manifest_path = get_manifest_path(run_id, div=div, quarter=quarter)
        if storage.exists(manifest_path):
            if not resume:
                raise PipelineError(
                    ErrorCategory.CONFIG,
                    f"run '{run_id}' already exists and is initialized. To resume remaining steps, invoke "
                    f"run_local.py with --run-id {run_id} --resume. To start a fresh run, omit --run-id or supply a new one.",
                )
        elif resume:
            raise PipelineError(
                ErrorCategory.CONFIG,
                f"cannot resume run '{run_id}': run does not exist or has not been initialized with init_run.",
            )
    elif resume:
        raise PipelineError(
            ErrorCategory.CONFIG,
            "cannot resume without specifying --run-id.",
        )

    run_id = run_id or new_run_id()
    manifest_path = get_manifest_path(run_id, div=div, quarter=quarter)

    def step(*argv) -> int:
        code = 1
        for attempt in range(1, retries + 2):
            code = run_pipeline.main(["--run-id", run_id, "--attempt", str(attempt), *argv], storage=storage)
            if code != 1:  # 0 ok, 2 = do not retry
                break
        return code

    if resume:
        # Sync latest repo configs into frozen config so downstream steps pick up updated dml_config.json
        frozen_cfg_path = get_master_config_path(run_id, div=div, quarter=quarter)
        storage.write_json(frozen_cfg_path, cfg)
        tasks = plan_tasks(cfg)
        manifest = storage.read_json(manifest_path)
        manifest["plan"] = {"tasks": tasks}
        storage.write_json(manifest_path, manifest)
    else:
        if step("--step", "init_run", *(["--config", config] if config else [])) != 0:
            return run_id  # nothing to plan from
        manifest = storage.read_json(manifest_path)
        tasks = manifest.get("plan", {}).get("tasks", plan_tasks(cfg))

    for stage in ("feature_selection", "confounder_selection", "dml"):
        for t in (t for t in tasks if t["step"] == stage):
            task_id = t["task_id"]
            if resume:
                st_path = get_task_status_path(run_id, task_id, div=div, quarter=quarter)
                if storage.exists(st_path):
                    st = storage.read_json(st_path)
                    if st.get("status") == "SUCCEEDED":
                        sys.stdout.write(f"\n[run_local] Task '{task_id}' already SUCCEEDED; skipping.\n")
                        sys.stdout.flush()
                        continue
            extra = ["--fold-id", t["fold_id"]] if t["fold_id"] else []
            step("--step", stage, "--target", t["target"], *extra)

    step("--step", "consolidate")  # trigger_rule=all_done
    return run_id


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Run the driver analysis causal pipeline locally")
    ap.add_argument("--run-id", help="Explicit run ID (resumed or new)")
    ap.add_argument("--resume", action="store_true", help="Resume an existing run, executing only remaining/incomplete steps")
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--config", help="alternative master_config.json (e.g. a bigger combination setup)")
    a = ap.parse_args()
    rid = run_all(run_id=a.run_id, retries=a.retries, config=a.config, resume=a.resume)
    print(file=sys.stderr)
    print(run_status.render(run_pipeline.get_storage_client(), rid), file=sys.stderr)
