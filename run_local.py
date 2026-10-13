"""
Run the whole pipeline locally in DAG order (mock storage), mirroring the Airflow trigger rules:
steps run in order, a failed task does not stop the run, consolidate always runs last.

    python run_local.py [--run-id ID] [--config path]   # prints the status table at the end (stderr)
    Logs (JSON lines) go to stdout; storage is .mock_blob/ (or $MOCK_BLOB_DIR). No Azure needed.
"""
import argparse
import sys

import run_pipeline
from pipelines import run_status
from shared.path_builder.paths import get_manifest_path
from shared.schema.contracts import new_run_id


def run_all(storage=None, run_id: str = None, retries: int = 0, config: str = None) -> str:
    """Returns the run_id. `retries` mimics Airflow retries for retryable (exit code 1) failures."""
    storage = storage or run_pipeline.get_storage_client()
    run_id = run_id or new_run_id()

    def step(*argv) -> int:
        code = 1
        for attempt in range(1, retries + 2):
            code = run_pipeline.main(["--run-id", run_id, "--attempt", str(attempt), *argv], storage=storage)
            if code != 1:  # 0 ok, 2 = do not retry
                break
        return code

    if step("--step", "init_run", *(["--config", config] if config else [])) != 0:
        return run_id  # nothing to plan from
    plan = storage.read_json(get_manifest_path(run_id))["plan"]
    for stage in ("feature_selection", "confounder_selection", "dml"):
        for t in (t for t in plan["tasks"] if t["step"] == stage):
            extra = ["--fold-id", t["fold_id"]] if t["fold_id"] else []
            step("--step", stage, "--target", t["target"], *extra)
    step("--step", "consolidate")  # trigger_rule=all_done
    return run_id


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id")
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--config", help="alternative master_config.json (e.g. a bigger combination setup)")
    a = ap.parse_args()
    rid = run_all(run_id=a.run_id, retries=a.retries, config=a.config)
    print(file=sys.stderr)
    print(run_status.render(run_pipeline.get_storage_client(), rid), file=sys.stderr)
