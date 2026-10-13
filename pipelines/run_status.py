"""
Live run viewer - works while a run is in flight (reads blob status files only).

    python -m pipelines.run_status --run-id <run_id> [--div EL] [--quarter 2026Q3] [--watch 10]

Shows every planned task (PENDING until its status file appears), duration, key counters and the
failure category, plus the run-level state from the manifest.
"""
import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared.path_builder.paths import get_manifest_path, get_task_status_path, set_active_scope
from shared.path_builder.storage import get_storage_client

COUNTER_KEYS = ("combinations", "units_total", "cache_hits", "units_fitted", "units_failed", "rows_failed",
                "present_rows", "expected_rows")


def _duration(status: dict) -> str:
    if not status.get("started_at"):
        return "-"
    start = datetime.fromisoformat(status["started_at"])
    end = datetime.fromisoformat(status["ended_at"]) if status.get("ended_at") else datetime.now(timezone.utc)
    return f"{(end - start).total_seconds():.1f}s"


def render(storage, run_id: str, div: Optional[str] = None, quarter: Optional[str] = None) -> str:
    manifest_path = get_manifest_path(run_id, div, quarter)
    if not storage.exists(manifest_path):
        # Discover manifest under any div/quarter
        matching = [b for b in storage.list_blobs("runs/") if b.replace("\\", "/").endswith(f"/{run_id}/run_manifest.json")]
        if matching:
            manifest_path = matching[0]
        else:
            return f"run {run_id}: no manifest yet (init_run has not completed)"
    
    manifest = storage.read_json(manifest_path)
    d = manifest.get("plan", {}).get("div", div)
    q = manifest.get("plan", {}).get("quarter", quarter)
    set_active_scope(d, q)

    lines = [f"run {run_id} (div={d}, quarter={q})  status={manifest['status']}  image={manifest.get('image')}  "
             f"{manifest['snapshot'].get('snapshot_id')}  {manifest['config'].get('config_version')}",
             f"{'TASK':<34}{'STATUS':<11}{'TRY':<5}{'TIME':<9}DETAIL"]
    for task in manifest["plan"]["tasks"]:
        path = get_task_status_path(run_id, task["task_id"], d, q)
        st = storage.read_json(path) if storage.exists(path) else {"status": "PENDING"}
        detail = ", ".join(f"{k}={v}" for k, v in st.get("counters", {}).items() if k in COUNTER_KEYS)
        if st.get("error"):
            detail = f"{st['error']['category']}: {st['error']['message']}"[:90]
        lines.append(f"{task['task_id']:<34}{st['status']:<11}{st.get('attempt', '-'):<5}{_duration(st):<9}{detail}")
    comp = manifest.get("completeness")
    if comp:
        lines.append(f"completeness: {comp['present_rows']}/{comp['expected_rows']} rows, "
                     f"{comp['completed_folds']}/{comp['expected_folds']} folds, failed tasks: {comp['failed_tasks']}")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Show pipeline run status")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--div", default=None)
    parser.add_argument("--quarter", default=None)
    parser.add_argument("--watch", type=int, default=0, help="refresh every N seconds until the run is closed")
    args = parser.parse_args(argv)

    storage = get_storage_client()
    while True:
        output = render(storage, args.run_id, args.div, args.quarter)
        if args.watch:
            sys.stdout.write("\033[2J\033[H")
        print(output)
        if not args.watch or "status=SUCCEEDED" in output or "status=FAILED" in output or "status=PARTIAL" in output:
            break
        time.sleep(args.watch)
    return 0


if __name__ == "__main__":
    sys.exit(main())
