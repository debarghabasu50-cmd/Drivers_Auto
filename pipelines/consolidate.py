"""
T5 consolidate - the LAST step of every run (always runs, even after failures or with steps switched off).

1. Union the fold results that exist and add cross-fold aggregates ->
   - published/{div}/{quarter}/runs/{run_id}/consolidated_result.parquet (immutable run record)
   - published/{div}/{quarter}/latest/consolidated_result.parquet (consumer serving pointer)
   - published/{div}/{quarter}/latest/run_info.json (serving provenance)
2. Check completeness (present vs expected) from the manifest plan and the task status files
3. Render the reviewer workbook (runs/{run_id} and latest)
4. Close the run: manifest (steps, completeness, lineage, final status, end_time) + run index

Final run status: SUCCEEDED (everything planned is complete) | PARTIAL (something failed/missing, results
exist) | FAILED (no usable output). The consolidate *task* itself succeeds in all three cases.
"""
from datetime import datetime, timezone
from typing import Any, Dict, List

import pandas as pd

from shared.combinations import expected_rows
from shared.context import StepContext, StepResult
from shared.logging import logger
from shared.path_builder.paths import (
    get_confounder_selection_path,
    get_dml_result_path,
    get_manifest_path,
    get_published_latest_path,
    get_published_latest_run_info_path,
    get_published_result_path,
    get_published_run_info_path,
    get_report_latest_path,
    get_report_path,
    get_run_index_path,
    get_task_status_path,
    set_active_scope,
)
from shared.reports import write_xlsx
from shared.schema.contracts import PipelineError

OK_STATUSES = {"SUCCEEDED", "PARTIAL", "SKIPPED"}


def _collect_statuses(ctx: StepContext, tasks: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Status file per planned task; a missing file means a hard crash before the wrapper could write it."""
    out = {}
    for t in tasks:
        path = get_task_status_path(ctx.run_id, t["task_id"], ctx.div, ctx.quarter)
        out[t["task_id"]] = ctx.storage.read_json(path) if ctx.storage.exists(path) else {
            "task_id": t["task_id"], "status": "MISSING", "error": {"category": "UNKNOWN", "message": "no status file"}}
    return out


def _expected_rows(ctx: StepContext, dml_tasks: List[Dict[str, Any]]) -> int:
    n = 0
    for t in dml_tasks:
        try:
            run = ctx.upstream_run_id("confounder_selection", t["target"])
            path = get_confounder_selection_path(run, t["target"], ctx.div, ctx.quarter)
            if ctx.storage.exists(path):
                n += expected_rows(len(ctx.storage.read_json(path)["treatments"]), ctx.config["combinations"])
        except PipelineError:
            pass
    return n


def _union_with_aggregates(frames: List[pd.DataFrame]) -> pd.DataFrame:
    df = pd.concat(frames, ignore_index=True)
    keys = ["target", "combination_id", "treatment_kpi"]
    ok = df[df["status"] == "SUCCEEDED"].groupby(keys)["elasticity"]
    agg = pd.DataFrame({"mean_elasticity": ok.mean(), "std_elasticity": ok.std(ddof=0), "n_folds": ok.count()}).reset_index()
    agg["cv_elasticity"] = agg["std_elasticity"] / agg["mean_elasticity"].abs()
    return df.merge(agg, on=keys, how="left")


def run_consolidation(ctx: StepContext) -> StepResult:
    storage, toggles = ctx.storage, ctx.config.get("steps", {})
    set_active_scope(ctx.div, ctx.quarter)
    dml_on = toggles.get("dml", "on") != "off"
    publish = toggles.get("consolidation", "on") != "off"

    manifest = ctx.manifest()
    tasks = [t for t in manifest["plan"]["tasks"] if t["step"] not in ("init_run", "consolidate")]
    dml_tasks = [t for t in tasks if t["step"] == "dml"]
    statuses = _collect_statuses(ctx, tasks)

    frames, unreadable = [], []
    for t in dml_tasks if dml_on else []:
        path = get_dml_result_path(ctx.run_id, t["target"], t["fold_id"], ctx.div, ctx.quarter)
        if statuses[t["task_id"]]["status"] in OK_STATUSES and storage.exists(path):
            try:
                frames.append(storage.read_parquet(path))
            except Exception as exc:
                unreadable.append(t["task_id"])
                logger.warning(f"cannot read {path}: {exc}", event="FOLD_RESULT_UNREADABLE", error_category="UNKNOWN")

    outputs, published_run_path = [], None
    result = None
    present = failed = 0
    if frames:
        result = _union_with_aggregates(frames)
        present, failed = len(result), int((result["status"] != "SUCCEEDED").sum())

    expected = _expected_rows(ctx, dml_tasks) if dml_on else 0
    bad_tasks = sorted({k for k, s in statuses.items() if s["status"] not in OK_STATUSES} | set(unreadable))
    partial_tasks = sorted(k for k, s in statuses.items() if s["status"] == "PARTIAL")
    missing_folds = sorted(t["task_id"] for t in dml_tasks
                           if dml_on and (statuses[t["task_id"]]["status"] not in OK_STATUSES or t["task_id"] in unreadable))
    completeness = {
        "expected_folds": len(dml_tasks) if dml_on else 0,
        "completed_folds": (len(dml_tasks) - len(missing_folds)) if dml_on else 0,
        "expected_rows": expected, "present_rows": present, "failed_rows": failed,
        "failed_tasks": bad_tasks, "partial_tasks": partial_tasks,
    }

    if dml_on and present == 0:
        run_status = "FAILED"
    elif not dml_on and bad_tasks:
        run_status = "FAILED"
    elif bad_tasks or partial_tasks or failed or present < expected:
        run_status = "PARTIAL"
    else:
        run_status = "SUCCEEDED"

    # Consumer serving layer publishing
    published_latest_path = None
    if publish and result is not None:
        published_run_path = get_published_result_path(ctx.run_id, ctx.div, ctx.quarter)
        published_latest_path = get_published_latest_path(ctx.div, ctx.quarter)
        
        # 1. Write immutable run-specific output
        storage.write_parquet(published_run_path, result)
        # 2. Update consumer serving 'latest' pointer
        storage.write_parquet(published_latest_path, result)

        # 3. Write metadata descriptor run_info.json
        run_info = {
            "run_id": ctx.run_id,
            "div": ctx.div,
            "quarter": ctx.quarter,
            "published_at": datetime.now(timezone.utc).isoformat(),
            "status": run_status,
            "present_rows": present,
            "expected_rows": expected,
            "failed_rows": failed,
            "manifest_uri": get_manifest_path(ctx.run_id, ctx.div, ctx.quarter)
        }
        storage.write_json(get_published_run_info_path(ctx.run_id, ctx.div, ctx.quarter), run_info)
        storage.write_json(get_published_latest_run_info_path(ctx.div, ctx.quarter), run_info)

        outputs.extend([published_run_path, published_latest_path])

    report_status = "SKIPPED"
    if publish and toggles.get("reports", "on") == "on":
        try:
            report_run_path = get_report_path("dml", ctx.run_id, ctx.div, ctx.quarter)
            report_latest_path = get_report_latest_path("dml", ctx.div, ctx.quarter)
            sheets = {
                "results": result if result is not None else pd.DataFrame({"note": ["no DML results"]}),
                "completeness": pd.DataFrame([{k: str(v) for k, v in completeness.items()}]),
                "tasks": pd.DataFrame([{"task_id": k, "status": s["status"],
                                        "error": (s.get("error") or {}).get("category")} for k, s in statuses.items()]),
            }
            write_xlsx(storage, report_run_path, sheets)
            write_xlsx(storage, report_latest_path, sheets)
            outputs.extend([report_run_path, report_latest_path])
            report_status = "SUCCEEDED"
        except Exception as exc:
            report_status = "FAILED"
            logger.warning(f"report generation failed (non-blocking): {exc}", event="REPORT_FAILED")

    reused = {k: s["counters"]["reused_from"] for k, s in statuses.items() if s.get("counters", {}).get("reused_from")}
    end = datetime.now(timezone.utc).isoformat()
    manifest.update(
        status=run_status, end_time=end, steps=statuses, completeness=completeness,
        lineage={
            "published": published_run_path,
            "published_latest": published_latest_path,
            "report_status": report_status,
            "reused_from": reused
        }
    )
    manifest_path = get_manifest_path(ctx.run_id, ctx.div, ctx.quarter)
    storage.write_json(manifest_path, manifest)
    
    index_path = get_run_index_path(ctx.run_id, ctx.div, ctx.quarter)
    index_record = {
        "run_id": ctx.run_id, "div": ctx.div, "quarter": ctx.quarter, "status": run_status,
        "start_time": manifest["start_time"], "end_time": end,
        "snapshot_id": manifest["snapshot"].get("snapshot_id"),
        "config_version": manifest["config"].get("config_version"),
        "image": manifest.get("image"), "published": published_run_path, "reused_from": reused,
    }
    storage.write_json(index_path, index_record)
    storage.write_json(f"runs/_index/{ctx.run_id}.json", index_record)
    outputs.append(manifest_path)

    logger.info(f"run {ctx.run_id} closed as {run_status} (div={ctx.div}, quarter={ctx.quarter})", event="RUN_CLOSED",
                counters={"present_rows": present, "expected_rows": expected, "failed_tasks": len(bad_tasks)})
    return StepResult(outputs=outputs, counters={"run_status": run_status, **{
        k: v for k, v in completeness.items() if isinstance(v, int)}, "report_status": report_status},
        warnings=[f"run finished {run_status}"] if run_status != "SUCCEEDED" else [])
