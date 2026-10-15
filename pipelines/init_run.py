"""
T1 init_run: merge + freeze config, validate input -> content-addressed snapshot, create the manifest
(RUNNING) and the run-index entry. After this step every other task reads only frozen artifacts.
"""
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from shared.context import StepContext, StepResult
from shared.logging import logger
from shared.path_builder.paths import (
    get_manifest_path,
    get_master_config_path,
    get_run_index_path,
    get_snapshot_path,
    plan_tasks,
    set_active_scope,
)
from shared.schema.contracts import ErrorCategory, PipelineError, RunManifestContract, data_fp, fp

CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"


def load_repo_config(master_path: str = None) -> Dict[str, Any]:
    """master_config.json + component configs merged under their own keys. Frozen per run, so the
    config_version covers every parameter that can change a result."""
    master = Path(master_path) if master_path else CONFIG_DIR / "master_config.json"
    config = json.loads(master.read_text(encoding="utf-8"))
    for key, name in (("feature_selection", "feature_selection_config.json"), ("dml", "dml_config.json")):
        config[key] = json.loads((CONFIG_DIR / name).read_text(encoding="utf-8"))
    
    div = config.get("div", config.get("pipeline", {}).get("div", "EL"))
    quarter = config.get("quarter", config.get("pipeline", {}).get("quarter", "2026Q3"))
    set_active_scope(div, quarter)

    # Derive folds if not explicitly provided in master_config
    if not config.get("folds"):
        cutoffs = config.get("train_end_dates") or config.get("dml", {}).get("train_end_dates", [])
        if cutoffs:
            config["folds"] = [{"fold_id": f"f{i+1}", "train_end": d} for i, d in enumerate(cutoffs)]

    return config


def _validate(df: pd.DataFrame, config: Dict[str, Any]) -> None:
    date_col = config["data"]["date_col"]
    needed = [date_col] + [t["column"] for t in config["targets"]] + config["kpis"]["extra"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise PipelineError(ErrorCategory.DATA_SCHEMA, f"input is missing columns: {missing}")
    if df[needed].isnull().any().any():
        raise PipelineError(ErrorCategory.DATA_QUALITY, "input has nulls in target/KPI columns")


def run_init(ctx: StepContext) -> StepResult:
    config, storage = ctx.config, ctx.storage
    set_active_scope(ctx.div, ctx.quarter)

    manifest_path = get_manifest_path(ctx.run_id, ctx.div, ctx.quarter)
    if storage.exists(manifest_path):
        if ctx.attempt == 1:
            raise PipelineError(
                ErrorCategory.CONFIG,
                f"run '{ctx.run_id}' is already initialized; to re-run, use a new run_id "
                f"or invoke individual steps with run_pipeline.py",
            )
        existing_manifest = storage.read_json(manifest_path)
        existing_status = existing_manifest.get("status")
        if existing_status in ("SUCCEEDED", "PARTIAL", "FAILED"):
            raise PipelineError(
                ErrorCategory.CONFIG,
                f"run '{ctx.run_id}' is already completed with status '{existing_status}'",
            )

    input_uri = config["data"]["input_uri"]
    if storage.exists(input_uri):
        df = storage.read_parquet(input_uri)
    elif storage.is_mock:
        repo_root = Path(__file__).resolve().parents[1]
        default_blob = repo_root / ".mock_blob" / "drivers-pipeline" / input_uri
        local_data = repo_root / "data" / Path(input_uri).name
        source = default_blob if default_blob.is_file() else (local_data if local_data.is_file() else None)
        if source:
            df = pd.read_parquet(source)
            storage.write_parquet(input_uri, df, overwrite=False)
        else:
            raise PipelineError(ErrorCategory.UPSTREAM_MISSING, f"input not found: {input_uri}")
    else:
        raise PipelineError(ErrorCategory.UPSTREAM_MISSING, f"input not found: {input_uri}")
    
    logger.log_provenance(
        source_run_id=ctx.run_id,
        path=input_uri,
        target=None,
        fingerprint=None,
        msg=f"loaded raw input: {input_uri}",
        rows=len(df),
        cols=len(df.columns),
    )
    _validate(df, config)

    snapshot_id = "snap_" + data_fp(df)
    snapshot_path = get_snapshot_path(snapshot_id)
    storage.write_parquet(snapshot_path, df, overwrite=False)  # content-addressed: write once
    logger.info(f"snapshot created: {snapshot_id}", event="SNAPSHOT_CREATED", snapshot_id=snapshot_id,
                path=snapshot_path, rows=len(df))

    config_version = "cfg_" + fp(config)
    config_path = get_master_config_path(ctx.run_id, ctx.div, ctx.quarter)
    storage.write_json(config_path, config)

    tasks = plan_tasks(config)
    start = datetime.now(timezone.utc).isoformat()
    manifest = RunManifestContract(
        run_id=ctx.run_id,
        status="RUNNING",
        start_time=start,
        image=os.environ.get("IMAGE_TAG", "local"),
        snapshot={"snapshot_id": snapshot_id, "uri": snapshot_path, "data_hash": snapshot_id[5:]},
        config={"config_version": config_version, "uri": config_path},
        step_toggles=config.get("steps", {}),
        plan={
            "div": ctx.div,
            "quarter": ctx.quarter,
            "targets": [t["target_id"] for t in config["targets"]],
            "folds": [f["fold_id"] for f in config["folds"]],
            "work_items": [f"{t['target']}|{t['fold_id']}" for t in tasks if t["step"] == "dml"],
            "tasks": tasks,
        },
    )
    manifest_path = get_manifest_path(ctx.run_id, ctx.div, ctx.quarter)
    storage.write_json(manifest_path, manifest.to_dict())
    
    # Write scoped index entry and flat index pointer for discovery
    index_path = get_run_index_path(ctx.run_id, ctx.div, ctx.quarter)
    index_entry = {
        "run_id": ctx.run_id, "div": ctx.div, "quarter": ctx.quarter,
        "status": "RUNNING", "start_time": start,
        "snapshot_id": snapshot_id, "config_version": config_version
    }
    storage.write_json(index_path, index_entry)
    storage.write_json(f"runs/_index/{ctx.run_id}.json", index_entry)

    outputs = [config_path, snapshot_path, manifest_path]
    logger.info("init_run outputs written", event="OUTPUT_WRITTEN", outputs=outputs)
    logger.info(f"run initialised: {snapshot_id}, {config_version}, {len(tasks)} planned tasks (div={ctx.div}, quarter={ctx.quarter})",
                event="RUN_INITIALISED", counters={"tasks": len(tasks)})
    return StepResult(outputs=outputs,
                      counters={"rows": len(df), "planned_tasks": len(tasks)})
