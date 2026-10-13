"""
Canonical Azure Blob and local file path resolver for the driver analysis pipeline.
Implements the multi-division, multi-quarter production blob layout:
  - Global content-addressed: snapshots/
  - Global cache (partitioned by div): cache/dml_stage1/{div}/{target}/{train_window}/{unit_key}/
  - Self-contained run workspace: runs/{div}/{quarter}/{run_id}/
  - Consumer serving layer: published/{div}/{quarter}/latest/ and published/{div}/{quarter}/runs/{run_id}/
  - Reviewer report layer: reports/{div}/{quarter}/latest/ and reports/{div}/{quarter}/runs/{run_id}/
"""
import os
from typing import Optional, Tuple

# Process-wide active scope fallback
_ACTIVE_DIV = "EL"
_ACTIVE_QUARTER = "2026Q3"


def set_active_scope(div: Optional[str] = None, quarter: Optional[str] = None) -> None:
    """Set process-wide division and quarter scope."""
    global _ACTIVE_DIV, _ACTIVE_QUARTER
    if div:
        _ACTIVE_DIV = div
    if quarter:
        _ACTIVE_QUARTER = quarter


def get_active_div() -> str:
    """Get active division identifier."""
    return os.environ.get("PIPELINE_DIV", _ACTIVE_DIV)


def get_active_quarter() -> str:
    """Get active quarter identifier."""
    return os.environ.get("PIPELINE_QUARTER", _ACTIVE_QUARTER)


def resolve_scope(div: Optional[str] = None, quarter: Optional[str] = None) -> Tuple[str, str]:
    """Resolve div and quarter using parameters or active process scope."""
    d = div or get_active_div()
    q = quarter or get_active_quarter()
    return d, q


# ---------------------------------------------------------------------------
# Global content-addressed data
# ---------------------------------------------------------------------------
def get_snapshot_path(snapshot_id: str) -> str:
    """Path to frozen, content-addressed input dataset."""
    return f"snapshots/{snapshot_id}/input.parquet"


# ---------------------------------------------------------------------------
# Self-contained run workspace: runs/{div}/{quarter}/{run_id}/
# ---------------------------------------------------------------------------
def get_run_dir(run_id: str, div: Optional[str] = None, quarter: Optional[str] = None) -> str:
    d, q = resolve_scope(div, quarter)
    return f"runs/{d}/{q}/{run_id}"


def get_manifest_path(run_id: str, div: Optional[str] = None, quarter: Optional[str] = None) -> str:
    """Path to the single source of truth manifest for a run."""
    d, q = resolve_scope(div, quarter)
    return f"runs/{d}/{q}/{run_id}/run_manifest.json"


def get_master_config_path(run_id: str, div: Optional[str] = None, quarter: Optional[str] = None) -> str:
    """Path to the frozen master config for a run."""
    d, q = resolve_scope(div, quarter)
    return f"runs/{d}/{q}/{run_id}/master_config.json"


def get_task_status_path(run_id: str, task_id: str, div: Optional[str] = None, quarter: Optional[str] = None) -> str:
    """Path to individual task completion status file."""
    d, q = resolve_scope(div, quarter)
    return f"runs/{d}/{q}/{run_id}/status/{task_id}.json"


def get_task_log_path(run_id: str, task_id: str, div: Optional[str] = None, quarter: Optional[str] = None) -> str:
    """Path to individual task execution .log file."""
    d, q = resolve_scope(div, quarter)
    return f"runs/{d}/{q}/{run_id}/logs/{task_id}.log"


def get_run_log_path(run_id: str, div: Optional[str] = None, quarter: Optional[str] = None) -> str:
    """Path to composite run execution .log file."""
    d, q = resolve_scope(div, quarter)
    return f"runs/{d}/{q}/{run_id}/logs/run.log"


def get_run_index_path(run_id: str, div: Optional[str] = None, quarter: Optional[str] = None) -> str:
    """Path to the lightweight discovery index file for a run."""
    d, q = resolve_scope(div, quarter)
    return f"runs/_index/{d}/{q}/{run_id}.json"


def get_feature_selection_dir(run_id: str, target: str, div: Optional[str] = None, quarter: Optional[str] = None) -> str:
    """Directory for feature selection outputs inside run workspace."""
    d, q = resolve_scope(div, quarter)
    return f"runs/{d}/{q}/{run_id}/feature_selection/{target}"


def get_feature_selection_parquet(run_id: str, target: str, div: Optional[str] = None, quarter: Optional[str] = None) -> str:
    """Path to selected features matrix."""
    d, q = resolve_scope(div, quarter)
    return f"runs/{d}/{q}/{run_id}/feature_selection/{target}/selected_features.parquet"


def get_lagged_features_path(run_id: str, target: str, div: Optional[str] = None, quarter: Optional[str] = None) -> str:
    """Path to dataset with lagged features emitted by feature selection for downstream tasks."""
    d, q = resolve_scope(div, quarter)
    return f"runs/{d}/{q}/{run_id}/feature_selection/{target}/data_with_lagged_features.parquet"


def get_feature_selection_summary(run_id: str, target: str, div: Optional[str] = None, quarter: Optional[str] = None) -> str:
    """Path to feature selection summary breakdown."""
    d, q = resolve_scope(div, quarter)
    return f"runs/{d}/{q}/{run_id}/feature_selection/{target}/fs_summary.json"


def get_confounder_selection_path(run_id: str, target: str, div: Optional[str] = None, quarter: Optional[str] = None) -> str:
    """Path to confounder set output inside run workspace."""
    d, q = resolve_scope(div, quarter)
    return f"runs/{d}/{q}/{run_id}/confounder_selection/{target}/confounder_set.json"


def get_dml_fold_dir(run_id: str, target: str, fold_id: str, div: Optional[str] = None, quarter: Optional[str] = None) -> str:
    """Directory for DML fold outputs inside run workspace."""
    d, q = resolve_scope(div, quarter)
    return f"runs/{d}/{q}/{run_id}/dml/{target}/fold={fold_id}"


def get_dml_result_path(run_id: str, target: str, fold_id: str, div: Optional[str] = None, quarter: Optional[str] = None) -> str:
    """Path to fold DML elasticity results inside run workspace."""
    d, q = resolve_scope(div, quarter)
    return f"runs/{d}/{q}/{run_id}/dml/{target}/fold={fold_id}/dml_result.parquet"


def get_stage1_refs_path(run_id: str, target: str, fold_id: str, div: Optional[str] = None, quarter: Optional[str] = None) -> str:
    """Path to fold Stage-1 residual cache references inside run workspace."""
    d, q = resolve_scope(div, quarter)
    return f"runs/{d}/{q}/{run_id}/dml/{target}/fold={fold_id}/stage1_refs.parquet"


# ---------------------------------------------------------------------------
# Consumer serving layer: published/{div}/{quarter}/
# ---------------------------------------------------------------------------
def get_published_result_path(run_id: str, div: Optional[str] = None, quarter: Optional[str] = None) -> str:
    """Path to consolidated cross-fold analytical parquet for a specific run."""
    d, q = resolve_scope(div, quarter)
    return f"published/{d}/{q}/runs/{run_id}/consolidated_result.parquet"


def get_published_latest_path(div: Optional[str] = None, quarter: Optional[str] = None) -> str:
    """Path to latest approved consolidated result for the division and quarter."""
    d, q = resolve_scope(div, quarter)
    return f"published/{d}/{q}/latest/consolidated_result.parquet"


def get_published_run_info_path(run_id: str, div: Optional[str] = None, quarter: Optional[str] = None) -> str:
    """Provenance metadata for a specific published run."""
    d, q = resolve_scope(div, quarter)
    return f"published/{d}/{q}/runs/{run_id}/run_info.json"


def get_published_latest_run_info_path(div: Optional[str] = None, quarter: Optional[str] = None) -> str:
    """Provenance metadata for the latest published run."""
    d, q = resolve_scope(div, quarter)
    return f"published/{d}/{q}/latest/run_info.json"


# ---------------------------------------------------------------------------
# Reviewer reports: reports/{div}/{quarter}/
# ---------------------------------------------------------------------------
def get_report_path(kind: str, run_id: str, div: Optional[str] = None, quarter: Optional[str] = None) -> str:
    """Path to run-specific reviewer sign-off report."""
    d, q = resolve_scope(div, quarter)
    name = "dml_report.xlsx" if kind == "dml" else "fs_report.xlsx"
    return f"reports/{d}/{q}/runs/{run_id}/{name}"


def get_report_latest_path(kind: str, div: Optional[str] = None, quarter: Optional[str] = None) -> str:
    """Path to latest reviewer report."""
    d, q = resolve_scope(div, quarter)
    name = "dml_report.xlsx" if kind == "dml" else "fs_report.xlsx"
    return f"reports/{d}/{q}/latest/{name}"


# ---------------------------------------------------------------------------
# Stage-1 residual cache (partitioned by div to eliminate cross-talk)
# ---------------------------------------------------------------------------
def get_residual_cache_prefix(target: str, train_window: str, div: Optional[str] = None) -> str:
    """Prefix for bulk cache prefix scan of Stage-1 units."""
    d, _ = resolve_scope(div, None)
    return f"cache/dml_stage1/{d}/{target}/{train_window}/"


def get_residual_cache_unit_dir(target: str, train_window: str, unit_key: str, div: Optional[str] = None) -> str:
    """Directory holding residual .npz and metadata.json for a unit."""
    d, _ = resolve_scope(div, None)
    return f"cache/dml_stage1/{d}/{target}/{train_window}/{unit_key}"


def get_residual_npz_path(target: str, train_window: str, unit_key: str, div: Optional[str] = None) -> str:
    """Path to cached Stage-1 residual arrays."""
    d, _ = resolve_scope(div, None)
    return f"cache/dml_stage1/{d}/{target}/{train_window}/{unit_key}/residuals.npz"


def get_residual_metadata_path(target: str, train_window: str, unit_key: str, div: Optional[str] = None) -> str:
    """Path to cached Stage-1 residual metadata."""
    d, _ = resolve_scope(div, None)
    return f"cache/dml_stage1/{d}/{target}/{train_window}/{unit_key}/metadata.json"


# ---------------------------------------------------------------------------
# Task identity & plan
# ---------------------------------------------------------------------------
def make_task_id(step: str, target: Optional[str] = None, fold_id: Optional[str] = None) -> str:
    if step == "dml":
        return f"dml_fold_{target}_{fold_id}"
    return f"{step}_{target}" if target else step


def plan_tasks(config: dict) -> list:
    """Every task a run is expected to execute, in DAG order."""
    targets = [t["target_id"] for t in config.get("targets", [])]
    folds = [f["fold_id"] for f in config.get("folds", [])]
    tasks = [{"task_id": "init_run", "step": "init_run", "target": None, "fold_id": None}]
    for step in ("feature_selection", "confounder_selection"):
        tasks += [{"task_id": make_task_id(step, t), "step": step, "target": t, "fold_id": None} for t in targets]
    tasks += [{"task_id": make_task_id("dml", t, f), "step": "dml", "target": t, "fold_id": f}
              for t in targets for f in folds]
    tasks.append({"task_id": "consolidate", "step": "consolidate", "target": None, "fold_id": None})
    return tasks
