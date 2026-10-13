"""
DML fold task: enumerate driver combinations -> plan Stage-1 units -> cache lookup -> fit misses
(optionally in parallel) -> Stage-2 per (combination, driver) -> fold artifacts.

Orchestration (planning, caching, failure isolation, artifacts) lives in `run_dml_fold`.
The two functions `fit_stage1_unit` and `fit_stage2` are the only science seams: real DML code
replaces their bodies and keeps the same inputs/outputs.
"""
import math
import os
from datetime import datetime, timezone
from typing import Any, Dict

import numpy as np
import pandas as pd

from components.residual_caching.residual_caching import (
    check_cache_hits, compute_unit_key, load_cached_unit, save_cached_unit,
)
from shared.combinations import combination_id, enumerate_combinations
from shared.context import StepContext, StepResult
from shared.logging import logger
from shared.path_builder.paths import get_confounder_selection_path, get_dml_result_path, get_stage1_refs_path
from shared.schema.contracts import CacheMetadataContract, ErrorCategory, PipelineError, data_fp, fp

# Errors a single unit can raise without failing the whole fold (Level 1 isolation).
UNIT_LEVEL_ERRORS = {ErrorCategory.MODEL_CONVERGENCE, ErrorCategory.NUMERIC}


# ----------------------------- science seams (mock) -------------------------
def fit_stage1_unit(unit: Dict[str, Any], train: pd.DataFrame, test: pd.DataFrame,
                    date_col: str, learner_spec: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """Mock Stage-1: OLS of `unit['column']` on its controls. Returns the arrays stored in residuals.npz."""
    def design(frame):
        return np.column_stack([np.ones(len(frame))] + [frame[c].to_numpy(float) for c in unit["controls"]])

    y_tr, y_te = train[unit["column"]].to_numpy(float), test[unit["column"]].to_numpy(float)
    beta, *_ = np.linalg.lstsq(design(train), y_tr, rcond=None)
    hat_tr, hat_te = design(train) @ beta, design(test) @ beta
    return {
        "train_dates": train[date_col].dt.strftime("%Y-%m-%d").to_numpy().astype("U10"),
        "train_y_true": y_tr, "train_y_hat": hat_tr, "train_residuals": y_tr - hat_tr,
        "test_dates": test[date_col].dt.strftime("%Y-%m-%d").to_numpy().astype("U10"),
        "test_y_true": y_te, "test_y_hat": hat_te, "test_residuals": y_te - hat_te,
    }


def fit_stage2(y_res: np.ndarray, t_res: np.ndarray, y_res_test: np.ndarray, t_res_test: np.ndarray) -> Dict[str, float]:
    """Mock Stage-2: residual-on-residual slope with a simple standard error."""
    denom = float(np.sum(t_res ** 2))
    if denom == 0.0 or len(t_res) < 3:
        raise PipelineError(ErrorCategory.NUMERIC, "degenerate treatment residuals")
    beta = float(np.sum(t_res * y_res) / denom)
    se = math.sqrt(float(np.sum((y_res - beta * t_res) ** 2)) / (len(t_res) - 1) / denom)
    sst = float(np.sum(y_res_test ** 2))
    r2 = 1.0 - float(np.sum((y_res_test - beta * t_res_test) ** 2)) / sst if sst > 0 else float("nan")
    p = math.erfc(abs(beta / se) / math.sqrt(2)) if se > 0 else float("nan")
    return {"elasticity": beta, "std_error": se, "p_value": p, "r2": r2, "n_obs": len(t_res)}


# --------------------------------- orchestration ----------------------------
def _fit_safe(unit, train, test, date_col):
    """Worker-side wrapper: unit-level science errors are returned (not raised) so one bad unit never aborts the pool."""
    try:
        return "ok", fit_stage1_unit(unit, train, test, date_col, unit["learner"])
    except PipelineError as exc:
        return "err", exc.category, str(exc)


def _iter_fits(units, train, test, date_col, n_jobs):
    """Yield fit results in input order. n_jobs>1 uses a loky process pool (streamed, so finished units can be
    cached while the rest still run); n_jobs=1 stays in-process (debuggable, patchable in tests)."""
    if n_jobs > 1 and len(units) > 1:
        from joblib import Parallel, delayed
        os.environ.setdefault("OMP_NUM_THREADS", "1")  # avoid n_jobs x BLAS threads oversubscription
        return Parallel(n_jobs=n_jobs, backend="loky", return_as="generator")(
            delayed(_fit_safe)(u, train, test, date_col) for u in units)
    return (_fit_safe(u, train, test, date_col) for u in units)


def _plan(ctx, confounders, combos, fold, col_hash, dml_cfg):
    """Stage-1 units (deduplicated by key) and the Stage-2 row plan.
    For every combination C and driver t in C: controls = (C minus t) + confounders(t);
    one outcome unit (Y ~ controls) and one treatment unit (t ~ controls)."""
    train_window = f"{fold['train_start']}_{fold['train_end']}"
    test_window = f"{fold['test_start']}_{fold['test_end']}"
    column_of = {t["target_id"]: t["column"] for t in ctx.config["targets"]}
    units, rows = {}, []
    for combo in combos:
        cid = combination_id(combo)
        for t in combo:
            controls = sorted(set(d for d in combo if d != t) | set(confounders[t]))
            keys = {}
            for role, variable in (("outcome", ctx.target), ("treatment", t)):
                learner = dml_cfg["outcome_learner" if role == "outcome" else "treatment_learner"]
                column = column_of.get(variable, variable)
                key = compute_unit_key(
                    target=ctx.target, role=role, variable=variable, controls=controls,
                    source_data_hash=fp({c: col_hash(c) for c in sorted([column] + controls)}),
                    train_window=train_window, test_window=test_window,
                    cache_version=dml_cfg.get("cache_version", 1))
                units.setdefault(key, {"unit_key": key, "role": role, "variable": variable, "column": column,
                                       "controls": controls, "learner": learner})
                keys[role] = key
            rows.append({"combination_id": cid, "treatment_kpi": t, "outcome_unit": keys["outcome"],
                         "treatment_unit": keys["treatment"]})
    return units, rows


def run_dml_fold(ctx: StepContext) -> StepResult:
    dml_cfg, fold, storage = ctx.config["dml"], ctx.fold_spec(), ctx.storage
    date_col = ctx.config["data"]["date_col"]
    train_window = f"{fold['train_start']}_{fold['train_end']}"
    n_jobs = int(ctx.config.get("execution", {}).get("n_jobs", 1))

    cs = ctx.require(get_confounder_selection_path(ctx.upstream_run_id("confounder_selection"), ctx.target))
    confounders = {t["treatment_kpi"]: [c["kpi"] for c in t["confounders"]]
                   for t in cs["treatments"] if t["status"] == "SUCCEEDED"}
    combos = enumerate_combinations(list(confounders), ctx.config["combinations"])

    manifest = ctx.manifest()
    df = ctx.load_lagged_dataset()
    train = df[(df[date_col] >= fold["train_start"]) & (df[date_col] <= fold["train_end"])]
    test = df[(df[date_col] >= fold["test_start"]) & (df[date_col] <= fold["test_end"])]
    source_hash = manifest["snapshot"]["snapshot_id"]

    hashes: Dict[str, str] = {}

    def col_hash(col):  # content hash of one training column, computed once per fold
        if col not in hashes:
            hashes[col] = data_fp(train[[col]])
        return hashes[col]

    by_key, row_plan = _plan(ctx, confounders, combos, fold, col_hash, dml_cfg)
    cache_mode = ctx.config.get("cache", {}).get("mode", "read_write")  # read_write | read_only | off
    if cache_mode == "off":
        hits, misses = set(), set(by_key)
    else:
        hits, misses = check_cache_hits(storage, ctx.target, train_window, list(by_key), div=ctx.div)
    logger.info(f"{len(combos)} combinations, {len(row_plan)} rows; stage-1 cache: {len(hits)} hits, "
                f"{len(misses)} misses of {len(by_key)} units", event="CACHE_EVAL",
                counters={"combinations": len(combos), "units_total": len(by_key), "cache_hits": len(hits),
                          "cache_misses": len(misses)})

    residuals: Dict[str, Dict[str, np.ndarray]] = {}
    unit_status: Dict[str, Dict[str, Any]] = {k: {"status": "HIT", "error": None} for k in hits}
    repair = set()
    for key in sorted(hits):
        try:
            residuals[key], _ = load_cached_unit(storage, ctx.target, train_window, key, div=ctx.div)
        except Exception as exc:  # unreadable/corrupt entry: heal by refitting instead of failing the fold
            logger.warning(f"cache entry unreadable, refitting: {exc}", event="CACHE_CORRUPT", unit_key=key)
            hits.discard(key), misses.add(key), repair.add(key)
            unit_status.pop(key, None)

    to_fit = [by_key[k] for k in sorted(misses)]
    every = max(1, len(to_fit) // 20)
    for i, (unit, res) in enumerate(zip(to_fit, _iter_fits(to_fit, train, test, date_col, n_jobs)), 1):
        key = unit["unit_key"]
        if res[0] == "err":
            _, category, message = res
            if category not in UNIT_LEVEL_ERRORS:
                raise PipelineError(category, message)  # infra/timeouts/config: fail the fold; Airflow retries it
            logger.warning(f"unit failed: {message}", event="UNIT_FAILED", error_category=category, unit_key=key)
            unit_status[key] = {"status": "FAILED", "error": f"{category}: {message}"}
            continue
        arrays = res[1]
        if cache_mode == "read_write":
            metadata = CacheMetadataContract(
                fingerprint=key, cache_version=dml_cfg["cache_version"], role=unit["role"], target=ctx.target,
                driver_col=unit["variable"], controls=unit["controls"],
                train_start=fold["train_start"], train_end=fold["train_end"],
                test_start=fold["test_start"], test_end=fold["test_end"], fold_id=ctx.fold_id,
                source_file=manifest["snapshot"]["uri"], source_data_hash=source_hash,
                learner_spec=unit["learner"], inner_cv_splits=dml_cfg["inner_cv_splits"], seed=dml_cfg["seed"],
                created_at=datetime.now(timezone.utc).isoformat(), producer_run_id=ctx.run_id).to_dict()
            save_cached_unit(storage, ctx.target, train_window, key, arrays, metadata, overwrite=key in repair, div=ctx.div)
        residuals[key] = arrays
        unit_status[key] = {"status": "FITTED", "error": None}
        if i % every == 0 or i == len(to_fit):
            logger.info(f"stage-1 progress {i}/{len(to_fit)}", event="STAGE1_PROGRESS",
                        counters={"fitted": i, "to_fit": len(to_fit)})

    # Stage 2: one row per (combination, driver). Rows whose units failed are marked FAILED, never dropped.
    rows = []
    for plan in row_plan:
        o_key, t_key = plan["outcome_unit"], plan["treatment_unit"]
        row = {"run_id": ctx.run_id, "target": ctx.target, "fold_id": ctx.fold_id, **plan,
                "elasticity": np.nan, "std_error": np.nan, "p_value": np.nan, "r2": np.nan, "n_obs": 0,
                "status": "FAILED", "error": None}
        try:
            if o_key not in residuals or t_key not in residuals:
                raise PipelineError(ErrorCategory.MODEL_CONVERGENCE, "stage-1 unit failed")
            o, t = residuals[o_key], residuals[t_key]
            row.update(fit_stage2(o["train_residuals"], t["train_residuals"],
                                  o["test_residuals"], t["test_residuals"]), status="SUCCEEDED")
        except PipelineError as exc:
            if exc.category not in UNIT_LEVEL_ERRORS:
                raise
            row["error"] = f"{exc.category}: {exc}"
        rows.append(row)

    refs = pd.DataFrame([{"unit_key": u["unit_key"], "role": u["role"], "variable": u["variable"],
                          "controls": ",".join(u["controls"]), **unit_status[u["unit_key"]]}
                         for u in by_key.values()])
    result_path = get_dml_result_path(ctx.run_id, ctx.target, ctx.fold_id, ctx.div, ctx.quarter)
    refs_path = get_stage1_refs_path(ctx.run_id, ctx.target, ctx.fold_id, ctx.div, ctx.quarter)
    storage.write_parquet(result_path, pd.DataFrame(rows))
    storage.write_parquet(refs_path, refs)

    failed_units = sum(1 for s in unit_status.values() if s["status"] == "FAILED")
    failed_rows = sum(1 for r in rows if r["status"] == "FAILED")
    return StepResult(
        outputs=[result_path, refs_path],
        counters={"combinations": len(combos), "rows": len(rows), "rows_failed": failed_rows,
                  "units_total": len(by_key), "cache_hits": len(hits),
                  "units_fitted": len(to_fit) - failed_units, "units_failed": failed_units},
        warnings=[f"{failed_rows} row(s) FAILED"] if failed_rows else [],
        status="PARTIAL" if failed_rows else "SUCCEEDED",
    )
