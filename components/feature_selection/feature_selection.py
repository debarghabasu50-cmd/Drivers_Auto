"""
Feature Selection (mock science). Resolves the driver KPIs for a target and writes the lagged matrix
plus a summary. Real selection logic replaces `select_kpis`; the I/O contract stays the same.
"""
from typing import List

import pandas as pd

from shared.context import StepContext, StepResult
from shared.logging import logger
from shared.path_builder.paths import (
    get_feature_selection_parquet,
    get_feature_selection_summary,
    get_lagged_features_path,
    get_report_path,
)
from shared.reports import write_xlsx


def select_kpis(ctx: StepContext, df) -> List[dict]:
    """Mock: every KPI in kpis.extra minus kpis.exclude is selected at lag 1 / tier P0."""
    kpis = ctx.config["kpis"]
    return [{"kpi": k, "tier": "P0", "lag": 1, "included": True, "reason": "mock: configured KPI"}
            for k in kpis["extra"] if k not in kpis.get("exclude", [])]


def run_feature_selection(ctx: StepContext) -> StepResult:
    df = ctx.load_snapshot()
    date_col = ctx.config["data"]["date_col"]
    column = next(t["column"] for t in ctx.config["targets"] if t["target_id"] == ctx.target)

    logger.info(f"evaluating KPI drivers for target '{ctx.target}'", event="COMPUTATION_STARTED",
                step="feature_selection", target=ctx.target,
                kpi_candidates=len(ctx.config.get("kpis", {}).get("extra", [])))
    decisions = select_kpis(ctx, df)
    selected = [d["kpi"] for d in decisions if d["included"]]
    logger.info(f"selected {len(selected)} KPI drivers for target '{ctx.target}'", event="COMPUTATION_COMPLETED",
                step="feature_selection", target=ctx.target,
                counters={"kpis_evaluated": len(decisions), "kpis_selected": len(selected)})

    matrix = df[[date_col, column]].rename(columns={column: ctx.target}).copy()
    lagged_df = df.copy()
    if ctx.target not in lagged_df.columns and column in lagged_df.columns:
        lagged_df[ctx.target] = lagged_df[column]
    for d in decisions:
        if d["included"]:
            lag_col = f"{d['kpi']}_lag{d['lag']}"
            matrix[lag_col] = df[d["kpi"]].shift(d["lag"])
            if lag_col not in lagged_df.columns:
                lagged_df[lag_col] = df[d["kpi"]].shift(d["lag"])

    matrix_path = get_feature_selection_parquet(ctx.run_id, ctx.target, ctx.div, ctx.quarter)
    lagged_path = get_lagged_features_path(ctx.run_id, ctx.target, ctx.div, ctx.quarter)
    summary_path = get_feature_selection_summary(ctx.run_id, ctx.target, ctx.div, ctx.quarter)
    ctx.storage.write_parquet(matrix_path, matrix)
    ctx.storage.write_parquet(lagged_path, lagged_df)
    ctx.storage.write_json(summary_path, {
        "run_id": ctx.run_id, "target": ctx.target, "step_fp": ctx.step_fp("feature_selection"),
        "kpis": decisions, "resolved_kpis": selected,
        "lagged_features_path": lagged_path,
    })
    outputs = [matrix_path, lagged_path, summary_path]
    if ctx.config.get("steps", {}).get("reports", "on") == "on":
        try:  # presentation only: never blocks the step
            report_path = get_report_path("feature_selection", ctx.run_id, ctx.div, ctx.quarter)
            write_xlsx(ctx.storage, report_path, {f"fs_{ctx.target}"[:31]: pd.DataFrame(decisions)})
            outputs.append(report_path)
        except Exception as exc:
            logger.warning(f"fs report failed (non-blocking): {exc}", event="REPORT_FAILED")
    logger.info(f"feature_selection outputs written for target '{ctx.target}'", event="OUTPUT_WRITTEN",
                outputs=outputs, target=ctx.target)
    return StepResult(outputs=outputs,
                      counters={"kpis_evaluated": len(decisions), "kpis_selected": len(selected)})
