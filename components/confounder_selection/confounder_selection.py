"""
Confounder Selection (mock science). Runs once per target on the common-train window and records,
for every treatment KPI, its confounders. Real logic replaces `select_confounders`.
"""
from typing import List

from shared.context import StepContext, StepResult
from shared.path_builder.paths import get_confounder_selection_path, get_feature_selection_summary


def select_confounders(ctx: StepContext, candidate_pool: List[str]) -> List[str]:
    """Select common confounders once from the candidate pool."""
    # Science seam: real selection logic replaces this (e.g. lasso/correlation on common train window)
    return list(candidate_pool)


def run_confounder_selection(ctx: StepContext) -> StepResult:
    fs_run = ctx.upstream_run_id("feature_selection")  # validated when reused from an earlier run
    fs_summary = ctx.require(get_feature_selection_summary(fs_run, ctx.target, ctx.div, ctx.quarter))
    treatments = fs_summary["resolved_kpis"]

    # Load enriched lagged features dataset emitted by feature_selection
    df = ctx.load_lagged_dataset()
    date_col = ctx.config["data"]["date_col"]
    target_col = next((t["column"] for t in ctx.config["targets"] if t["target_id"] == ctx.target), ctx.target)

    # Confounder pool: candidate_pool from config or all other columns in lagged dataset except target and date
    configured_pool = ctx.config.get("confounders", {}).get("candidate_pool", [])
    if configured_pool:
        candidate_pool = [c for c in configured_pool if c not in (target_col, ctx.target, date_col)]
    else:
        candidate_pool = [c for c in df.columns if c not in (target_col, ctx.target, date_col) and c not in treatments]

    # Confounder selection runs ONCE: selects the common confounder set
    common_confounders = select_confounders(ctx, candidate_pool)

    # Use the same set for each treatment (excluding the treatment driver itself)
    treatment_confounders = [
        {
            "treatment_kpi": t,
            "status": "SUCCEEDED",
            "confounders": [{"kpi": c, "lag": 1, "reason": "common confounder set"}
                            for c in common_confounders if c != t],
        }
        for t in treatments
    ]

    payload = {
        "run_id": ctx.run_id,
        "target": ctx.target,
        "step_fp": ctx.step_fp("confounder_selection"),
        "window": ctx.config.get("confounders", {}).get("window", "common_train"),
        "feature_selection_run_id": fs_run,
        "candidate_pool": candidate_pool,
        "common_confounders": common_confounders,
        "confounder_set": common_confounders,
        "treatments": treatment_confounders,
    }
    path = get_confounder_selection_path(ctx.run_id, ctx.target, ctx.div, ctx.quarter)
    ctx.storage.write_json(path, payload)
    return StepResult(outputs=[path], counters={
        "treatments_analyzed": len(treatments),
        "confounders_selected": len(common_confounders),
        "confounder_pool_size": len(candidate_pool),
    })
