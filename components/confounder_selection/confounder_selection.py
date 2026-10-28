# confounder_selection.py
from __future__ import annotations

import numpy as np
import pandas as pd

from shared.context import StepContext, StepResult
from shared.logging import logger
from shared.path_builder.paths import get_confounder_selection_path, get_feature_selection_summary

FREQ_MAP = {"M": 12, "Q": 4, "W": 52, "D": 365}


# =============================================================================
# PARTIAL CORRELATION
# =============================================================================

def calculate_partial_correlation(data, target_col, candidate_col, control_cols):
    """
    Calculate partial correlation between target_col and candidate_col
    while controlling for control_cols.
    """
    cols = [target_col, candidate_col] + control_cols
    temp = data[cols].dropna()

    if len(temp) < 5:
        return np.nan

    corr_matrix = temp.corr().values
    precision_matrix = np.linalg.pinv(corr_matrix)

    numerator = -precision_matrix[0, 1]
    denominator = np.sqrt(precision_matrix[0, 0] * precision_matrix[1, 1])

    if denominator <= 0 or not np.isfinite(denominator):
        return np.nan

    partial_corr = numerator / denominator
    return float(np.clip(partial_corr, -1.0, 1.0))


# =============================================================================
# CONFOUNDER SELECTION
# =============================================================================

def select_confounders(df, train_df, config, drivers, time_confs, yoy_log_diff):
    """
    Select confounders for:
        1. Every treatment/driver
        2. Outcome Y

    Returns
    -------
    driver_confounders : dict
    y_confounders      : list
    yoy_df             : pd.DataFrame
    """

    # =========================================================================
    # BUILD EXTERNAL CONFOUNDER POOL
    # =========================================================================

    piecewise_generated_cols = set()

    for label in config.get("piecewise_drivers", {}):
        piecewise_generated_cols.add(label + "_T_low")
        piecewise_generated_cols.add(label + "_T_high")

    exclude = (
        {
            "date",
            "modelling_level",
            config["outcome_col"],
            "Y",
            "trend",
            "regime_dummy",
        }
        | set(config["drivers"].values())
        | set(config["drivers"].keys())
        | {label + "_T" for label in config["drivers"]}
        | piecewise_generated_cols
    )

    external_confounder_pool = [
        col
        for col in df.columns
        if col not in exclude and pd.api.types.is_numeric_dtype(df[col])
    ]

    logger.info(
        f"Candidate confounder pool: {len(external_confounder_pool)} numeric variables",
        event="CANDIDATE_POOL",
        candidate_count=len(external_confounder_pool),
    )

    # =========================================================================
    # CREATE YoY CONFOUNDER DATA
    # =========================================================================

    yoy_df = pd.DataFrame(index=df.index)

    for col in external_confounder_pool:
        yoy_df[col] = yoy_log_diff(df[col])

    # =========================================================================
    # PARTIAL CORRELATION CONFIG
    # =========================================================================

    partial_corr_cfg = config.get("partial_correlation", {})

    partial_corr_enabled = partial_corr_cfg.get("enabled", True)
    relative_threshold = partial_corr_cfg.get("relative_threshold", 0.15)
    stop_when_below = partial_corr_cfg.get("stop_when_below_threshold", True)

    # =========================================================================
    # DRIVER CONFOUNDER SELECTION
    # =========================================================================

    driver_confounders = {}

    for driver_label in drivers:
        logger.info(
            f"Confounder selection for driver: {driver_label}",
            event="DRIVER_SELECTION_START",
            driver=driver_label,
        )

        correlations = (
            yoy_df.loc[train_df.index]
            .corrwith(train_df[driver_label + "_T"])
            .abs()
            .dropna()
            .sort_values(ascending=False)
        )

        max_k = config.get("confounder_count", {}).get(driver_label, 0)
        max_k = min(max_k, len(correlations))

        candidate_pool = correlations.head(max_k).index.tolist()
        selected_external = []

        # ---------------------------------------------------------------------
        # NO EXTERNAL CANDIDATES
        # ---------------------------------------------------------------------

        if not candidate_pool:
            logger.info(f"No eligible external confounders found for {driver_label}.", event="NO_CONFOUNDERS", driver=driver_label)
            driver_confounders[driver_label] = list(time_confs)
            continue

        # ---------------------------------------------------------------------
        # STRONGEST CANDIDATE
        # ---------------------------------------------------------------------

        first_candidate = candidate_pool[0]
        strongest_corr = correlations[first_candidate]

        selected_external.append(first_candidate)

        dynamic_threshold = relative_threshold * strongest_corr

        logger.info(
            f"Driver {driver_label} candidates: {len(candidate_pool)}, strongest={strongest_corr:.4f}, threshold={dynamic_threshold:.4f}",
            event="SELECTION_RULE",
            driver=driver_label,
            strongest_corr=round(float(strongest_corr), 4),
            dynamic_threshold=round(float(dynamic_threshold), 4),
        )

        # ---------------------------------------------------------------------
        # REMAINING CANDIDATES
        # ---------------------------------------------------------------------

        for candidate in candidate_pool[1:]:
            control_cols = list(time_confs) + selected_external

            partial_data = pd.DataFrame(index=train_df.index)
            partial_data["__treatment__"] = train_df[driver_label + "_T"]
            partial_data["__candidate__"] = yoy_df.loc[train_df.index, candidate]

            for control in time_confs:
                partial_data[control] = train_df[control]

            for control in selected_external:
                partial_data[control] = yoy_df.loc[train_df.index, control]

            partial_data = partial_data.loc[:, ~partial_data.columns.duplicated()]

            if partial_corr_enabled:
                partial_corr = calculate_partial_correlation(
                    data=partial_data,
                    target_col="__treatment__",
                    candidate_col="__candidate__",
                    control_cols=control_cols,
                )
            else:
                partial_corr = correlations[candidate]

            abs_partial_corr = abs(partial_corr) if pd.notna(partial_corr) else np.nan

            # -----------------------------------------------------------------
            # DECISION
            # -----------------------------------------------------------------

            if pd.isna(abs_partial_corr):
                keep_candidate = False
                decision = "REJECTED"

            elif abs_partial_corr >= dynamic_threshold:
                keep_candidate = True
                decision = "SELECTED"

            else:
                keep_candidate = False
                decision = "REJECTED"

            logger.info(
                f"{driver_label} candidate {candidate}: corr={correlations[candidate]:.4f}, partial_corr={abs_partial_corr:.4f}, decision={decision}",
                event="CANDIDATE_DECISION",
                driver=driver_label,
                candidate=candidate,
                decision=decision,
            )

            if keep_candidate:
                selected_external.append(candidate)

            elif stop_when_below:
                logger.info(
                    f"Stopping selection for {driver_label}: candidate fell below threshold.",
                    event="SELECTION_STOPPED",
                    driver=driver_label,
                )
                break

        driver_confounders[driver_label] = list(time_confs) + selected_external
        logger.info(
            f"Final confounders for {driver_label}: {driver_confounders[driver_label]}",
            event="DRIVER_CONFOUNDERS_RESOLVED",
            driver=driver_label,
            confounders=driver_confounders[driver_label],
        )

    # =========================================================================
    # PIECEWISE TIME CONFOUNDERS
    # =========================================================================

    for driver_label in config.get("piecewise_drivers", {}):
        # Skip piecewise drivers that are not present
        # in the current driver combination.
        if driver_label not in driver_confounders:
            continue

        externals = [
            confounder
            for confounder in driver_confounders[driver_label]
            if confounder not in time_confs
        ]

        piecewise_time_confs = config.get(
            "piecewise_time_confounders",
            ["trend", "regime_dummy"],
        )

        driver_confounders[driver_label] = list(piecewise_time_confs) + externals

    # =========================================================================
    # Y CONFOUNDER SELECTION
    # =========================================================================

    y_cfg = config.get("y_confounder_selection", {})

    y_enabled = y_cfg.get("enabled", True)
    y_max = y_cfg.get("max_confounders", 3)
    y_relative_threshold = y_cfg.get("relative_threshold", 0.5)
    y_stop = y_cfg.get("stop_when_below_threshold", True)

    y_confounders = list(time_confs)

    if y_enabled:
        y_correlations = (
            yoy_df.loc[train_df.index]
            .corrwith(train_df["Y"])
            .abs()
            .dropna()
            .sort_values(ascending=False)
        )

        max_y_k = min(y_max, len(y_correlations))
        candidate_pool_y = y_correlations.head(max_y_k).index.tolist()
        selected_y_external = []

        if candidate_pool_y:
            first_candidate = candidate_pool_y[0]
            strongest_y_corr = y_correlations[first_candidate]

            selected_y_external.append(first_candidate)

            dynamic_y_threshold = y_relative_threshold * strongest_y_corr

            for candidate in candidate_pool_y[1:]:
                control_cols = list(time_confs) + selected_y_external

                partial_data = pd.DataFrame(index=train_df.index)
                partial_data["__Y__"] = train_df["Y"]
                partial_data["__candidate__"] = yoy_df.loc[train_df.index, candidate]

                for control in time_confs:
                    partial_data[control] = train_df[control]

                for control in selected_y_external:
                    partial_data[control] = yoy_df.loc[train_df.index, control]

                partial_data = partial_data.loc[
                    :,
                    ~partial_data.columns.duplicated(),
                ]

                partial_corr_y = calculate_partial_correlation(
                    data=partial_data,
                    target_col="__Y__",
                    candidate_col="__candidate__",
                    control_cols=control_cols,
                )

                abs_partial = abs(partial_corr_y) if pd.notna(partial_corr_y) else np.nan

                if pd.notna(abs_partial) and abs_partial >= dynamic_y_threshold:
                    selected_y_external.append(candidate)

                elif y_stop:
                    break

        y_confounders = list(time_confs) + selected_y_external

    logger.info(
        f"Final Y confounders: {y_confounders}",
        event="Y_CONFOUNDERS_RESOLVED",
        y_confounders=y_confounders,
    )

    return driver_confounders, y_confounders, yoy_df


# =============================================================================
# PIPELINE INTEGRATION
# =============================================================================


def run_confounder_selection(ctx: StepContext) -> StepResult:
    """Pipeline entry point for confounder selection.
    Loads upstream feature-selection summary and lagged dataset, sets up time-series cutoffs,
    and runs select_confounders before writing confounder_set.json.
    """
    fs_run = ctx.upstream_run_id("feature_selection")
    summary_path = get_feature_selection_summary(fs_run, ctx.target, ctx.div, ctx.quarter)
    if ctx.storage and not ctx.storage.exists(summary_path):
        legacy_path = summary_path.replace("feature_prioritization.json", "fs_summary.json")
        if ctx.storage.exists(legacy_path):
            summary_path = legacy_path
    fs_summary = ctx.require(summary_path)
    treatments = fs_summary.get("resolved_kpis", [])

    # Load enriched lagged features dataset emitted by feature_selection
    df = ctx.load_lagged_dataset().copy()
    date_col = ctx.config.get("data", {}).get("date_col", "date")
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.sort_values(date_col).reset_index(drop=True)

    # Resolve target column
    target_col = next((t["column"] for t in ctx.config.get("targets", []) if t["target_id"] == ctx.target), ctx.target)

    # Resolve drivers: support curated driver superset from dml_config
    dml_cfg = ctx.config.get("dml", {})
    configured_drivers = dml_cfg.get("drivers", {})

    if treatments and configured_drivers:
        # Check if configured_drivers filters or curates upstream treatments
        matching = {
            label: col for label, col in configured_drivers.items()
            if (label in treatments or col in treatments) and (col in df.columns or label in df.columns)
        }
        if matching:
            drivers = {label: (col if col in df.columns else label) for label, col in matching.items()}
        elif ctx.config.get("steps", {}).get("feature_selection", "on") == "off" and not treatments:
            drivers = {label: (col if col in df.columns else label) for label, col in configured_drivers.items() if (col in df.columns or label in df.columns)}
        else:
            drivers = {t: t for t in treatments if t in df.columns}
    elif treatments:
        drivers = {t: t for t in treatments if t in df.columns}
    elif configured_drivers:
        drivers = {label: (col if col in df.columns else label) for label, col in configured_drivers.items() if (col in df.columns or label in df.columns)}
    else:
        drivers = {}

    if not drivers:
        drivers = {c: c for c in df.columns if c not in (target_col, date_col, "trend", "regime_dummy") and pd.api.types.is_numeric_dtype(df[c])}

    # Time confounders
    time_confs = dml_cfg.get("time_confounders", ["trend"])
    if "trend" in time_confs and "trend" not in df.columns:
        df["trend"] = np.arange(len(df))
    if "regime_dummy" in time_confs and "regime_dummy" not in df.columns:
        regime_date = dml_cfg.get("regime_date", "2023-01-01")
        df["regime_dummy"] = (df[date_col] >= pd.Timestamp(regime_date)).astype(int)

    # YoY log-diff function
    freq_val = ctx.config.get("data", {}).get("freq", "M")
    freq = FREQ_MAP.get(freq_val, 12) if isinstance(freq_val, str) else int(freq_val)

    def _yoy(series: pd.Series) -> pd.Series:
        shifted = series.shift(freq)
        # Safe log-diff avoiding div by zero or negative
        ratio = np.where((series > 0) & (shifted > 0), series / shifted, np.nan)
        return pd.Series(np.log(np.where(ratio > 0, ratio, 1.0)), index=series.index)

    # Add driver transform columns {label}_T and outcome Y if missing
    if "Y" not in df.columns and target_col in df.columns:
        df["Y"] = _yoy(df[target_col])
    for label, col in drivers.items():
        if f"{label}_T" not in df.columns and col in df.columns:
            df[f"{label}_T"] = _yoy(df[col])
        if label in dml_cfg.get("piecewise_drivers", {}):
            knot = float(dml_cfg["piecewise_drivers"][label])
            df[f"{label}_T_low"] = np.minimum(df[f"{label}_T"], knot)
            df[f"{label}_T_high"] = np.maximum(df[f"{label}_T"] - knot, 0.0)

    # Confounder-selection training window
    conf_cfg = ctx.config.get("confounders", {})
    end_date = conf_cfg.get("selection_end_date") or conf_cfg.get("confounder_selection_end_date")
    if end_date and date_col in df.columns:
        cutoff = pd.Timestamp(end_date)
        train_df = df[df[date_col] < cutoff].copy().reset_index(drop=True)
    else:
        folds = ctx.config.get("folds", [])
        if folds and "train_end" in folds[0] and date_col in df.columns:
            train_df = df[df[date_col] <= pd.Timestamp(folds[0]["train_end"])].copy().reset_index(drop=True)
        else:
            train_df = df.copy()

    # Merged config for select_confounders
    merged_cfg = {**dml_cfg, **conf_cfg}
    merged_cfg["outcome_col"] = target_col
    merged_cfg["drivers"] = drivers

    logger.info(f"running confounder selection for target '{ctx.target}' with {len(drivers)} drivers",
                event="CONFOUNDER_SELECTION_STARTED", target=ctx.target)

    # Call science core
    driver_confounders, y_confounders, yoy_df = select_confounders(
        df=df,
        train_df=train_df,
        config=merged_cfg,
        drivers=drivers,
        time_confs=time_confs,
        yoy_log_diff=_yoy,
    )

    treatment_confounders = [
        {
            "treatment_kpi": driver_label,
            "status": "SUCCEEDED",
            "confounders": [{"kpi": c, "lag": 1, "reason": "selected"} for c in confs if c != driver_label and c != drivers.get(driver_label)],
        }
        for driver_label, confs in driver_confounders.items()
    ]

    payload = {
        "run_id": ctx.run_id,
        "target": ctx.target,
        "step_fp": ctx.step_fp("confounder_selection"),
        "window": conf_cfg.get("window", "common_train"),
        "feature_selection_run_id": fs_run,
        "candidate_pool": [c for c in yoy_df.columns if c not in time_confs],
        "common_confounders": y_confounders,
        "confounder_set": y_confounders,
        "y_confounders": y_confounders,
        "driver_confounders": driver_confounders,
        "treatments": treatment_confounders,
    }

    path = get_confounder_selection_path(ctx.run_id, ctx.target, ctx.div, ctx.quarter)
    ctx.storage.write_json(path, payload)

    logger.info(f"confounder selection completed for target '{ctx.target}': {len(y_confounders)} Y confounders selected",
                event="CONFOUNDER_SELECTION_COMPLETED", target=ctx.target,
                counters={"treatments_analyzed": len(drivers), "confounders_selected": len(y_confounders)})

    return StepResult(outputs=[path], counters={
        "treatments_analyzed": len(drivers),
        "confounders_selected": len(y_confounders),
        "confounder_pool_size": len(yoy_df.columns),
    })