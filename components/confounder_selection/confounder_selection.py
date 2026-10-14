# confounder_selection.py

import numpy as np
import pandas as pd


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
        | {label + "_T" for label in config["drivers"]}
        | piecewise_generated_cols
    )

    external_confounder_pool = [
        col
        for col in df.columns
        if col not in exclude and pd.api.types.is_numeric_dtype(df[col])
    ]

    print(
        f"Candidate confounder pool: "
        f"{len(external_confounder_pool)} numeric variables"
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
        print("\n" + "=" * 110)
        print(f"CONFOUNDER SELECTION FOR DRIVER: {driver_label}")
        print("=" * 110)

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
            print("No eligible external confounders found.")
            driver_confounders[driver_label] = list(time_confs)
            continue

        # ---------------------------------------------------------------------
        # STRONGEST CANDIDATE
        # ---------------------------------------------------------------------

        first_candidate = candidate_pool[0]
        strongest_corr = correlations[first_candidate]

        selected_external.append(first_candidate)

        dynamic_threshold = relative_threshold * strongest_corr

        print("\nCandidate ranking based on simple correlation:")

        for rank, candidate in enumerate(candidate_pool, start=1):
            print(
                f"  {rank}. {candidate} "
                f"(abs correlation = {correlations[candidate]:.4f})"
            )
            

        print("\nSelection rule:")
        print(f"  Strongest correlation = {strongest_corr:.4f}")
        print(f"  Relative threshold = {relative_threshold:.1%}")
        print(f"  Dynamic partial-correlation threshold = {dynamic_threshold:.4f}")

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

            print(
                f"{candidate:<45}"
                f"{correlations[candidate]:>15.4f}"
                f"{abs_partial_corr:>17.4f}"
                f"{dynamic_threshold:>15.4f}"
                f"{decision:>12}"
            )

            if keep_candidate:
                selected_external.append(candidate)

            elif stop_when_below:
                print(
                    f"Stopping selection for {driver_label}: "
                    "candidate fell below threshold."
                )
                break

        driver_confounders[driver_label] = list(time_confs) + selected_external

        print(
            f"\nFinal confounders for {driver_label}: "
            f"{driver_confounders[driver_label]}"
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

    print(f"\nFinal Y confounders: {y_confounders}")

    return driver_confounders, y_confounders, yoy_df