"""
Double Machine Learning (DML) pipeline for causal elasticity estimation.

Run:
    python dml_analysis.py --config config_dml.json
"""

from __future__ import annotations

import argparse
import json
import warnings
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from statsmodels.tsa.stattools import adfuller

from confounder_selection import select_confounders
from stage1_cache import (
    calculate_file_hash,
    load_driver_stage1_cache,
    make_driver_cache_fingerprint,
    save_driver_stage1_cache,
)

warnings.filterwarnings("ignore")


# =============================================================================
# CONFIGURATION
# =============================================================================


def load_config(config_path: str | Path) -> dict:
    """Load and validate configuration."""
    path = Path(config_path)

    with path.open("r", encoding="utf-8") as file:
        config = json.load(file)

    assert config["alpha_stage1y"] >= 0.1, "alpha_stage1y too low (min 0.1)"
    assert config["alpha_stage1y"] <= 500, "alpha_stage1y too high (max 500)"
    assert 3 <= config["val_rows"] <= 12, "val_rows must be 3-12"

    if config["alpha_stage2"] is not None:
        assert (
            0.001 <= config["alpha_stage2"] <= 1000
        ), "alpha_stage2 out of range [0.001, 1000]"

    config.setdefault("output_dir", "dml_outputs")
    config.setdefault("alpha_stage1t", 5.0)

    return config


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run the DML elasticity pipeline.")
    parser.add_argument("--config", required=True, help="Path to JSON config file.")
    return parser.parse_args()


# =============================================================================
# BASIC UTILITIES
# =============================================================================


def yoy_log_diff(series: pd.Series, freq: int) -> pd.Series:
    """Calculate year-over-year log difference."""
    return np.log(series / series.shift(freq))


def calculate_partial_correlation(data, target_col, candidate_col, control_cols):
    """Calculate partial correlation using the inverse correlation matrix."""
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

    return float(np.clip(numerator / denominator, -1.0, 1.0))


def get_stability_status(swing: float, ok_swing: float, warn_swing: float) -> str:
    """Classify coefficient stability."""
    if swing < ok_swing:
        return "OK"

    if swing < warn_swing:
        return "WARN"

    return "UNSTABLE"


# =============================================================================
# DATA PREPARATION
# =============================================================================


def prepare_data(config: dict):
    """Load data, create transformations and create train/test datasets."""
    df = pd.read_excel(config["file"])
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values("date").reset_index(drop=True)

    freq = config["data_freq"]

    # Outcome transformation
    df["Y"] = yoy_log_diff(df[config["outcome_col"]], freq)

    # Driver transformations
    for label, col in config["drivers"].items():
        df[f"{label}_T"] = yoy_log_diff(df[col], freq)

        if label in config.get("piecewise_drivers", {}):
            knot = float(config["piecewise_drivers"][label])
            df[f"{label}_T_low"] = np.minimum(df[f"{label}_T"], knot)
            df[f"{label}_T_high"] = np.maximum(df[f"{label}_T"] - knot, 0.0)

    # Required columns
    required_cols = ["Y"]

    for label in config["drivers"]:
        required_cols.append(f"{label}_T")

        if label in config.get("piecewise_drivers", {}):
            required_cols.extend([f"{label}_T_low", f"{label}_T_high"])

    df = df.dropna(subset=required_cols).reset_index(drop=True)

    # Time confounders
    df["trend"] = np.arange(len(df))
    df["regime_dummy"] = (df["date"] >= config["regime_date"]).astype(int)

    # Confounder-selection dataset
    max_train_end = pd.Timestamp(config["confounder_selection_end_date"])
    master_confounder_train_df = (
        df[df["date"] < max_train_end].copy().reset_index(drop=True)
    )

    # Model train/test datasets
    train_df = df[df["date"] < config["train_end"]].copy().reset_index(drop=True)
    test_df = df[df["date"] >= config["train_end"]].copy().reset_index(drop=True)

    return df, master_confounder_train_df, train_df, test_df


def print_dataset_summary(config, train_df, test_df, output_dir):
    """Print dataset information."""
    print(f"Division    : {config['division_name']}")
    print(f"All Drivers : {list(config['drivers'].keys())}")

    print(
        f"Training set: {len(train_df)} rows  "
        f"({train_df['date'].min().date()} to {train_df['date'].max().date()})"
    )

    if test_df.empty:
        print("Test set    : skipped - entire available period is used for training")
    else:
        print(
            f"Test set    : {len(test_df)} rows  "
            f"({test_df['date'].min().date()} to {test_df['date'].max().date()})"
        )

    print(f"Outputs will be saved to: {output_dir.resolve()}")


# =============================================================================
# DESIGN MATRIX
# =============================================================================


def build_design_matrix(df_subset, yoy_df, confounders):
    """Create Stage 1 confounder design matrix."""
    X = pd.DataFrame(index=df_subset.index)

    for col in confounders:
        if col in ["trend", "regime_dummy"]:
            X[col] = df_subset[col]
        else:
            X[col] = yoy_df.loc[df_subset.index, col]

    return X.fillna(0)


# =============================================================================
# LOO RESIDUALIZATION
# =============================================================================


def compute_loo_residuals(
    X,
    y,
    gbm_config,
    use_gradient_boosting=True,
    ridge_alpha=5.0,
):
    """Compute leave-one-out cross-fitted residuals."""
    n_obs = len(y)
    residuals = np.zeros(n_obs)

    for i in range(n_obs):
        train_mask = np.ones(n_obs, dtype=bool)
        train_mask[i] = False

        if use_gradient_boosting:
            model = GradientBoostingRegressor(
                n_estimators=gbm_config.get("n_estimators", 30),
                max_depth=gbm_config.get("max_depth", 2),
                learning_rate=gbm_config.get("learning_rate", 0.1),
                random_state=gbm_config.get("random_state", 42),
            )
        else:
            model = Ridge(alpha=ridge_alpha)

        model.fit(X[train_mask], y[train_mask])
        residuals[i] = y[i] - model.predict(X[i : i + 1])[0]

    return residuals


def residualize_treatment_component(
    X_train,
    X_test,
    T_train,
    T_test,
    component_name,
    config,
    min_var_ratio,
    ridge_alpha=5.0,
):
    """Residualize a normal or piecewise treatment component."""
    gbm_config = config.get("gbm_stage1t", {})

    gbm_residuals = compute_loo_residuals(
        X_train,
        T_train,
        gbm_config,
        use_gradient_boosting=True,
    )

    gbm_var_ratio = np.var(gbm_residuals) / (np.var(T_train) + 1e-10)
    use_gbm = gbm_var_ratio >= min_var_ratio

    if use_gbm:
        train_residuals = gbm_residuals
    else:
        train_residuals = compute_loo_residuals(
            X_train,
            T_train,
            gbm_config,
            use_gradient_boosting=False,
        )

    var_ratio = np.var(train_residuals) / (np.var(T_train) + 1e-10)

    try:
        adf_pvalue = adfuller(train_residuals)[1]
    except Exception:
        adf_pvalue = np.nan

    if use_gbm:
        full_model = GradientBoostingRegressor(
            n_estimators=gbm_config.get("n_estimators", 30),
            max_depth=gbm_config.get("max_depth", 2),
            learning_rate=gbm_config.get("learning_rate", 0.1),
            random_state=gbm_config.get("random_state", 42),
        )
    else:
        full_model = Ridge(alpha=ridge_alpha)

    full_model.fit(X_train, T_train)

    if len(T_test) > 0:
        test_residuals = T_test - full_model.predict(X_test)
    else:
        test_residuals = np.array([])

    return {
        "train_residuals": train_residuals,
        "test_residuals": test_residuals,
        "var_ratio": var_ratio,
        "adf_pvalue": adf_pvalue,
        "model_used": "GBM" if use_gbm else "Ridge",
        "component_name": component_name,
    }


# =============================================================================
# STAGE 1T
# =============================================================================


def calculate_stage1_driver(
    driver_label,
    train_df,
    test_df,
    yoy_df,
    confounders,
    config,
    min_var_ratio,
):
    """Calculate Stage 1T residuals for one driver."""
    X_train = build_design_matrix(train_df, yoy_df, confounders).values
    X_test = build_design_matrix(test_df, yoy_df, confounders).values

    if driver_label in config.get("piecewise_drivers", {}):
        low_res = residualize_treatment_component(
            X_train=X_train,
            X_test=X_test,
            T_train=train_df[f"{driver_label}_T_low"].values,
            T_test=test_df[f"{driver_label}_T_low"].values,
            component_name=f"{driver_label}_low",
            config=config,
            min_var_ratio=min_var_ratio,
            ridge_alpha=config.get("alpha_stage1t", 5.0),
        )

        high_res = residualize_treatment_component(
            X_train=X_train,
            X_test=X_test,
            T_train=train_df[f"{driver_label}_T_high"].values,
            T_test=test_df[f"{driver_label}_T_high"].values,
            component_name=f"{driver_label}_high",
            config=config,
            min_var_ratio=min_var_ratio,
            ridge_alpha=config.get("alpha_stage1t", 5.0),
        )

        result = {
            "train_residuals": low_res["train_residuals"],
            "test_residuals": low_res["test_residuals"],
            "var_ratio": low_res["var_ratio"],
            "adf_pvalue": low_res["adf_pvalue"],
            "model_used": f"low={low_res['model_used']}, high={high_res['model_used']}",
            "train_residuals_low": low_res["train_residuals"],
            "test_residuals_low": low_res["test_residuals"],
            "train_residuals_high": high_res["train_residuals"],
            "test_residuals_high": high_res["test_residuals"],
            "var_ratio_low": low_res["var_ratio"],
            "adf_pvalue_low": low_res["adf_pvalue"],
            "var_ratio_high": high_res["var_ratio"],
            "adf_pvalue_high": high_res["adf_pvalue"],
        }

        return result

    normal_res = residualize_treatment_component(
        X_train=X_train,
        X_test=X_test,
        T_train=train_df[f"{driver_label}_T"].values,
        T_test=test_df[f"{driver_label}_T"].values,
        component_name=driver_label,
        config=config,
        min_var_ratio=min_var_ratio,
        ridge_alpha=config.get("alpha_stage1t", 5.0),
    )

    return {
        "train_residuals": normal_res["train_residuals"],
        "test_residuals": normal_res["test_residuals"],
        "var_ratio": normal_res["var_ratio"],
        "adf_pvalue": normal_res["adf_pvalue"],
        "model_used": normal_res["model_used"],
    }


def print_stage1_driver_gate(
    driver_label, result, config, min_var_ratio, max_adf_pvalue
):
    """Print Stage 1T quality gate."""
    if driver_label in config.get("piecewise_drivers", {}):
        gate_low = (
            "OK"
            if result["var_ratio_low"] >= min_var_ratio
            and pd.notna(result["adf_pvalue_low"])
            and result["adf_pvalue_low"] < max_adf_pvalue
            else "WARN"
        )

        gate_high = (
            "OK"
            if result["var_ratio_high"] >= min_var_ratio
            and pd.notna(result["adf_pvalue_high"])
            and result["adf_pvalue_high"] < max_adf_pvalue
            else "WARN"
        )

        print(
            f"[{gate_low}] {driver_label + ' low':<15} "
            f"VarRatio={result['var_ratio_low']:.3f} "
            f"ADF-p={result['adf_pvalue_low']:.3f}"
        )

        print(
            f"[{gate_high}] {driver_label + ' high':<15} "
            f"VarRatio={result['var_ratio_high']:.3f} "
            f"ADF-p={result['adf_pvalue_high']:.3f}"
        )

        return

    gate = (
        "OK"
        if result["var_ratio"] >= min_var_ratio
        and pd.notna(result["adf_pvalue"])
        and result["adf_pvalue"] < max_adf_pvalue
        else "WARN"
    )

    print(
        f"[{gate}] {driver_label:<15} "
        f"model={result['model_used']:<5} "
        f"VarRatio={result['var_ratio']:.3f} "
        f"ADF-p={result['adf_pvalue']:.3f}"
    )


def run_stage1_drivers(
    combination_id,
    drivers,
    driver_confounders,
    train_df,
    test_df,
    yoy_df,
    config,
    cache_root,
    source_data_hash,
    global_driver_residual_cache,
    min_var_ratio,
    max_adf_pvalue,
):
    """Run Stage 1T for all drivers in a combination."""
    cache_config = config.get("stage1_cache", {})
    cache_enabled = cache_config.get("enabled", True)
    cache_force_rebuild = cache_config.get("force_rebuild", False)
    cache_version = cache_config.get("cache_version", 1)

    stage1_results = {}

    for driver_label in drivers:
        print(f"\n--- Stage 1T: {driver_label} ---")

        confounders = driver_confounders[driver_label]
        reuse_key = (driver_label, tuple(sorted(confounders)))

        fingerprint = make_driver_cache_fingerprint(
            driver_label=driver_label,
            confounders=confounders,
            config=config,
            source_data_hash=source_data_hash,
            cache_version=cache_version,
        )

        if reuse_key in global_driver_residual_cache:
            print(f"[RESIDUAL REUSE] {driver_label}")
            result = global_driver_residual_cache[reuse_key]

        else:
            result = load_driver_stage1_cache(
                combination_id=combination_id,
                driver_label=driver_label,
                expected_fingerprint=fingerprint,
                cache_root=cache_root,
                cache_enabled=cache_enabled,
                force_rebuild=cache_force_rebuild,
            )

            if result is not None:
                print(f"[USING CACHE] {driver_label}: Stage 1T calculation skipped")

            else:
                print(
                    f"[CALCULATING] {driver_label}: running Stage 1T LOO residualization"
                )

                result = calculate_stage1_driver(
                    driver_label=driver_label,
                    train_df=train_df,
                    test_df=test_df,
                    yoy_df=yoy_df,
                    confounders=confounders,
                    config=config,
                    min_var_ratio=min_var_ratio,
                )

            global_driver_residual_cache[reuse_key] = result

        stage1_results[driver_label] = result

        save_driver_stage1_cache(
            combination_id=combination_id,
            driver_label=driver_label,
            fingerprint=fingerprint,
            result=result,
            cache_root=cache_root,
            config=config,
            source_data_hash=source_data_hash,
            cache_version=cache_version,
            confounders=confounders,
            cache_enabled=cache_enabled,
        )

        print_stage1_driver_gate(
            driver_label,
            result,
            config,
            min_var_ratio,
            max_adf_pvalue,
        )

    return stage1_results


# =============================================================================
# STAGE 1Y
# =============================================================================


def run_stage1_y(
    combination_id,
    y_confounders,
    train_df,
    test_df,
    yoy_df,
    config,
    global_y_residual_cache,
    min_var_ratio,
    max_adf_pvalue,
):
    """Run Stage 1Y with global in-memory residual reuse."""
    y_cache_key = tuple(sorted(y_confounders))

    if y_cache_key in global_y_residual_cache:
        print(f"[Y RESIDUAL REUSE] {y_cache_key}")
        return global_y_residual_cache[y_cache_key]

    ridge_alpha_y = config["alpha_stage1y"]

    X_y_train = build_design_matrix(train_df, yoy_df, y_confounders)
    X_y_test = build_design_matrix(test_df, yoy_df, y_confounders)

    outcome_model = Ridge(alpha=ridge_alpha_y).fit(
        X_y_train.values,
        train_df["Y"].values,
    )

    Y_residuals_train = compute_loo_residuals(
        X_y_train.values,
        train_df["Y"].values,
        config.get("gbm_stage1t", {}),
        use_gradient_boosting=False,
        ridge_alpha=ridge_alpha_y,
    )

    if test_df.empty:
        Y_residuals_test = np.array([])
    else:
        Y_residuals_test = test_df["Y"].values - outcome_model.predict(X_y_test.values)

    var_ratio_y = np.var(Y_residuals_train) / (np.var(train_df["Y"].values) + 1e-10)

    try:
        adf_pvalue_y = adfuller(Y_residuals_train)[1]
    except Exception:
        adf_pvalue_y = np.nan

    gate_y = (
        "OK  "
        if var_ratio_y >= min_var_ratio
        and pd.notna(adf_pvalue_y)
        and adf_pvalue_y < max_adf_pvalue
        else "WARN"
    )

    print(
        f"\n[{gate_y}] {'Outcome Y':<15} model=Ridge  "
        f"VarRatio={var_ratio_y:.3f}  ADF-p={adf_pvalue_y:.3f}"
    )

    print(f"Y confounders used: {y_confounders}")
    print(f"{combination_id} -> {y_confounders}")

    result = (
        Y_residuals_train,
        Y_residuals_test,
        var_ratio_y,
        adf_pvalue_y,
        gate_y,
    )

    global_y_residual_cache[y_cache_key] = result

    return result


# =============================================================================
# STAGE 2 ALPHA SELECTION
# =============================================================================


def select_stage2_alpha(T_residuals_train, Y_residuals_train, config):
    """Select Stage 2 Ridge alpha."""
    alpha_config = config.get("stage2_alpha_search", {})

    alphas_to_try = np.logspace(
        alpha_config.get("min_power", -2),
        alpha_config.get("max_power", 3),
        alpha_config.get("num_values", 100),
    )

    val_n = config["val_rows"]

    T_fit = T_residuals_train[:-val_n]
    Y_fit = Y_residuals_train[:-val_n]
    T_val = T_residuals_train[-val_n:]
    Y_val = Y_residuals_train[-val_n:]

    if config["alpha_stage2"] is not None:
        best_alpha = config["alpha_stage2"]
        print(f"Using manual alpha = {best_alpha} (set via CONFIG['alpha_stage2'])")
        return best_alpha, alphas_to_try, val_n

    validation_r2 = [
        r2_score(
            Y_val,
            Ridge(alpha=alpha, fit_intercept=False).fit(T_fit, Y_fit).predict(T_val),
        )
        for alpha in alphas_to_try
    ]

    best_alpha = float(alphas_to_try[int(np.argmax(validation_r2))])

    print(
        f"Auto-selected alpha = {best_alpha:.4f} "
        f"(best R² on last {val_n} training rows)"
    )

    return best_alpha, alphas_to_try, val_n


# =============================================================================
# STAGE 2
# =============================================================================


def run_stage2(
    drivers,
    stage1_results,
    Y_residuals_train,
    Y_residuals_test,
    config,
    train_only,
):
    """Estimate Stage 2 elasticities."""
    piecewise = config.get("piecewise_drivers", {})
    fallback_thresholds = config.get("fallback_thresholds", {})

    pw_drivers = [d for d in drivers if d in piecewise]
    joint_drivers = [d for d in drivers if d not in pw_drivers]

    T_residuals_train = np.column_stack(
        [stage1_results[d]["train_residuals"] for d in joint_drivers]
    )

    if train_only:
        T_residuals_test = np.empty((0, len(joint_drivers)))
    else:
        T_residuals_test = np.column_stack(
            [stage1_results[d]["test_residuals"] for d in joint_drivers]
        )

    best_alpha, alphas_to_try, val_n = select_stage2_alpha(
        T_residuals_train,
        Y_residuals_train,
        config,
    )

    joint_model = Ridge(alpha=best_alpha, fit_intercept=False)
    joint_model.fit(T_residuals_train, Y_residuals_train)

    r2_train = r2_score(
        Y_residuals_train,
        joint_model.predict(T_residuals_train),
    )

    if train_only:
        r2_test = None
        print(
            f"Joint Ridge alpha={best_alpha:.4f} "
            f"R²_train={r2_train:.4f} R²_test=SKIPPED"
        )
    else:
        r2_test = (
            r2_score(
                Y_residuals_test,
                joint_model.predict(T_residuals_test),
            )
            if len(Y_residuals_test) > 1
            else None
        )

        r2_test_text = f"{r2_test:.4f}" if r2_test is not None else "N/A"

        print(
            f"Joint Ridge alpha={best_alpha:.4f} "
            f"R²_train={r2_train:.4f} R²_test={r2_test_text}"
        )

    joint_elasticities = dict(zip(joint_drivers, joint_model.coef_))

    final_elasticities = {}
    diagnostics = {}

    for driver_label in drivers:
        method = "joint Ridge"
        T_res_train = stage1_results[driver_label]["train_residuals"]
        T_res_test = stage1_results[driver_label]["test_residuals"]
        elasticity = joint_elasticities.get(driver_label)

        if driver_label in piecewise:
            T_low_res_train = stage1_results[driver_label]["train_residuals_low"]
            T_high_res_train = stage1_results[driver_label]["train_residuals_high"]
            T_pw = np.column_stack([T_low_res_train, T_high_res_train])

            pw_model = Ridge(alpha=best_alpha, fit_intercept=False)
            pw_model.fit(T_pw, Y_residuals_train)

            e_low, e_high = pw_model.coef_
            elasticity = e_low
            method = "piecewise Ridge (log-knot before Stage 1)"

        elif abs(elasticity) > fallback_thresholds.get("max_abs_elasticity", 1.5):
            elasticity = float(
                np.dot(T_res_train, Y_residuals_train)
                / (np.dot(T_res_train, T_res_train) + 1e-10)
            )
            method = "per-driver OLS (fallback)"

        expected_sign = config["sign_priors"].get(driver_label)
        sign_as_expected = not (
            expected_sign is not None and np.sign(elasticity) != expected_sign
        )

        if train_only:
            r2_test_driver = None

        else:
            if driver_label in piecewise:
                T_low_res_test = stage1_results[driver_label]["test_residuals_low"]
                T_high_res_test = stage1_results[driver_label]["test_residuals_high"]

                Y_pred_test_driver = e_low * T_low_res_test + e_high * T_high_res_test
            else:
                Y_pred_test_driver = elasticity * T_res_test

            r2_test_driver = (
                r2_score(Y_residuals_test, Y_pred_test_driver)
                if len(T_res_test) > 1
                else None
            )

        final_elasticities[driver_label] = elasticity

        diagnostics[driver_label] = {
            "elasticity": round(elasticity, 4),
            "sign_ok": sign_as_expected,
            "r2_test": (
                round(float(r2_test_driver), 3) if r2_test_driver is not None else None
            ),
            "stage1_model": stage1_results[driver_label]["model_used"],
            "var_ratio": round(stage1_results[driver_label]["var_ratio"], 3),
            "adf_pvalue": round(stage1_results[driver_label]["adf_pvalue"], 3),
            "method": method,
        }

        if driver_label in piecewise:
            diagnostics[driver_label]["e_low"] = round(float(e_low), 4)
            diagnostics[driver_label]["e_high"] = round(float(e_high), 4)
            diagnostics[driver_label]["knot"] = float(piecewise[driver_label])

    return {
        "pw_drivers": pw_drivers,
        "joint_drivers": joint_drivers,
        "T_residuals_train": T_residuals_train,
        "T_residuals_test": T_residuals_test,
        "best_alpha": best_alpha,
        "alphas_to_try": alphas_to_try,
        "val_n": val_n,
        "joint_model": joint_model,
        "r2_train": r2_train,
        "r2_test": r2_test,
        "final_elasticities": final_elasticities,
        "diagnostics": diagnostics,
    }


def print_elasticity_results(stage2_results):
    """Print Stage 2 elasticity output."""
    diagnostics = stage2_results["diagnostics"]
    pw_drivers = stage2_results["pw_drivers"]

    print("\n" + "=" * 78)
    print(f"{'Driver':<15} {'Elasticity':>12} {'Sign OK':>8}  {'R²_test':>8}  Method")
    print("=" * 78)

    for driver_label, diag in diagnostics.items():
        sign_flag = "YES" if diag["sign_ok"] else "NO *"
        r2_str = f"{diag['r2_test']:.3f}" if diag["r2_test"] is not None else "N/A"

        print(
            f"{driver_label:<15} {diag['elasticity']:>12.4f} "
            f"{sign_flag:>8} {r2_str:>8}  {diag['method']}"
        )

    if not pw_drivers:
        return

    print("\n=== Piecewise Driver Elasticities (value-knot) ===")

    for d in pw_drivers:
        knot = diagnostics[d]["knot"]
        e_low = diagnostics[d]["e_low"]
        e_high = diagnostics[d]["e_high"]

        def contribution(t):
            return e_low * min(t, knot) + e_high * max(t - knot, 0.0)

        print(
            f"{d:<15} knot={knot:.2f} "
            f"e_low={e_low:.4f} e_high={e_high:.4f} "
            f"@40%={contribution(0.40):.4f} "
            f"@60%={contribution(0.60):.4f} "
            f"@80%={contribution(0.80):.4f}"
        )


# =============================================================================
# ELASTICITY PLOT
# =============================================================================


def create_elasticity_scatter_plot(
    stage1_results,
    diagnostics,
    Y_residuals_train,
    drivers,
    config,
    output_dir,
):
    """Create Stage 2 elasticity scatter plots."""
    n_drivers = len(drivers)
    n_cols = 4
    n_rows = (n_drivers + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    axes = np.atleast_1d(axes).flatten()

    for idx, (driver_label, diag) in enumerate(diagnostics.items()):
        ax = axes[idx]
        T_res = stage1_results[driver_label]["train_residuals"]
        sign_ok = diag["sign_ok"]

        point_colour = "#1976D2" if sign_ok else "#C62828"
        bg_colour = "#e3f2fd" if sign_ok else "#ffebee"

        ax.scatter(
            T_res,
            Y_residuals_train,
            alpha=0.65,
            s=35,
            color=point_colour,
            zorder=3,
        )

        x_range = np.linspace(T_res.min(), T_res.max(), 100)

        if driver_label in config.get("piecewise_drivers", {}):
            knot = diag["knot"]
            e_low = diag["e_low"]
            e_high = diag["e_high"]

            y_piecewise = e_low * np.minimum(x_range, knot) + e_high * np.maximum(
                x_range - knot, 0.0
            )

            ax.plot(x_range, y_piecewise, "k--", lw=1.5, label="piecewise")
            ax.axvline(knot, color="gray", lw=0.8, ls="--", alpha=0.6)

            ax.plot(
                x_range[x_range <= knot],
                e_low * np.minimum(x_range[x_range <= knot], knot),
                lw=1.2,
                ls=":",
                label=f"e_low={e_low:.3f}",
            )

            ax.plot(
                x_range[x_range >= knot],
                e_low * knot
                + e_high * np.maximum(x_range[x_range >= knot] - knot, 0.0),
                lw=1.2,
                ls=":",
                label=f"e_high={e_high:.3f}",
            )

        else:
            ax.plot(
                x_range,
                diag["elasticity"] * x_range,
                "k--",
                lw=1.5,
                label=f"e = {diag['elasticity']:.3f}",
            )

        ax.axhline(0, color="gray", lw=0.5)
        ax.axvline(0, color="gray", lw=0.5)
        ax.set_title(driver_label, fontsize=9, fontweight="bold")
        ax.set_xlabel("Driver residual (Stage 1T)", fontsize=8)
        ax.set_ylabel("Orders residual (Stage 1Y)", fontsize=8)
        ax.legend(fontsize=8, loc="upper left")
        ax.set_facecolor(bg_colour)

    for idx in range(n_drivers, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle(
        f"DML Stage 2 - {config['division_name']}: Driver vs Orders Residuals\n"
        "Blue = expected sign | Red = unexpected sign",
        fontsize=11,
        y=1.01,
    )

    plt.tight_layout()

    output_path = (
        output_dir
        / f"DML_elasticity_scatter_{config['division_name'].replace(' ', '_')}.png"
    )

    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)

    print(f"Plot saved: {output_path}")


# =============================================================================
# DATA QUALITY DIAGNOSTICS
# =============================================================================


def run_data_quality_diagnostics(
    train_df,
    test_df,
    drivers,
    stage1_results,
    diagnostics,
    min_var_ratio,
    max_adf_pvalue,
    config,
):
    """Print pre-stage and Stage 1 diagnostic information."""
    print("=== Pre-Stage Data Quality Checks ===")

    if test_df.empty:
        y_var_ratio = None
        y_mean_shift = None
        print("  Skipped - entire available period is used for training.")

    else:
        thresholds = config.get("data_quality_thresholds", {})

        y_var_ratio = np.var(test_df["Y"].values) / (
            np.var(train_df["Y"].values) + 1e-10
        )

        y_mean_shift = np.mean(
            np.abs(test_df["Y"].values - np.mean(train_df["Y"].values))
        ) / (np.std(train_df["Y"].values) + 1e-10)

        var_status = (
            "PASS"
            if y_var_ratio > thresholds.get("min_y_var_ratio", 0.30)
            else "FAIL - test period unusually calm"
        )

        shift_status = (
            "PASS"
            if y_mean_shift < thresholds.get("max_y_mean_shift", 0.50)
            else "WARN - test period looks different from train"
        )

        print(f"  Y variance ratio (test/train) : {y_var_ratio:.3f}  {var_status}")
        print(f"  Y distribution shift          : {y_mean_shift:.3f}  {shift_status}")

    print("\n=== Stage 1 Gate (per driver) ===")

    for driver_label in drivers:
        result = stage1_results[driver_label]

        both = (
            result["var_ratio"] >= min_var_ratio
            and result["adf_pvalue"] < max_adf_pvalue
        )

        one = (
            result["var_ratio"] >= min_var_ratio
            or result["adf_pvalue"] < max_adf_pvalue
        )

        gate = "PASS" if both else ("WARN" if one else "FAIL")

        print(
            f"{driver_label:<15} "
            f"VarRatio={result['var_ratio']:.3f} "
            f"ADF-p={result['adf_pvalue']:.3f} "
            f"Model={result['model_used']} "
            f"Gate={gate}"
        )

    print("\n=== Final Elasticity Summary ===")

    summary_df = pd.DataFrame.from_dict(diagnostics, orient="index")

    print(
        summary_df[
            [
                "elasticity",
                "sign_ok",
                "r2_test",
                "stage1_model",
                "var_ratio",
                "adf_pvalue",
                "method",
            ]
        ].to_string()
    )


# =============================================================================
# BACKTEST
# =============================================================================


def run_backtest(
    train_df,
    stage1_results,
    stage2_results,
    Y_residuals_train,
    config,
    output_dir,
):
    """Run Stage 2 training backtest."""
    joint_drivers = stage2_results["joint_drivers"]
    pw_drivers = stage2_results["pw_drivers"]
    T_residuals_train = stage2_results["T_residuals_train"]
    final_elasticities = stage2_results["final_elasticities"]
    diagnostics = stage2_results["diagnostics"]

    Y_fitted = T_residuals_train @ np.array(
        [final_elasticities[d] for d in joint_drivers]
    )

    for d in pw_drivers:
        Y_fitted += (
            diagnostics[d]["e_low"] * stage1_results[d]["train_residuals_low"]
            + diagnostics[d]["e_high"] * stage1_results[d]["train_residuals_high"]
        )

    r2_backtest = r2_score(Y_residuals_train, Y_fitted)

    fig, ax = plt.subplots(figsize=(13, 4))

    ax.plot(
        train_df["date"],
        Y_residuals_train,
        label="Actual (Y residual)",
        lw=2,
        color="#1565C0",
    )

    ax.plot(
        train_df["date"],
        Y_fitted,
        label=f"Model fit R²={r2_backtest:.3f}",
        lw=1.5,
        ls="--",
        color="#C62828",
    )

    ax.axhline(0, color="gray", lw=0.5)
    ax.set_xlabel("Date")
    ax.set_ylabel("YoY log-diff residual")
    ax.legend(fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    plt.xticks(rotation=30)
    plt.tight_layout()

    backtest_path = (
        output_dir / f"DML_backtest_{config['division_name'].replace(' ', '_')}.png"
    )

    plt.savefig(backtest_path, dpi=120, bbox_inches="tight")
    plt.close(fig)

    print(f"Back-test R² (training residuals): {r2_backtest:.4f}")

    return r2_backtest


# =============================================================================
# CONTIGUOUS TIME-SERIES SUBSAMPLING
# =============================================================================


def run_contiguous_subsampling(
    stage1_results,
    stage2_results,
    Y_residuals_train,
    config,
):
    """Run contiguous time-series subsampling and calculate CIs."""
    bootstrap_config = config.get("bootstrap", {})
    bootstrap_ci = config.get("bootstrap_ci", {})

    joint_drivers = stage2_results["joint_drivers"]
    pw_drivers = stage2_results["pw_drivers"]
    T_residuals_train = stage2_results["T_residuals_train"]
    final_elasticities = stage2_results["final_elasticities"]
    diagnostics = stage2_results["diagnostics"]
    best_alpha = stage2_results["best_alpha"]

    n_train = len(Y_residuals_train)

    sample_fraction = bootstrap_config.get("sample_fraction", 0.75)
    max_iterations = bootstrap_config.get("n_resamples", 200)
    random_seed = bootstrap_config.get("random_seed", 42)

    ci_lower = bootstrap_ci.get("lower_percentile", 2.5)
    ci_upper = bootstrap_ci.get("upper_percentile", 97.5)
    ci_level = ci_upper - ci_lower

    sample_size = int(np.ceil(n_train * sample_fraction))

    if sample_size < 2:
        raise ValueError(
            "At least 2 observations are required for contiguous subsampling."
        )

    n_possible_slices = n_train - sample_size + 1
    n_bootstrap = min(max_iterations, n_possible_slices)

    possible_slices = [
        np.arange(start_idx, start_idx + sample_size)
        for start_idx in range(n_possible_slices)
    ]

    rng = np.random.default_rng(random_seed)

    selected_slice_ids = rng.choice(
        len(possible_slices),
        size=n_bootstrap,
        replace=False,
    )

    bootstrap_coefs = np.zeros((n_bootstrap, len(joint_drivers)))
    bootstrap_pw = {d: np.zeros((n_bootstrap, 2)) for d in pw_drivers}

    for b, slice_id in enumerate(selected_slice_ids):
        sample_idx = possible_slices[slice_id]

        if joint_drivers:
            bootstrap_coefs[b] = (
                Ridge(
                    alpha=best_alpha,
                    fit_intercept=False,
                )
                .fit(
                    T_residuals_train[sample_idx],
                    Y_residuals_train[sample_idx],
                )
                .coef_
            )

        for d in pw_drivers:
            T_pw = np.column_stack(
                [
                    stage1_results[d]["train_residuals_low"],
                    stage1_results[d]["train_residuals_high"],
                ]
            )

            bootstrap_pw[d][b] = (
                Ridge(
                    alpha=best_alpha,
                    fit_intercept=False,
                )
                .fit(
                    T_pw[sample_idx],
                    Y_residuals_train[sample_idx],
                )
                .coef_
            )

    bootstrap_rows = []

    for i, driver_label in enumerate(joint_drivers):
        distribution = bootstrap_coefs[:, i]
        standard_error = np.std(distribution, ddof=1)
        lower, upper = np.percentile(distribution, [ci_lower, ci_upper])

        bootstrap_rows.append(
            {
                "driver": driver_label,
                "slope_type": "single",
                "elasticity": final_elasticities[driver_label],
                "standard_error": standard_error,
                "ci_level": ci_level,
                "ci_lower_percentile": ci_lower,
                "ci_upper_percentile": ci_upper,
                "ci_lower": lower,
                "ci_upper": upper,
                "significant": lower > 0 or upper < 0,
                "sample_fraction": sample_fraction,
                "sample_size": sample_size,
                "n_slices": n_bootstrap,
            }
        )

    for d in pw_drivers:
        for slope_index, slope_name in enumerate(["low", "high"]):
            distribution = bootstrap_pw[d][:, slope_index]
            standard_error = np.std(distribution, ddof=1)
            lower, upper = np.percentile(distribution, [ci_lower, ci_upper])

            elasticity = (
                diagnostics[d]["e_low"]
                if slope_name == "low"
                else diagnostics[d]["e_high"]
            )

            bootstrap_rows.append(
                {
                    "driver": d,
                    "slope_type": slope_name,
                    "elasticity": elasticity,
                    "standard_error": standard_error,
                    "ci_level": ci_level,
                    "ci_lower_percentile": ci_lower,
                    "ci_upper_percentile": ci_upper,
                    "ci_lower": lower,
                    "ci_upper": upper,
                    "significant": lower > 0 or upper < 0,
                    "sample_fraction": sample_fraction,
                    "sample_size": sample_size,
                    "n_slices": n_bootstrap,
                }
            )

    bootstrap_df = pd.DataFrame(bootstrap_rows)

    print(
        f"\n=== {ci_level:.0f}% CI from 75% contiguous slices " f"(n={n_bootstrap}) ==="
    )

    for _, row in bootstrap_df.iterrows():
        print(
            f"{row['driver']:<20} "
            f"{row['elasticity']:>12.4f} "
            f"{row['standard_error']:>12.4f} "
            f"{row['ci_lower']:>12.4f} "
            f"{row['ci_upper']:>12.4f}"
        )

    return {
        "bootstrap_df": bootstrap_df,
        "bootstrap_coefs": bootstrap_coefs,
        "bootstrap_pw": bootstrap_pw,
        "sample_fraction": sample_fraction,
        "sample_size": sample_size,
        "n_bootstrap": n_bootstrap,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
    }


# =============================================================================
# ROLLING STABILITY
# =============================================================================


def run_rolling_stability(
    stage1_results,
    stage2_results,
    Y_residuals_train,
    config,
):
    """Run rolling Stage 2 stability analysis."""
    joint_drivers = stage2_results["joint_drivers"]
    pw_drivers = stage2_results["pw_drivers"]
    T_residuals_train = stage2_results["T_residuals_train"]
    alphas_to_try = stage2_results["alphas_to_try"]
    val_n = stage2_results["val_n"]

    rolling_thresholds = config.get("rolling_stability_thresholds", {})
    ok_swing = rolling_thresholds.get("ok_swing", 0.15)
    warn_swing = rolling_thresholds.get("warn_swing", 0.35)

    window_sizes = config.get("rolling_windows", [30, 33, 36])
    window_coefs = {}
    window_coefs_pw = {d: {} for d in pw_drivers}

    for window in window_sizes:
        T_window = T_residuals_train[-window:]
        Y_window = Y_residuals_train[-window:]

        T_fit_w = T_window[:-val_n]
        Y_fit_w = Y_window[:-val_n]
        T_val_w = T_window[-val_n:]
        Y_val_w = Y_window[-val_n:]

        r2_per_alpha = [
            r2_score(
                Y_val_w,
                Ridge(alpha=alpha, fit_intercept=False)
                .fit(T_fit_w, Y_fit_w)
                .predict(T_val_w),
            )
            for alpha in alphas_to_try
        ]

        alpha_w = float(alphas_to_try[int(np.argmax(r2_per_alpha))])

        window_coefs[window] = (
            Ridge(
                alpha=alpha_w,
                fit_intercept=False,
            )
            .fit(
                T_window,
                Y_window,
            )
            .coef_
        )

        for d in pw_drivers:
            T_pw_w = np.column_stack(
                [
                    stage1_results[d]["train_residuals_low"][-window:],
                    stage1_results[d]["train_residuals_high"][-window:],
                ]
            )

            window_coefs_pw[d][window] = (
                Ridge(
                    alpha=alpha_w,
                    fit_intercept=False,
                )
                .fit(
                    T_pw_w,
                    Y_residuals_train[-window:],
                )
                .coef_
            )

    rolling_rows = []

    for i, driver_label in enumerate(joint_drivers):
        values = [window_coefs[w][i] for w in window_sizes]
        swing = max(values) - min(values)

        row = {
            "driver": driver_label,
            "slope_type": "single",
            "swing": swing,
            "status": get_stability_status(swing, ok_swing, warn_swing),
        }

        for window, value in zip(window_sizes, values):
            row[f"coef_{window}"] = value

        rolling_rows.append(row)

    for d in pw_drivers:
        for slope_idx, slope_name in enumerate(["low", "high"]):
            values = [window_coefs_pw[d][w][slope_idx] for w in window_sizes]
            swing = max(values) - min(values)

            row = {
                "driver": d,
                "slope_type": slope_name,
                "swing": swing,
                "status": get_stability_status(swing, ok_swing, warn_swing),
            }

            for window, value in zip(window_sizes, values):
                row[f"coef_{window}"] = value

            rolling_rows.append(row)

    return {
        "window_sizes": window_sizes,
        "window_coefs": window_coefs,
        "window_coefs_pw": window_coefs_pw,
        "rolling_df": pd.DataFrame(rolling_rows),
        "ok_swing": ok_swing,
        "warn_swing": warn_swing,
    }


# =============================================================================
# DRIVER SELECTION
# =============================================================================


def run_driver_selection(
    stage1_results,
    stage2_results,
    bootstrap_results,
    rolling_results,
    train_df,
    config,
    output_dir,
):
    """Run residual correlation, VIF and KEEP/REVIEW/DROP framework."""
    driver_labels = stage2_results["joint_drivers"]
    pw_drivers = stage2_results["pw_drivers"]
    T_residuals_train = stage2_results["T_residuals_train"]
    final_elasticities = stage2_results["final_elasticities"]
    diagnostics = stage2_results["diagnostics"]

    bootstrap_coefs = bootstrap_results["bootstrap_coefs"]
    bootstrap_pw = bootstrap_results["bootstrap_pw"]
    ci_lower = bootstrap_results["ci_lower"]
    ci_upper = bootstrap_results["ci_upper"]

    window_sizes = rolling_results["window_sizes"]
    window_coefs = rolling_results["window_coefs"]
    window_coefs_pw = rolling_results["window_coefs_pw"]
    ok_swing = rolling_results["ok_swing"]
    warn_swing = rolling_results["warn_swing"]

    thresholds = config.get("decision_thresholds", {})
    vif_config = config.get("vif_config", {})
    piecewise_thresholds = config.get("piecewise_decision_thresholds", {})

    CORR_FLAG_HIGH = thresholds.get("corr_flag_high", 0.70)
    CORR_FLAG_STRONG = thresholds.get("corr_flag_strong", 0.85)
    VIF_WARN = thresholds.get("vif_warn", 5.0)
    VIF_ACTION = thresholds.get("vif_action", 10.0)
    ELAST_STRONG = thresholds.get("elast_strong", 0.50)
    ELAST_WEAK = thresholds.get("elast_weak", 0.10)

    NEAR_ZERO_BOTH = piecewise_thresholds.get("near_zero_both", 0.01)
    HIGH_SLOPE_ZERO = piecewise_thresholds.get("high_slope_zero", 1e-6)

    # -------------------------------------------------------------------------
    # Correlation
    # -------------------------------------------------------------------------

    resid_df = pd.DataFrame(T_residuals_train, columns=driver_labels)
    corr_mat = resid_df.corr()
    n_d = len(driver_labels)

    fig, ax = plt.subplots(figsize=(max(5, n_d * 1.5), max(4, n_d * 1.3)))

    image = ax.imshow(
        corr_mat.values,
        vmin=-1,
        vmax=1,
        cmap="RdBu_r",
        aspect="auto",
    )

    plt.colorbar(image, ax=ax, shrink=0.8)

    ax.set_xticks(range(n_d))
    ax.set_xticklabels(driver_labels, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(n_d))
    ax.set_yticklabels(driver_labels, fontsize=9)

    for i in range(n_d):
        for j in range(n_d):
            value = corr_mat.values[i, j]

            ax.text(
                j,
                i,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if abs(value) > 0.6 else "black",
            )

    plt.tight_layout()

    residual_corr_path = (
        output_dir
        / f"DML_residual_corr_{config['division_name'].replace(' ', '_')}.png"
    )

    plt.savefig(residual_corr_path, dpi=120, bbox_inches="tight")
    plt.close(fig)

    flagged_pairs = []

    for (_, a), (_, b) in combinations(enumerate(driver_labels), 2):
        correlation = corr_mat.loc[a, b]

        if abs(correlation) >= CORR_FLAG_HIGH:
            severity = (
                "STRONG -- drop one"
                if abs(correlation) >= CORR_FLAG_STRONG
                else "HIGH -- review"
            )

            flagged_pairs.append((a, b, correlation, severity))

    # -------------------------------------------------------------------------
    # VIF
    # -------------------------------------------------------------------------

    vif_scores = {}

    for i, driver in enumerate(driver_labels):
        others = [j for j in range(n_d) if j != i]
        X_other = T_residuals_train[:, others]
        y_i = T_residuals_train[:, i]

        vif_model = Ridge(
            alpha=vif_config.get("ridge_alpha", 1e-4),
            fit_intercept=True,
        )

        vif_model.fit(X_other, y_i)
        r2_vif = r2_score(y_i, vif_model.predict(X_other))
        r2_vif = min(r2_vif, vif_config.get("max_r2_cap", 0.9999))

        vif_scores[driver] = 1.0 / (1.0 - r2_vif)

    # -------------------------------------------------------------------------
    # Elasticity tiers
    # -------------------------------------------------------------------------

    elast_tiers = {}

    for driver in driver_labels:
        elasticity = final_elasticities[driver]

        if abs(elasticity) > ELAST_STRONG:
            tier = "STRONG"
        elif abs(elasticity) >= ELAST_WEAK:
            tier = "MODERATE"
        else:
            tier = "WEAK"

        elast_tiers[driver] = tier

    # -------------------------------------------------------------------------
    # Bootstrap significance + rolling stability
    # -------------------------------------------------------------------------

    bootstrap_sig = {}
    rolling_stable = {}

    for i, driver in enumerate(driver_labels):
        lower, upper = np.percentile(
            bootstrap_coefs[:, i],
            [ci_lower, ci_upper],
        )

        bootstrap_sig[driver] = lower > 0 or upper < 0

        swing = max(window_coefs[w][i] for w in window_sizes) - min(
            window_coefs[w][i] for w in window_sizes
        )

        rolling_stable[driver] = get_stability_status(
            swing,
            ok_swing,
            warn_swing,
        )

    # -------------------------------------------------------------------------
    # Strong collinearity drops
    # -------------------------------------------------------------------------

    drop_from_collinear = set()

    for a, b, correlation, _ in flagged_pairs:
        if abs(correlation) >= CORR_FLAG_STRONG:
            vr_a = diagnostics[a]["var_ratio"]
            vr_b = diagnostics[b]["var_ratio"]
            drop_from_collinear.add(b if vr_a >= vr_b else a)

    # -------------------------------------------------------------------------
    # Decision matrix
    # -------------------------------------------------------------------------

    verdicts = {}
    reasons = {}

    for driver in driver_labels:
        elasticity = final_elasticities[driver]
        tier = elast_tiers[driver]
        vif = vif_scores[driver]
        significant = bootstrap_sig[driver]
        stability = rolling_stable[driver]
        sign_ok = diagnostics[driver]["sign_ok"]

        in_high_pair = any(
            driver in (a, b)
            for a, b, correlation, _ in flagged_pairs
            if abs(correlation) >= CORR_FLAG_HIGH
        )

        if driver in drop_from_collinear:
            verdict = "DROP"

            partner = next(
                a if b == driver else b
                for a, b, correlation, _ in flagged_pairs
                if abs(correlation) >= CORR_FLAG_STRONG and driver in (a, b)
            )

            reason = f"D1: strongly collinear with {partner}, lower VarRatio"

        elif not significant and tier == "WEAK" and stability == "UNSTABLE":
            verdict = "DROP"
            reason = "D4: weak + not significant + unstable"

        elif vif > VIF_ACTION:
            verdict = "REVIEW"
            reason = f"D3: VIF={vif:.1f} > 10"

        elif in_high_pair and driver not in drop_from_collinear:
            r_max = max(
                abs(correlation)
                for a, b, correlation, _ in flagged_pairs
                if driver in (a, b)
            )

            verdict = "REVIEW"
            reason = f"D2: pairwise |r|={r_max:.2f} -- test sensitivity"

        elif not significant and tier == "WEAK":
            verdict = "REVIEW"
            reason = "D5: weak + CI crosses zero -- directional only"

        elif tier == "STRONG" and stability == "UNSTABLE":
            verdict = "REVIEW"
            reason = "D6: high elasticity but unstable"

        elif not sign_ok:
            verdict = "REVIEW"
            reason = "D7: unexpected sign"

        else:
            verdict = "KEEP"
            reason = "significant, stable, sign OK, low multicollinearity"

        verdicts[driver] = verdict
        reasons[driver] = reason

    # -------------------------------------------------------------------------
    # Piecewise decisions
    # -------------------------------------------------------------------------

    piecewise_verdicts = {}
    piecewise_reasons = {}

    for driver in pw_drivers:
        e_low = diagnostics[driver]["e_low"]
        e_high = diagnostics[driver]["e_high"]

        lo_low, hi_low = np.percentile(
            bootstrap_pw[driver][:, 0],
            [ci_lower, ci_upper],
        )

        lo_high, hi_high = np.percentile(
            bootstrap_pw[driver][:, 1],
            [ci_lower, ci_upper],
        )

        low_sig = lo_low > 0 or hi_low < 0
        high_sig = lo_high > 0 or hi_high < 0

        low_values = [window_coefs_pw[driver][w][0] for w in window_sizes]

        high_values = [window_coefs_pw[driver][w][1] for w in window_sizes]

        low_stable = get_stability_status(
            max(low_values) - min(low_values),
            ok_swing,
            warn_swing,
        )

        high_stable = get_stability_status(
            max(high_values) - min(high_values),
            ok_swing,
            warn_swing,
        )

        sign_ok = diagnostics[driver]["sign_ok"]

        if not sign_ok:
            verdict = "REVIEW"
            reason = "unexpected sign"

        elif abs(e_low) < NEAR_ZERO_BOTH and abs(e_high) < NEAR_ZERO_BOTH:
            verdict = "DROP"
            reason = "both low and high slopes are near zero"

        elif not low_sig and not high_sig:
            verdict = "REVIEW"
            reason = "both low and high CI cross zero"

        elif high_stable == "UNSTABLE" or low_stable == "UNSTABLE":
            verdict = "REVIEW"
            reason = "piecewise slope unstable"

        elif abs(e_high) < HIGH_SLOPE_ZERO:
            verdict = "REVIEW"
            reason = "above-knot slope not identified / zero"

        else:
            verdict = "KEEP"
            reason = "piecewise slopes acceptable"

        piecewise_verdicts[driver] = verdict
        piecewise_reasons[driver] = reason

        knot = float(config["piecewise_drivers"][driver])
        T_log = train_df[f"{driver}_T"].values

        print(f"\n=== Knot Diagnostics: {driver} ===")
        print(f"knot = {knot}")
        print(f"min log-YoY T = {np.nanmin(T_log):.4f}")
        print(f"max log-YoY T = {np.nanmax(T_log):.4f}")
        print(f"rows above knot = {(T_log > knot).sum()} out of {len(T_log)}")
        print(f"share above knot = {(T_log > knot).mean():.2%}")

    keep_list = [d for d, verdict in verdicts.items() if verdict == "KEEP"]
    review_list = [d for d, verdict in verdicts.items() if verdict == "REVIEW"]
    drop_list = [d for d, verdict in verdicts.items() if verdict == "DROP"]

    return {
        "corr_mat": corr_mat,
        "flagged_pairs": flagged_pairs,
        "vif_scores": vif_scores,
        "elast_tiers": elast_tiers,
        "bootstrap_sig": bootstrap_sig,
        "rolling_stable": rolling_stable,
        "verdicts": verdicts,
        "reasons": reasons,
        "piecewise_verdicts": piecewise_verdicts,
        "piecewise_reasons": piecewise_reasons,
        "keep_list": keep_list,
        "review_list": review_list,
        "drop_list": drop_list,
        "CORR_FLAG_HIGH": CORR_FLAG_HIGH,
        "CORR_FLAG_STRONG": CORR_FLAG_STRONG,
        "VIF_WARN": VIF_WARN,
        "VIF_ACTION": VIF_ACTION,
    }


# =============================================================================
# REPORT TABLES
# =============================================================================


def build_report_tables(
    combination_id,
    config,
    driver_confounders,
    stage1_results,
    stage2_results,
    bootstrap_results,
    rolling_results,
    selection_results,
    train_df,
    test_df,
    Y_residuals_train,
    Y_residuals_test,
    var_ratio_y,
    adf_pvalue_y,
    gate_y,
    r2_backtest,
):
    """Create all DataFrames used by the Excel report."""
    diagnostics = stage2_results["diagnostics"]
    final_elasticities = stage2_results["final_elasticities"]
    joint_drivers = stage2_results["joint_drivers"]
    drivers = list(stage1_results.keys())

    bootstrap_df = bootstrap_results["bootstrap_df"]

    config_df = pd.DataFrame(
        {
            "key": list(config.keys()),
            "value": [str(value) for value in config.values()],
        }
    )

    confounders_df = pd.DataFrame(
        [
            {
                "driver": driver_label,
                "confounders": ", ".join(confounders),
            }
            for driver_label, confounders in driver_confounders.items()
        ]
    )

    stage1_summary_df = pd.DataFrame(
        [
            {
                "driver": driver_label,
                "model_used": result["model_used"],
                "var_ratio": result["var_ratio"],
                "adf_pvalue": result["adf_pvalue"],
            }
            for driver_label, result in stage1_results.items()
        ]
    )

    outcome_stage1y_df = pd.DataFrame(
        [
            {
                "var_ratio_y": var_ratio_y,
                "adf_pvalue_y": adf_pvalue_y,
                "gate_y": gate_y.strip(),
            }
        ]
    )

    elasticity_df = (
        pd.DataFrame.from_dict(diagnostics, orient="index")
        .reset_index()
        .rename(columns={"index": "driver"})
    )

    bootstrap_merge = bootstrap_df[
        [
            "driver",
            "standard_error",
            "ci_lower",
            "ci_upper",
            "significant",
        ]
    ].copy()

    elasticity_df = elasticity_df.merge(
        bootstrap_merge,
        on="driver",
        how="left",
    )

    elasticity_df["combination_id"] = combination_id
    elasticity_df["sample_fraction"] = bootstrap_results["sample_fraction"]
    elasticity_df["sample_size"] = bootstrap_results["sample_size"]
    elasticity_df["n_slices"] = bootstrap_results["n_bootstrap"]

    backtest_df = pd.DataFrame(
        [
            {
                "r2_backtest": r2_backtest,
                "r2_train_joint": stage2_results["r2_train"],
                "r2_test_joint": stage2_results["r2_test"],
                "best_alpha": stage2_results["best_alpha"],
            }
        ]
    )

    corr_matrix_df = selection_results["corr_mat"].copy()
    corr_matrix_df.index.name = "driver"

    pairwise_corr_rows = []

    for (_, a), (_, b) in combinations(enumerate(joint_drivers), 2):
        correlation = selection_results["corr_mat"].loc[a, b]

        severity = (
            "STRONG -- drop one"
            if abs(correlation) >= selection_results["CORR_FLAG_STRONG"]
            else (
                "HIGH -- review"
                if abs(correlation) >= selection_results["CORR_FLAG_HIGH"]
                else "OK"
            )
        )

        pairwise_corr_rows.append(
            {
                "driver_1": a,
                "driver_2": b,
                "correlation": correlation,
                "severity": severity,
            }
        )

    pairwise_corr_df = pd.DataFrame(pairwise_corr_rows)

    vif_df = pd.DataFrame(
        [
            {
                "driver": driver,
                "vif": selection_results["vif_scores"][driver],
                "status": (
                    "ACTION NEEDED"
                    if selection_results["vif_scores"][driver]
                    > selection_results["VIF_ACTION"]
                    else (
                        "WARN"
                        if selection_results["vif_scores"][driver]
                        > selection_results["VIF_WARN"]
                        else "OK"
                    )
                ),
            }
            for driver in joint_drivers
        ]
    )

    elasticity_tiers_df = pd.DataFrame(
        [
            {
                "driver": driver,
                "elasticity": final_elasticities[driver],
                "tier": selection_results["elast_tiers"][driver],
            }
            for driver in joint_drivers
        ]
    )

    decision_matrix_df = pd.DataFrame(
        [
            {
                "driver": driver,
                "elasticity": final_elasticities[driver],
                "tier": selection_results["elast_tiers"][driver],
                "vif": selection_results["vif_scores"][driver],
                "significant": selection_results["bootstrap_sig"][driver],
                "stability": selection_results["rolling_stable"][driver],
                "sign_ok": diagnostics[driver]["sign_ok"],
                "verdict": selection_results["verdicts"][driver],
                "reason": selection_results["reasons"][driver],
            }
            for driver in joint_drivers
        ]
    )

    actionable_rows = []

    for driver in selection_results["keep_list"]:
        actionable_rows.append(
            {
                "group": "KEEP",
                "driver": driver,
                "elasticity": final_elasticities[driver],
                "reason": "",
            }
        )

    for driver in selection_results["review_list"]:
        actionable_rows.append(
            {
                "group": "REVIEW",
                "driver": driver,
                "elasticity": final_elasticities[driver],
                "reason": selection_results["reasons"][driver],
            }
        )

    for driver in selection_results["drop_list"]:
        actionable_rows.append(
            {
                "group": "DROP",
                "driver": driver,
                "elasticity": final_elasticities[driver],
                "reason": selection_results["reasons"][driver],
            }
        )

    actionable_df = pd.DataFrame(actionable_rows)

    train_residuals_df = pd.DataFrame(
        {
            "date": train_df["date"].values,
            "Y_residual_train": Y_residuals_train,
        }
    )

    for driver_label in drivers:
        train_residuals_df[f"{driver_label}_residual_train"] = stage1_results[
            driver_label
        ]["train_residuals"]

    if test_df.empty:
        test_residuals_df = None

    else:
        test_residuals_df = pd.DataFrame(
            {
                "date": test_df["date"].values,
                "Y_residual_test": Y_residuals_test,
            }
        )

        for driver_label in drivers:
            test_residuals_df[f"{driver_label}_residual_test"] = stage1_results[
                driver_label
            ]["test_residuals"]

    return {
        "config": config_df,
        "confounders": confounders_df,
        "stage1_summary": stage1_summary_df,
        "outcome_stage1y": outcome_stage1y_df,
        "elasticities": elasticity_df,
        "backtest": backtest_df,
        "bootstrap_ci": bootstrap_df,
        "rolling_stability": rolling_results["rolling_df"],
        "corr_matrix": corr_matrix_df,
        "pairwise_corr": pairwise_corr_df,
        "vif": vif_df,
        "elasticity_tiers": elasticity_tiers_df,
        "decision_matrix": decision_matrix_df,
        "actionable_summary": actionable_df,
        "train_residuals": train_residuals_df,
        "test_residuals": test_residuals_df,
    }


# =============================================================================
# EXCEL REPORT
# =============================================================================


def write_excel_report(report_tables, output_excel):
    """Write one combination report."""
    with pd.ExcelWriter(output_excel, engine="openpyxl") as writer:
        report_tables["config"].to_excel(writer, sheet_name="config", index=False)
        report_tables["confounders"].to_excel(
            writer, sheet_name="confounders", index=False
        )
        report_tables["stage1_summary"].to_excel(
            writer, sheet_name="stage1_summary", index=False
        )
        report_tables["outcome_stage1y"].to_excel(
            writer, sheet_name="outcome_stage1y", index=False
        )
        report_tables["elasticities"].to_excel(
            writer, sheet_name="elasticities", index=False
        )
        report_tables["backtest"].to_excel(writer, sheet_name="backtest", index=False)
        report_tables["bootstrap_ci"].to_excel(
            writer, sheet_name="bootstrap_ci", index=False
        )
        report_tables["rolling_stability"].to_excel(
            writer, sheet_name="rolling_stability", index=False
        )
        report_tables["corr_matrix"].to_excel(writer, sheet_name="corr_matrix")
        report_tables["pairwise_corr"].to_excel(
            writer, sheet_name="pairwise_corr", index=False
        )
        report_tables["vif"].to_excel(writer, sheet_name="vif", index=False)
        report_tables["elasticity_tiers"].to_excel(
            writer, sheet_name="elasticity_tiers", index=False
        )
        report_tables["decision_matrix"].to_excel(
            writer, sheet_name="decision_matrix", index=False
        )
        report_tables["actionable_summary"].to_excel(
            writer, sheet_name="actionable_summary", index=False
        )
        report_tables["train_residuals"].to_excel(
            writer, sheet_name="train_residuals", index=False
        )

        if report_tables["test_residuals"] is not None:
            report_tables["test_residuals"].to_excel(
                writer,
                sheet_name="test_residuals",
                index=False,
            )

    print(f"\nExcel report saved to: {output_excel}")


# =============================================================================
# RUN ONE COMBINATION
# =============================================================================


def run_combination(
    combination_id,
    combo,
    all_drivers,
    global_driver_confounders,
    global_y_confounders,
    yoy_df,
    train_df,
    test_df,
    config,
    output_dir,
    cache_root,
    source_data_hash,
    global_driver_residual_cache,
    global_y_residual_cache,
    min_var_ratio,
    max_adf_pvalue,
):
    """Run the complete DML workflow for one driver combination."""
    print("\n\n" + "=" * 120)
    print(f"RUNNING {combination_id}")
    print("=" * 120)

    drivers = {key: value for key, value in all_drivers.items() if key in combo}
    driver_confounders = {d: global_driver_confounders[d] for d in drivers}
    y_confounders = global_y_confounders.copy()

    print(f"Drivers in combination: {list(drivers.keys())}")
    print(f"Y confounders: {y_confounders}")

    # -------------------------------------------------------------------------
    # Stage 1T
    # -------------------------------------------------------------------------

    print("\n[Step 1] Running Stage 1T...")

    stage1_results = run_stage1_drivers(
        combination_id=combination_id,
        drivers=drivers,
        driver_confounders=driver_confounders,
        train_df=train_df,
        test_df=test_df,
        yoy_df=yoy_df,
        config=config,
        cache_root=cache_root,
        source_data_hash=source_data_hash,
        global_driver_residual_cache=global_driver_residual_cache,
        min_var_ratio=min_var_ratio,
        max_adf_pvalue=max_adf_pvalue,
    )

    # -------------------------------------------------------------------------
    # Stage 1Y
    # -------------------------------------------------------------------------

    print("\n[Step 2] Running Stage 1Y...")

    (
        Y_residuals_train,
        Y_residuals_test,
        var_ratio_y,
        adf_pvalue_y,
        gate_y,
    ) = run_stage1_y(
        combination_id=combination_id,
        y_confounders=y_confounders,
        train_df=train_df,
        test_df=test_df,
        yoy_df=yoy_df,
        config=config,
        global_y_residual_cache=global_y_residual_cache,
        min_var_ratio=min_var_ratio,
        max_adf_pvalue=max_adf_pvalue,
    )

    # -------------------------------------------------------------------------
    # Stage 2
    # -------------------------------------------------------------------------

    print("\n[Step 3] Estimating Stage 2 elasticities...")

    stage2_results = run_stage2(
        drivers=drivers,
        stage1_results=stage1_results,
        Y_residuals_train=Y_residuals_train,
        Y_residuals_test=Y_residuals_test,
        config=config,
        train_only=test_df.empty,
    )

    print_elasticity_results(stage2_results)

    # -------------------------------------------------------------------------
    # Plots
    # -------------------------------------------------------------------------

    print("\n[Step 4] Creating elasticity scatter plots...")

    create_elasticity_scatter_plot(
        stage1_results=stage1_results,
        diagnostics=stage2_results["diagnostics"],
        Y_residuals_train=Y_residuals_train,
        drivers=drivers,
        config=config,
        output_dir=output_dir,
    )

    # -------------------------------------------------------------------------
    # Diagnostics
    # -------------------------------------------------------------------------

    print("\n[Step 5] Printing diagnostics...")

    run_data_quality_diagnostics(
        train_df=train_df,
        test_df=test_df,
        drivers=drivers,
        stage1_results=stage1_results,
        diagnostics=stage2_results["diagnostics"],
        min_var_ratio=min_var_ratio,
        max_adf_pvalue=max_adf_pvalue,
        config=config,
    )

    # -------------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------------

    print("\n[Step 6] Running validation checks...")

    r2_backtest = run_backtest(
        train_df=train_df,
        stage1_results=stage1_results,
        stage2_results=stage2_results,
        Y_residuals_train=Y_residuals_train,
        config=config,
        output_dir=output_dir,
    )

    bootstrap_results = run_contiguous_subsampling(
        stage1_results=stage1_results,
        stage2_results=stage2_results,
        Y_residuals_train=Y_residuals_train,
        config=config,
    )

    rolling_results = run_rolling_stability(
        stage1_results=stage1_results,
        stage2_results=stage2_results,
        Y_residuals_train=Y_residuals_train,
        config=config,
    )

    # -------------------------------------------------------------------------
    # Driver selection
    # -------------------------------------------------------------------------

    print("\n[Step 7] Evaluating driver selection framework...")

    selection_results = run_driver_selection(
        stage1_results=stage1_results,
        stage2_results=stage2_results,
        bootstrap_results=bootstrap_results,
        rolling_results=rolling_results,
        train_df=train_df,
        config=config,
        output_dir=output_dir,
    )

    # -------------------------------------------------------------------------
    # Report
    # -------------------------------------------------------------------------

    report_tables = build_report_tables(
        combination_id=combination_id,
        config=config,
        driver_confounders=driver_confounders,
        stage1_results=stage1_results,
        stage2_results=stage2_results,
        bootstrap_results=bootstrap_results,
        rolling_results=rolling_results,
        selection_results=selection_results,
        train_df=train_df,
        test_df=test_df,
        Y_residuals_train=Y_residuals_train,
        Y_residuals_test=Y_residuals_test,
        var_ratio_y=var_ratio_y,
        adf_pvalue_y=adf_pvalue_y,
        gate_y=gate_y,
        r2_backtest=r2_backtest,
    )

    train_end_label = pd.Timestamp(config["train_end"]).strftime("%Y-%m-%d")

    output_excel = output_dir / (
        f"DML_full_report_"
        f"{config['division_name'].replace(' ', '_')}_"
        f"{combination_id}_"
        f"{train_end_label}.xlsx"
    )

    write_excel_report(report_tables, output_excel)

    return report_tables["elasticities"]


# =============================================================================
# COMBINATION COMPARISON
# =============================================================================


def write_combination_comparison(all_combination_results, config, output_dir):
    """Write final workbook containing results for every combination."""
    comparison_excel = (
        output_dir / f"DML_window_comparison_{config['division_name']}.xlsx"
    )

    master_summary_rows = []
    master_elasticity_rows = []

    with pd.ExcelWriter(comparison_excel, engine="openpyxl") as writer:
        for combo_id, combo_df in all_combination_results.items():
            master_elasticity_rows.append(combo_df.copy())

            combo_df.to_excel(
                writer,
                sheet_name=combo_id[:31],
                index=False,
            )

            master_summary_rows.append(
                {
                    "Combination": combo_id,
                    "Driver_Count": len(combo_df),
                    "Drivers": ", ".join(combo_df["driver"].astype(str)),
                }
            )

        all_combinations_df = pd.concat(
            master_elasticity_rows,
            ignore_index=True,
        )

        all_combinations_df.to_excel(
            writer,
            sheet_name="ALL_COMBINATIONS",
            index=False,
        )

        pd.DataFrame(master_summary_rows).to_excel(
            writer,
            sheet_name="MASTER_SUMMARY",
            index=False,
        )

    print(f"\nWindow comparison workbook saved:\n{comparison_excel}")


# =============================================================================
# MAIN
# =============================================================================


def main():
    """Run complete DML pipeline."""
    args = parse_args()
    config = load_config(args.config)

    # -------------------------------------------------------------------------
    # Configuration
    # -------------------------------------------------------------------------

    stage1_thresholds = config.get("stage1_thresholds", {})
    cache_config = config.get("stage1_cache", {})

    min_var_ratio = stage1_thresholds.get("min_var_ratio", 0.15)
    max_adf_pvalue = stage1_thresholds.get("max_adf_pvalue", 0.10)

    cache_enabled = cache_config.get("enabled", True)
    cache_subdir = cache_config.get("subdir", "_stage1_cache")

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    cache_root = output_dir / cache_subdir

    if cache_enabled:
        cache_root.mkdir(parents=True, exist_ok=True)
        print(f"Stage 1 cache directory: {cache_root.resolve()}")

    # -------------------------------------------------------------------------
    # Data
    # -------------------------------------------------------------------------

    df, master_confounder_train_df, train_df, test_df = prepare_data(config)

    all_drivers = config["drivers"]
    time_confs = config.get("time_confounders", ["trend"])

    driver_combinations = list(
        combinations(
            all_drivers.keys(),
            config["combination_size"],
        )
    )

    print(f"Created {len(driver_combinations)} driver combinations")

    print_dataset_summary(
        config,
        train_df,
        test_df,
        output_dir,
    )

    # -------------------------------------------------------------------------
    # GLOBAL CONFOUNDER SELECTION
    # -------------------------------------------------------------------------

    print("\n" + "=" * 120)
    print("GLOBAL CONFOUNDER SELECTION")
    print("=" * 120)

    (
        global_driver_confounders,
        global_y_confounders,
        global_yoy_df,
    ) = select_confounders(
        df=df,
        train_df=master_confounder_train_df,
        config=config,
        drivers=all_drivers,
        time_confs=time_confs,
        yoy_log_diff=lambda series: yoy_log_diff(
            series,
            config["data_freq"],
        ),
    )

    print("\nGlobal confounder selection completed.")

    # -------------------------------------------------------------------------
    # GLOBAL RESIDUAL CACHES
    # -------------------------------------------------------------------------

    global_driver_residual_cache = {}
    global_y_residual_cache = {}

    source_data_hash = calculate_file_hash(config["file"])

    # -------------------------------------------------------------------------
    # RUN COMBINATIONS
    # -------------------------------------------------------------------------

    all_combination_results = {}

    for combo_idx, combo in enumerate(driver_combinations, start=1):
        combination_id = f"COMB_{combo_idx:03d}"

        elasticity_df = run_combination(
            combination_id=combination_id,
            combo=combo,
            all_drivers=all_drivers,
            global_driver_confounders=global_driver_confounders,
            global_y_confounders=global_y_confounders,
            yoy_df=global_yoy_df,
            train_df=train_df,
            test_df=test_df,
            config=config,
            output_dir=output_dir,
            cache_root=cache_root,
            source_data_hash=source_data_hash,
            global_driver_residual_cache=global_driver_residual_cache,
            global_y_residual_cache=global_y_residual_cache,
            min_var_ratio=min_var_ratio,
            max_adf_pvalue=max_adf_pvalue,
        )

        all_combination_results[combination_id] = elasticity_df.copy()

        print(
            f"\nStored {combination_id} | "
            f"Total combinations stored = {len(all_combination_results)}"
        )

    # -------------------------------------------------------------------------
    # FINAL COMBINATION REPORT
    # -------------------------------------------------------------------------

    print("\nAll combinations finished.")
    print(f"Total combinations stored = {len(all_combination_results)}")

    print("\nALL COMBINATION KEYS")
    print(list(all_combination_results.keys()))
    print("COUNT =", len(all_combination_results))

    write_combination_comparison(
        all_combination_results=all_combination_results,
        config=config,
        output_dir=output_dir,
    )


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()
