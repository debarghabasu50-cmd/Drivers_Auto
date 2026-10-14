# =============================================================================
# FEATURE SELECTION FRAMEWORK — Driver/KPI Selection for DML
# =============================================================================
#
# PURPOSE
#   Select a robust, business-mapped set of candidate KPIs/drivers for downstream
#   DML modelling. The pipeline screens noisy KPIs, chooses meaningful lags,
#   scores relationship strength, checks stability, applies business rules, and
#   creates a final selected KPI set with diagnostics.
#
# METHOD — read top-to-bottom like the DML pipeline
#   Step 1 : Load data, create rolling target, and prefilter usable numeric KPIs.
#   Step 2 : Map every KPI to business Category/Theme using the mapping file.
#   Step 3 : Select the best lag for every KPI using Spearman + optional DTW tie-break.
#   Step 4 : Build the modelling dataframe with one lagged column per KPI.
#   Step 5 : Compute MI + correlation signal score.
#   Step 6 : Apply cointegration floor to protect long-run economic relationships.
#   Step 7 : Compute LCEN stability and FCS_pre.
#   Step 8 : Compute DTW alignment score and final joint score.
#   Step 9 : Apply family survival and country contribution multiplier.
#   Step 10: Pick top KPI per category and compute within-category partial correlation.
#   Step 11: Apply country diversity so top contributing countries are represented.
#   Step 12: Create non-overlapping priority sets P0/P1/P2/P3.
#   Step 13: Save all diagnostics and selected KPI outputs to Excel.
#
# INPUTS
#   - Excel file with monthly target + KPI columns
#   - KPI mapping file with KPI, Category, Theme columns
#   - Optional country contribution file
#
# OUTPUTS
#   - Excel report with prefilter, lag selection, scoring, selection,
#     country-diversity actions, priority sets, and partial-correlation diagnostics
#
# TO USE FOR A NEW DIVISION
#   1. Update feature_selection_config.json
#   2. Run: python feature_selection_dml_style.py --config feature_selection_config.json
#   3. Review summary, final_feature_table, P0/P1/P2/P3 sheets
# =============================================================================

# HOW TO READ THIS FILE
#   This script is intentionally written as a sequential pipeline.
#   Each STEP produces either:
#     1. a filtered/scored dataframe,
#     2. an audit/diagnostic table, or
#     3. a final selected KPI output.
#
#   A new reader should first understand the STEP blocks,
#   then go inside individual functions only when debugging.

from __future__ import annotations

import argparse
import json
import re
import warnings
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import coint

warnings.filterwarnings("ignore")


# This class loads the settings from our json config file.
# Each loader keeps the file path we give it.
# Calling load() reads the file and adds the same defaults as before.


class ConfigLoader:
    """
    Load the feature-selection JSON config and apply existing defaults.

    OOP explanation
    ---------------
    This class describes a configuration loader.

    Each object created from this class remembers its own config-file path.
    Its load() method reads that file and returns the settings.

    The default below are unchanged from the original load_config() function.
    """

    def __init__(self, config_path: str | Path):
        """
        Load settings from a JSON file.
        """
        self.config_path = Path(config_path)

    def load(self) -> dict:
        """
        Read the file and returns ots settings.
        """
        with self.config_path.open("r", encoding="utf-8") as file:
            return json.load(file)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the feature selection pipeline.")
    parser.add_argument("--config", required=True, help="Path to JSON config file.")
    parser.add_argument(
        "--run-id", required=True, help="Run ID supplied by the pipeline runner."
    )
    parser.add_argument(
        "--quarter",
        required=True,
        help="Quarter label supplied by the pipeline runner.",
    )
    return parser.parse_args()


# =============================================================================
# SMALL UTILITY FUNCTIONS
# =============================================================================
# These functions keep the main steps readable. They do small generic tasks:
# safe text cleanup, safe numeric conversion, compact printing, and basic metrics.
# =============================================================================


def print_step(
    step_no: int,
    title: str,
    purpose: str | None = None,
    *,
    logger: logging.Logger,
) -> None:
    """Write a step heading to the run log."""

    logger.info("=" * 78)
    logger.info("[Step %s] %s", step_no, title)

    if purpose:
        logger.info("%s", purpose)

    logger.info("=" * 78)


def print_top_rows(
    title: str,
    df: pd.DataFrame,
    columns: list[str],
    n: int = 10,
    sort_col: str | None = None,
    ascending: bool = False,
    *,
    logger: logging.Logger,
) -> None:
    """Write a compact diagnostics table to the run log."""

    logger.info("-" * 78)
    logger.info("%s", title)
    logger.info("-" * 78)

    if df is None or df.empty:
        logger.info("No rows to display.")
        return

    cols = [col for col in columns if col in df.columns]

    if not cols:
        logger.info("Requested columns not found in dataframe.")
        return

    view = df.copy()

    if sort_col is not None and sort_col in view.columns:
        view = view.sort_values(sort_col, ascending=ascending)

    logger.info("\n%s", view[cols].head(n).to_string(index=False))


def to_numeric_series(s: pd.Series) -> pd.Series:
    """Convert a pandas Series to numeric safely."""
    return pd.to_numeric(s, errors="coerce")


def normalize_text(s: str) -> str:
    """Lowercase and collapse whitespace for robust text matching."""
    s = str(s).lower()
    return re.sub(r"\s+", " ", s).strip()


def clean_key(s: str) -> str:
    """Clean country/category text so matching does not fail due to spacing/case."""
    return re.sub(r"\s+", " ", str(s).strip().lower())


def get_lags_for_feature_category(
    feature: str,
    feature_to_category: dict,
    category_lags: dict,
    default_lags: list[int] | tuple[int, ...],
) -> tuple[int, ...]:
    """
    Select candidate lags based on the KPI's mapped Category.

    Priority:
      1. If feature's Category exists in category_lags, use those lags.
      2. If not found, try case-insensitive category matching.
      3. If still not found, use default_lags.
    """

    category = feature_to_category.get(feature)

    if category is None or pd.isna(category):
        return tuple(default_lags)

    category_key = str(category).strip()

    # Exact match
    if category_key in category_lags:
        return tuple(category_lags[category_key])

    # Case-insensitive fallback
    category_lags_lower = {str(k).strip().lower(): v for k, v in category_lags.items()}

    matched_lags = category_lags_lower.get(category_key.lower())

    if matched_lags is not None:
        return tuple(matched_lags)

    return tuple(default_lags)


def extract_country(feature: str) -> str | None:
    """Extract the first pipe-delimited token as the country name."""
    s = str(feature)
    if "|" not in s:
        return None
    return s.split("|", 1)[0].strip()


def append_reason(existing: str, addition: str) -> str:
    """Append an audit-trail reason without creating messy separators."""
    existing = "" if pd.isna(existing) else str(existing).strip()
    return addition if not existing else existing + " | " + addition


def get_mandatory_kpi_config(config: dict) -> dict:
    """Read mandatory KPI settings from the supplied config."""
    cfg = config.get("mandatory_kpis", {})
    return {
        "enabled": bool(cfg.get("enabled", False)),
        "match_mode": cfg.get("match_mode", "contains"),
        "items": cfg.get("items", []),
        "override_prefilter": bool(cfg.get("override_prefilter", True)),
        "override_lag_rejection": bool(cfg.get("override_lag_rejection", True)),
        "override_family_survival": bool(cfg.get("override_family_survival", True)),
        "override_top_selection": bool(cfg.get("override_top_selection", True)),
        "priority_set": cfg.get("priority_set", "P0"),
    }


def is_mandatory_kpi(feature: str, mandatory_cfg: dict) -> bool:
    """
    Check whether a feature should be force-included based on mandatory_kpis config.

    Supports:
      - contains: case-insensitive substring match
      - exact: case-insensitive exact match
      - regex: regex search
    """
    if not mandatory_cfg.get("enabled", False):
        return False

    feature_text = normalize_text(feature)
    items = mandatory_cfg.get("items", [])
    match_mode = mandatory_cfg.get("match_mode", "contains")

    for item in items:
        item_text = normalize_text(item)

        if match_mode == "contains":
            if item_text in feature_text:
                return True

        elif match_mode == "exact":
            if item_text == feature_text:
                return True

        elif match_mode == "regex":
            if re.search(str(item), str(feature), flags=re.IGNORECASE):
                return True

        else:
            raise ValueError(f"Unknown mandatory_kpis.match_mode: {match_mode}")

    return False


def add_mandatory_flags(
    df: pd.DataFrame,
    feature_col: str = "Feature",
    *,
    mandatory_cfg: dict,
) -> pd.DataFrame:
    """Mark mandatory KPIs using the settings supplied by the caller."""
    out = df.copy()
    out["Mandatory_KPI"] = (
        out[feature_col].astype(str).apply(lambda x: is_mandatory_kpi(x, mandatory_cfg))
    )

    out["Override_Reason"] = np.where(
        out["Mandatory_KPI"], "Mandatory KPI override configured by business", ""
    )
    return out


def robust_cv_mad(x: pd.Series, eps: float = 1e-9) -> float:
    """Robust CV (%) using MAD around median. Used to filter flat KPIs."""
    x = to_numeric_series(x).dropna()
    if len(x) == 0:
        return np.nan
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    robust_sigma = 1.4826 * mad
    return 100.0 * (robust_sigma / (abs(med) + eps))


def transform_series(
    s: pd.Series, method: str = "raw", yoy_period: int = 12
) -> pd.Series:
    """
    Transform a time series consistently across lag/correlation/MI/DTW steps.

    raw     : original values
    log     : log(level), only valid for positive values
    yoy     : year-over-year percent change
    log_yoy : log(x_t) - log(x_t-12), approx. YoY % change
    """
    s = to_numeric_series(s)

    if method == "raw":
        return s
    if method == "log":
        return np.log(s.where(s > 0))
    if method == "yoy":
        return s.pct_change(yoy_period)
    if method == "log_yoy":
        log_s = np.log(s.where(s > 0))
        return log_s - log_s.shift(yoy_period)

    raise ValueError(f"Unknown transform method: {method}")


def safe_rank_pct(s: pd.Series) -> pd.Series:
    """Percentile rank in [0, 1], safely filling NaNs with 0."""
    return s.astype(float).rank(pct=True).fillna(0.0)


def simple_dtw_distance(a: pd.Series, b: pd.Series) -> float:
    """Simple Dynamic Time Warping distance between two aligned numeric series."""
    tmp = pd.DataFrame({"a": a, "b": b}).dropna()
    if len(tmp) == 0:
        return np.inf
    x = tmp["a"].to_numpy(dtype=float)
    y = tmp["b"].to_numpy(dtype=float)
    n, m = len(x), len(y)
    dp = np.full((n + 1, m + 1), np.inf)
    dp[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(x[i - 1] - y[j - 1])
            dp[i, j] = cost + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])
    return float(dp[n, m])


def safe_spearman(a: pd.Series, b: pd.Series, min_points: int = 10) -> float:
    """Spearman correlation with checks for too few points or constant series."""
    tmp = pd.DataFrame({"a": a, "b": b}).dropna()
    if len(tmp) < min_points or tmp["a"].nunique() < 2 or tmp["b"].nunique() < 2:
        return np.nan
    r, _ = spearmanr(tmp["a"], tmp["b"])
    return float(r) if pd.notna(r) else np.nan


# =============================================================================
# This class holds the data and settings used by the feature-selection pipleline.
class FeatureSelectionPipeline:
    """
    Run feature selection using the settings provided in the config.
    """

    def __init__(self, config: dict):
        # keep the settings so the methods in this class can use them.
        self.config = config

        # Remember the column names used throughout the pipeline.
        self.date_col = config["date_col"]
        self.raw_target_col = config["raw_target_col"]
        self.target_col = config["target_col"]
        self.modelling_level_col = config["modelling_level_col"]

    # =============================================================================
    # STEP 1 — LOAD DATA AND PREFILTER KPIs
    # =============================================================================
    # PURPOSE
    #   Create the rolling target and remove KPI columns that are not usable for
    #   downstream modelling.
    #
    # WHY
    #   Feature selection should not spend effort on variables that are mostly
    #   missing, flat, constant, or invalid for log transformation.
    # =============================================================================

    def load_data(self) -> None:
        """
        Read the input files and prepare the target column.
        """
        print_step(
            1,
            "Loading data and prefiltering KPIs",
            "Create rolling tarrget and keep usuable numeric KPI candidates.",
            logger=self.logger,
        )
        input_path = Path(self.config["input_file"])
        mapping_path = Path(self.config["mapping_file"])

        # Check that both files exist before trying to read them.
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        if not mapping_path.exists():
            raise FileNotFoundError(f"Mapping file not found: {mapping_path}")

        # Keep both tables in this object for the later steps.
        self.df = pd.read_excel(input_path)
        self.mapping_df = pd.read_excel(mapping_path)

        self.df[self.date_col] = pd.to_datetime(self.df[self.date_col], errors="coerce")
        self.df = self.df.sort_values(self.date_col).reset_index(drop=True)

        rolling_window = int(self.config.get("rolling_window", 3))

        if rolling_window < 1:
            raise ValueError("rolling_window must be at least 1.")

        if rolling_window == 1:
            # Use the original target directly when no smoothing is needed.
            self.target_col = self.raw_target_col

        else:
            # Store the rolling average under the configured target name.
            self.target_col = self.config["target_col"]

            if self.target_col == self.raw_target_col:
                raise ValueError(
                    "For rolling windows greater than 1, target_col must "
                    "differ from raw_target_col."
                )

            self.df[self.target_col] = (
                self.df[self.raw_target_col]
                .rolling(
                    window=rolling_window,
                    min_periods=rolling_window,
                )
                .mean()
            )

            if self.config.get("drop_raw_target_after_rolling", True):
                self.df = self.df.drop(columns=[self.raw_target_col])

        # Keep rows where the target is available.
        self.df = self.df.dropna(subset=[self.target_col]).reset_index(drop=True)
        self.df_full = self.df.copy()
        # Use only the training period for feature selection.
        train_end = pd.Timestamp(self.config["train_end"])

        self.df = (
            self.df.loc[self.df[self.date_col] <= train_end]
            .copy()
            .reset_index(drop=True)
        )

        if self.df.empty:
            raise ValueError(f"No training rows found on or before {train_end.date()}.")

        self.logger.info(f"Training end    : {train_end.date()}")
        self.logger.info(f"Full data rows  : {len(self.df_full)}")
        self.logger.info(f"Division        : {self.config.get('division_name', 'N/A')}")
        self.logger.info(f"Input rows      : {len(self.df)}")
        self.logger.info(f"Target Column   : {self.target_col}")
        self.logger.info(f"Rolling window  : {rolling_window}")

    def prefilter_kpis(self) -> None:
        """
        Check the KPI columns and keep those that pass the existing filters.
        """

        pref_cfg = self.config["prefilter"]

        # Check missing values, unique values, negatives, and variability.
        self.prefilter_df = build_prefilter_diagnostics(
            self.df,
            date_col=self.date_col,
            target_col=self.target_col,
            missing_threshold=float(pref_cfg["missing_threshold"]),
            cv_threshold=float(pref_cfg["cv_threshold"]),
            min_unique=int(pref_cfg["min_unique"]),
            pref_cfg=pref_cfg,
            ignore_cols=self.config.get("ignore_cols", []),
        )
        mandatory_cfg = get_mandatory_kpi_config(self.config)

        # Mark the KPIs listed as mandatory in the config.
        self.prefilter_df = add_mandatory_flags(
            self.prefilter_df,
            feature_col="Feature",
            mandatory_cfg=mandatory_cfg,
        )
        if mandatory_cfg["enabled"] and mandatory_cfg["override_prefilter"]:
            mandatory_prefilter_mask = self.prefilter_df["Mandatory_KPI"]

            # Let mandatory KPIs contibue even if they failed a filter.
            self.prefilter_df.loc[mandatory_prefilter_mask, "overall_pass"] = True
            # Keep the original reason and record why was it allowed
            self.prefilter_df.loc[
                mandatory_prefilter_mask, "Drop_Reason"
            ] = self.prefilter_df.loc[mandatory_prefilter_mask, "Drop_Reason"].apply(
                lambda x: append_reason(
                    x, "OVERRIDDEN: MANDATORY KPI passed prefilter by business rule"
                )
            )
        # Build the candidate list after applying the mandatory overrides.
        self.candidate_features = self.prefilter_df.query("overall_pass")[
            "Feature"
        ].to_list()

        self.logger.info(f"Candidate numeric KPIs : {len(self.prefilter_df)}")
        self.logger.info(f"Passed prefilter KPIs  : {len(self.candidate_features)}")

    def map_category_theme(self, feature_name: str) -> pd.Series:
        """
        Find the longest mapping-file KPI that matches the feature name.
        """

        s = normalize_text(feature_name)
        hits = []

        for kpi_raw, pat, cat, theme in self.patterns:
            if pat.search(s):
                hits.append((len(str(kpi_raw)), kpi_raw, cat, theme))

        if not hits:
            return pd.Series(
                {
                    "Mapped_KPI": None,
                    "Category": None,
                    "Theme": None,
                }
            )

        # Use the longest match, just as we did in the original function.
        hits.sort(reverse=True)
        _, kpi_raw, cat, theme = hits[0]
        return pd.Series({"Mapped_KPI": kpi_raw, "Category": cat, "Theme": theme})

    # ================================================================================
    # STEP 2 — BUILD KPI CATEGORY/THEME MAPPING
    # ================================================================================
    def build_kpi_mapping(self) -> None:
        """
        Prepare the mapping patterns and map the candidate KPIs.
        """

        print_step(
            2,
            "Building KPI category/theme mapping",
            "Prepare regex patterns for robust KPI-to-category matching.",
            logger=self.logger,
        )

        map_cfg = self.config["mapping"]
        kpi_col = map_cfg["kpi_col"]
        category_col = map_cfg["category_col"]
        theme_col = map_cfg["theme_col"]

        self.mapping_df = self.mapping_df.copy()
        self.mapping_df[kpi_col] = self.mapping_df[kpi_col].astype(str).str.strip()

        # keep the patterns in this object so we can map features again later.
        self.patterns = []

        for _, row in self.mapping_df.iterrows():
            kpi_raw = str(row[kpi_col]).strip()

            if not kpi_raw or kpi_raw.lower() == "nan":
                continue

            k = normalize_text(kpi_raw)
            k_esc = re.escape(k)
            k_esc = k_esc.replace(r"\ \|\ ", r"\s*\|\s*")
            k_esc = k_esc.replace(r"\|", r"\s*\|\s*")

            pat = re.compile(k_esc, flags=re.IGNORECASE)

            self.patterns.append(
                (
                    kpi_raw,
                    pat,
                    row.get(category_col, None),
                    row.get(theme_col, None),
                )
            )

        self.logger.info(f"Mapping patterns loaded: {len(self.patterns)}")

        self.candidate_feature_mapping_df = pd.DataFrame(
            {
                "Feature": self.candidate_features,
            }
        )

        # Map every candidate before choosing its category-specific lags.
        self.candidate_feature_mapping_df = pd.concat(
            [
                self.candidate_feature_mapping_df,
                self.candidate_feature_mapping_df["Feature"].apply(
                    self.map_category_theme
                ),
            ],
            axis=1,
        )

        self.feature_to_category = dict(
            zip(
                self.candidate_feature_mapping_df["Feature"],
                self.candidate_feature_mapping_df["Category"],
            )
        )

        self.feature_to_theme = dict(
            zip(
                self.candidate_feature_mapping_df["Feature"],
                self.candidate_feature_mapping_df["Theme"],
            )
        )

        self.feature_to_mapped_kpi = dict(
            zip(
                self.candidate_feature_mapping_df["Feature"],
                self.candidate_feature_mapping_df["Mapped_KPI"],
            )
        )

        self.logger.info(
            "Candidate KPIs mapped before lag selection: "
            f"{self.candidate_feature_mapping_df['Category'].notna().sum()}/"
            f"{len(self.candidate_feature_mapping_df)}"
        )

    # =========================================================================
    # STEP 3 — LAG SELECTION USING SPEARMAN + DTW TIE-BREAK
    # =========================================================================
    def select_best_lags(self) -> None:
        """
        Choose the lag for each candidate KPI and keep the diagnostics.
        """

        print_step(
            3,
            "Selecting best lags",
            "Spearman chooses candidate lags; DTW resolves close ties.",
            logger=self.logger,
        )

        lag_cfg = self.config["lag_selection"]

        # Create the full diagnostic table only when requested in the config.
        self.corr_long_df = pd.DataFrame()

        if lag_cfg.get("run_full_lag_diagnostics", False):
            self.corr_long_df = lag_corr_table(
                df=self.df,
                date_col=self.date_col,
                target_col=self.target_col,
                candidate_features=self.candidate_features,
                max_lag=int(lag_cfg.get("full_lag_max_lag", 12)),
                min_points=int(lag_cfg["min_points"]),
                transform_method=lag_cfg.get("full_lag_transform_method", "yoy"),
                yoy_period=int(lag_cfg["yoy_period"]),
            )
            self.logger.info(
                f"Full lag diagnostics created: {len(self.corr_long_df)} rows"
            )

        self.lag_corr_df = best_lag_corr_table(
            df=self.df,
            date_col=self.date_col,
            target_col=self.target_col,
            candidate_features=self.candidate_features,
            feature_to_category=self.feature_to_category,
            feature_to_theme=self.feature_to_theme,
            feature_to_mapped_kpi=self.feature_to_mapped_kpi,
            min_points=int(lag_cfg["min_points"]),
            default_lags=tuple(lag_cfg.get("default_lags", [0, 3, 6, 9, 12])),
            category_lags=lag_cfg.get("category_lags", {}),
            transform_method=lag_cfg["transform_method"],
            yoy_period=int(lag_cfg["yoy_period"]),
            lag_gap_threshold=float(lag_cfg["lag_gap_threshold"]),
            min_abs_corr_threshold=float(lag_cfg["min_abs_corr_threshold"]),
            min_qualifying_lags=int(lag_cfg["min_qualifying_lags"]),
            use_dtw_tiebreak=bool(lag_cfg.get("use_dtw_tiebreak", True)),
            mandatory_cfg=get_mandatory_kpi_config(self.config),
        )
        self.logger.info(f" Best lag rows    : {len(self.lag_corr_df)}")
        self.logger.info(
            "OK lag selections : "
            f"{self.lag_corr_df['Reason'].astype(str).str.startswith('OK').sum()}"
        )
        self.logger.info(
            f"DTW tie-break used: {self.lag_corr_df['DTW_Selected'].sum()}"
        )

        print_top_rows(
            "TOP LAG SELECTIONS - Spearman decision vs final DTW decision",
            self.lag_corr_df,
            [
                "Feature",
                "Spearman_Best_Lag",
                "Spearman_Second_Best_Lag",
                "Best_Lag",
                "Lag_Gap",
                "DTW_Selected",
                "DTW_Changed_Lag",
                "Spearman_Corr",
                "Abs_Spearman_Corr",
                "n_points",
            ],
            n=int(self.config.get("print_top_n", 15)),
            sort_col="Abs_Spearman_Corr",
            logger=self.logger,
        )

    # =========================================================================
    # STEP 4 — BUILD LAGGED MODEL DATAFRAME
    # =========================================================================
    def build_model_data(self) -> None:
        """
        Build the lagged data and separate accepted KPIs from rejected ones.
        """

        print_step(
            4,
            "Building lagged model dataframe",
            "Create one lagged feature column per selected best lag.",
            logger=self.logger,
        )

        # Build the lagged columns using the decisions from step 3.
        self.model_df = build_model_df_from_best_lags(
            df=self.df,
            lag_table=self.lag_corr_df,
            date_col=self.date_col,
            modelling_level_col=self.modelling_level_col,
            target_col=self.target_col,
        )

        self.model_df_after_dropna = self.model_df.dropna().copy()
        self.model_df = self.model_df.dropna().reset_index(drop=True)

        self.logger.info(f"Model dataframe rows after dropna: {len(self.model_df)}")
        self.logger.info(
            f"Lagged KPI columns               : {len(self.model_df.columns) - 3}"
        )

        self.lag_corr_df_original_features = self.lag_corr_df.copy()

        ok_lag_mask = (
            self.lag_corr_df["Reason"].astype(str).str.startswith("OK")
            & self.lag_corr_df["Best_Lag"].notna()
        )

        self.rejected_lag_kpis_df = self.lag_corr_df_original_features.loc[
            ~ok_lag_mask
        ].copy()

        rejection_cols = [
            "Feature",
            "Category_Used_For_Lag",
            "Theme_Used_For_Lag",
            "Mapped_KPI_Used_For_Lag",
            "Lags_Checked",
            "Reason",
            "Sig_Count",
            "Min_Significant_Lags",
            "Spearman_Best_Lag",
            "Spearman_Best_Corr",
            "Spearman_Best_Abs_Corr",
            "Spearman_Best_PValue",
            "Spearman_Second_Best_Lag",
            "Spearman_Second_Best_Corr",
            "Spearman_Second_Best_Abs_Corr",
            "Spearman_Second_Best_PValue",
            "Best_Lag",
            "Spearman_Corr",
            "Abs_Spearman_Corr",
            "Spearman_PValue",
            "Lag_Gap",
            "DTW_Selected",
            "DTW_Changed_Lag",
            "DTW_Best",
            "DTW_Second",
            "n_points",
            "Transform_Method",
            "Lag_0_Corr",
            "Lag_3_Corr",
            "Lag_6_Corr",
            "Lag_9_Corr",
            "Lag_12_Corr",
            "Lag_0_PValue",
            "Lag_3_PValue",
            "Lag_6_PValue",
            "Lag_9_PValue",
            "Lag_12_PValue",
            "Lag_0_Significant",
            "Lag_3_Significant",
            "Lag_6_Significant",
            "Lag_9_Significant",
            "Lag_12_Significant",
            "Sign_Stability_Ratio",
            "N_Positive_Sig_Lags",
            "N_Negative_Sig_Lags",
        ]
        self.rejected_lag_kpis_df = self.rejected_lag_kpis_df[
            [col for col in rejection_cols if col in self.rejected_lag_kpis_df.columns]
        ].copy()

        self.logger.info(
            "Rejected KPIs at lag-selection stage: " f"{len(self.rejected_lag_kpis_df)}"
        )

        # Only accepted KPIs continue to the scoring steps.
        self.lag_corr_df = self.lag_corr_df.loc[ok_lag_mask].copy()

        self.lag_corr_df["Original_Feature"] = self.lag_corr_df["Feature"].astype(str)

        # Match the feature names to the lagged columns in the modelling data.
        self.lag_corr_df["Feature"] = (
            self.lag_corr_df["Feature"].astype(str)
            + " | lag_"
            + self.lag_corr_df["Best_Lag"].astype(int).astype(str)
        )

        self.lag_corr_df = pd.concat(
            [
                self.lag_corr_df,
                self.lag_corr_df["Feature"].apply(self.map_category_theme),
            ],
            axis=1,
        )

        self.logger.info(
            "Mapped KPIs to category/theme: "
            f"{self.lag_corr_df['Category'].notna().sum()}/"
            f"{len(self.lag_corr_df)}"
        )

    # =========================================================================
    # STEP 5 — MI + CORRELATION SIGNAL
    # =========================================================================
    def compute_mi_signal(self) -> None:
        """
        Compute MI and combine it with the correlation score.
        """

        print_step(
            5,
            "Computing MI and signal scores",
            "Combine Spearman and mutual information into one initial signal score.",
            logger=self.logger,
        )

        mi_cfg = self.config["mi_signal"]

        # Use the lagged data prepared in Step 4.
        self.lag_corr_df = add_mi_and_signal(
            df=self.model_df,
            lag_corr_df=self.lag_corr_df,
            date_col=self.date_col,
            target_col=self.target_col,
            n_neighbors=int(mi_cfg["n_neighbors"]),
            random_state=int(mi_cfg["random_state"]),
            min_points=int(mi_cfg["min_points"]),
            transform_method=mi_cfg["transform_method"],
            yoy_period=int(mi_cfg["yoy_period"]),
        )

        self.logger.info(
            "MI computed successfully for: "
            f"{(self.lag_corr_df['mi_reason'] == 'ok').sum()} KPIs"
        )

        print_top_rows(
            "TOP SIGNAL SCORES — Spearman + MI",
            self.lag_corr_df,
            [
                "Feature",
                "Category",
                "Spearman_Corr",
                "corr_pct",
                "MI_Raw",
                "mi_pct",
                "signal",
            ],
            n=int(self.config.get("print_top_n", 15)),
            sort_col="signal",
            logger=self.logger,
        )

    # =========================================================================
    # STEP 6 — COINTEGRATION FLOOR
    # =========================================================================
    def apply_cointegration(self) -> None:
        """
        Check cointegration and apply the existing signal floor.
        """

        print_step(
            6,
            "Applying cointegration floor",
            "If a KPI is cointegrated with target, lift its signal to a minimum floor.",
            logger=self.logger,
        )

        coint_cfg = self.config["cointegration"]

        # Add the cointegration results to our current feature table.
        self.lag_corr_df = apply_cointegration_floor(
            df=self.model_df,
            lag_corr_df=self.lag_corr_df,
            date_col=self.date_col,
            target_col=self.target_col,
            coint_alpha=float(coint_cfg["alpha"]),
            floor_value=float(coint_cfg["floor_value"]),
            min_points=int(coint_cfg["min_points"]),
        )

        self.logger.info(f"Cointegrated KPIs: {self.lag_corr_df['cointegrated'].sum()}")

        print_top_rows(
            "COINTEGRATION FLOOR CHECK",
            self.lag_corr_df,
            [
                "Feature",
                "Category",
                "signal",
                "coint_pvalue",
                "cointegrated",
                "signal_after_coint",
            ],
            n=int(self.config.get("print_top_n", 15)),
            sort_col="signal_after_coint",
            logger=self.logger,
        )

    # =========================================================================
    # STEP 7 — LCEN STABILITY + FCS_PRE
    # =========================================================================
    def compute_lcen_stability(self) -> None:
        """
        Compute LCEN stability and add FCS_pre to the feature table.
        """

        print_step(
            7,
            "Computing LCEN stability and FCS_pre",
            "Combine signal_after_coint with ElasticNet stability.",
            logger=self.logger,
        )

        lcen_cfg = self.config["lcen"]

        # Build the same regularisation values used in the original code.
        alphas = list(
            np.logspace(
                float(lcen_cfg["alphas_logspace_start"]),
                float(lcen_cfg["alphas_logspace_end"]),
                int(lcen_cfg["alphas_count"]),
            )
        )

        # Use our existing bootstrap and ElasticNet calculation.
        self.lag_corr_df = compute_lcen_and_fcs_pre(
            df=self.model_df,
            lag_corr_df=self.lag_corr_df,
            date_col=self.date_col,
            target_col=self.target_col,
            lcen_weight=float(lcen_cfg["weight"]),
            signal_weight=float(lcen_cfg.get("signal_weight", 1.0)),
            n_bootstraps=int(lcen_cfg["n_bootstraps"]),
            alphas=alphas,
            l1_ratios=lcen_cfg["l1_ratios"],
            random_state=int(lcen_cfg["random_state"]),
            block_size=lcen_cfg.get("block_size"),
            coef_tol=float(lcen_cfg["coef_tol"]),
        )

        self.logger.info(
            "FCS_pre available for: "
            f"{self.lag_corr_df['FCS_pre'].notna().sum()} KPIs"
        )
        self.logger.info(
            "FCS_pre capped at 1 : " f"{(self.lag_corr_df['FCS_pre'] == 1).sum()} KPIs"
        )

        print_top_rows(
            "TOP FCS_PRE SCORES — signal_after_coint + LCEN",
            self.lag_corr_df,
            [
                "Feature",
                "Category",
                "signal_after_coint",
                "LCEN",
                "FCS_pre",
            ],
            n=int(self.config.get("print_top_n", 15)),
            sort_col="FCS_pre",
            logger=self.logger,
        )

    # ==========================================================================
    # STEP 8 — DTW SCORE + JOINT SCORE
    # ==========================================================================
    def compute_dtw_score(self) -> None:
        """
        Compute DTW alignment and add the joint score to the feature table.
        """
        print_step(
            8,
            "Computing DTW score and joint score",
            "Combine FCS_pre with target-alignment score.",
            logger=self.logger,
        )

        dtw_cfg = self.config["dtw_joint_score"]

        # Use our existing function to calculate alignment and the joint score.
        self.lag_corr_df = compute_dtw_and_joint_score(
            df=self.model_df,
            lag_corr_df=self.lag_corr_df,
            date_col=self.date_col,
            target_col=self.target_col,
            fcs_col=dtw_cfg.get("fcs_col", "FCS_pre"),
            transform_method=dtw_cfg["transform_method"],
            yoy_period=int(dtw_cfg["yoy_period"]),
            min_points=int(dtw_cfg["min_points"]),
            dtw_cfg=dtw_cfg,
        )

        self.logger.info(
            "DTW computed successfully for: "
            f"{(self.lag_corr_df['dtw_reason'] == 'ok').sum()} KPIs"
        )
        self.logger.info(
            "Joint score available for    : "
            f"{self.lag_corr_df['joint_score'].notna().sum()} KPIs"
        )

        print_top_rows(
            f"TOP JOINT SCORES — "
            f"{float(dtw_cfg.get('fcs_weight', 0.7)):g} × "
            f"{dtw_cfg.get('fcs_col', 'FCS_pre')} + "
            f"{float(dtw_cfg.get('dtw_weight', 0.3)):g} × DTW score",
            self.lag_corr_df,
            [
                "Feature",
                "Category",
                "FCS_pre",
                "dtw_raw",
                "dtw_score",
                "joint_score",
            ],
            n=int(self.config.get("print_top_n", 15)),
            sort_col="joint_score",
            logger=self.logger,
        )

    # =========================================================================
    # STEP 9 — FAMILY SURVIVAL + COUNTRY MULTIPLIER
    # =========================================================================
    def apply_business_scores(self) -> None:
        """
        Apply family survival and adjust scores using country contributions.
        """
        print_step(
            9,
            "Applying family survival and country multiplier",
            "Filter weak categories and apply contribution-aware score adjustment.",
            logger=self.logger,
        )

        # Check which categories pass the existing family-survival rule.
        self.lag_corr_df, self.family_info = apply_family_survival(
            self.lag_corr_df,
            self.config["family_survival"],
            mandatory_cfg=get_mandatory_kpi_config(self.config),
        )

        self.logger.info(f"Family threshold used : {self.family_info['threshold']:.4f}")
        self.logger.info(
            f"Dropped families      : {self.family_info['dropped_families']}"
        )

        # Adjust each KPI's score using its country's contribution.
        self.lag_corr_df = apply_country_multiplier(
            self.lag_corr_df,
            contribution_file=self.config["country_contribution_file"],
            config=self.config["country_multiplier"],
            logger=self.logger,
        )

        self.logger.info("Country multiplier applied.")

        print_top_rows(
            "TOP FINAL SCORES BEFORE SELECTION — joint_score × country_multiplier",
            self.lag_corr_df,
            [
                "Feature",
                "Category",
                "joint_score",
                "country_raw",
                "Contribution",
                "country_multiplier",
                "FCS_star",
                "Status",
                "Reason",
            ],
            n=int(self.config.get("print_top_n", 15)),
            sort_col="FCS_star",
            logger=self.logger,
        )

    # ==========================================================================
    # STEP 10 — TOP-1 CATEGORY SELECTION + PARTIAL CORRELATION
    # ==========================================================================
    def select_category_kpis(self) -> None:
        """
        Select category winners and check the remaining KPIs for added signal.
        """

        print_step(
            10,
            "Selecting top KPI per category and running partial correlation",
            "Create selected base set and within-category diagnostics.",
            logger=self.logger,
        )

        top_cfg = self.config["top_selection"]

        # Choose the highest-scoring eligible KPI in each category.
        self.lag_corr_df, self.selected_step10 = pick_top1_per_category_simple(
            self.lag_corr_df,
            feature_col=top_cfg.get("feature_col", "Feature"),
            category_col=top_cfg.get("category_col", "Category"),
            fcs_star_col=top_cfg.get("score_col", "FCS_star"),
            min_score=top_cfg.get("min_score"),
        )

        mandatory_cfg = get_mandatory_kpi_config(self.config)

        if mandatory_cfg["enabled"] and mandatory_cfg["override_top_selection"]:
            feature_col = top_cfg.get("feature_col", "Feature")

            self.lag_corr_df = add_mandatory_flags(
                self.lag_corr_df,
                feature_col=feature_col,
                mandatory_cfg=mandatory_cfg,
            )

            mandatory_mask = self.lag_corr_df["Mandatory_KPI"]

            # Keep mandatory KPIs even when another KPI wins their category.
            self.lag_corr_df.loc[mandatory_mask, "Status"] = "Selected"

            self.lag_corr_df.loc[mandatory_mask, "Reason"] = self.lag_corr_df.loc[
                mandatory_mask, "Reason"
            ].apply(
                lambda x: append_reason(
                    x,
                    "OVERRIDDEN: mandatory KPI selected by business rule",
                )
            )

            self.lag_corr_df.loc[
                mandatory_mask, "Override_Reason"
            ] = self.lag_corr_df.loc[mandatory_mask, "Override_Reason"].apply(
                lambda x: append_reason(
                    x,
                    "Mandatory KPI override at top-selection",
                )
            )

            # Keep the selected results in the same form as the existing code.
            self.selected_step10 = self.lag_corr_df[
                self.lag_corr_df["Status"].eq("Selected")
            ].copy()

        self.logger.info(f"Selected top-1 KPIs: {len(self.selected_step10)}")

        print_top_rows(
            "SELECTED KPIS — top 1 per category",
            self.lag_corr_df[self.lag_corr_df["Status"].eq("Selected")].copy(),
            [
                "Category",
                "Feature",
                "Best_Lag",
                "Spearman_Corr",
                "MI_Raw",
                "FCS_pre",
                "dtw_score",
                "joint_score",
                "FCS_star",
                "Status",
                "Reason",
            ],
            n=100,
            sort_col="FCS_star",
            logger=self.logger,
        )

        pc_cfg = self.config["partial_corr"]

        # Check each dropped KPI against the selected KPI in its category.
        (
            self.lag_corr_df_updated,
            self.selected_rank1_df,
            self.decision_df,
        ) = apply_within_category_partial_corr_using_status(
            model_df=self.model_df,
            lag_corr_df=self.lag_corr_df,
            target_col=self.target_col,
            feature_col=pc_cfg.get("feature_col", "Feature"),
            category_col=pc_cfg.get("category_col", "Category"),
            best_lag_col=pc_cfg.get("best_lag_col", "Best_Lag"),
            score_col=pc_cfg.get("score_col", "FCS_star"),
            status_col=pc_cfg.get("status_col", "Status"),
            partial_corr_threshold=float(pc_cfg.get("partial_corr_threshold", 0.10)),
            min_points=int(pc_cfg.get("min_points", 12)),
            logger=self.logger,
        )

        self.logger.info(f"Partial-corr decisions: {len(self.decision_df)}")
        self.logger.info(
            "Partial-corr additions flagged: "
            f"{self.decision_df['Add_Flag'].sum() if not self.decision_df.empty else 0}"
        )

        print_top_rows(
            "PARTIAL CORRELATION DECISIONS",
            self.decision_df,
            [
                "Category",
                "Top1_Feature",
                "Feature",
                "Candidate_Top1_Spearman",
                "Partial_Corr",
                "Add_Flag",
                "Reason",
            ],
            n=int(self.config.get("print_top_n", 20)),
            sort_col="Partial_Corr",
            logger=self.logger,
        )

    # ==========================================================================
    # STEP 11 — COUNTRY DIVERSITY ADJUSTMENT
    # ==========================================================================
    def apply_country_diversity(self) -> None:
        """
        Adjust country representation and refresh the partial correlations.
        """

        print_step(
            11,
            "Applying country diversity adjustment",
            "Ensure top contributing countries are represented in selected KPIs.",
            logger=self.logger,
        )

        # Apply the existing country-diversity rules to the selected set.
        (
            self.lag_corr_df_country_adjusted,
            self.selected_after_country_diversity_df,
            self.country_diversity_log_df,
            self.top_contributing_countries_df,
        ) = apply_country_diversity_adjustment(
            self.lag_corr_df_updated,
            contribution_file=self.config["country_contribution_file"],
            config=self.config["country_diversity"],
            logger=self.logger,
        )

        pc_cfg = self.config["partial_corr"]

        # Country replacements can change P0, so check candidates against
        # the final selected set before assigning the other priority sets.
        (
            self.lag_corr_df_country_adjusted,
            self.decision_df,
        ) = recompute_partial_corr_against_final_p0(
            model_df=self.model_df,
            lag_corr_df=self.lag_corr_df_country_adjusted,
            target_col=self.target_col,
            feature_col=pc_cfg.get("feature_col", "Feature"),
            status_col=pc_cfg.get("status_col", "Status"),
            partial_corr_threshold=float(pc_cfg.get("partial_corr_threshold", 0.10)),
            min_points=int(pc_cfg.get("min_points", 12)),
        )

        # Refresh the selected table using the updated results.
        status_col = pc_cfg.get("status_col", "Status")

        self.selected_after_country_diversity_df = self.lag_corr_df_country_adjusted[
            self.lag_corr_df_country_adjusted[status_col].eq("Selected")
        ].copy()

        self.logger.info(
            "Partial correlations refreshed against final P0: "
            f"{len(self.decision_df)}"
        )

        # Keep the earlier selected set so the report can show before and after.
        self.selected_before_country_diversity_df = self.lag_corr_df_updated[
            self.lag_corr_df_updated["Status"].eq("Selected")
        ].copy()

        self.logger.info(
            "Selected before country diversity: "
            f"{len(self.selected_before_country_diversity_df)}"
        )
        self.logger.info(
            "Selected after country diversity : "
            f"{len(self.selected_after_country_diversity_df)}"
        )

        print_top_rows(
            "COUNTRY DIVERSITY LOG",
            self.country_diversity_log_df,
            [
                "Action",
                "Missing_Country",
                "Added_Feature",
                "Added_Score",
                "Removed_Feature",
                "Removed_Country",
                "Removed_Score",
                "Reason",
            ],
            n=100,
            logger=self.logger,
        )

    # ==========================================================================
    # STEP 12 — PRIORITY SET CREATION: P0, P1, P2, P3
    # ==========================================================================
    def assign_priority_sets(self) -> None:
        """
        Assign priority sets and apply the mandatory KPI settings.
        """
        print_step(
            12,
            "Creating priority sets P0/P1/P2/P3",
            "Assign non-overlapping interpretation sets.",
            logger=self.logger,
        )

        post_set_cfg = self.config.get("post_selection_sets", {})

        # Assign the sets using our existing thresholds and rules.
        (
            self.lag_corr_df_priority_sets,
            self.priority_set_summary,
        ) = create_priority_sets(
            self.lag_corr_df_country_adjusted,
            post_set_cfg,
        )

        mandatory_cfg = get_mandatory_kpi_config(self.config)

        if mandatory_cfg["enabled"]:
            self.lag_corr_df_priority_sets = add_mandatory_flags(
                self.lag_corr_df_priority_sets,
                feature_col=post_set_cfg.get("feature_col", "Feature"),
                mandatory_cfg=mandatory_cfg,
            )

            mandatory_mask = self.lag_corr_df_priority_sets["Mandatory_KPI"]
            set_col = post_set_cfg.get("set_col", "Priority_Set")

            # Put mandatory KPIs in the set specified by their config.
            self.lag_corr_df_priority_sets.loc[mandatory_mask, set_col] = (
                mandatory_cfg.get("priority_set", "P0")
            )

            self.lag_corr_df_priority_sets.loc[
                mandatory_mask, "Priority_Set_Reason"
            ] = self.lag_corr_df_priority_sets.loc[
                mandatory_mask, "Priority_Set_Reason"
            ].apply(
                lambda x: append_reason(
                    x,
                    "OVERRIDDEN: mandatory KPI assigned to priority set",
                )
            )
        # Count the sets after all mandatory assignments are finished.
        self.priority_set_summary = (
            self.lag_corr_df_priority_sets[post_set_cfg.get("set_col", "Priority_Set")]
            .value_counts(dropna=False)
            .to_dict()
        )
        self.logger.info("Priority set summary: %s", self.priority_set_summary)

        print_top_rows(
            "PRIORITY SETS — P0/P1/P2/P3",
            self.lag_corr_df_priority_sets,
            [
                "Priority_Set",
                "Category",
                "Feature",
                "FCS_star",
                "dtw_score",
                "partial_corr",
                "Status",
                "Priority_Set_Reason",
            ],
            n=100,
            sort_col="FCS_star",
            logger=self.logger,
        )

    # =========================================================================
    # STEP 13 — SAVE EXCEL REPORT
    # =========================================================================
    # PURPOSE
    #   Save every important intermediate table so the feature selection decision can
    #   be audited without rerunning the code.
    # =============================================================================
    def save_report(
        self,
        full_data: pd.DataFrame,
        *,
        run_id: str,
        quarter: str,
    ) -> Path:
        """
        Save the existing Excel report and print the final summary.
        """

        # Use the settings and results already stored in this object.
        CONFIG = self.config
        division = str(self.config["division_name"])
        target = self.target_col

        # Save the tables under this run's division, quarter, and target.
        results_dir = (
            Path("runs") / division / quarter / run_id / "feature_selection" / target
        )

        # Keep the Excel report in the separate reports folder.
        report_dir = Path("reports") / division / quarter / "runs" / run_id

        results_dir.mkdir(parents=True, exist_ok=True)
        report_dir.mkdir(parents=True, exist_ok=True)

        # Save the tables under this run's division, quarter and target.

        DATE_COL = self.date_col
        TARGET_COL = self.target_col
        MODELLING_LEVEL_COL = self.modelling_level_col

        df = self.df
        df_full = full_data

        prefilter_df = self.prefilter_df
        candidate_features = self.candidate_features
        corr_long_df = self.corr_long_df
        model_df = self.model_df

        lag_corr_df = self.lag_corr_df
        lag_corr_df_original_features = self.lag_corr_df_original_features
        rejected_lag_kpis_df = self.rejected_lag_kpis_df
        family_info = self.family_info

        selected_step10 = self.selected_step10
        selected_rank1_df = self.selected_rank1_df
        decision_df = self.decision_df
        lag_corr_df_updated = self.lag_corr_df_updated

        selected_before_country_diversity_df = self.selected_before_country_diversity_df
        selected_after_country_diversity_df = self.selected_after_country_diversity_df
        top_contributing_countries_df = self.top_contributing_countries_df
        country_diversity_log_df = self.country_diversity_log_df

        lag_corr_df_priority_sets = self.lag_corr_df_priority_sets
        priority_set_summary = self.priority_set_summary

        print_step(
            13,
            "Saving feature selection report",
            "Write config, diagnostics, selected features, priority sets, and logs to Excel.",
            logger=self.logger,
        )

        output_excel = report_dir / "fs_report.xlsx"

        config_df = pd.DataFrame(
            [
                {
                    "config_key": key,
                    "config_value": json.dumps(value, ensure_ascii=False),
                }
                for key, value in CONFIG.items()
            ]
        )
        final_feature_table_df = lag_corr_df_priority_sets.copy()
        final_feature_cols = (
            final_feature_table_df["Feature"]
            .dropna()
            .astype(str)
            .drop_duplicates()
            .tolist()
        )
        base_cols = [MODELLING_LEVEL_COL, DATE_COL, TARGET_COL]

        # Apply the lag decisions learned from the training period to the complete
        # dataset. No feature or lag is re-selected using the post-training period.
        full_model_df = build_model_df_from_best_lags(
            df=df_full,
            lag_table=lag_corr_df_original_features,
            date_col=DATE_COL,
            modelling_level_col=MODELLING_LEVEL_COL,
            target_col=TARGET_COL,
        )

        # Build lagged data across the complete prepared period.
        # The export below keeps only rows with all required values available.
        full_model_df = full_model_df.reset_index(drop=True)

        # Check which final selected features exist in the full lagged dataset.
        available_final_feature_cols = [
            c for c in final_feature_cols if c in full_model_df.columns
        ]

        data_with_lagged_features_df = (
            full_model_df[base_cols + available_final_feature_cols]
            .dropna()
            .reset_index(drop=True)
        )

        selected_features_cols = [
            "Priority_Set",
            "Category",
            "Theme",
            "Mapped_KPI",
            "Feature",
            "Original_Feature",
            "Best_Lag",
            "Spearman_Corr",
            "Spearman_PValue",
            "MI_Raw",
            "LCEN",
            "FCS_pre",
            "dtw_score",
            "joint_score",
            "FCS_star",
            "country_raw",
            "Contribution",
            "Status",
            "Mandatory_KPI",
            "Override_Reason",
            "partial_corr",
            "partial_corr_reason",
            "Reason",
        ]
        # Use the final table so selected features include their priority sets.
        selected_rows = final_feature_table_df.loc[
            final_feature_table_df["Status"].eq("Selected")
        ]

        selected_features_df = selected_rows.loc[
            :,
            [col for col in selected_features_cols if col in selected_rows.columns],
        ].copy()
        family_info_df = pd.DataFrame([family_info])
        # Count all accepted lag decisions, including mandatory overrides.
        best_lag_ok_count = int(
            lag_corr_df_original_features["Reason"]
            .astype(str)
            .str.startswith("OK")
            .sum()
        )
        summary_df = pd.DataFrame(
            [
                {
                    "division_name": CONFIG.get("division_name", "N/A"),
                    "input_rows_after_target_prep": len(df),
                    "candidate_kpis": len(prefilter_df),
                    "passed_prefilter_kpis": len(candidate_features),
                    "best_lag_ok": best_lag_ok_count,
                    "dtw_tiebreak_used": int(
                        lag_corr_df_original_features["DTW_Selected"].sum()
                    ),
                    "model_df_rows": len(model_df),
                    "selected_top1_kpis": len(selected_step10),
                    "selected_before_country_diversity": len(
                        selected_before_country_diversity_df
                    ),
                    "selected_after_country_diversity": len(
                        selected_after_country_diversity_df
                    ),
                    "country_diversity_actions": len(country_diversity_log_df),
                    "partial_corr_decisions": len(decision_df),
                    "partial_corr_additions": (
                        int(decision_df["Add_Flag"].sum())
                        if not decision_df.empty
                        else 0
                    ),
                }
            ]
        )

        # Collect the report tables once, then save them in both formats.
        report_tables = {
            "config": config_df,
            "summary": summary_df,
            "prefilter": prefilter_df,
        }

        if not corr_long_df.empty:
            report_tables["full_lag_diagnostics"] = corr_long_df

        report_tables.update(
            {
                "data_with_lagged_features": data_with_lagged_features_df,
                "scored_features": lag_corr_df,
                "lag_selection_all": lag_corr_df_original_features,
                "rejected_lag_kpis": rejected_lag_kpis_df,
                "family_survival_info": family_info_df,
                "selected_features": selected_features_df,
                "selected_before_country": selected_before_country_diversity_df,
                "selected_after_country": selected_after_country_diversity_df,
                "top_contributing_countries": top_contributing_countries_df,
                "country_diversity_log": country_diversity_log_df,
                "selected_rank1": selected_rank1_df,
                "partial_corr_decisions": decision_df,
                "final_feature_before_country": lag_corr_df_updated,
                "final_feature_table": final_feature_table_df,
            }
        )

        set_col = self.config["post_selection_sets"].get("set_col", "Priority_Set")

        priority_features = {}

        for set_name in ["P0", "P1", "P2", "P3"]:
            table = final_feature_table_df[
                final_feature_table_df[set_col].eq(set_name)
            ].copy()

            # Keep the same sorting used in the existing Excel report.
            if set_name == "P1" and "partial_corr" in table.columns:
                table["partial_corr"] = pd.to_numeric(
                    table["partial_corr"], errors="coerce"
                )
                table["abs_partial_corr"] = table["partial_corr"].abs()
                table = table.sort_values(
                    "abs_partial_corr", ascending=False
                ).reset_index(drop=True)

            elif "FCS_star" in table.columns:
                table["FCS_star"] = pd.to_numeric(table["FCS_star"], errors="coerce")
                table = table.sort_values("FCS_star", ascending=False).reset_index(
                    drop=True
                )

            report_tables[f"{set_name}_set"] = table

            # The JSON contains feature names only, including their lag suffixes.
            priority_features[set_name] = (
                table["Feature"].dropna().astype(str).drop_duplicates().tolist()
            )

        # Save every report table as a separate Parquet file.
        for table_name, table in report_tables.items():
            table.to_parquet(
                results_dir / f"{table_name}.parquet",
                engine="pyarrow",
                index=False,
            )

        # Save the variable lists for downstream modules.
        with (results_dir / "fs_summary.json").open("w", encoding="utf-8") as file:
            json.dump(
                priority_features,
                file,
                indent=2,
                ensure_ascii=False,
            )

        # Save the same tables together in the Excel report.
        with pd.ExcelWriter(output_excel, engine="openpyxl") as writer:
            for table_name, table in report_tables.items():
                table.to_excel(
                    writer,
                    sheet_name=table_name,
                    index=False,
                )

        self.logger.info("\n" + "=" * 78)
        self.logger.info("FEATURE SELECTION SUMMARY")
        self.logger.info("=" * 78)
        self.logger.info(f"Passed prefilter KPIs     : {len(candidate_features)}")
        self.logger.info(f"Best lag OK               : {best_lag_ok_count}")
        self.logger.info(
            f"DTW tie-break used        : {lag_corr_df_original_features['DTW_Selected'].sum()}"
        )
        self.logger.info(f"Selected top-1 KPIs       : {len(selected_step10)}")
        self.logger.info(
            f"Selected before country   : {len(selected_before_country_diversity_df)}"
        )
        self.logger.info(
            f"Selected after country    : {len(selected_after_country_diversity_df)}"
        )
        self.logger.info(f"Country diversity actions : {len(country_diversity_log_df)}")
        self.logger.info(
            f"Partial-corr additions    : {decision_df['Add_Flag'].sum() if not decision_df.empty else 0}"
        )
        self.logger.info(f"Priority set summary      : {priority_set_summary}")
        self.logger.info(f"Excel report saved to     : {output_excel}")
        return output_excel

    # ==========================================================================
    # STEP 14
    # ==========================================================================
    def run(self, *, run_id: str, quarter: str) -> Path:
        """
        Run feature selection and record progress and failures.
        """

        self.setup_logger(run_id=run_id, quarter=quarter)

        try:
            self.logger.info(
                "Starting feature selection: division=%s, quarter=%s, run_id=%s",
                self.config["division_name"],
                quarter,
                run_id,
            )
            self.logger.info("Log file: %s", self.log_path.resolve())

            self.load_data()
            self.prefilter_kpis()
            self.build_kpi_mapping()
            self.select_best_lags()
            self.build_model_data()
            self.compute_mi_signal()
            self.apply_cointegration()
            self.compute_lcen_stability()
            self.compute_dtw_score()
            self.apply_business_scores()
            self.select_category_kpis()
            self.apply_country_diversity()
            self.assign_priority_sets()

            report_path = self.save_report(
                full_data=self.df_full,
                run_id=run_id,
                quarter=quarter,
            )

            self.logger.info("Feature selection completed successfully.")
            return report_path

        except Exception:
            # Save the traceback and let the caller know the run failed.
            self.logger.exception("Feature selection failed.")
            raise

        finally:
            # Release the file when the run finishes or fails.
            for handler in list(self.logger.handlers):
                handler.close()
                self.logger.removeHandler(handler)

    # =========================================================================
    # Logging
    # =========================================================================
    def setup_logger(self, *, run_id: str, quarter: str) -> None:
        """Write progress messages to the terminal and this run's log file."""

        # Work out the target name before loading the data.
        rolling_window = int(self.config.get("rolling_window", 3))
        self.target_col = (
            self.raw_target_col if rolling_window == 1 else self.config["target_col"]
        )

        log_dir = (
            Path("runs")
            / str(self.config["division_name"])
            / quarter
            / run_id
            / "feature_selection"
            / self.target_col
        )
        log_dir.mkdir(parents=True, exist_ok=True)

        self.log_path = log_dir / "feature_selection.log"

        # Give this pipeline object its own logger.
        self.logger = logging.Logger(
            f"feature_selection.{run_id}.{id(self)}",
            level=logging.INFO,
        )
        self.logger.propagate = False

        formatter = logging.Formatter(
            "%(asctime)s %(levelname)s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # Append so retrying the same run keeps the earlier messages.
        file_handler = logging.FileHandler(
            self.log_path,
            mode="a",
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)

        terminal_handler = logging.StreamHandler(sys.stdout)
        terminal_handler.setFormatter(formatter)

        self.logger.addHandler(file_handler)
        self.logger.addHandler(terminal_handler)


# =============================================================================
# STEP 2 — BUILD KPI CATEGORY/THEME MAPPING
# =============================================================================
# PURPOSE
#   Attach business Category/Theme metadata to each KPI.
#
# WHY
#   Later selection is category-aware. We usually want the best KPI per business
#   category, not simply the top N statistical variables overall.
# =============================================================================
def build_prefilter_diagnostics(
    df: pd.DataFrame,
    *,
    date_col: str,
    target_col: str,
    missing_threshold: float,
    cv_threshold: float,
    min_unique: int,
    pref_cfg: dict,
    ignore_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Build pass/fail diagnostics for all numeric KPI columns.

    New CV logic:
      - DROP: very low robust CV
      - WARN: borderline low robust CV, but allowed to continue
      - KEEP: acceptable robust CV
    """
    ignore_cols = ignore_cols or []
    exclude = set([date_col, target_col] + ignore_cols)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c not in exclude]

    rows = []

    for col in feature_cols:
        s = to_numeric_series(df[col])

        missing_frac = float(s.isna().mean())
        nunique = int(s.nunique(dropna=True))
        has_negative = bool((s.dropna() < 0).any())
        rcv = robust_cv_mad(s)

        mean = float(np.nanmean(s))
        std = float(np.nanstd(s, ddof=1))
        classic_cv = 100.0 * (std / (abs(mean) + 1e-9)) if np.isfinite(std) else np.nan

        pass_missing = missing_frac <= missing_threshold
        pass_unique = nunique >= min_unique
        pass_negative = not has_negative

        rows.append(
            {
                "Feature": col,
                "Missing_Fraction": missing_frac,
                "Unique_Values": nunique,
                "Has_Negative": has_negative,
                "Robust_CV_%": rcv,
                "Classic_CV_%": classic_cv,
                "pass_missing": pass_missing,
                "pass_unique": pass_unique,
                "pass_non_negative": pass_negative,
            }
        )

    out = pd.DataFrame(rows)

    if out.empty:
        return out

    # ---------------------------------------------------------
    # NEW ROBUST CV LOGIC
    # ---------------------------------------------------------

    cv_mode = pref_cfg.get("cv_mode", "threshold")

    if cv_mode == "percentile":
        absolute_min_cv = float(pref_cfg.get("absolute_min_cv", 1.0))
        drop_pct = float(pref_cfg.get("cv_drop_percentile", 0.10))
        warn_pct = float(pref_cfg.get("cv_warn_percentile", 0.25))

        out["Robust_CV_Percentile"] = out["Robust_CV_%"].rank(pct=True)

        out["CV_Status"] = "KEEP"

        out.loc[out["Robust_CV_%"] < absolute_min_cv, "CV_Status"] = "DROP"

        out.loc[
            (out["Robust_CV_%"] >= absolute_min_cv)
            & (out["Robust_CV_Percentile"] < drop_pct),
            "CV_Status",
        ] = "DROP"

        out.loc[
            (out["CV_Status"] != "DROP") & (out["Robust_CV_Percentile"] < warn_pct),
            "CV_Status",
        ] = "WARN"

    else:
        out["Robust_CV_Percentile"] = np.nan
        out["CV_Status"] = np.where(out["Robust_CV_%"] >= cv_threshold, "KEEP", "DROP")

    # WARN is allowed to pass
    out["pass_robust_cv"] = out["CV_Status"].isin(["KEEP", "WARN"])

    # Final pass logic
    out["overall_pass"] = (
        out["pass_missing"]
        & out["pass_unique"]
        & out["pass_non_negative"]
        & out["pass_robust_cv"]
    )

    # ---------------------------------------------------------
    # REASON LOGIC
    # ---------------------------------------------------------
    reasons_list = []

    for _, row in out.iterrows():
        reasons = []

        if not row["pass_non_negative"]:
            reasons.append("Contains negative values (cannot log-transform)")

        if not row["pass_missing"]:
            reasons.append(
                f"High missingness: {row['Missing_Fraction']:.1%} > {missing_threshold:.1%}"
            )

        if not row["pass_unique"]:
            reasons.append(
                f"Constant/near-constant: nunique={int(row['Unique_Values'])} < {min_unique}"
            )

        if row["CV_Status"] == "DROP":
            reasons.append(
                f"Low variability: Robust_CV={row['Robust_CV_%']:.2f}%, "
                f"CV_Status=DROP"
            )

        elif row["CV_Status"] == "WARN":
            reasons.append(
                f"PASS with warning: borderline low variability; "
                f"Robust_CV={row['Robust_CV_%']:.2f}%, CV_Status=WARN"
            )

        if not reasons:
            reasons.append("PASS")

        reasons_list.append("; ".join(reasons))

    out["Drop_Reason"] = reasons_list

    return out.sort_values(
        by=["overall_pass", "CV_Status", "Robust_CV_%", "Missing_Fraction"],
        ascending=[True, True, True, False],
        na_position="last",
    ).reset_index(drop=True)


# =============================================================================
# STEP 3 — LAG SELECTION USING SPEARMAN + DTW TIE-BREAK
# =============================================================================
# PURPOSE
#   Identify the lag at which each KPI best relates to the rolling target.
#
# WHY
#   Macro/business KPIs usually affect orders with delay. Picking the wrong lag
#   can hide a meaningful relationship or create misleading correlations.
#
# METHOD
#   1. Transform target and KPI consistently, usually log_yoy.
#   2. Compute Spearman correlation across candidate lags.
#   3. Pick the lag with highest absolute correlation.
#   4. If the best two lags are close, use DTW as a tie-breaker.
# =============================================================================


def compute_lag_correlations(
    x_base: pd.Series,
    y: pd.Series,
    lags: tuple[int, ...],
    min_points: int,
    min_abs_corr_threshold: float = 0.10,
) -> list[dict]:
    """
    Return Spearman correlation + p-value for one KPI across candidate lags.

    New rule:
      - Do NOT use p-value/significance for rejection.
      - A lag qualifies only if abs(correlation) >= min_abs_corr_threshold.
    """
    results = []

    for lag in lags:
        x = x_base.shift(lag)
        tmp = pd.DataFrame({"x": x, "y": y}).dropna()

        if len(tmp) < min_points:
            continue

        if tmp["x"].nunique() < 2 or tmp["y"].nunique() < 2:
            continue

        corr, p_value = spearmanr(tmp["x"], tmp["y"])

        if np.isnan(corr):
            continue

        abs_corr = abs(float(corr))
        # Use the same qualification decision for selection and diagnostics.
        qualified = bool(round(abs_corr, 1) >= min_abs_corr_threshold)

        results.append(
            {
                "lag": int(lag),
                "corr": float(corr),
                "abs_corr": abs_corr,
                "p_value": float(p_value) if pd.notna(p_value) else np.nan,
                "qualified": qualified,
                "significant": qualified,
                "n": int(len(tmp)),
            }
        )

    return sorted(results, key=lambda d: d["abs_corr"], reverse=True)


def choose_final_lag_with_dtw(
    x_base: pd.Series,
    y: pd.Series,
    lag_results: list[dict],
    lag_gap_threshold: float,
) -> tuple[dict, dict | None, float, bool, float, float]:
    """Pick final lag. Use DTW only if top two Spearman lags are close."""
    best = lag_results[0]
    second = lag_results[1] if len(lag_results) > 1 else None
    final = best
    lag_gap = np.nan
    dtw_selected = False
    dtw_best = np.nan
    dtw_second = np.nan

    if second is not None:
        lag_gap = best["abs_corr"] - second["abs_corr"]
        if lag_gap < lag_gap_threshold:
            dtw_best = simple_dtw_distance(x_base.shift(best["lag"]), y)
            dtw_second = simple_dtw_distance(x_base.shift(second["lag"]), y)
            if dtw_second < dtw_best:
                final = second
            dtw_selected = True

    return final, second, lag_gap, dtw_selected, dtw_best, dtw_second


def build_lag_result_row(
    feature: str, best: dict | None, second: dict | None, final: dict | None, **extras
) -> dict:
    """Create one standardized row for the best-lag table."""

    lag_results = extras.get("lag_results", [])
    sig_count = int(sum(r.get("significant", False) for r in lag_results))

    # Convert lag_results into dictionary so diagnostics sheet can show lag-wise values
    lag_map = {int(r["lag"]): r for r in lag_results}

    base_row = {
        "Feature": feature,
        "Mandatory_KPI": bool(extras.get("mandatory_kpi", False)),
        "Override_Reason": extras.get("override_reason", ""),
        # Category-based lag audit
        "Category_Used_For_Lag": extras.get("category_used_for_lag"),
        "Theme_Used_For_Lag": extras.get("theme_used_for_lag"),
        "Mapped_KPI_Used_For_Lag": extras.get("mapped_kpi_used_for_lag"),
        "Lags_Checked": extras.get("lags_checked"),
        # Lag-wise correlations
        "Lag_0_Corr": lag_map.get(0, {}).get("corr", np.nan),
        "Lag_3_Corr": lag_map.get(3, {}).get("corr", np.nan),
        "Lag_6_Corr": lag_map.get(6, {}).get("corr", np.nan),
        "Lag_9_Corr": lag_map.get(9, {}).get("corr", np.nan),
        "Lag_12_Corr": lag_map.get(12, {}).get("corr", np.nan),
        # Lag-wise p-values
        "Lag_0_PValue": lag_map.get(0, {}).get("p_value", np.nan),
        "Lag_3_PValue": lag_map.get(3, {}).get("p_value", np.nan),
        "Lag_6_PValue": lag_map.get(6, {}).get("p_value", np.nan),
        "Lag_9_PValue": lag_map.get(9, {}).get("p_value", np.nan),
        "Lag_12_PValue": lag_map.get(12, {}).get("p_value", np.nan),
        # Lag-wise significance flags
        "Lag_0_Significant": lag_map.get(0, {}).get("significant", False),
        "Lag_3_Significant": lag_map.get(3, {}).get("significant", False),
        "Lag_6_Significant": lag_map.get(6, {}).get("significant", False),
        "Lag_9_Significant": lag_map.get(9, {}).get("significant", False),
        "Lag_12_Significant": lag_map.get(12, {}).get("significant", False),
        # KPI-level diagnostics
        "Sig_Count": sig_count,
        "Min_Significant_Lags": extras.get("min_significant_lags"),
        "Sign_Stability_Ratio": extras.get("sign_stability_ratio", np.nan),
        "N_Positive_Sig_Lags": extras.get("n_pos_sig_lags", np.nan),
        "N_Negative_Sig_Lags": extras.get("n_neg_sig_lags", np.nan),
    }
    if best is None or final is None:
        base_row.update(
            {
                "Spearman_Best_Lag": np.nan,
                "Spearman_Best_Corr": np.nan,
                "Spearman_Best_Abs_Corr": np.nan,
                "Spearman_Best_PValue": np.nan,
                "Spearman_Second_Best_Lag": np.nan,
                "Spearman_Second_Best_Corr": np.nan,
                "Spearman_Second_Best_Abs_Corr": np.nan,
                "Spearman_Second_Best_PValue": np.nan,
                "Best_Lag": np.nan,
                "Spearman_Corr": np.nan,
                "Abs_Spearman_Corr": np.nan,
                "Spearman_PValue": np.nan,
                "Lag_Gap": np.nan,
                "DTW_Selected": False,
                "DTW_Changed_Lag": False,
                "DTW_Best": np.nan,
                "DTW_Second": np.nan,
                "n_points": 0,
                "Transform_Method": extras.get("transform_method"),
                "Reason": extras.get("reason", "No valid lag"),
            }
        )
        return base_row
    base_row.update(
        {
            "Spearman_Best_Lag": int(best["lag"]),
            "Spearman_Best_Corr": round(float(best["corr"]), 3),
            "Spearman_Best_Abs_Corr": round(float(best["abs_corr"]), 3),
            "Spearman_Best_PValue": round(float(best["p_value"]), 4),
            "Spearman_Second_Best_Lag": (
                int(second["lag"]) if second is not None else np.nan
            ),
            "Spearman_Second_Best_Corr": (
                round(float(second["corr"]), 3) if second is not None else np.nan
            ),
            "Spearman_Second_Best_Abs_Corr": (
                round(float(second["abs_corr"]), 3) if second is not None else np.nan
            ),
            "Spearman_Second_Best_PValue": (
                round(float(second["p_value"]), 4) if second is not None else np.nan
            ),
            "Best_Lag": int(final["lag"]),
            "Spearman_Corr": round(float(final["corr"]), 3),
            "Abs_Spearman_Corr": round(float(final["abs_corr"]), 3),
            "Spearman_PValue": round(float(final["p_value"]), 4),
            "Lag_Gap": (
                round(float(extras.get("lag_gap", np.nan)), 3)
                if pd.notna(extras.get("lag_gap", np.nan))
                else np.nan
            ),
            "DTW_Selected": bool(extras.get("dtw_selected", False)),
            "DTW_Changed_Lag": bool(
                extras.get("dtw_selected", False) and final["lag"] != best["lag"]
            ),
            "DTW_Best": (
                round(float(extras.get("dtw_best", np.nan)), 4)
                if np.isfinite(extras.get("dtw_best", np.nan))
                else np.nan
            ),
            "DTW_Second": (
                round(float(extras.get("dtw_second", np.nan)), 4)
                if np.isfinite(extras.get("dtw_second", np.nan))
                else np.nan
            ),
            "n_points": int(final["n"]),
            "Transform_Method": extras.get("transform_method"),
            "Reason": extras.get("reason", "OK"),
        }
    )
    return base_row


def lag_corr_table(
    *,
    df: pd.DataFrame,
    date_col: str,
    target_col: str,
    candidate_features: list[str],
    max_lag: int,
    min_points: int,
    transform_method: str,
    yoy_period: int,
) -> pd.DataFrame:
    """Optional full lag diagnostic table for all lags from 0..max_lag."""
    work = df.copy()
    work[target_col] = to_numeric_series(work[target_col])
    if date_col in work.columns:
        work[date_col] = pd.to_datetime(work[date_col], errors="coerce")
        work = work.sort_values(date_col).reset_index(drop=True)

    y = transform_series(
        work[target_col], method=transform_method, yoy_period=yoy_period
    )
    rows = []
    for feat in candidate_features:
        if feat not in work.columns:
            for lag in range(max_lag + 1):
                rows.append(
                    {
                        "Feature": feat,
                        "Lag": lag,
                        "Spearman_Corr": np.nan,
                        "Abs_Spearman_Corr": np.nan,
                        "p_value": np.nan,
                        "n_points": 0,
                        "Transform_Method": transform_method,
                        "Reason": "Feature not in dataframe",
                    }
                )
            continue
        x_base = transform_series(
            work[feat], method=transform_method, yoy_period=yoy_period
        )
        for lag in range(max_lag + 1):
            tmp = pd.DataFrame({"x": x_base.shift(lag), "y": y}).dropna()
            n = len(tmp)
            if n < min_points:
                corr, pval, reason = (
                    np.nan,
                    np.nan,
                    f"Insufficient valid data (min_points={min_points})",
                )
            elif tmp["x"].nunique() < 2 or tmp["y"].nunique() < 2:
                corr, pval, reason = np.nan, np.nan, "Constant series after dropna"
            else:
                corr, pval = spearmanr(tmp["x"], tmp["y"])
                reason = "NaN correlation" if np.isnan(corr) else "OK"
            rows.append(
                {
                    "Feature": feat,
                    "Lag": lag,
                    "Spearman_Corr": float(corr) if pd.notna(corr) else np.nan,
                    "Abs_Spearman_Corr": float(abs(corr)) if pd.notna(corr) else np.nan,
                    "p_value": float(pval) if pd.notna(pval) else np.nan,
                    "n_points": int(n),
                    "Transform_Method": transform_method,
                    "Reason": reason,
                }
            )
    return pd.DataFrame(rows).sort_values(by=["Feature", "Lag"]).reset_index(drop=True)


# =============================================================================
# LAG SELECTION BUSINESS RULES
# =============================================================================
# For every KPI, we do not simply pick the lag with highest correlation.
#
# Rule 1:
#   A KPI must show significance across at least N lags.
#   This avoids selecting a KPI that works only at one random lag.
#
# Rule 2:
#   Significant lags should mostly have the same sign direction.
#   Example: if lag 3 is positive but lag 6 is negative, the KPI is unstable.
#
# Rule 3:
#   Final lag is selected only from significant lags.
#
# Rule 4:
#   If top lags are close, DTW is used as a tie-breaker to check movement alignment.
# =============================================================================
def best_lag_corr_table(
    *,
    df: pd.DataFrame,
    date_col: str,
    target_col: str,
    candidate_features: list[str],
    feature_to_category: dict | None,
    feature_to_theme: dict | None,
    feature_to_mapped_kpi: dict | None,
    min_points: int,
    default_lags: tuple[int, ...],
    category_lags: dict | None,
    transform_method: str,
    yoy_period: int,
    lag_gap_threshold: float,
    min_abs_corr_threshold: float,
    min_qualifying_lags: int,
    use_dtw_tiebreak: bool,
    mandatory_cfg: dict,
) -> pd.DataFrame:
    """
    Select best lag for every candidate feature.

    New logic:
    1. Sweep all configured lags.
    2. Keep lags where abs(Spearman correlation) > min_abs_corr_threshold.
    3. If qualifying lag count < min_qualifying_lags, reject.
    4. If qualifying lags have mixed signs, reject.
    5. Select the lag with highest abs(Spearman) among same-sign qualifying lags.
    """
    work = df.copy()
    work[target_col] = to_numeric_series(work[target_col])
    if date_col in work.columns:
        work[date_col] = pd.to_datetime(work[date_col], errors="coerce")
        work = work.sort_values(date_col).reset_index(drop=True)
    feature_to_category = feature_to_category or {}
    feature_to_theme = feature_to_theme or {}
    feature_to_mapped_kpi = feature_to_mapped_kpi or {}
    category_lags = category_lags or {}
    y = transform_series(
        work[target_col], method=transform_method, yoy_period=yoy_period
    )
    rows = []
    for feat in candidate_features:
        category_used_for_lag = feature_to_category.get(feat)
        theme_used_for_lag = feature_to_theme.get(feat)
        mapped_kpi_used_for_lag = feature_to_mapped_kpi.get(feat)
        mandatory_kpi = is_mandatory_kpi(feat, mandatory_cfg)
        override_lag_rejection = mandatory_kpi and mandatory_cfg.get(
            "override_lag_rejection", True
        )
        lags_to_check = get_lags_for_feature_category(
            feature=feat,
            feature_to_category=feature_to_category,
            category_lags=category_lags,
            default_lags=default_lags,
        )
        if feat not in work.columns:
            rows.append(
                build_lag_result_row(
                    feat,
                    None,
                    None,
                    None,
                    transform_method=transform_method,
                    reason="Feature not in dataframe",
                    min_significant_lags=min_qualifying_lags,
                    lag_results=[],
                    category_used_for_lag=category_used_for_lag,
                    theme_used_for_lag=theme_used_for_lag,
                    mapped_kpi_used_for_lag=mapped_kpi_used_for_lag,
                    lags_checked=list(lags_to_check),
                )
            )
            continue
        x_base = transform_series(
            work[feat], method=transform_method, yoy_period=yoy_period
        )
        lag_results = compute_lag_correlations(
            x_base=x_base,
            y=y,
            lags=lags_to_check,
            min_points=min_points,
            min_abs_corr_threshold=min_abs_corr_threshold,
        )
        if not lag_results:
            rows.append(
                build_lag_result_row(
                    feat,
                    None,
                    None,
                    None,
                    transform_method=transform_method,
                    reason=f"Insufficient valid data across lags {lags_to_check}",
                    min_significant_lags=min_qualifying_lags,
                    lag_results=[],
                    category_used_for_lag=category_used_for_lag,
                    theme_used_for_lag=theme_used_for_lag,
                    mapped_kpi_used_for_lag=mapped_kpi_used_for_lag,
                    lags_checked=list(lags_to_check),
                )
            )
            continue
        # -------------------------------------------------------------------------
        # NEW RULE 1 — Keep only lags where abs(correlation) >= threshold
        # -------------------------------------------------------------------------
        qualified_lag_results = [r for r in lag_results if r.get("qualified", False)]
        qualified_count = len(qualified_lag_results)

        # -------------------------------------------------------------------------
        # DROP if fewer than 3 qualifying lags
        # -------------------------------------------------------------------------
        if qualified_count < min_qualifying_lags and not override_lag_rejection:
            best = lag_results[0]
            second = lag_results[1] if len(lag_results) > 1 else None
            rows.append(
                build_lag_result_row(
                    feat,
                    best,
                    second,
                    best,
                    transform_method=transform_method,
                    reason=(
                        f"Rejected: fewer than {min_qualifying_lags} lags with "
                        f"|r| > {min_abs_corr_threshold}; "
                        f"qualified_count={qualified_count}"
                    ),
                    min_significant_lags=min_qualifying_lags,
                    lag_results=lag_results,
                    category_used_for_lag=category_used_for_lag,
                    theme_used_for_lag=theme_used_for_lag,
                    mapped_kpi_used_for_lag=mapped_kpi_used_for_lag,
                    lags_checked=list(lags_to_check),
                )
            )
            continue
        # Mandatory fallback: if business forces KPI, select best available lag
        if override_lag_rejection and qualified_count == 0:
            qualified_lag_results = [lag_results[0]]
            qualified_count = 1
        # -------------------------------------------------------------------------
        # NEW RULE 2 — Among qualifying lags, signs must be consistent
        # Requirement:
        #   >= 3 lags with |r| > 0.1 and same sign => KEEP
        #   >= 3 lags with |r| > 0.1 but mixed signs => DROP
        # -------------------------------------------------------------------------
        positive_lags = [r for r in qualified_lag_results if r["corr"] > 0]
        negative_lags = [r for r in qualified_lag_results if r["corr"] < 0]
        n_pos = len(positive_lags)
        n_neg = len(negative_lags)
        if n_pos >= min_qualifying_lags:
            same_sign_lag_results = positive_lags
            sign_direction = "positive"
        elif n_neg >= min_qualifying_lags:
            same_sign_lag_results = negative_lags
            sign_direction = "negative"
        else:
            best = qualified_lag_results[0]
            second = (
                qualified_lag_results[1] if len(qualified_lag_results) > 1 else None
            )
            if not override_lag_rejection:
                rows.append(
                    build_lag_result_row(
                        feat,
                        best,
                        second,
                        best,
                        transform_method=transform_method,
                        reason=(
                            f"Rejected: mixed signs among lags with "
                            f"|r| > {min_abs_corr_threshold}; "
                            f"positive_lags={n_pos}, negative_lags={n_neg}, "
                            f"required_same_sign_lags={min_qualifying_lags}"
                        ),
                        min_significant_lags=min_qualifying_lags,
                        lag_results=lag_results,
                        category_used_for_lag=category_used_for_lag,
                        theme_used_for_lag=theme_used_for_lag,
                        mapped_kpi_used_for_lag=mapped_kpi_used_for_lag,
                        lags_checked=list(lags_to_check),
                        sign_stability_ratio=(
                            max(n_pos, n_neg) / qualified_count
                            if qualified_count
                            else np.nan
                        ),
                        n_pos_sig_lags=n_pos,
                        n_neg_sig_lags=n_neg,
                    )
                )
                continue
            # If mandatory override, still select best qualifying lag
            same_sign_lag_results = qualified_lag_results
            sign_direction = "mandatory_override_mixed_sign"
        # -------------------------------------------------------------------------
        # NEW LAG SELECTION — highest |r| among qualifying same-sign lags
        # -------------------------------------------------------------------------
        same_sign_lag_results = sorted(
            same_sign_lag_results, key=lambda d: d["abs_corr"], reverse=True
        )
        best = same_sign_lag_results[0]
        if use_dtw_tiebreak:
            final, second, lag_gap, dtw_selected, dtw_best, dtw_second = (
                choose_final_lag_with_dtw(
                    x_base=x_base,
                    y=y,
                    lag_results=same_sign_lag_results,
                    lag_gap_threshold=lag_gap_threshold,
                )
            )
        else:
            final = best
            second = (
                same_sign_lag_results[1] if len(same_sign_lag_results) > 1 else None
            )
            lag_gap = (
                best["abs_corr"] - second["abs_corr"] if second is not None else np.nan
            )
            dtw_selected = False
            dtw_best = np.nan
            dtw_second = np.nan
        rows.append(
            build_lag_result_row(
                feat,
                best,
                second,
                final,
                transform_method=transform_method,
                lag_gap=lag_gap,
                dtw_selected=dtw_selected,
                dtw_best=dtw_best,
                dtw_second=dtw_second,
                reason=(
                    f"OK: selected highest |r| lag among {sign_direction} qualifying lags"
                    if not override_lag_rejection
                    else "OK | OVERRIDDEN: mandatory KPI selected by business rule"
                ),
                mandatory_kpi=mandatory_kpi,
                override_reason=(
                    "Mandatory KPI override at lag-selection gate"
                    if override_lag_rejection
                    else ""
                ),
                min_significant_lags=min_qualifying_lags,
                lag_results=lag_results,
                category_used_for_lag=category_used_for_lag,
                theme_used_for_lag=theme_used_for_lag,
                mapped_kpi_used_for_lag=mapped_kpi_used_for_lag,
                lags_checked=list(lags_to_check),
                sign_stability_ratio=(
                    max(n_pos, n_neg) / qualified_count if qualified_count else np.nan
                ),
                n_pos_sig_lags=n_pos,
                n_neg_sig_lags=n_neg,
            )
        )
    return (
        pd.DataFrame(rows)
        .sort_values(
            by=["Reason", "Abs_Spearman_Corr", "n_points"],
            ascending=[True, False, False],
        )
        .reset_index(drop=True)
    )


# =============================================================================
# STEP 4 — BUILD LAGGED MODEL DATAFRAME
# =============================================================================
# PURPOSE
#   Convert the best-lag decision table into an actual modelling dataframe.
#
# WHY
#   From this point onward, each feature column must refer to the selected lag.
#   For example, "China IIP" with lag 6 becomes "China IIP | lag_6".
# =============================================================================


def build_model_df_from_best_lags(
    *,
    df: pd.DataFrame,
    lag_table: pd.DataFrame,
    date_col: str,
    modelling_level_col: str,
    target_col: str,
    feature_col: str = "Feature",
    best_lag_col: str = "Best_Lag",
    only_ok: bool = True,
) -> pd.DataFrame:
    """Build model dataframe with selected lagged KPI columns."""
    work = df.copy()
    work[date_col] = pd.to_datetime(work[date_col], errors="coerce")
    work = work.sort_values([modelling_level_col, date_col]).reset_index(drop=True)
    out = work[[modelling_level_col, date_col, target_col]].copy()
    lt = lag_table.copy()
    if only_ok and "Reason" in lt.columns:
        lt = lt[lt["Reason"].astype(str).str.startswith("OK")].copy()
    lt = lt.dropna(subset=[best_lag_col]).copy()
    lt[best_lag_col] = lt[best_lag_col].astype(int)
    for _, r in lt.iterrows():
        feat = r[feature_col]
        lag = int(r[best_lag_col])
        if feat not in work.columns:
            continue
        s = to_numeric_series(work[feat])
        out[f"{feat} | lag_{lag}"] = s.groupby(work[modelling_level_col]).shift(lag)
    return out


# =============================================================================
# STEP 5 — MI + CORRELATION SIGNAL
# =============================================================================
# PURPOSE
#   Create a first signal score using both correlation and mutual information.
#
# WHY
#   Spearman captures monotonic association. MI can capture broader non-linear
#   dependency. Taking the better percentile rank protects signals that are
#   missed by one of the two methods.
# =============================================================================


def compute_single_mi(
    x: pd.Series, y: pd.Series, min_points: int, n_neighbors: int, random_state: int
) -> tuple[float, str]:
    """Compute MI for one feature with readable failure reasons."""
    tmp = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(tmp) < min_points:
        return np.nan, f"insufficient_points={len(tmp)}"
    if tmp["x"].nunique() <= 1:
        return np.nan, "x_constant_after_transform"
    if tmp["y"].nunique() <= 1:
        return np.nan, "y_constant_after_transform"
    try:
        mi = mutual_info_regression(
            tmp[["x"]].to_numpy(),
            tmp["y"].to_numpy(),
            n_neighbors=n_neighbors,
            random_state=random_state,
        )[0]
        return float(mi), "ok"
    except Exception as e:
        return np.nan, f"mi_failed:{type(e).__name__}"


def add_mi_and_signal(
    df: pd.DataFrame,
    lag_corr_df: pd.DataFrame,
    *,
    date_col: str,
    target_col: str,
    feature_col: str = "Feature",
    corr_col: str = "Spearman_Corr",
    n_neighbors: int,
    random_state: int,
    min_points: int,
    transform_method: str,
    yoy_period: int,
) -> pd.DataFrame:
    """Compute MI for lagged features and combine MI/correlation percentile ranks."""
    out = lag_corr_df.copy()
    out["corr_abs"] = pd.to_numeric(out[corr_col], errors="coerce").abs()
    work = df.copy()
    if date_col in work.columns:
        work[date_col] = pd.to_datetime(work[date_col], errors="coerce")
        work = work.sort_values(date_col).reset_index(drop=True)
    y_full = transform_series(
        work[target_col], method=transform_method, yoy_period=yoy_period
    )
    mi_vals, mi_reasons = [], []
    for _, row in out.iterrows():
        feat = row[feature_col]
        if feat not in work.columns:
            mi_vals.append(np.nan)
            mi_reasons.append("feature_missing_in_df")
            continue
        x = transform_series(work[feat], method=transform_method, yoy_period=yoy_period)
        mi, reason = compute_single_mi(x, y_full, min_points, n_neighbors, random_state)
        mi_vals.append(mi)
        mi_reasons.append(reason)
    out["MI_Raw"] = mi_vals
    out["mi_reason"] = mi_reasons
    out["corr_pct"] = safe_rank_pct(out["corr_abs"])
    out["mi_pct"] = safe_rank_pct(out["MI_Raw"])
    out["signal"] = out[["corr_pct", "mi_pct"]].max(axis=1)
    out["MI_Transform_Method"] = transform_method
    out["MI_Already_Lagged"] = True
    return out


# =============================================================================
# STEP 6 — COINTEGRATION FLOOR
# =============================================================================
# PURPOSE
#   Protect long-run economic relationships that may look weak in short-term
#   correlation/MI scoring.
#
# BUSINESS INTUITION
#   Some macro KPIs and the target can move together over the long run even if
#   month-to-month changes are noisy. Cointegration detects this long-run link.
# =============================================================================


def apply_cointegration_floor(
    df: pd.DataFrame,
    lag_corr_df: pd.DataFrame,
    *,
    date_col: str,
    target_col: str,
    feature_col: str = "Feature",
    signal_col: str = "signal",
    coint_alpha: float,
    floor_value: float,
    min_points: int,
) -> pd.DataFrame:
    """Apply a minimum signal floor where target and feature are cointegrated."""
    out = lag_corr_df.copy()
    work = df.copy()
    if date_col in work.columns:
        work[date_col] = pd.to_datetime(work[date_col], errors="coerce")
        work = work.sort_values(date_col).reset_index(drop=True)
    y_full = to_numeric_series(work[target_col])
    pvals, coint_flags, new_signals = [], [], []
    for _, row in out.iterrows():
        feat = row[feature_col]
        original_signal = row[signal_col]
        pvals.append(np.nan)
        coint_flags.append(False)
        new_signals.append(original_signal)
        if feat not in work.columns:
            continue
        tmp = pd.DataFrame({"x": to_numeric_series(work[feat]), "y": y_full}).dropna()
        if len(tmp) < min_points:
            continue
        try:
            _, pvalue, _ = coint(tmp["y"], tmp["x"])
        except Exception:
            continue
        pvals[-1] = pvalue
        if pvalue < coint_alpha:
            coint_flags[-1] = True
            new_signals[-1] = max(original_signal, floor_value)
    out["coint_pvalue"] = pvals
    out["cointegrated"] = coint_flags
    out["signal_after_coint"] = new_signals
    return out


# =============================================================================
# STEP 7 — LCEN STABILITY + FCS_PRE
# =============================================================================
# PURPOSE
#   Reward features that are repeatedly selected by ElasticNet across bootstrap
#   samples and regularisation settings.
#
# WHY
#   A KPI with high signal but unstable selection is risky. LCEN helps separate
#   stable drivers from one-off noisy correlations.
# =============================================================================


def moving_block_bootstrap_indices(n_obs: int, block: int, rng_obj) -> np.ndarray:
    """Sample time-series blocks so bootstrap preserves local temporal structure."""
    starts = rng_obj.integers(
        0, max(1, n_obs - block + 1), size=max(1, n_obs // block + 1)
    )
    return np.concatenate([np.arange(s, min(s + block, n_obs)) for s in starts])[:n_obs]


# =============================================================================
# LCEN INTERPRETATION
# =============================================================================
# LCEN is a stability score.
# It checks how often a KPI is selected by ElasticNet across bootstrap samples
# and different regularisation strengths.
#
# High LCEN:
#   KPI is repeatedly useful across many samples/settings.
#
# Low LCEN:
#   KPI may be noisy, unstable, or dependent on a specific sample split.
#
# FCS_pre combines:
#   signal_after_coint + sqrt(LCEN)
# =============================================================================
def compute_lcen_and_fcs_pre(
    df: pd.DataFrame,
    lag_corr_df: pd.DataFrame,
    *,
    date_col: str,
    target_col: str,
    feature_col: str = "Feature",
    signal_col: str = "signal_after_coint",
    lcen_weight: float,
    signal_weight: float,
    n_bootstraps: int,
    alphas: list[float],
    l1_ratios: list[float],
    random_state: int,
    block_size: int | None,
    coef_tol: float,
) -> pd.DataFrame:
    """Compute LCEN stability and FCS_pre."""
    out = lag_corr_df.copy()
    work = df.copy()
    if date_col in work.columns:
        work = work.sort_values(date_col).reset_index(drop=True)
    work[target_col] = to_numeric_series(work[target_col])
    valid_features = [
        f for f in out[feature_col].dropna().astype(str).tolist() if f in work.columns
    ]
    if not valid_features:
        out["LCEN"] = np.nan
        out["FCS_pre"] = np.nan
        return out
    lcen_model_df = work[valid_features + [target_col]].dropna().reset_index(drop=True)
    if len(lcen_model_df) < 10:
        out["LCEN"] = np.nan
        out["FCS_pre"] = np.nan
        return out
    X_df = lcen_model_df[valid_features].copy()
    y = lcen_model_df[target_col].to_numpy()
    non_constant_features = [c for c in X_df.columns if X_df[c].nunique() > 1]
    if not non_constant_features:
        out["LCEN"] = np.nan
        out["FCS_pre"] = np.nan
        return out
    X_df = X_df[non_constant_features]
    n = len(X_df)
    block_size = block_size or max(5, n // 20)
    rng = np.random.default_rng(random_state)

    selection_counts = {feat: 0 for feat in valid_features}
    total_runs = 0
    for _ in range(n_bootstraps):
        idx = moving_block_bootstrap_indices(n, block_size, rng)
        X_boot = X_df.iloc[idx].copy()
        y_boot = y[idx]
        X_boot_scaled = StandardScaler().fit_transform(X_boot)
        for alpha in alphas:
            for l1_ratio in l1_ratios:
                try:
                    model = ElasticNet(
                        alpha=alpha,
                        l1_ratio=l1_ratio,
                        max_iter=2000,
                        random_state=random_state,
                    )
                    model.fit(X_boot_scaled, y_boot)
                    selected = X_boot.columns[np.abs(model.coef_) > coef_tol]
                    for feat in selected:
                        selection_counts[feat] += 1
                    total_runs += 1
                except Exception:
                    continue
    lcen_map = {
        feat: (
            selection_counts[feat] / total_runs
            if total_runs > 0 and feat in non_constant_features
            else np.nan
        )
        for feat in valid_features
    }
    lcen_vals, fcs_pre_vals = [], []
    for _, row in out.iterrows():
        feat = row[feature_col]
        signal = row[signal_col]
        lcen = lcen_map.get(feat, np.nan)
        lcen_vals.append(lcen)
        if pd.isna(signal) or pd.isna(lcen):
            fcs_pre_vals.append(np.nan)
        else:
            fcs_pre = signal_weight * signal + lcen_weight * np.sqrt(lcen)
            fcs_pre_vals.append(max(0, min(1, fcs_pre)))
    out["LCEN"] = lcen_vals
    out["FCS_pre"] = fcs_pre_vals
    return out


# =============================================================================
# STEP 8 — DTW SCORE + JOINT SCORE
# =============================================================================
# PURPOSE
#   Score temporal alignment between each KPI and the target.
#
# WHY
#   Some KPIs may have strong statistical score but poor movement-shape alignment.
#   DTW checks whether the KPI pattern actually moves like the target pattern.
# =============================================================================


def compute_dtw_and_joint_score(
    df: pd.DataFrame,
    lag_corr_df: pd.DataFrame,
    *,
    date_col: str,
    target_col: str,
    feature_col: str = "Feature",
    fcs_col: str = "FCS_pre",
    transform_method: str,
    yoy_period: int,
    min_points: int,
    dtw_cfg: dict,
) -> pd.DataFrame:
    """Compute exponential DTW alignment score and weighted joint score."""
    out = lag_corr_df.copy()
    work = df.copy()
    if date_col in work.columns:
        work[date_col] = pd.to_datetime(work[date_col], errors="coerce")
        work = work.sort_values(date_col).reset_index(drop=True)
    y = transform_series(
        work[target_col], method=transform_method, yoy_period=yoy_period
    )
    dtw_raw_vals, dtw_reasons = [], []
    for _, row in out.iterrows():
        feat = row[feature_col]
        if feat not in work.columns:
            dtw_raw_vals.append(np.nan)
            dtw_reasons.append("feature_missing_in_df")
            continue
        x = transform_series(work[feat], method=transform_method, yoy_period=yoy_period)
        tmp = pd.DataFrame({"x": x, "y": y}).dropna()
        if len(tmp) < min_points:
            dtw_raw_vals.append(np.nan)
            dtw_reasons.append(f"insufficient_points={len(tmp)}")
            continue
        if tmp["x"].nunique() <= 1 or tmp["y"].nunique() <= 1:
            dtw_raw_vals.append(np.nan)
            dtw_reasons.append("constant_series_after_transform")
            continue
        dtw_raw_vals.append(float(simple_dtw_distance(tmp["x"], tmp["y"])))
        dtw_reasons.append("ok")
    out["dtw_raw"] = dtw_raw_vals
    out["dtw_reason"] = dtw_reasons
    valid_dtw = out["dtw_raw"].replace([np.inf, -np.inf], np.nan).dropna()
    if len(valid_dtw) == 0:
        out["dtw_score"] = np.nan
        out["joint_score"] = np.nan
        return out

    # -------------------------------------------------------------------------
    # DTW scaling
    # -------------------------------------------------------------------------
    # dtw_raw is a distance:
    # lower dtw_raw = better alignment
    # higher dtw_raw = weaker alignment
    # -------------------------------------------------------------------------
    # Read the scaling settings supplied by the pipeline
    dtw_scaling_method = dtw_cfg.get("dtw_scaling_method", "exponential")
    manual_lambda = dtw_cfg.get("dtw_lambda", None)
    lambda_strategy = dtw_cfg.get("dtw_lambda_strategy", "median_inverse")
    if dtw_scaling_method == "exponential":
        if manual_lambda is not None:
            dtw_lambda = float(manual_lambda)
        else:
            if lambda_strategy == "median_inverse":
                dtw_lambda = 1 / (valid_dtw.median() + 1e-9)
            elif lambda_strategy == "mean_inverse":
                dtw_lambda = 1 / (valid_dtw.mean() + 1e-9)
            else:
                raise ValueError(f"Unknown dtw_lambda_strategy: {lambda_strategy}")
        out["dtw_lambda"] = dtw_lambda
        out["dtw_score"] = np.exp(-dtw_lambda * out["dtw_raw"])

    elif dtw_scaling_method == "minmax":
        dtw_min_all = valid_dtw.min()
        dtw_max_all = valid_dtw.max()
        out["dtw_min_all"] = dtw_min_all
        out["dtw_max_all"] = dtw_max_all
        if np.isclose(dtw_max_all, dtw_min_all):
            out["dtw_score"] = 1.0
        else:
            out["dtw_score"] = (dtw_max_all - out["dtw_raw"]) / (
                dtw_max_all - dtw_min_all
            )

    else:
        raise ValueError(f"Unknown dtw_scaling_method: {dtw_scaling_method}")
    out["dtw_score"] = out["dtw_score"].clip(0, 1)

    # -------------------------------------------------------------------------
    # Weighted joint score
    # -------------------------------------------------------------------------

    fcs_weight = float(dtw_cfg.get("fcs_weight", 0.7))
    dtw_weight = float(dtw_cfg.get("dtw_weight", 0.3))
    weight_sum = fcs_weight + dtw_weight
    if not np.isclose(weight_sum, 1.0):
        raise ValueError(
            f"fcs_weight + dtw_weight must equal 1. "
            f"Got {fcs_weight} + {dtw_weight} = {weight_sum}"
        )
    fcs_values = pd.to_numeric(out[fcs_col], errors="coerce")
    dtw_values = pd.to_numeric(out["dtw_score"], errors="coerce")
    out["joint_score"] = (fcs_weight * fcs_values + dtw_weight * dtw_values).clip(0, 1)
    out["DTW_Transform_Method"] = transform_method
    return out


# =============================================================================
# STEP 9 — FAMILY SURVIVAL + COUNTRY MULTIPLIER
# =============================================================================
# PURPOSE
#   Apply two business-aware adjustments before final category selection.
#
# FAMILY SURVIVAL
#   Drop entire weak categories if even their best KPI is below the score floor.
#
# COUNTRY MULTIPLIER
#   Slightly increase scores for KPIs from high-contribution countries.
# =============================================================================

# =============================================================================
# FAMILY SURVIVAL INTERPRETATION
# =============================================================================
# Category-level filtering happens before choosing final KPIs.
#
# If the best KPI inside a category is still weak, the whole category is dropped.
# This prevents weak business families from entering the final DML candidate pool
# just because every category needs one winner.
# =============================================================================


def apply_family_survival(
    lag_corr_df: pd.DataFrame,
    config: dict,
    *,
    mandatory_cfg: dict,
) -> tuple[pd.DataFrame, dict]:
    """Drop categories where max score is below max(config quantile, floor)."""
    df_out = lag_corr_df.copy()
    category_col = config["category_col"]
    score_col = config["score_col"]
    quantile = float(config["quantile"])
    min_floor = float(config["min_floor"])
    df_out[category_col] = (
        df_out[category_col]
        .astype(str)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    df_out = add_mandatory_flags(
        df_out, feature_col="Feature", mandatory_cfg=mandatory_cfg
    )
    q_val = float(df_out[score_col].astype(float).quantile(quantile))
    thresh = max(q_val, min_floor)
    df_out["family_max_score"] = df_out.groupby(category_col)[score_col].transform(
        "max"
    )
    # Apply the normal family-survival check first.
    df_out["family_survival_pass"] = df_out["family_max_score"] >= thresh
    df_out.setdefault("Status", "Active") if hasattr(df_out, "setdefault") else None
    if "Status" not in df_out.columns:
        df_out["Status"] = "Active"
    if "Reason" not in df_out.columns:
        df_out["Reason"] = ""
    # Then allow mandatory KPIs to pass, even if their family failed.
    if mandatory_cfg["enabled"] and mandatory_cfg["override_family_survival"]:
        mandatory_mask = df_out["Mandatory_KPI"]
        df_out.loc[mandatory_mask, "family_survival_pass"] = True
        df_out.loc[mandatory_mask, "Reason"] = df_out.loc[
            mandatory_mask, "Reason"
        ].apply(
            lambda x: append_reason(
                x,
                "OVERRIDDEN: mandatory KPI bypassed family survival",
            )
        )
        df_out.loc[mandatory_mask, "Override_Reason"] = df_out.loc[
            mandatory_mask, "Override_Reason"
        ].apply(
            lambda x: append_reason(
                x,
                "Mandatory KPI override at family survival",
            )
        )
    step_name = "Step 9a: Family survival"
    drop_mask = ~df_out["family_survival_pass"]
    df_out.loc[drop_mask, "Status"] = "Dropped"
    df_out.loc[drop_mask, "Reason"] = (
        f"[{step_name}] Family dropped: max({score_col}) < max(P{int(quantile * 100)}={q_val:.3f}, {min_floor:.2f})={thresh:.2f}"
    )
    df_out.loc[
        ~drop_mask & (df_out["Reason"].astype(str).str.strip().eq("")), "Reason"
    ] = f"[{step_name}] Passed"
    info = {
        "quantile_value": q_val,
        "threshold": thresh,
        "dropped_families": int(df_out.loc[drop_mask, category_col].nunique()),
    }
    return df_out, info


def apply_country_multiplier(
    lag_corr_df: pd.DataFrame,
    contribution_file: str,
    config: dict,
    *,
    logger: logging.Logger,
) -> pd.DataFrame:
    """Apply country contribution multiplier and create FCS_star."""
    out = lag_corr_df.copy()
    path = Path(contribution_file)
    score_col = config.get("score_col", "joint_score")
    output_col = config.get("output_col", "FCS_star")
    out["country_raw"] = out["Feature"].apply(extract_country)
    if not path.exists():
        logger.warning(
            "Country contribution file not found: %s. Applying default multiplier.",
            path,
        )
        out["Contribution"] = np.nan
        out["normalized_share"] = np.nan
        out["country_multiplier"] = float(config.get("default_multiplier", 1.0))
        out[output_col] = out[score_col] * out["country_multiplier"]
        return out
    cc = pd.read_excel(path)
    country_col = config["country_col"]
    contribution_col = config["contribution_col"]
    cc["country_key"] = cc[country_col].apply(clean_key)
    cc[contribution_col] = pd.to_numeric(cc[contribution_col], errors="coerce")
    max_share = float(cc[contribution_col].max())
    share_map = dict(zip(cc["country_key"], cc[contribution_col]))
    out["country_key"] = out["country_raw"].apply(
        lambda x: clean_key(x) if x is not None else None
    )
    out["Contribution"] = out["country_key"].map(share_map)
    out["normalized_share"] = out["Contribution"] / max_share
    base = float(config.get("base", 0.85))
    variable_weight = float(config.get("variable_weight", 0.15))
    default_multiplier = float(config.get("default_multiplier", 1.0))
    out["country_multiplier"] = base + variable_weight * out["normalized_share"]
    out.loc[out["Contribution"].isna(), "country_multiplier"] = default_multiplier
    out[output_col] = out[score_col] * out["country_multiplier"]
    return out


# =============================================================================
# STEP 10 — TOP-1 CATEGORY SELECTION + PARTIAL CORRELATION
# =============================================================================
# PURPOSE
#   Select a compact base set by choosing the strongest KPI in each category,
#   then check whether dropped KPIs add independent within-category signal.
#
# WHY
#   This avoids overloading DML with many similar variables from the same family.
# =============================================================================


def pick_top1_per_category_simple(
    lag_corr_df: pd.DataFrame,
    *,
    feature_col: str = "Feature",
    category_col: str = "Category",
    fcs_star_col: str = "FCS_star",
    min_score: float | None = None,
    step_name: str = "Step 10a: Top-1 per Category",
) -> tuple[pd.DataFrame, list[str]]:
    """Select one KPI per category using row-index-safe marking."""
    df_out = lag_corr_df.copy()
    if "Status" not in df_out.columns:
        df_out["Status"] = "Active"
    if "Reason" not in df_out.columns:
        df_out["Reason"] = ""
    pool_mask = df_out["Status"].isin(["Active", "Selected"])
    pool = df_out.loc[pool_mask].copy()
    pool[fcs_star_col] = pd.to_numeric(pool[fcs_star_col], errors="coerce")
    pool = pool.dropna(subset=[category_col, fcs_star_col])
    if min_score is not None:
        pool = pool[pool[fcs_star_col] >= min_score]
    if pool.empty:
        return df_out, []
    winners = (
        pool.sort_values(fcs_star_col, ascending=False)
        .groupby(category_col, as_index=False)
        .head(1)
    )
    winner_idx = winners.index
    selected_features = winners[feature_col].astype(str).tolist()
    df_out.loc[pool_mask, "Status"] = "Active"
    df_out.loc[winner_idx, "Status"] = "Selected"
    df_out.loc[winner_idx, "Reason"] = (
        df_out.loc[winner_idx, "Reason"].astype(str)
        + f" | [{step_name}] Picked as top KPI in category by highest {fcs_star_col}"
    ).str.strip(" |")
    drop_idx = pool.index.difference(winner_idx)
    df_out.loc[drop_idx, "Status"] = "Dropped"
    df_out.loc[drop_idx, "Reason"] = (
        df_out.loc[drop_idx, "Reason"].astype(str)
        + f" | [{step_name}] Not picked (another KPI chosen for this category)"
    ).str.strip(" |")
    return df_out, selected_features


def compute_partial_corr_against_top1(
    model_df: pd.DataFrame,
    target_col: str,
    top1_col: str,
    cand_col: str,
    min_points: int,
) -> tuple[float, int, str]:
    """Partial correlation of candidate with target after controlling for selected top1 KPI."""
    tmp = model_df[[target_col, top1_col, cand_col]].dropna().copy()
    if len(tmp) < min_points:
        return np.nan, len(tmp), "too_few_points"
    x = to_numeric_series(tmp[cand_col]).values
    y = to_numeric_series(tmp[target_col]).values
    z = to_numeric_series(tmp[top1_col]).values.reshape(-1, 1)
    if np.std(x) < 1e-12 or np.std(y) < 1e-12 or np.std(z) < 1e-12:
        return np.nan, len(tmp), "constant_series"
    rx = x - LinearRegression().fit(z, x).predict(z)
    ry = y - LinearRegression().fit(z, y).predict(z)
    if np.std(rx) < 1e-12 or np.std(ry) < 1e-12:
        return 0.0, len(tmp), "no_residual_signal"
    r, _ = pearsonr(rx, ry)
    return float(r), len(tmp), "exclude pc < threshold"


def recompute_partial_corr_against_final_p0(
    model_df: pd.DataFrame,
    lag_corr_df: pd.DataFrame,
    *,
    target_col: str,
    feature_col: str = "Feature",
    status_col: str = "Status",
    partial_corr_threshold: float = 0.10,
    min_points: int = 12,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Refresh candidate partial correlations using the final P0 composition.

    Both the target and each non-P0 candidate are residualized against every
    final P0 KPI. A KPI removed from P0 by country diversity is therefore tested
    again after its replacement, before P1/P2/P3 assignment.
    """
    out = lag_corr_df.copy()
    for col, default in [
        ("partial_corr", np.nan),
        ("partial_corr_n_points", np.nan),
        ("partial_corr_reason", ""),
    ]:
        if col not in out.columns:
            out[col] = default
    # Final P0 KPIs after country-diversity adjustment
    final_p0_features = (
        out.loc[
            out[status_col].astype(str).eq("Selected"),
            feature_col,
        ]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .tolist()
    )
    final_p0_features = [
        feature for feature in final_p0_features if feature in model_df.columns
    ]
    # All KPIs outside final P0 are reconsidered
    candidate_rows = out.loc[
        ~out[status_col].astype(str).eq("Selected")
        & out[feature_col].isin(model_df.columns)
    ].copy()
    decision_rows = []
    for idx, row in candidate_rows.iterrows():
        candidate = row[feature_col]
        controls = [feature for feature in final_p0_features if feature != candidate]
        required_cols = [target_col, candidate] + controls
        tmp = model_df[required_cols].apply(pd.to_numeric, errors="coerce").dropna()
        if len(tmp) < min_points:
            partial_corr = np.nan
            reason = "too_few_points_against_final_p0"
        elif tmp[target_col].std() < 1e-12 or tmp[candidate].std() < 1e-12:
            partial_corr = np.nan
            reason = "constant_series_against_final_p0"
        elif controls:
            z = tmp[controls].values
            candidate_values = tmp[candidate].values
            target_values = tmp[target_col].values
            candidate_resid = candidate_values - LinearRegression().fit(
                z, candidate_values
            ).predict(z)
            target_resid = target_values - LinearRegression().fit(
                z, target_values
            ).predict(z)
            if np.std(candidate_resid) < 1e-12 or np.std(target_resid) < 1e-12:
                partial_corr = 0.0
                reason = "no_residual_signal_against_final_p0"
            else:
                partial_corr = float(pearsonr(candidate_resid, target_resid)[0])
                reason = "recomputed_against_final_p0"
        else:
            partial_corr = float(
                pearsonr(
                    tmp[candidate],
                    tmp[target_col],
                )[0]
            )
            reason = "recomputed_without_p0_controls"
        add_flag = (
            pd.notna(partial_corr) and abs(partial_corr) >= partial_corr_threshold
        )
        out.loc[idx, "partial_corr"] = partial_corr
        out.loc[idx, "partial_corr_n_points"] = len(tmp)
        out.loc[idx, "partial_corr_reason"] = reason
        decision_rows.append(
            {
                "Feature": candidate,
                "Partial_Corr": partial_corr,
                "n_points": len(tmp),
                "Add_Flag": add_flag,
                "Reason": reason,
                "Final_P0_Control_Count": len(controls),
                "Final_P0_Controls": " | ".join(controls),
            }
        )
    return out, pd.DataFrame(decision_rows)


def apply_within_category_partial_corr_using_status(
    model_df: pd.DataFrame,
    lag_corr_df: pd.DataFrame,
    *,
    target_col: str,
    feature_col: str,
    category_col: str,
    best_lag_col: str,
    score_col: str,
    status_col: str,
    partial_corr_threshold: float,
    min_points: int,
    logger: logging.Logger,
):
    """For dropped KPIs, test added value versus selected top1 KPI in same category."""
    meta = lag_corr_df.copy()
    meta = meta.dropna(subset=[feature_col, category_col, score_col, status_col]).copy()
    if best_lag_col in meta.columns:
        meta[best_lag_col] = pd.to_numeric(meta[best_lag_col], errors="coerce")
    meta = meta[meta[feature_col].isin(model_df.columns)].copy()
    if meta.empty:
        logger.warning("No valid lagged features found in model_df.")
        return meta, pd.DataFrame(), pd.DataFrame()

    meta[status_col] = meta[status_col].astype(str).str.strip()
    selected_pool = meta[meta[status_col].str.lower() == "selected"].copy()
    dropped_pool = meta[meta[status_col].str.lower() == "dropped"].copy()
    if selected_pool.empty:
        logger.warning("No selected KPIs found in Status column.")
        return meta, pd.DataFrame(), pd.DataFrame()

    meta["Within_Category_Rank"] = np.nan
    meta["top1_in_category"] = False
    meta["partial_corr"] = np.nan
    meta["partial_corr_n_points"] = np.nan
    meta["partial_corr_reason"] = ""

    selected_pool = selected_pool.sort_values(
        [category_col, score_col], ascending=[True, False]
    ).copy()
    selected_pool["Within_Category_Rank"] = (
        selected_pool.groupby(category_col)[score_col]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    top1_df = selected_pool[selected_pool["Within_Category_Rank"] == 1].copy()

    top1_map = {
        row[category_col]: {
            "feature": row[feature_col],
            "lag": row[best_lag_col] if best_lag_col in row else np.nan,
            "score": row[score_col],
        }
        for _, row in top1_df.iterrows()
    }

    meta = meta.merge(
        selected_pool[[feature_col, "Within_Category_Rank"]],
        on=feature_col,
        how="left",
        suffixes=("", "_sel"),
    )
    meta["Within_Category_Rank"] = meta["Within_Category_Rank_sel"].combine_first(
        meta["Within_Category_Rank"]
    )
    meta.drop(columns=["Within_Category_Rank_sel"], inplace=True)
    meta["top1_in_category"] = meta[feature_col].isin(top1_df[feature_col])

    decision_rows = []
    for cat, grp in dropped_pool.groupby(category_col):
        if cat not in top1_map:
            continue
        top1_feat = top1_map[cat]["feature"]
        top1_lag = top1_map[cat]["lag"]
        top1_series = to_numeric_series(model_df[top1_feat])
        y_series = to_numeric_series(model_df[target_col])
        top1_target_spearman = safe_spearman(
            top1_series, y_series, min_points=min_points
        )

        for _, row in grp.iterrows():
            feat = row[feature_col]
            cand_series = to_numeric_series(model_df[feat])
            cand_top1_spearman = safe_spearman(
                cand_series, top1_series, min_points=min_points
            )
            r, n, reason = compute_partial_corr_against_top1(
                model_df, target_col, top1_feat, feat, min_points
            )
            add_flag = pd.notna(r) and abs(r) >= partial_corr_threshold
            final_reason = "include_after_partial_corr" if add_flag else reason

            decision_rows.append(
                {
                    "Category": cat,
                    "Top1_Feature": top1_feat,
                    "Top1_Lag": top1_lag,
                    "Top1_Target_Spearman": top1_target_spearman,
                    "Feature": feat,
                    "Candidate_Lag": (
                        row[best_lag_col] if best_lag_col in row else np.nan
                    ),
                    "Candidate_Top1_Spearman": cand_top1_spearman,
                    "FCS_star": row[score_col],
                    "Dropped_by_FCS": True,
                    "Partial_Corr": r,
                    "n_points": n,
                    "Add_Flag": add_flag,
                    "Reason": final_reason,
                }
            )
            meta.loc[meta[feature_col] == feat, "partial_corr"] = r
            meta.loc[meta[feature_col] == feat, "partial_corr_n_points"] = n
            meta.loc[meta[feature_col] == feat, "partial_corr_reason"] = final_reason

    selected_rank1_df = meta[meta["top1_in_category"] == True].copy()
    decision_df = pd.DataFrame(decision_rows)
    return meta, selected_rank1_df, decision_df


# =============================================================================
# STEP 11 — COUNTRY DIVERSITY ADJUSTMENT
# =============================================================================
# PURPOSE
#   Ensure top contributing countries are represented in the final selected set.
#
# WHY
#   Pure statistical selection can over-select KPIs from one dominant geography
#   and ignore other important business-contributing countries.
#
# BUSINESS RULE
#   For each top contributing country:
#     1. If already represented, do nothing.
#     2. If missing, pick the highest FCS_star KPI from that country.
#     3. If there is room, add it.
#     4. If max_selected_features is reached, replace the weakest selected KPI
#        from the highest contributing country already represented.
#     5. Do not replace internal/competitor KPIs unless explicitly allowed.
# =============================================================================


def kpi_concept_key(feature: str, mapped_kpi: str | None = None) -> str:
    """Create a country-agnostic KPI concept key for diagnostics."""
    base = (
        str(mapped_kpi)
        if mapped_kpi is not None and str(mapped_kpi).lower() != "nan"
        else str(feature)
    )
    base = re.sub(r"\s*\|\s*lag_\d+\s*$", "", base, flags=re.IGNORECASE).strip()
    parts = [p.strip() for p in base.split("|")]
    if len(parts) > 1:
        base = " | ".join(parts[1:]).strip()
    return clean_key(base)


def selected_view(df: pd.DataFrame, status_col: str) -> pd.DataFrame:
    """Return current selected KPIs."""
    return df[df[status_col].eq("Selected")].copy()


def get_top_contributing_countries(
    contribution_file: str, config: dict
) -> pd.DataFrame:
    """Read and rank top contributing countries from the contribution file."""
    cc = pd.read_excel(contribution_file)
    country_name_col = config.get("contribution_country_col", "Country")
    contribution_col = config.get("contribution_col", "Contribution")
    top_n = int(config.get("top_n_countries", 3))

    cc = cc.copy()
    cc["country_key_for_diversity"] = cc[country_name_col].apply(clean_key)
    cc[contribution_col] = pd.to_numeric(cc[contribution_col], errors="coerce")
    return (
        cc.dropna(subset=[contribution_col])
        .sort_values(contribution_col, ascending=False)
        .head(top_n)[[country_name_col, "country_key_for_diversity", contribution_col]]
        .reset_index(drop=True)
    )


def get_best_country_candidate(
    out: pd.DataFrame,
    missing_key: str,
    score_col: str,
    status_col: str,
    min_score: float | None,
) -> pd.DataFrame:
    """Pick the highest FCS_star non-selected KPI from a missing country."""
    candidates = out[
        (out["country_key_for_diversity"] == missing_key)
        & (~out[status_col].eq("Selected"))
        & (out[score_col].notna())
    ].copy()
    if min_score is not None:
        candidates = candidates[candidates[score_col] >= float(min_score)]
    return candidates.sort_values(score_col, ascending=False)


def get_replacement_pool(
    selected_now: pd.DataFrame,
    missing_key: str,
    top_country_keys: list[str],
    score_col: str,
) -> pd.DataFrame:
    """Select replaceable KPIs from the highest contributing represented country."""
    selected_keys = set(selected_now["country_key_for_diversity"].dropna().astype(str))
    represented_top_keys = [
        k for k in top_country_keys if k != missing_key and k in selected_keys
    ]

    if represented_top_keys:
        replacement_country_key = represented_top_keys[0]
        return selected_now[
            selected_now["country_key_for_diversity"] == replacement_country_key
        ].copy()

    return selected_now[
        selected_now["country_key_for_diversity"].notna()
        & (selected_now["country_key_for_diversity"] != missing_key)
    ].copy()


# =============================================================================
# COUNTRY DIVERSITY INTERPRETATION
# =============================================================================
# Final selection should not be dominated by only one country if the business
# contribution file says multiple countries are important.
#
# This step checks whether high-contribution countries are represented.
# If a top country is missing, the logic can replace a weaker selected KPI
# with an eligible KPI from the missing country.
# ==============================================================================


def apply_country_diversity_adjustment(
    lag_corr_df: pd.DataFrame,
    *,
    contribution_file: str,
    config: dict,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Apply country diversity business rule and return updated selected set + audit log."""
    out = lag_corr_df.copy()
    if not config.get("enabled", True):
        selected = out[out.get("Status", pd.Series(dtype=str)).eq("Selected")].copy()
        return out, selected, pd.DataFrame(), pd.DataFrame()

    status_col = config.get("status_col", "Status")
    score_col = config.get("score_col", "FCS_star")
    feature_col = config.get("feature_col", "Feature")
    category_col = config.get("category_col", "Category")
    country_col = config.get("country_col", "country_raw")
    mapped_kpi_col = config.get("mapped_kpi_col", "Mapped_KPI")
    max_selected_features = int(config.get("max_selected_features", 8))
    min_score = config.get("min_candidate_score", None)

    if status_col not in out.columns:
        out[status_col] = "Active"
    if "Reason" not in out.columns:
        out["Reason"] = ""
    out[score_col] = pd.to_numeric(out[score_col], errors="coerce")

    if country_col not in out.columns:
        out[country_col] = out[feature_col].apply(extract_country)

    out["country_key_for_diversity"] = out[country_col].apply(
        lambda x: clean_key(x) if pd.notna(x) else None
    )
    out["kpi_concept_key"] = out.apply(
        lambda r: kpi_concept_key(r.get(feature_col), r.get(mapped_kpi_col)), axis=1
    )
    out["Country_Diversity_Action"] = ""

    path = Path(contribution_file)
    if not path.exists():
        log_df = pd.DataFrame(
            [
                {
                    "Action": "SKIP",
                    "Reason": f"Country contribution file not found: {path}",
                }
            ]
        )
        logger.warning(
            "Country contribution file not found: %s. Skipping country diversity.",
            path,
        )
        return out, selected_view(out, status_col), log_df, pd.DataFrame()

    top_countries_df = get_top_contributing_countries(str(path), config)
    top_country_keys = top_countries_df["country_key_for_diversity"].tolist()
    top_country_name_map = dict(
        zip(
            top_countries_df["country_key_for_diversity"],
            top_countries_df[config.get("contribution_country_col", "Country")],
        )
    )
    log_rows = []

    for missing_key in top_country_keys:
        selected_now = selected_view(out, status_col)
        selected_country_keys = set(
            selected_now["country_key_for_diversity"].dropna().astype(str)
        )
        country_name = top_country_name_map.get(missing_key, missing_key)

        if missing_key in selected_country_keys:
            log_rows.append(
                {
                    "Action": "ALREADY_PRESENT",
                    "Missing_Country": country_name,
                    "Reason": "Top contributing country already represented in selected KPIs",
                }
            )
            continue

        candidates = get_best_country_candidate(
            out, missing_key, score_col, status_col, min_score
        )
        if candidates.empty:
            log_rows.append(
                {
                    "Action": "SKIP_NO_CANDIDATE",
                    "Missing_Country": country_name,
                    "Reason": "No non-selected KPI available from missing country",
                }
            )
            continue

        add_idx = candidates.index[0]
        add_row = candidates.loc[add_idx]
        add_feature = add_row[feature_col]
        add_score = add_row[score_col]
        add_reason = f"[Step 11: Country diversity] Added to represent top contributing country {country_name}; best available KPI from missing top country"
        selected_count = len(selected_now)

        if selected_count < max_selected_features:
            out.loc[add_idx, status_col] = "Selected"
            out.loc[add_idx, "Reason"] = append_reason(
                out.loc[add_idx, "Reason"], add_reason
            )
            out.loc[add_idx, "Country_Diversity_Action"] = (
                "ADDED_FOR_MISSING_TOP_COUNTRY"
            )
            log_rows.append(
                {
                    "Action": "ADD",
                    "Missing_Country": country_name,
                    "Added_Feature": add_feature,
                    "Added_Category": add_row.get(category_col),
                    "Added_KPI_Concept": add_row.get("kpi_concept_key"),
                    "Added_Score": add_score,
                    "Removed_Feature": None,
                    "Removed_Country": None,
                    "Removed_Score": None,
                    "Reason": f"Selected count {selected_count} < max_selected_features {max_selected_features}; added without replacement",
                }
            )
            continue

        selected_now = selected_view(out, status_col)
        selected_now[score_col] = pd.to_numeric(
            selected_now[score_col], errors="coerce"
        )
        replacement_pool = get_replacement_pool(
            selected_now, missing_key, top_country_keys, score_col
        )
        # Protect Internal and Competitor KPIs unless replacement is allowed.
        if not config.get("allow_replace_internal", False):
            protected_categories = (
                replacement_pool[category_col]
                .astype(str)
                .map(clean_key)
                .isin(["internal", "competitor"])
            )

            replacement_pool = replacement_pool.loc[~protected_categories].copy()

        if replacement_pool.empty:
            log_rows.append(
                {
                    "Action": "SKIP_NO_REPLACEMENT_POOL",
                    "Missing_Country": country_name,
                    "Added_Feature": add_feature,
                    "Reason": "No replaceable selected external KPI found; internal replacement disabled",
                }
            )
            continue

        remove_idx = replacement_pool.sort_values(score_col, ascending=True).index[0]
        remove_row = out.loc[remove_idx].copy()

        out.loc[remove_idx, status_col] = "Dropped"
        out.loc[remove_idx, "Reason"] = append_reason(
            out.loc[remove_idx, "Reason"],
            f"[Step 11: Country diversity] Replaced by {add_feature} to represent missing top contributing country {country_name}; removed as weakest selected KPI from highest contributing represented country",
        )
        out.loc[remove_idx, "Country_Diversity_Action"] = (
            "REMOVED_FOR_COUNTRY_REPLACEMENT"
        )

        out.loc[add_idx, status_col] = "Selected"
        out.loc[add_idx, "Reason"] = append_reason(
            out.loc[add_idx, "Reason"], add_reason
        )
        out.loc[add_idx, "Country_Diversity_Action"] = (
            "ADDED_BY_REPLACING_WEAKEST_TOP_COUNTRY"
        )

        log_rows.append(
            {
                "Action": "REPLACE",
                "Missing_Country": country_name,
                "Added_Feature": add_feature,
                "Added_Category": add_row.get(category_col),
                "Added_KPI_Concept": add_row.get("kpi_concept_key"),
                "Added_Score": add_score,
                "Removed_Feature": remove_row[feature_col],
                "Removed_Country": remove_row.get(country_col),
                "Removed_Category": remove_row.get(category_col),
                "Removed_KPI_Concept": remove_row.get("kpi_concept_key"),
                "Removed_Score": remove_row.get(score_col),
                "Reason": f"Selected count {selected_count} >= max_selected_features {max_selected_features}; replaced weakest selected KPI from highest contributing represented country",
            }
        )

    selected_after = selected_view(out, status_col)
    log_df = pd.DataFrame(log_rows)
    return out, selected_after, log_df, top_countries_df


# =============================================================================
# STEP 12 — PRIORITY SET CREATION: P0, P1, P2, P3
# =============================================================================
# PURPOSE
#   Convert the scored feature table into non-overlapping review sets.
#
# BUSINESS MEANING
#   P0 = Final selected/base KPIs after country diversity.
#   P1 = Remaining KPIs with strong FCS_star and strong partial correlation.
#   P2 = Remaining KPIs with high FCS_star but weak DTW alignment.
#   P3 = Remaining KPIs with low FCS_star but strong DTW alignment.
#
# IMPORTANT
#   - Priority sets are mutually exclusive.
#   - One KPI should belong to only one set.
#   - Assignment should happen top-down: P0 first, then P1, then P2, then P3.
#   - If P2 is empty, it may simply mean no KPI satisfied the P2 rule after
#     higher-priority sets already captured the stronger KPIs.
# =============================================================================


def create_priority_sets(
    lag_corr_df: pd.DataFrame, config: dict
) -> tuple[pd.DataFrame, dict]:
    """
    Create non-overlapping priority sets P0/P1/P2/P3.

    P0:


      Base selected set after category/country diversity.

    P1:
      Remaining candidates with high FCS_star and strong abs(partial_corr).
      No separate DTW condition here.

    P2:
      Remaining candidates with high FCS_star but weak DTW alignment.
      Means signal quality is high, but timing/pattern is uncertain.

    P3:
      Remaining candidates with low FCS_star but strong DTW alignment.
      Means pattern aligns, but statistical/selection strength is weaker.
    """
    out = lag_corr_df.copy()

    if not config.get("enabled", True):
        out[config.get("set_col", "Priority_Set")] = None
        return out, {}

    score_col = config.get("score_col", "FCS_star")
    dtw_col = config.get("dtw_col", "dtw_score")
    pc_col = config.get("partial_corr_col", "partial_corr")
    status_col = config.get("status_col", "Status")
    set_col = config.get("set_col", "Priority_Set")

    # Separate thresholds by set
    fcs_th_p1 = float(config.get("fcs_threshold_p1", config.get("fcs_threshold", 0.65)))
    pc_th_p1 = float(
        config.get(
            "partial_corr_threshold_p1", config.get("partial_corr_threshold", 0.2)
        )
    )

    fcs_th_p2 = float(config.get("fcs_threshold_p2", config.get("fcs_threshold", 0.65)))
    dtw_th_p2 = float(config.get("dtw_threshold_p2", config.get("dtw_threshold", 0.5)))

    fcs_th_p3 = float(config.get("fcs_threshold_p3", config.get("fcs_threshold", 0.65)))
    dtw_th_p3 = float(config.get("dtw_threshold_p3", config.get("dtw_threshold", 0.5)))

    for col in [score_col, dtw_col, pc_col]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    # Round only for priority-set threshold comparisons
    # This makes Excel-visible values match set assignment logic
    if score_col in out.columns:
        out[score_col] = out[score_col].round(2)

    if dtw_col in out.columns:
        out[dtw_col] = out[dtw_col].round(2)

    if pc_col in out.columns:
        out[pc_col] = out[pc_col].round(2)
    out[set_col] = None
    out["Priority_Set_Reason"] = ""

    # P0: base selected set
    p0_mask = out[status_col].astype(str).eq("Selected")
    out.loc[p0_mask, set_col] = "P0"
    out.loc[p0_mask, "Priority_Set_Reason"] = (
        "P0 base set: selected after top-category selection and country diversity"
    )

    # P1: high FCS_star + strong partial correlation
    # No DTW check here.
    remaining = out[set_col].isna()
    p1_mask = (
        remaining
        & (round(out[score_col], 1) >= fcs_th_p1)
        & (round(out[pc_col].abs(), 1) >= pc_th_p1)
    )
    out.loc[p1_mask, set_col] = "P1"
    out.loc[p1_mask, "Priority_Set_Reason"] = (
        f"P1: remaining candidate with {score_col} >= {fcs_th_p1} "
        f"and abs({pc_col}) >= {pc_th_p1}; rank by abs({pc_col})"
    )

    # P2: high FCS_star but low DTW
    remaining = out[set_col].isna()
    p2_mask = (
        remaining
        & (round(out[score_col], 1) >= fcs_th_p2)
        & (round(out[dtw_col], 1) < dtw_th_p2)
    )
    out.loc[p2_mask, set_col] = "P2"
    out.loc[p2_mask, "Priority_Set_Reason"] = (
        f"P2: high {score_col} >= {fcs_th_p2}, "
        f"but timing uncertain because {dtw_col} < {dtw_th_p2}"
    )

    # P3: low FCS_star but high DTW
    remaining = out[set_col].isna()
    p3_mask = (
        remaining
        & (round(out[score_col], 1) < fcs_th_p3)
        & (round(out[dtw_col], 1) >= dtw_th_p3)
    )
    out.loc[p3_mask, set_col] = "P3"
    out.loc[p3_mask, "Priority_Set_Reason"] = (
        f"P3: strong pattern alignment {dtw_col} >= {dtw_th_p3}, "
        f"but weaker {score_col} < {fcs_th_p3}"
    )

    summary = out[set_col].value_counts(dropna=False).to_dict()
    return out, summary


if __name__ == "__main__":
    # Read the config path supplied in the terminal.
    args = parse_args()

    # Load the settings and run feature selection.
    config_loader = ConfigLoader(args.config)
    config = config_loader.load()

    pipeline = FeatureSelectionPipeline(config)
    output_excel = pipeline.run(
        run_id=args.run_id,
        quarter=args.quarter,
    )
