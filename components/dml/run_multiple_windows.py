"""
Multi-window runner for the DML elasticity pipeline.

Runs dml_analysis.py independently for multiple training cutoffs.

Example:
    python run_multiple_windows.py --config config_dml.json

Each training cutoff:
    1. Gets its own output folder.
    2. Gets its own config_used.json.
    3. Runs dml_analysis.py in a separate subprocess.
    4. Can run in parallel with other training cutoffs.

Date interpretation:
    train_df = df[df["date"] < train_end]
    test_df = df[df["date"] >= train_end]

Therefore, train_end represents the first test-period date.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from shared.logging import logger

# =============================================================================
# CONSTANTS
# =============================================================================

DEFAULT_DML_SCRIPT = str(Path(__file__).parent / "dml_analysis.py")
DEFAULT_OUTPUT_DIR = "dml_outputs"
MAX_PARALLEL_WINDOWS = 2


# =============================================================================
# COMMAND-LINE ARGUMENTS
# =============================================================================


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run the DML pipeline for multiple training cutoffs."
    )

    parser.add_argument(
        "--config",
        required=True,
        help="Path to the main DML JSON configuration.",
    )

    parser.add_argument(
        "--dml-script",
        default=DEFAULT_DML_SCRIPT,
        help=f"Path to the DML script. Default: {DEFAULT_DML_SCRIPT}",
    )

    return parser.parse_args()


# =============================================================================
# CONFIGURATION
# =============================================================================


def load_config(config_path: Path) -> dict:
    """Load the main JSON configuration."""
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    from components.dml.dml_analysis import load_config as load_dml_config

    return load_dml_config(config_path)


def validate_dml_script(dml_script_path: Path) -> None:
    """Check that the DML analysis script exists."""
    if not dml_script_path.exists():
        raise FileNotFoundError(f"DML analysis script not found: {dml_script_path}")


def validate_train_end_dates(values: list[str]) -> list[str]:
    """Validate, deduplicate and sort training cutoff dates."""
    if not isinstance(values, list) or not values:
        raise ValueError("'train_end_dates' must be a non-empty list of dates.")

    valid_dates = []

    for value in values:
        try:
            parsed_date = datetime.strptime(value, "%Y-%m-%d")

        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid train_end date '{value}'. " "Expected format: YYYY-MM-DD."
            ) from exc

        valid_dates.append(parsed_date.strftime("%Y-%m-%d"))

    return sorted(set(valid_dates))


def get_train_end_dates(config: dict) -> list[str]:
    """Get training cutoffs from either train_end_dates, folds, or train_end."""
    if "train_end_dates" in config:
        return validate_train_end_dates(config["train_end_dates"])

    if "folds" in config:
        fold_dates = [f["train_end"] for f in config["folds"] if isinstance(f, dict) and "train_end" in f]
        if fold_dates:
            return validate_train_end_dates(fold_dates)

    if "train_end" in config:
        return validate_train_end_dates([config["train_end"]])

    raise KeyError("Config must contain either 'train_end_dates', 'folds', or 'train_end'.")


# =============================================================================
# OUTPUT DIRECTORIES
# =============================================================================


def get_division_output_dir(base_config: dict) -> Path:
    """Create and return the division-level output directory."""
    base_output_dir = Path(base_config.get("output_dir", DEFAULT_OUTPUT_DIR)).resolve()

    safe_division_name = base_config["division_name"].strip().replace(" ", "_")

    division_output_dir = base_output_dir / safe_division_name

    division_output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    return division_output_dir


def get_window_output_dir(
    division_output_dir: Path,
    train_end: str,
) -> Path:
    """Create and return the output folder for one training window."""
    window_output_dir = division_output_dir / f"train_end_{train_end}"

    window_output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    return window_output_dir


# =============================================================================
# WINDOW CONFIGURATION
# =============================================================================


def create_window_config(
    base_config: dict,
    train_end: str,
    window_output_dir: Path,
) -> dict:
    """Create the single-window config passed to dml_analysis.py."""
    window_config = base_config.copy()

    window_config["train_end"] = train_end
    window_config["output_dir"] = str(window_output_dir)

    window_config.pop("train_end_dates", None)

    return window_config


def save_window_config(
    window_config: dict,
    window_output_dir: Path,
) -> Path:
    """Save the exact configuration used by one training window."""
    config_path = window_output_dir / "config_used.json"

    with config_path.open("w", encoding="utf-8") as file:
        json.dump(
            window_config,
            file,
            indent=2,
        )

    return config_path


# =============================================================================
# DML SUBPROCESS
# =============================================================================


def execute_dml(
    dml_script_path: Path,
    window_config_path: Path,
) -> None:
    """Execute dml_analysis.py for one training window."""
    command = [
        sys.executable,
        str(dml_script_path),
        "--config",
        str(window_config_path),
    ]

    import os
    repo_root = str(Path(__file__).resolve().parents[2])
    env = os.environ.copy()
    current_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{repo_root}{os.pathsep}{current_pp}" if current_pp else repo_root

    subprocess.run(
        command,
        env=env,
        check=True,
    )


# =============================================================================
# SINGLE WINDOW
# =============================================================================


def run_single_window(
    train_end: str,
    base_config: dict,
    division_output_dir: Path,
    dml_script_path: Path,
) -> dict:
    """Prepare and execute one independent DML training window."""
    logger.info(
        f"STARTING WINDOW: {train_end} (train < {train_end}, test >= {train_end})",
        event="WINDOW_START",
        train_end=train_end,
    )

    window_output_dir = get_window_output_dir(
        division_output_dir,
        train_end,
    )

    window_config = create_window_config(
        base_config,
        train_end,
        window_output_dir,
    )

    window_config_path = save_window_config(
        window_config,
        window_output_dir,
    )

    logger.info(
        f"Window {train_end} | Config: {window_config_path} | Output: {window_output_dir}",
        event="WINDOW_CONFIG_READY",
        train_end=train_end,
        config_path=str(window_config_path),
        output_dir=str(window_output_dir),
    )

    execute_dml(
        dml_script_path,
        window_config_path,
    )

    logger.info(f"COMPLETED WINDOW: {train_end}", event="WINDOW_COMPLETE", train_end=train_end)

    return {
        "train_end": train_end,
        "output_dir": window_output_dir,
        "config_path": window_config_path,
    }


# =============================================================================
# PARALLEL WINDOW EXECUTION
# =============================================================================


def run_windows_parallel(
    train_end_dates: list[str],
    base_config: dict,
    division_output_dir: Path,
    dml_script_path: Path,
) -> list[dict]:
    """Run independent training windows in parallel."""
    max_workers = min(
        len(train_end_dates),
        MAX_PARALLEL_WINDOWS,
    )

    logger.info(
        f"PARALLEL WINDOW EXECUTION | Windows: {len(train_end_dates)} | Workers: {max_workers} | Cutoffs: {train_end_dates}",
        event="PARALLEL_EXECUTION_START",
        num_windows=len(train_end_dates),
        max_workers=max_workers,
    )

    completed_runs = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_train_end = {
            executor.submit(
                run_single_window,
                train_end,
                base_config,
                division_output_dir,
                dml_script_path,
            ): train_end
            for train_end in train_end_dates
        }

        for future in as_completed(future_to_train_end):
            train_end = future_to_train_end[future]

            try:
                result = future.result()

            except Exception as exc:
                logger.error(
                    f"FAILED WINDOW: {train_end} - {exc}",
                    event="WINDOW_FAILED",
                    error_category="WINDOW_EXECUTION_ERROR",
                    train_end=train_end,
                    exc=str(exc),
                )
                raise

            completed_runs.append(result)

            logger.info(
                f"Finished {train_end} | {len(completed_runs)}/{len(train_end_dates)} complete",
                event="WINDOW_PROGRESS",
                train_end=train_end,
                completed=len(completed_runs),
                total=len(train_end_dates),
            )

    return sorted(
        completed_runs,
        key=lambda run: run["train_end"],
    )


# =============================================================================
# EXECUTION SUMMARY
# =============================================================================


def print_execution_summary(completed_runs: list[dict]) -> None:
    """Log a summary after all training windows finish."""
    logger.info("ALL DML TRAINING WINDOWS COMPLETED SUCCESSFULLY", event="ALL_WINDOWS_COMPLETE", count=len(completed_runs))

    for run in completed_runs:
        logger.info(
            f"Window: {run['train_end']} | Output: {run['output_dir']} | Config: {run['config_path']}",
            event="WINDOW_SUMMARY",
            train_end=run["train_end"],
            output_dir=str(run["output_dir"]),
            config_path=str(run["config_path"]),
        )


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    """Run the complete multi-window DML workflow."""
    args = parse_args()

    config_path = Path(args.config).resolve()
    dml_script_path = Path(args.dml_script).resolve()

    # -------------------------------------------------------------------------
    # 1. Load and validate
    # -------------------------------------------------------------------------

    base_config = load_config(config_path)

    validate_dml_script(dml_script_path)

    train_end_dates = get_train_end_dates(base_config)

    # -------------------------------------------------------------------------
    # 2. Prepare output directory
    # -------------------------------------------------------------------------

    division_output_dir = get_division_output_dir(base_config)

    # -------------------------------------------------------------------------
    # 3. Run training windows in parallel
    # -------------------------------------------------------------------------

    completed_runs = run_windows_parallel(
        train_end_dates=train_end_dates,
        base_config=base_config,
        division_output_dir=division_output_dir,
        dml_script_path=dml_script_path,
    )

    # -------------------------------------------------------------------------
    # 4. Final summary
    # -------------------------------------------------------------------------

    print_execution_summary(completed_runs)


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()
