# stage1_cache.py

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


def safe_cache_name(value):

    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")


def calculate_file_hash(file_path):

    hasher = hashlib.sha256()

    with open(file_path, "rb") as f:

        while True:

            chunk = f.read(1024 * 1024)

            if not chunk:
                break

            hasher.update(chunk)

    return hasher.hexdigest()


def make_driver_cache_fingerprint(
    driver_label, confounders, config, source_data_hash, cache_version
):

    fingerprint_data = {
        "cache_version": cache_version,
        "source_file": str(config["file"]),
        "source_data_hash": source_data_hash,
        "data_freq": config["data_freq"],
        "train_end": str(config["train_end"]),
        "driver_label": driver_label,
        "driver_column": config["drivers"][driver_label],
        "alpha_stage1t": config.get("alpha_stage1t", 5.0),
        "gbm_stage1t": config.get("gbm_stage1t", {}),
        "stage1_thresholds": config.get("stage1_thresholds", {}),
        "time_confounders": list(config.get("time_confounders", ["trend"])),
        "piecewise_time_confounders": list(
            config.get("piecewise_time_confounders", [])
        ),
        "regime_date": str(config.get("regime_date", "")),
        "confounders": sorted(list(confounders)),
        "partial_correlation": config.get("partial_correlation", {}),
        "piecewise_knot": config.get("piecewise_drivers", {}).get(driver_label),
    }

    serialized = json.dumps(fingerprint_data, sort_keys=True, default=str)

    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def get_driver_cache_dir(cache_root, driver_label):

    return Path(cache_root) / safe_cache_name(driver_label)


def load_driver_stage1_cache(
    driver_label,
    expected_fingerprint,
    cache_root,
    cache_enabled=True,
    force_rebuild=False,
):

    if not cache_enabled:
        return None

    if force_rebuild:

        print(f"[CACHE BYPASS] " f"{driver_label}: " "force_rebuild=True")

        return None

    cache_dir = get_driver_cache_dir(cache_root, driver_label)

    residual_file = cache_dir / "residuals.npz"

    metadata_file = cache_dir / "metadata.json"

    if not residual_file.exists() or not metadata_file.exists():

        print(f"[CACHE MISS] " f"{driver_label}: " "cache not found")

        return None

    try:

        with metadata_file.open("r", encoding="utf-8") as f:

            metadata = json.load(f)

    except Exception as exc:

        print(f"[CACHE INVALID] " f"{driver_label}: {exc}")

        return None

    if metadata.get("fingerprint") != expected_fingerprint:

        print(f"[CACHE STALE] " f"{driver_label}: " "fingerprint changed")

        return None

    try:

        with np.load(residual_file, allow_pickle=False) as data:

            result = {key: data[key] for key in data.files}

    except Exception as exc:

        print(f"[CACHE INVALID] " f"{driver_label}: {exc}")

        return None

    # Restore metadata diagnostics

    for key in [
        "model_used",
        "var_ratio",
        "adf_pvalue",
        "var_ratio_low",
        "adf_pvalue_low",
        "var_ratio_high",
        "adf_pvalue_high",
    ]:

        if key in metadata:
            result[key] = metadata[key]

    print(f"[CACHE HIT] " f"{driver_label}: " "loaded Stage 1T residuals")

    return result


def save_driver_stage1_cache(
    driver_label,
    fingerprint,
    result,
    cache_root,
    config,
    source_data_hash,
    cache_version,
    confounders,
    cache_enabled=True,
):

    if not cache_enabled:
        return

    cache_dir = get_driver_cache_dir(cache_root, driver_label)

    cache_dir.mkdir(parents=True, exist_ok=True)

    residual_file = cache_dir / "residuals.npz"

    metadata_file = cache_dir / "metadata.json"

    is_piecewise = driver_label in config.get("piecewise_drivers", {})

    # ----------------------------------------------------------
    # Arrays
    # ----------------------------------------------------------

    arrays = {
        "train_residuals": result["train_residuals"],
        "test_residuals": result["test_residuals"],
    }

    if is_piecewise:

        arrays.update(
            {
                "train_residuals_low": result["train_residuals_low"],
                "test_residuals_low": result["test_residuals_low"],
                "train_residuals_high": result["train_residuals_high"],
                "test_residuals_high": result["test_residuals_high"],
            }
        )

    np.savez_compressed(residual_file, **arrays)

    # ----------------------------------------------------------
    # Metadata
    # ----------------------------------------------------------

    metadata = {
        "cache_version": cache_version,
        "fingerprint": fingerprint,
        "driver": driver_label,
        "driver_column": config["drivers"][driver_label],
        "train_end": str(config["train_end"]),
        "source_file": str(config["file"]),
        "source_data_hash": source_data_hash,
        "piecewise": is_piecewise,
        "knot": config.get("piecewise_drivers", {}).get(driver_label),
        "confounders": list(confounders),
        "model_used": result["model_used"],
        "var_ratio": float(result["var_ratio"]),
        "adf_pvalue": (
            float(result["adf_pvalue"]) if pd.notna(result["adf_pvalue"]) else None
        ),
    }

    if is_piecewise:

        metadata.update(
            {
                "var_ratio_low": float(result["var_ratio_low"]),
                "adf_pvalue_low": (
                    float(result["adf_pvalue_low"])
                    if pd.notna(result["adf_pvalue_low"])
                    else None
                ),
                "var_ratio_high": float(result["var_ratio_high"]),
                "adf_pvalue_high": (
                    float(result["adf_pvalue_high"])
                    if pd.notna(result["adf_pvalue_high"])
                    else None
                ),
            }
        )

    with metadata_file.open("w", encoding="utf-8") as f:

        json.dump(metadata, f, indent=2)

    print(f"[CACHE SAVE] " f"{driver_label}: " f"{residual_file}")
