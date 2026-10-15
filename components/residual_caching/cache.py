# stage1_cache.py

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from shared.logging import logger
from shared.path_builder.paths import (
    get_residual_cache_prefix,
    get_residual_metadata_path,
    get_residual_npz_path,
)
from shared.path_builder.storage import BlobStorageClient
from shared.schema.contracts import fp

FREQ_MAP = {"M": 12, "Q": 4, "W": 52, "D": 365}


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

    drivers_cfg = config.get("drivers", {})
    driver_col = drivers_cfg.get(driver_label, driver_label) if isinstance(drivers_cfg, dict) else driver_label

    freq_val = config.get("data_freq") or FREQ_MAP.get(config.get("data", {}).get("freq", "M"), 12)

    fingerprint_data = {
        "cache_version": cache_version,
        "source_file": str(config.get("data", {}).get("input_uri", config.get("file", ""))),
        "source_data_hash": source_data_hash,
        "data_freq": int(freq_val),
        "train_end": str(config.get("train_end", "")),
        "driver_label": driver_label,
        "driver_column": driver_col,
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
    combination_id=None,
    storage=None,
    **kwargs,
):

    if not cache_enabled:
        return None

    if force_rebuild:
        logger.info(f"Stage 1 cache bypass for {driver_label}: force_rebuild=True", event="CACHE_BYPASS", driver=driver_label)
        return None

    # Storage client mode (BlobStorageClient or local filesystem)
    if storage is not None and hasattr(storage, "exists"):
        prefix = str(cache_root).replace("\\", "/").rstrip("/")
        driver_dir = f"{prefix}/{safe_cache_name(driver_label)}"
        residual_file = f"{driver_dir}/residuals.npz"
        metadata_file = f"{driver_dir}/metadata.json"

        if not storage.exists(residual_file) or not storage.exists(metadata_file):
            logger.info(f"Stage 1 cache miss for {driver_label}: entry not found", event="CACHE_MISS", driver=driver_label)
            return None

        try:
            metadata = storage.read_json(metadata_file)
        except Exception as exc:
            logger.warning(f"Stage 1 cache invalid for {driver_label}: {exc}", event="CACHE_INVALID", driver=driver_label)
            return None

        if metadata.get("fingerprint") != expected_fingerprint:
            logger.info(f"Stage 1 cache stale for {driver_label}: fingerprint changed", event="CACHE_STALE", driver=driver_label)
            return None

        try:
            result = storage.read_npz(residual_file)
        except Exception as exc:
            logger.warning(f"Stage 1 cache invalid for {driver_label}: {exc}", event="CACHE_INVALID", driver=driver_label)
            return None

        for key in [
            "model_used", "var_ratio", "adf_pvalue",
            "var_ratio_low", "adf_pvalue_low", "var_ratio_high", "adf_pvalue_high",
        ]:
            if key in metadata:
                result[key] = metadata[key]

        logger.info(f"Stage 1 cache hit for {driver_label}: loaded Stage 1T residuals", event="CACHE_HIT", driver=driver_label)
        return result

    cache_dir = get_driver_cache_dir(cache_root, driver_label)
    residual_file = cache_dir / "residuals.npz"
    metadata_file = cache_dir / "metadata.json"

    if not residual_file.exists() or not metadata_file.exists():
        logger.info(f"Stage 1 cache miss for {driver_label}: cache not found", event="CACHE_MISS", driver=driver_label)
        return None

    try:
        with metadata_file.open("r", encoding="utf-8") as f:
            metadata = json.load(f)
    except Exception as exc:
        logger.warning(f"Stage 1 cache invalid for {driver_label}: {exc}", event="CACHE_INVALID", driver=driver_label)
        return None

    if metadata.get("fingerprint") != expected_fingerprint:
        logger.info(f"Stage 1 cache stale for {driver_label}: fingerprint changed", event="CACHE_STALE", driver=driver_label)
        return None

    try:
        with np.load(residual_file, allow_pickle=False) as data:
            result = {key: data[key] for key in data.files}
    except Exception as exc:
        logger.warning(f"Stage 1 cache invalid for {driver_label}: {exc}", event="CACHE_INVALID", driver=driver_label)
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

    logger.info(f"Stage 1 cache hit for {driver_label}: loaded Stage 1T residuals", event="CACHE_HIT", driver=driver_label)
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
    combination_id=None,
    storage=None,
    **kwargs,
):

    if not cache_enabled:
        return

    if storage is not None and hasattr(storage, "write_npz"):
        prefix = str(cache_root).replace("\\", "/").rstrip("/")
        driver_dir = f"{prefix}/{safe_cache_name(driver_label)}"
        residual_file = f"{driver_dir}/residuals.npz"
        metadata_file = f"{driver_dir}/metadata.json"

        is_piecewise = driver_label in config.get("piecewise_drivers", {})
        arrays = {
            "train_residuals": result["train_residuals"],
            "test_residuals": result["test_residuals"],
        }
        if is_piecewise:
            arrays.update({
                "train_residuals_low": result["train_residuals_low"],
                "test_residuals_low": result["test_residuals_low"],
                "train_residuals_high": result["train_residuals_high"],
                "test_residuals_high": result["test_residuals_high"],
            })

        metadata = {
            "cache_version": cache_version,
            "fingerprint": fingerprint,
            "driver": driver_label,
            "driver_column": config.get("drivers", {}).get(driver_label, driver_label),
            "train_end": str(config.get("train_end", "")),
            "source_file": str(config.get("data", {}).get("input_uri", config.get("file", ""))),
            "source_data_hash": source_data_hash,
            "piecewise": is_piecewise,
            "knot": config.get("piecewise_drivers", {}).get(driver_label),
            "confounders": list(confounders),
            "model_used": result["model_used"],
            "var_ratio": float(result["var_ratio"]),
            "adf_pvalue": float(result["adf_pvalue"]) if pd.notna(result["adf_pvalue"]) else None,
        }
        if is_piecewise:
            metadata.update({
                "var_ratio_low": float(result["var_ratio_low"]),
                "adf_pvalue_low": float(result["adf_pvalue_low"]) if pd.notna(result["adf_pvalue_low"]) else None,
                "var_ratio_high": float(result["var_ratio_high"]),
                "adf_pvalue_high": float(result["adf_pvalue_high"]) if pd.notna(result["adf_pvalue_high"]) else None,
            })

        storage.write_npz(residual_file, arrays)
        storage.write_json(metadata_file, metadata)
        logger.info(f"Stage 1 cache saved for {driver_label}: {residual_file}", event="CACHE_SAVE", driver=driver_label)
        return

    cache_dir = get_driver_cache_dir(cache_root, driver_label)
    cache_dir.mkdir(parents=True, exist_ok=True)
    residual_file = cache_dir / "residuals.npz"
    metadata_file = cache_dir / "metadata.json"

    is_piecewise = driver_label in config.get("piecewise_drivers", {})
    arrays = {
        "train_residuals": result["train_residuals"],
        "test_residuals": result["test_residuals"],
    }
    if is_piecewise:
        arrays.update({
            "train_residuals_low": result["train_residuals_low"],
            "test_residuals_low": result["test_residuals_low"],
            "train_residuals_high": result["train_residuals_high"],
            "test_residuals_high": result["test_residuals_high"],
        })

    np.savez_compressed(residual_file, **arrays)

    metadata = {
        "cache_version": cache_version,
        "fingerprint": fingerprint,
        "driver": driver_label,
        "driver_column": config.get("drivers", {}).get(driver_label, driver_label),
        "train_end": str(config.get("train_end", "")),
        "source_file": str(config.get("data", {}).get("input_uri", config.get("file", ""))),
        "source_data_hash": source_data_hash,
        "piecewise": is_piecewise,
        "knot": config.get("piecewise_drivers", {}).get(driver_label),
        "confounders": list(confounders),
        "model_used": result["model_used"],
        "var_ratio": float(result["var_ratio"]),
        "adf_pvalue": float(result["adf_pvalue"]) if pd.notna(result["adf_pvalue"]) else None,
    }
    if is_piecewise:
        metadata.update({
            "var_ratio_low": float(result["var_ratio_low"]),
            "adf_pvalue_low": float(result["adf_pvalue_low"]) if pd.notna(result["adf_pvalue_low"]) else None,
            "var_ratio_high": float(result["var_ratio_high"]),
            "adf_pvalue_high": float(result["adf_pvalue_high"]) if pd.notna(result["adf_pvalue_high"]) else None,
        })

    with metadata_file.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Stage 1 cache saved for {driver_label}: {residual_file}", event="CACHE_SAVE", driver=driver_label)


# =============================================================================
# PIPELINE UNIT CACHE PROTOCOL
# =============================================================================


def compute_unit_key(
    target: str,
    role: str,
    variable: str,
    controls: List[str],
    source_data_hash: str,
    train_window: str,
    test_window: str,
    cache_version: int = 1,
    **kwargs: Any,
) -> str:
    """16-hex fingerprint of Stage-1 unit identity."""
    return fp({
        "target": target,
        "role": role,
        "variable": variable,
        "controls": sorted(controls),
        "source_data_hash": source_data_hash,
        "train_window": train_window,
        "test_window": test_window,
        "cache_version": cache_version,
    })


def check_cache_hits(
    storage: BlobStorageClient,
    target: str,
    train_window: str,
    unit_keys: List[str],
    div: Optional[str] = None,
) -> Tuple[Set[str], Set[str]]:
    """One prefix listing -> (hits, misses). Hit = residuals.npz AND metadata.json both present."""
    present: Dict[str, Set[str]] = {}
    for blob in storage.list_blobs(get_residual_cache_prefix(target, train_window, div=div)):
        parts = blob.replace("\\", "/").split("/")
        if len(parts) >= 2:
            present.setdefault(parts[-2], set()).add(parts[-1])
    complete = {k for k, files in present.items() if {"residuals.npz", "metadata.json"} <= files}
    requested = set(unit_keys)
    hits = requested & complete
    return hits, requested - hits


def load_cached_unit(
    storage: BlobStorageClient,
    target: str,
    train_window: str,
    unit_key: str,
    div: Optional[str] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    residuals = storage.read_npz(get_residual_npz_path(target, train_window, unit_key, div=div))
    metadata = storage.read_json(get_residual_metadata_path(target, train_window, unit_key, div=div))
    return residuals, metadata


def save_cached_unit(
    storage: BlobStorageClient,
    target: str,
    train_window: str,
    unit_key: str,
    residuals_dict: Dict[str, Any],
    metadata: Dict[str, Any],
    overwrite: bool = False,
    div: Optional[str] = None,
) -> bool:
    """Write npz then metadata. Both overwrite=False so concurrent folds computing the same shared unit are safe."""
    wrote_npz = storage.write_npz(
        get_residual_npz_path(target, train_window, unit_key, div=div),
        residuals_dict,
        overwrite=overwrite,
    )
    wrote_meta = storage.write_json(
        get_residual_metadata_path(target, train_window, unit_key, div=div),
        metadata,
        overwrite=overwrite,
    )
    return wrote_npz or wrote_meta

