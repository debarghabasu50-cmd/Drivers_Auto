"""
Stage-1 residual cache: content-addressed, shared across folds and runs, partitioned by division.

Layout: cache/dml_stage1/{div}/{target}/{train_window}/{unit_key}/{residuals.npz, metadata.json}
Commit protocol: residuals.npz first, metadata.json LAST. A unit is a HIT only when both exist,
so a crash between the two writes is simply a miss that gets repaired on the next fit.
"""
from typing import Any, Dict, List, Optional, Set, Tuple

from shared.path_builder.paths import (
    get_residual_cache_prefix,
    get_residual_metadata_path,
    get_residual_npz_path,
)
from shared.path_builder.storage import BlobStorageClient
from shared.schema.contracts import fp


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
    """16-hex fingerprint of Stage-1 unit identity.
    Simplified: includes target, role, variable, controls, data hash, train/test window, and cache_version.
    Modelling parameters (learner_spec, inner_cv_splits, seed) are removed to keep it simple.
    """
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


def check_cache_hits(storage: BlobStorageClient, target: str, train_window: str,
                     unit_keys: List[str], div: Optional[str] = None) -> Tuple[Set[str], Set[str]]:
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


def load_cached_unit(storage: BlobStorageClient, target: str, train_window: str,
                     unit_key: str, div: Optional[str] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    residuals = storage.read_npz(get_residual_npz_path(target, train_window, unit_key, div=div))
    metadata = storage.read_json(get_residual_metadata_path(target, train_window, unit_key, div=div))
    return residuals, metadata


def save_cached_unit(storage: BlobStorageClient, target: str, train_window: str, unit_key: str,
                     residuals_dict: Dict[str, Any], metadata: Dict[str, Any], overwrite: bool = False,
                     div: Optional[str] = None) -> bool:
    """Write npz then metadata (both overwrite=False so concurrent folds computing the same shared
    unit are safe; overwrite=True only to repair a corrupt entry). Returns True if this call wrote anything."""
    wrote_npz = storage.write_npz(get_residual_npz_path(target, train_window, unit_key, div=div),
                                  residuals_dict, overwrite=overwrite)
    wrote_meta = storage.write_json(get_residual_metadata_path(target, train_window, unit_key, div=div),
                                    metadata, overwrite=overwrite)
    return wrote_npz or wrote_meta
