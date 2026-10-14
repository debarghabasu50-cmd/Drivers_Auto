"""
Core contracts, hashing functions, and canonical schemas for the driver analysis pipeline.
"""
import hashlib
import json
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any


# Canonical Error Categories
class ErrorCategory:
    DATA_SCHEMA = "DATA_SCHEMA"
    DATA_QUALITY = "DATA_QUALITY"
    CONFIG = "CONFIG"
    UPSTREAM_MISSING = "UPSTREAM_MISSING"
    MODEL_CONVERGENCE = "MODEL_CONVERGENCE"
    NUMERIC = "NUMERIC"
    TIMEOUT = "TIMEOUT"
    INFRA_TRANSIENT = "INFRA_TRANSIENT"
    CODE_BUG = "CODE_BUG"
    UNKNOWN = "UNKNOWN"


# Only transient infrastructure problems are worth retrying; everything else fails fast.
RETRYABLE_CATEGORIES = {ErrorCategory.INFRA_TRANSIENT, ErrorCategory.TIMEOUT}


def is_retryable(category: Optional[str]) -> bool:
    return category in RETRYABLE_CATEGORIES


class PipelineError(Exception):
    """Raise from any component to fail a task with a canonical ErrorCategory."""

    def __init__(self, category: str, message: str):
        super().__init__(message)
        self.category = category
        self.message = message


def new_run_id(now=None, seed: str = None) -> str:
    """run_id = dra_{YYYYMMDDTHHMM}_{6 hex}. With `seed` (e.g. the Airflow dag-run id) the suffix is
    deterministic, so a retried init_run task keeps the same run_id instead of orphaning a new one."""
    import uuid
    from datetime import datetime, timezone
    now = now or datetime.now(timezone.utc)
    suffix = hashlib.sha256(seed.encode("utf-8")).hexdigest()[:6] if seed else uuid.uuid4().hex[:6]
    return f"dra_{now.strftime('%Y%m%dT%H%M')}_{suffix}"


def fp(obj: Any) -> str:
    """
    Compute 16-character hex hash of canonical JSON (sorted keys).
    Used for config, metadata, and fingerprint generation.
    """
    serialized = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]


def data_fp(data: Any) -> str:
    """
    Compute content fingerprint of tabular data.
    If pandas DataFrame is passed and pandas is installed, uses hash_pandas_object.
    Otherwise hashes serialized tabular rows or bytes.
    """
    try:
        import pandas as pd
        if isinstance(data, pd.DataFrame):
            # Sum of row hashes as defined in the spec
            row_hashes = pd.util.hash_pandas_object(data, index=True)
            total_sum = int(row_hashes.sum())
            return fp({"data_sum": total_sum, "shape": list(data.shape)})
    except ImportError:
        pass

    if isinstance(data, (dict, list)):
        return fp(data)
    elif isinstance(data, (bytes, bytearray)):
        return hashlib.sha256(data).hexdigest()[:16]
    else:
        return hashlib.sha256(str(data).encode("utf-8")).hexdigest()[:16]


@dataclass
class CacheMetadataContract:
    fingerprint: str
    cache_version: int
    role: str
    target: str
    driver_col: str
    controls: List[str]
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    fold_id: str
    source_file: str
    source_data_hash: str
    learner_spec: Dict[str, Any]
    inner_cv_splits: int
    seed: int
    created_at: str
    producer_run_id: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TaskStatusContract:
    run_id: str
    task_id: str
    step: str
    status: str  # RUNNING, SUCCEEDED, PARTIAL, FAILED, SKIPPED
    target: Optional[str] = None
    fold_id: Optional[str] = None
    attempt: int = 1
    started_at: Optional[str] = None
    ended_at: Optional[str] = None
    outputs: List[str] = field(default_factory=list)
    counters: Dict[str, Any] = field(default_factory=dict)
    cache_stats: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    error: Optional[Dict[str, str]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RunManifestContract:
    schema_version: str = "1.0"
    run_id: str = ""
    status: str = "RUNNING"
    start_time: str = ""
    end_time: Optional[str] = None
    image: str = "unknown"
    snapshot: Dict[str, Any] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)
    step_toggles: Dict[str, str] = field(default_factory=dict)
    plan: Dict[str, Any] = field(default_factory=dict)
    steps: Dict[str, Any] = field(default_factory=dict)
    completeness: Dict[str, Any] = field(default_factory=dict)
    lineage: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
