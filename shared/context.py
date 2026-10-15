"""
Step contract shared by every pipeline component.

A component is a function ``run_xxx(ctx: StepContext) -> StepResult``. It reads inputs through
``ctx`` (config, storage, snapshot, upstream run ids) and returns a StepResult. It never writes
its own status file, binds logging context or catches errors for bookkeeping - ``run_pipeline``
does that uniformly. Swapping in real science code means replacing the body of the function,
not touching orchestration.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from datetime import datetime

from shared.logging import logger
from shared.path_builder.paths import (
    get_confounder_selection_path,
    get_feature_selection_summary,
    get_manifest_path,
    get_lagged_features_path,
    get_task_status_path,
    make_task_id,
    set_active_scope,
)
from shared.schema.contracts import ErrorCategory, PipelineError, fp

# Steps whose artifacts can be reused from an earlier run (switch the step "off" in config).
REUSABLE_STEPS = {
    "feature_selection": ("feature_selection", get_feature_selection_summary),
    "confounder_selection": ("confounder_selection", get_confounder_selection_path),
}


@dataclass
class StepResult:
    outputs: List[str] = field(default_factory=list)
    counters: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    status: str = "SUCCEEDED"  # SUCCEEDED | PARTIAL | SKIPPED


@dataclass
class StepContext:
    run_id: str
    step: str
    config: Dict[str, Any]
    storage: Any
    target: Optional[str] = None
    fold_id: Optional[str] = None
    attempt: int = 1

    def __post_init__(self):
        # Set active scope in path builder so all derived paths match division and quarter
        set_active_scope(self.div, self.quarter)

    @property
    def div(self) -> str:
        return self.config.get("div") or self.config.get("pipeline", {}).get("div", "EL")

    @property
    def quarter(self) -> str:
        return self.config.get("quarter") or self.config.get("pipeline", {}).get("quarter", "2026Q3")

    @property
    def task_id(self) -> str:
        return make_task_id(self.step, self.target, self.fold_id)

    # ---- artifact reuse -------------------------------------------------
    def step_fp(self, step: str) -> str:
        """Fingerprint of everything that determines a reusable step's output: the data snapshot and the
        config sections it reads (chained for downstream steps). Stored inside the step's artifact, so a
        later run can prove an old artifact is still valid for its own data + config."""
        c = self.config
        if step == "feature_selection":
            return fp({"snapshot": self.manifest()["snapshot"]["snapshot_id"], "data": c["data"],
                       "targets": c["targets"], "kpis": c["kpis"], "params": c["feature_selection"]})
        if step == "confounder_selection":
            return fp({"upstream": self.step_fp("feature_selection"), "confounders": c.get("confounders", {})})
        raise ValueError(f"step '{step}' is not reusable")

    def upstream_run_id(self, upstream_step: str, target: Optional[str] = None) -> str:
        if self.config.get("steps", {}).get(upstream_step, "on") != "off":
            return self.run_id

        target = target or self.target
        if not target:
            raise PipelineError(
                ErrorCategory.CONFIG, f"target required for '{upstream_step}'"
            )

        expected = self.step_fp(upstream_step)
        src = self.config.get("source_artifacts", {}).get(f"{upstream_step}_run_id")

        if src:
            if (
                self._compatible_source_completion(upstream_step, src, target, expected)
                is None
            ):
                raise PipelineError(
                    ErrorCategory.CONFIG,
                    f"source run {src} is not compatible with this run for '{upstream_step}' "
                    f"(no completed, compatible artifact found for target '{target}')",
                )
            return src

        src = self._latest_compatible_run(upstream_step, target, expected)
        if src is None:
            raise PipelineError(
                ErrorCategory.CONFIG,
                f"steps.{upstream_step}='off' but no compatible earlier run was found "
                f"for target '{target}'",
            )

        return src

    def _compatible_source_completion(
        self, step: str, run_id: str, target: str, expected: str
    ) -> Optional[datetime]:
        """Return task completion time if this run has a usable artifact set."""
        _, path_of = REUSABLE_STEPS[step]
        artifact_path = path_of(run_id, target, div=self.div, quarter=self.quarter)
        status_path = get_task_status_path(
            run_id, make_task_id(step, target), div=self.div, quarter=self.quarter
        )

        if not self.storage.exists(artifact_path) or not self.storage.exists(
            status_path
        ):
            return None

        artifact = self.storage.read_json(artifact_path)
        status = self.storage.read_json(status_path)

        if artifact.get("step_fp") != expected or status.get("status") != "SUCCEEDED":
            return None

        if step == "feature_selection":
            lagged_path = get_lagged_features_path(
                run_id, target, div=self.div, quarter=self.quarter
            )
            if not self.storage.exists(lagged_path):
                return None

        ended_at = status.get("ended_at")
        if not ended_at:
            return None

        try:
            completed = datetime.fromisoformat(ended_at.replace("Z", "+00:00"))
            return completed if completed.tzinfo is not None else None
        except ValueError:
            return None

    def _latest_compatible_run(
        self, step: str, target: str, expected: str
    ) -> Optional[str]:
        prefix = f"runs/{self.div}/{self.quarter}/"
        candidate_runs = set()

        for blob in self.storage.list_blobs(prefix):
            parts = blob.replace("\\", "/").split("/")
            if len(parts) >= 4 and parts[:3] == ["runs", self.div, self.quarter]:
                if parts[3] != self.run_id:
                    candidate_runs.add(parts[3])

        matches = []
        for run_id in candidate_runs:
            completed = self._compatible_source_completion(
                step, run_id, target, expected
            )
            if completed is not None:
                matches.append((completed, run_id))

        return max(matches)[1] if matches else None

    # ---- inputs ---------------------------------------------------------
    def manifest(self) -> Dict[str, Any]:
        return self.require(get_manifest_path(self.run_id, div=self.div, quarter=self.quarter))

    def load_snapshot(self):
        """Frozen, validated input dataframe of this run."""
        manifest = self.manifest()
        snap = manifest.get("snapshot", {})
        uri = snap.get("uri")
        snap_id = snap.get("snapshot_id")
        df = self.storage.read_parquet(uri)
        logger.log_provenance(
            source_run_id=self.run_id,
            path=uri,
            target=self.target,
            fingerprint=snap_id,
            msg=f"loaded snapshot: {uri}",
            rows=len(df),
            cols=len(df.columns),
        )
        return df

    def load_lagged_dataset(self, target: Optional[str] = None):
        """Dataset with lagged features emitted by feature_selection (falls back to snapshot if absent)."""
        import pandas as pd

        t = target or self.target
        if not t:
            raise PipelineError(ErrorCategory.CONFIG, "target required for lagged dataset")

        source_run = self.upstream_run_id("feature_selection", t)
        path = get_lagged_features_path(
            source_run, t, div=self.div, quarter=self.quarter
        )
        if not self.storage.exists(path):
            raise PipelineError(
                ErrorCategory.UPSTREAM_MISSING,
                f"required lagged dataset missing: {path}"
            )

        df = self.storage.read_parquet(path)
        date_col = self.config.get("data", {}).get("date_col", self.config.get("date_col", "date"))
        if date_col and date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col])

        logger.log_provenance(
            source_run_id=source_run,
            path=path,
            target=t,
            fingerprint=self.step_fp("feature_selection"),
            msg=f"loaded lagged dataset: {path}",
            rows=len(df),
            cols=len(df.columns),
        )
        return df
        

    def fold_spec(self) -> Dict[str, Any]:
        spec = next((f for f in self.config.get("folds", []) if f["fold_id"] == self.fold_id), None)
        if spec is None:
            raise PipelineError(ErrorCategory.CONFIG, f"unknown fold_id '{self.fold_id}'")
        return spec

    def require(self, path: str) -> Any:
        """Read an upstream JSON artifact or fail with UPSTREAM_MISSING."""
        if not self.storage.exists(path):
            raise PipelineError(ErrorCategory.UPSTREAM_MISSING, f"missing upstream artifact: {path}")
        data = self.storage.read_json(path)
        if isinstance(data, dict):
            source_run = data.get("run_id")
            step_fp = data.get("step_fp")
            target_val = data.get("target") or self.target
            if source_run or step_fp:
                logger.log_provenance(
                    source_run_id=source_run or self.run_id,
                    path=path,
                    target=target_val,
                    fingerprint=step_fp,
                    msg=f"loaded upstream artifact: {path}",
                )
        return data
