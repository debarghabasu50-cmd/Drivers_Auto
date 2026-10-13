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

from shared.path_builder.paths import (
    get_confounder_selection_path,
    get_feature_selection_summary,
    get_manifest_path,
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
            return fp({"upstream": self.step_fp("feature_selection"), "confounders": c["confounders"]})
        raise ValueError(f"step '{step}' is not reusable")

    def upstream_run_id(self, upstream_step: str, target: Optional[str] = None) -> str:
        """Run that owns the artifacts of `upstream_step`: this run when the step is on; otherwise a previous
        run - `source_artifacts.<step>_run_id`, or (if null) the latest run whose artifact fingerprint matches.
        An incompatible source (other data or config) is rejected, never silently used."""
        if self.config.get("steps", {}).get(upstream_step, "on") != "off":
            return self.run_id
        target = target or self.target
        _, path_of = REUSABLE_STEPS[upstream_step]
        expected = self.step_fp(upstream_step)
        src = self.config.get("source_artifacts", {}).get(f"{upstream_step}_run_id")
        if src:
            found = self.require(path_of(src, target, div=self.div, quarter=self.quarter))
            if found.get("step_fp") != expected:
                raise PipelineError(
                    ErrorCategory.CONFIG,
                    f"source run {src} is not compatible with this run for '{upstream_step}' "
                    f"(different data snapshot or config); re-run the step or point to a compatible run")
            return src
        src = self._latest_compatible_run(upstream_step, target, expected)
        if src is None:
            raise PipelineError(ErrorCategory.CONFIG,
                                f"steps.{upstream_step}='off' but no compatible earlier run was found; "
                                f"set source_artifacts.{upstream_step}_run_id or run the step")
        return src

    def _latest_compatible_run(self, step: str, target: str, expected: str) -> Optional[str]:
        prefix = f"runs/{self.div}/{self.quarter}/"
        blobs = self.storage.list_blobs(prefix)
        candidate_runs = set()
        for b in blobs:
            parts = b.replace("\\", "/").split("/")
            if len(parts) >= 4 and parts[0] == "runs" and parts[1] == self.div and parts[2] == self.quarter:
                r_id = parts[3]
                if r_id != self.run_id and not r_id.startswith("_index"):
                    candidate_runs.add(r_id)

        sorted_runs = sorted(candidate_runs, reverse=True)
        _, path_of = REUSABLE_STEPS[step]
        for run in sorted_runs:
            artifact_path = path_of(run, target, div=self.div, quarter=self.quarter)
            if self.storage.exists(artifact_path):
                if self.storage.read_json(artifact_path).get("step_fp") == expected:
                    return run
        return None

    # ---- inputs ---------------------------------------------------------
    def manifest(self) -> Dict[str, Any]:
        return self.require(get_manifest_path(self.run_id, div=self.div, quarter=self.quarter))

    def load_snapshot(self):
        """Frozen, validated input dataframe of this run."""
        return self.storage.read_parquet(self.manifest()["snapshot"]["uri"])

    def load_lagged_dataset(self, target: Optional[str] = None):
        """Dataset with lagged features emitted by feature_selection (falls back to snapshot if absent)."""
        import pandas as pd
        t = target or self.target
        df = None
        if t:
            try:
                fs_run = self.upstream_run_id("feature_selection", t)
                from shared.path_builder.paths import get_lagged_features_path
                path = get_lagged_features_path(fs_run, t, div=self.div, quarter=self.quarter)
                if self.storage.exists(path):
                    df = self.storage.read_parquet(path)
            except Exception:
                pass
        if df is None:
            df = self.load_snapshot()
        
        date_col = self.config.get("data", {}).get("date_col")
        if date_col and date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col])
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
        return self.storage.read_json(path)
