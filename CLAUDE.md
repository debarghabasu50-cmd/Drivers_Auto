# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Status

Causal driver-analysis pipeline (Double Machine Learning of driver KPIs on target KPIs such as `orders`). The **orchestration, caching, status/logging and failure handling are real and tested**; the **science is mock** (`select_kpis`, `select_confounders`, `fit_stage1_unit`, `fit_stage2` — these four functions are the seams where real code goes in). Storage is a local mock by default and live Azure Blob when `AZURE_STORAGE_CONNECTION_STRING` is set; AML submission is implemented in `shared/aml.py`. **The live Blob and AML code paths are only tested against fakes (no Azure account was available)** - verify them once against dev before relying on them. Dockerfile exists but has not been built here. Not a git repo.

Design docs: `plan.md` (blueprint: layout, blob layout, DAG, failure matrix) and `pipeline_arc.md` (original brief; priorities reproducibility > traceability > simplicity > performance; don't invent methodology). Known doc conflict: arc says Airflow is the only manifest writer; we follow plan.md (init creates → consolidate closes, strictly sequential). `pipeline_arc.md` mentions `src/methods/`/`python -m src.main`; the repo uses `components/` and `run_pipeline.py`.

## Commands

```bash
pip install -r requirements.txt
python -m pytest                                   # all tests (tests/test_e2e.py runs the whole flow on tmp mock storage)
python -m pytest tests/test_e2e.py::test_rerun_is_reproducible_and_fully_cached

python run_local.py [--run-id ID] [--retries 2] [--config big.json]   # full run in DAG order, no Azure; storage = .mock_blob/ (or $MOCK_BLOB_DIR); JSON logs on stdout, status table on stderr
python -m pipelines.run_status --run-id ID [--watch 5]   # live per-task table (PENDING/RUNNING/SUCCEEDED/...)
python run_pipeline.py --step dml --run-id ID --target orders --fold-id f1   # one step, as AML invokes it
```
Steps: `init_run | feature_selection | confounder_selection | dml | consolidate | validate`. Exit code 0 ok, 1 retryable failure, 2 non-retryable.

## Architecture

Flow: `init_run` → `feature_selection[target]` → `confounder_selection[target]` → `dml_fold[target|fold]` → `consolidate` (last step; always runs, closes the run). The Airflow DAG (`airflow/dags/driver_analysis_dag.py`) is a thin dispatcher: it reads targets/work items from the manifest written by init_run, calls `run_step`, reads the status file and raises `AirflowFailException` (non-retryable) or a normal exception (retry) by error category. `run_local.py` mimics the same trigger rules without Airflow. Don't put `airflow/` ahead of site-packages on `sys.path` (the local package would shadow real Airflow).

**Artifact reuse across runs** (`shared/context.py`): switch a step `"off"` in `steps` to reuse an earlier run's output instead of re-running it (feature_selection, confounder_selection). Each of those artifacts stores a `step_fp` = fp(data snapshot id + the config sections it reads, chained downstream). The reusing run recomputes the expected `step_fp` from *its own* snapshot/config and only accepts `source_artifacts.<step>_run_id` - or, when that is null, the latest earlier run - if the fingerprints match; otherwise the task fails with `CONFIG` (never silently reuses stale output). Reused steps get a SKIPPED status whose `counters.reused_from` is recorded in the manifest `lineage.reused_from`. Switched-off steps run in-process in the DAG (no AML job). `consolidate` always runs and closes the run: `steps.dml=off` -> nothing to union (an "up to selection" run closes SUCCEEDED); `steps.consolidation=off` -> close but do not publish/report.

**Azure execution**: `AML_MODE=aml` makes the DAG submit each step as an AML command job (`shared/aml.py`, env settings from `config/env/$AML_ENV.yaml`, secrets from Airflow Variables, job tags = run_id/step/task_id/attempt, studio URL logged, poll with timeout then cancel); default `local` runs steps in-process. Outcome is always decided from the step's status file: no/RUNNING final status after the job ended = transient -> Airflow retry; FAILED with a non-retryable category -> `AirflowFailException`. Azure SDK errors are classified in `run_pipeline._classify`. `new_run_id(seed=<dag run id>)` keeps run_id stable across init_run retries.

**Component contract** (`shared/context.py`): every step is `run_xxx(ctx: StepContext) -> StepResult`. Components read via `ctx` (`config`, `storage`, `load_snapshot()`, `upstream_run_id(step)`, `require(path)`), return outputs/counters/warnings, and raise `PipelineError(category, msg)` to fail. They never write status files, bind logging, or time themselves. Register new steps in `STEPS` in `run_pipeline.py`.

**`run_pipeline.main()` is the only status writer**: writes `runs/{run_id}/status/{task_id}.json` as RUNNING at start and final at end, classifies exceptions into `ErrorCategory` (traceback stored), honors step toggles (`"off"` → SKIPPED; upstream artifacts then come from `source_artifacts.*_run_id`), and emits `TASK_STARTED`/`TASK_SUMMARY` log events. Non-init steps require the frozen config in blob (`UPSTREAM_MISSING` otherwise — never fall back to repo config).

**Identity and paths**: all blob paths and `make_task_id`/`plan_tasks` live in `shared/path_builder/paths.py`; never inline path strings or task ids. Hashing: `fp` (16-hex sha256 of sorted JSON) and `data_fp` only. `snapshot_id = snap_+data_fp(input)` (content-addressed parquet), `config_version = cfg_+fp(frozen config)`; `init_run` merges `dml_config.json` and `feature_selection_config.json` into the frozen config under `dml` / `feature_selection`, so everything that affects results is covered by `config_version`.

**Manifest lifecycle**: `init_run` creates `run_manifest.json` (RUNNING, with `plan.tasks`) + run index; `consolidate` reads every planned task's status file, unions `dml_result.parquet` per fold with cross-fold aggregates into `published/{run_id}/consolidated_result.parquet`, then writes final status (SUCCEEDED | PARTIAL | FAILED), `steps`, `completeness`, `lineage`, `end_time` into manifest and index. A missing status file = hard crash = `MISSING`.

**Combinations (the heavy part)**: `shared/combinations.py` enumerates every k-of-N set of the selected drivers (`combinations.drivers_per_combo`, `must_include`, deterministic cap `max_combinations` by smallest `combination_id` hash) and computes `expected_rows` (= combos × k per fold) for consolidate. DML plans, per combination and driver t: controls = (combination − t) + confounders(t), one outcome unit and one treatment unit; units are deduplicated by key across combinations (10-choose-7: 1680 planned → 1050 fitted per fold, ~4 s mock, fully cached on rerun). Fits run in-process by default; `execution.n_jobs > 1` uses a streamed loky pool (results are cached as they arrive). Output: one `dml_result` row per (combination, driver); consolidate aggregates mean/std/CV/n_folds per (target, combination, driver) across folds.

**Stage-1 residual cache** (`components/residual_caching/`): `cache/dml_stage1/{target}/{train_window}/{unit_key}/{residuals.npz,metadata.json}`. `unit_key` includes train *and test* window, data hash of the exact training columns, controls, learner spec, seed, `cache_version` (all from `dml_config.json`). Commit protocol: npz then metadata last; a HIT needs both. Units are deduped before fitting. Unit errors of category MODEL_CONVERGENCE/NUMERIC only fail the affected treatments (fold → PARTIAL); any other error fails the fold so Airflow retries it (finished units are cache hits).

**Logging/RCA**: JSON lines to stdout (`shared/logging/logger.py`), every record carries `run_id, task_id, component, target, fold_id, attempt, image`; errors carry `error_category` and `exc` traceback. RCA path: filter logs by `run_id` → first ERROR by `ts` → `status/{task_id}.json` → manifest `steps`/`completeness` → fold `stage1_refs.parquet` → cache `metadata.json`.

Out of scope for now: live Azure Blob, per-unit timeout (`execution.unit_timeout_s` is not enforced), `step_fp` auto-reuse, the Validation component (`components/validation/evaluate.py` is a stub).
