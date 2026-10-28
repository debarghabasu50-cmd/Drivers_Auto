# Production Causal DML Pipeline Runbook

A complete operational guide for data scientists and MLOps engineers running, monitoring, resuming, and configuring the causal Double Machine Learning (DML) driver analysis pipeline.

---

## Table of Contents

1. [Pipeline Overview & Architecture](#1-pipeline-overview--architecture)
2. [Environment & Prerequisites](#2-environment--prerequisites)
3. [Quick Start: Automated End-to-End Run](#3-quick-start-automated-end-to-end-run)
4. [Primary Workflow: Feature Selection → Update DML Config → Resume Remaining Steps](#4-primary-workflow-feature-selection--update-dml-config--resume-remaining-steps)
5. [Single Component CLI Reference](#5-single-component-cli-reference)
6. [Artifact Reuse & Step Toggling](#6-artifact-reuse--step-toggling)
7. [Observability & Live Status Monitoring](#7-observability--live-status-monitoring)
8. [Stage-1 Residual Caching Operations](#8-stage-1-residual-caching-operations)
9. [Failure Recovery & Retries](#9-failure-recovery--retries)
10. [Command Cheat Sheet](#10-command-cheat-sheet)

---

## 1. Pipeline Overview & Architecture

The pipeline evaluates macroeconomic and industry drivers against company targets (e.g. `orders`) using Double Machine Learning with Frisch-Waugh-Lovell residualization:

```
[Raw Snapshot Input]
       │
       ▼
   init_run                      (Freezes input parquet snapshot + merged config)
       │
       ▼
feature_selection                (13-step screening: lags, MI, cointegration, FCS*, P0-P3 tiers)
       │                         Emits: feature_prioritization.json, data_with_lagged_features.parquet, fs_report.xlsx
       ▼
confounder_selection             (Runs ONCE per target; selects common confounder set across treatments)
       │                         Emits: confounder_set.json
       ▼
  dml_fold × F                   (Runs per target per fold: f1, f2, ...)
       │                         Emits: dml_result.parquet, stage1_refs.parquet, caches Stage-1 residuals
       ▼
  consolidate                    (Aggregates across folds, checks completeness, generates reports)
       │                         Emits: published/ consolidated_result.parquet, reports/ dml_report.xlsx
       ▼
   validate                      (Enforces strict causal signs, stability, and significance thresholds)
```

### Storage Directory Layout

All artifacts adhere to the multi-division, multi-quarter blob structure:

```
{container}/
├── snapshots/{snapshot_id}/input.parquet                 # Content-addressed raw data
├── runs/{div}/{quarter}/{run_id}/
│   ├── master_config.json                               # Frozen config for the run
│   ├── run_manifest.json                                # Run state machine & task plan
│   ├── status/{task_id}.json                            # Task execution status & counters
│   ├── logs/{task_id}.log                               # Task execution log
│   ├── feature_selection/{target}/
│   │   ├── feature_prioritization.json                  # P0-P3 tiers, resolved KPIs, lags
│   │   ├── data_with_lagged_features.parquet            # Lagged features dataset
│   │   └── selected_features.parquet                    # Feature matrix
│   ├── confounder_selection/{target}/
│   │   └── confounder_set.json                          # Common confounder set
│   └── dml/{target}/fold={fold_id}/
│       ├── dml_result.parquet                           # Elasticity point estimates & p-values
│       └── stage1_refs.parquet                          # Pointers to Stage-1 unit keys
├── cache/dml_stage1/{div}/{target}/{window}/{unit_key}/  # Cross-run reusable residual arrays
├── published/{div}/{quarter}/{latest|runs/{run_id}}/    # Consumer serving layer (BI/Tableau)
└── reports/{div}/{quarter}/{latest|runs/{run_id}}/      # Reviewer Excel workbooks
```

---

## 2. Environment & Prerequisites

### Windows PowerShell Setup
```powershell
# Set Python path to dependencies
$env:PYTHONPATH="C:\Users\Debargha\drivers_pipeline\Drivers_Auto\.venv\Lib\site-packages"
$PYTHON="C:\Users\Debargha\AppData\Roaming\uv\python\cpython-3.12.15-windows-x86_64-none\python.exe"

# Optional: Set division and quarter scope (defaults: div=EL, quarter=2026Q3)
$env:PIPELINE_DIV="EL"
$env:PIPELINE_QUARTER="2026Q3"
```

### Storage Backend
- **Local Mock (default for local runs/tests)**: Uses `.mock_blob/drivers-pipeline/` (or `$env:MOCK_BLOB_DIR`).
- **Live Azure Blob**: Set connection string:
  ```powershell
  $env:AZURE_STORAGE_CONNECTION_STRING="DefaultEndpointsProtocol=https;AccountName=...;AccountKey=...;"
  $env:AZURE_STORAGE_CONTAINER="drivers-pipeline"
  ```

---

## 3. Quick Start: Automated End-to-End Run

To run the complete pipeline locally in DAG order (mirroring Airflow trigger rules):

```powershell
# Run with automatically generated run_id
& $PYTHON run_local.py

# Run with custom run_id and retry attempts
& $PYTHON run_local.py --run-id dra_20261027_full --retries 2

# Run with alternative config file
& $PYTHON run_local.py --config config/master_config.json
```

---

## 4. Primary Workflow: Feature Selection → Update DML Config → Resume Remaining Steps

This is the standard data science operating rhythm:
1. Run feature selection to screen hundreds of raw KPIs and produce prioritization tiers.
2. Inspect the feature prioritization decisions and Excel report.
3. Update `config/dml_config.json` with the chosen drivers, combinations, controls, and folds.
4. Run the downstream jobs reusing the feature selection outputs.

### Step 1: Initialize the Run and Execute Feature Selection

```powershell
# 1. Initialize the run workspace
& $PYTHON run_pipeline.py --step init_run --run-id dra_20261027_run1

# 2. Run feature selection for the target
& $PYTHON run_pipeline.py --step feature_selection --run-id dra_20261027_run1 --target orders
```

### Step 2: Review Feature Prioritization Results

Check the emitted outputs in `.mock_blob/drivers-pipeline/` (or Azure Blob):
- **JSON Breakdown**:
  `runs/EL/2026Q3/dra_20261027_run1/feature_selection/orders/feature_prioritization.json`
- **Reviewer Excel Workbook**:
  `reports/EL/2026Q3/runs/dra_20261027_run1/fs_report.xlsx`

Key sections in `feature_prioritization.json`:
- `priority_sets`: Summary of candidates in each tier:
  - **`P0`**: Primary selected drivers (high signal, optimal lag, category & country diversity satisfied).
  - **`P1`**: High FCS* score + strong partial correlation (recommended alternative drivers).
  - **`P2`**: High signal strength, timing uncertain (low DTW pattern alignment).
  - **`P3`**: Strong visual/pattern alignment, lower statistical significance.
- `resolved_kpis`: Clean list of driver names selected at tier `P0`.
- `kpis`: Full diagnostic rows per candidate with Best_Lag, Category, Theme, and audit reason.

### Step 3: Update `config/dml_config.json`

Open `config/dml_config.json` and adjust driver treatments, controls, and combination settings based on the `feature_prioritization.json` insights:

```json
{
  "drivers": {
    "price_index": "price_index | lag_3",
    "competitor_index": "competitor_index | lag_3",
    "gdp_growth": "gdp_growth | lag_6"
  },
  "combinations": {
    "drivers_per_combo": 2,
    "max_combinations": 20
  }
}
```

### Step 4: Resume Downstream Steps

There are two supported methods to resume:

#### Option A: Native Step Toggle (Recommended for Production & Airflow)
Keep full lineage by launching the second run with feature selection toggled off and pointing to `dra_20261027_run1`:

1. In `config/master_config.json`, set:
   ```json
   {
     "steps": {
       "feature_selection": "off"
     },
     "source_artifacts": {
       "feature_selection_run_id": "dra_20261027_run1"
     }
   }
   ```
2. Execute the downstream run:
   ```powershell
   & $PYTHON run_local.py --run-id dra_20261027_run2
   ```
   *Result*: `feature_selection` is marked `SKIPPED`, its artifacts are loaded directly from `dra_20261027_run1`, and `confounder_selection`, `dml`, and `consolidate` run with the updated `dml_config.json`!

#### Option B: Component-by-Component CLI (Same Run ID)
If continuing within the same run ID `dra_20261027_run1`:

```powershell
# 1. Run confounder selection
& $PYTHON run_pipeline.py --step confounder_selection --run-id dra_20261027_run1 --target orders

# 2. Run DML folds
& $PYTHON run_pipeline.py --step dml --run-id dra_20261027_run1 --target orders --fold-id f1
& $PYTHON run_pipeline.py --step dml --run-id dra_20261027_run1 --target orders --fold-id f2

# 3. Consolidate results & generate published artifacts
& $PYTHON run_pipeline.py --step consolidate --run-id dra_20261027_run1
```

---

## 5. Single Component CLI Reference

Every step can be invoked independently using `run_pipeline.py`.

### Step 1: `init_run`
Initializes run workspace, creates content-addressed input snapshot, and registers the run index.
```powershell
& $PYTHON run_pipeline.py --step init_run --run-id <run_id> [--config <master_config_path>]
```
- **Inputs**: `config/master_config.json`, `data.input_uri`
- **Outputs**: `snapshots/{snapshot_id}/input.parquet`, `master_config.json`, `run_manifest.json`

### Step 2: `feature_selection`
Executes 13-step sequential screening pipeline for candidate drivers.
```powershell
& $PYTHON run_pipeline.py --step feature_selection --run-id <run_id> --target <target_id>
```
- **Inputs**: Snapshot input table, frozen config
- **Outputs**:
  - `feature_prioritization.json`
  - `data_with_lagged_features.parquet`
  - `selected_features.parquet`
  - `reports/.../fs_report.xlsx`

### Step 3: `confounder_selection`
Selects confounders across all treatments for the given target. Runs ONCE per target.
```powershell
& $PYTHON run_pipeline.py --step confounder_selection --run-id <run_id> --target <target_id>
```
- **Inputs**: `data_with_lagged_features.parquet`, `feature_prioritization.json`
- **Outputs**: `confounder_selection/{target}/confounder_set.json`

### Step 4: `dml` (per fold)
Runs Double Machine Learning estimation for all driver combinations on a specific fold.
```powershell
& $PYTHON run_pipeline.py --step dml --run-id <run_id> --target <target_id> --fold-id <fold_id>
```
- **Inputs**: `data_with_lagged_features.parquet`, `confounder_set.json`, Stage-1 cache
- **Outputs**:
  - `dml/{target}/fold={fold_id}/dml_result.parquet`
  - `dml/{target}/fold={fold_id}/stage1_refs.parquet`
  - Updates `cache/dml_stage1/...`

### Step 5: `consolidate`
Unions fold results, computes cross-fold elasticity aggregates, writes published serving endpoints and final sign-off workbook.
```powershell
& $PYTHON run_pipeline.py --step consolidate --run-id <run_id>
```
- **Inputs**: All fold `dml_result.parquet` files, `run_manifest.json`
- **Outputs**:
  - `published/{div}/{quarter}/runs/{run_id}/consolidated_result.parquet`
  - `published/{div}/{quarter}/latest/consolidated_result.parquet`
  - `reports/{div}/{quarter}/runs/{run_id}/dml_report.xlsx`
  - Updates `run_manifest.json` status to `SUCCEEDED`

### Step 6: `validate`
Verifies sign consistency, causal significance, and model stability against configured acceptance thresholds.
```powershell
& $PYTHON run_pipeline.py --step validate --run-id <run_id>
```

---

## 6. Artifact Reuse & Step Toggling

The pipeline includes built-in caching for coarse steps (`feature_selection` and `confounder_selection`).

### Configuration Switches in `master_config.json`

```json
{
  "steps": {
    "feature_selection": "off",
    "confounder_selection": "off",
    "dml": "on",
    "consolidation": "on"
  },
  "source_artifacts": {
    "feature_selection_run_id": "dra_20261027_run1",
    "confounder_selection_run_id": "dra_20261027_run1"
  }
}
```

### Resolution Rules:
- **Explicit Source Run**: If `source_artifacts.feature_selection_run_id` is set, the pipeline loads artifacts directly from that run ID.
- **Automatic Fallback**: If `source_artifacts.*` is set to `null` while the step is `"off"`, the context automatically searches `runs/_index/{div}/{quarter}/` for the latest terminal `SUCCEEDED` run with a matching fingerprint.

---

## 7. Observability & Live Status Monitoring

### Live CLI Status Table
Inspect active or historical runs with task-level durations, counters, and failure categories:

```powershell
# Single check
& $PYTHON -m pipelines.run_status --run-id <run_id>

# Watch mode (polls every 5 seconds until terminal status)
& $PYTHON -m pipelines.run_status --run-id <run_id> --watch 5

# Inspect specific division/quarter
& $PYTHON -m pipelines.run_status --run-id <run_id> --div EL --quarter 2026Q3
```

**Example Output:**
```
run dra_20261027_run1 (div=EL, quarter=2026Q3)  status=SUCCEEDED  snap_d361f8e51eb9572c
TASK                              STATUS     TRY  TIME     DETAIL
init_run                          SUCCEEDED  1    0.1s     
feature_selection_orders          SUCCEEDED  1    24.4s    
confounder_selection_orders       SUCCEEDED  1    0.1s     
dml_fold_orders_f1                SUCCEEDED  1    0.3s     cache_hits=40, combinations=10
dml_fold_orders_f2                SUCCEEDED  1    0.2s     cache_hits=40, combinations=10
consolidate                       SUCCEEDED  1    0.1s     expected_rows=60, present_rows=60
completeness: 60/60 rows, 2/2 folds, failed tasks: []
```

### Structured JSON Logging Contract
All pipeline steps emit single-line JSON log events to `sys.stdout`:
```json
{"timestamp": "2026-10-27T08:01:15.598Z", "level": "INFO", "event": "CACHE_EVAL", "message": "10 combinations, 30 rows; stage-1 cache: 40 hits, 0 misses", "component": "dml", "task_id": "dml_fold_orders_f1", "run_id": "dra_20261027_run1", "target": "orders", "fold_id": "f1"}
```
Detailed step progress logs are simultaneously saved to:
`runs/{div}/{quarter}/{run_id}/logs/{task_id}.log`

---

## 8. Stage-1 Residual Caching Operations

DML Stage-1 fits nuisance models (predicting outcome $Y$ and treatment $T$ from confounders $W$). Because multiple driver combinations share the same control sets across folds, Stage-1 residual arrays are automatically cached and content-addressed:

### Cache Path
```
cache/dml_stage1/{div}/{target}/{train_window}/{unit_key}/
├── residuals.npz    # Compressed numpy residual arrays
└── metadata.json   # Producer run_id, variables, controls, timestamp
```

### Cache Configuration (`master_config.json`)
```json
{
  "cache": {
    "mode": "read_write"   // Options: "read_write", "read_only", "write_only", "bypass"
  }
}
```
- `read_write`: Use existing cached units; compute and write misses.
- `bypass`: Recompute all units from scratch (useful for benchmarking).
- `read_only`: Fail if a cache miss occurs (strict reproducibility).

---

## 9. Failure Recovery & Retries

### Error Classification & Exit Codes

| Exit Code | Classification | Meaning | Action |
| :---: | :--- | :--- | :--- |
| **0** | `SUCCEEDED` / `SKIPPED` | Normal task completion | Proceed to next task |
| **1** | `INFRA_TRANSIENT`, `TIMEOUT` | Transient network/disk failure | Retry with `--attempt N+1` |
| **2** | `CONFIG`, `DATA_SCHEMA`, `CODE_BUG` | Deterministic configuration or code error | Fix config/data before retrying |

### Re-running a Failed Task
If a task fails (e.g. `dml_fold_orders_f2`):

1. Check the error detail in `status/dml_fold_orders_f2.json`:
   ```powershell
   Get-Content .mock_blob/drivers-pipeline/runs/EL/2026Q3/<run_id>/status/dml_fold_orders_f2.json
   ```
2. Re-run only that specific task:
   ```powershell
   & $PYTHON run_pipeline.py --step dml --run-id <run_id> --target orders --fold-id f2 --attempt 2
   ```
3. After the task succeeds, seal the run by executing `consolidate`:
   ```powershell
   & $PYTHON run_pipeline.py --step consolidate --run-id <run_id>
   ```

---

## 10. Command Cheat Sheet

| Task | Command |
| :--- | :--- |
| **Run local end-to-end** | `python run_local.py [--run-id <id>]` |
| **Check live status** | `python -m pipelines.run_status --run-id <id> [--watch 5]` |
| **Init run** | `python run_pipeline.py --step init_run --run-id <id>` |
| **Run Feature Selection** | `python run_pipeline.py --step feature_selection --run-id <id> --target <target>` |
| **Run Confounder Selection** | `python run_pipeline.py --step confounder_selection --run-id <id> --target <target>` |
| **Run DML fold** | `python run_pipeline.py --step dml --run-id <id> --target <target> --fold-id <fold>` |
| **Consolidate & Publish** | `python run_pipeline.py --step consolidate --run-id <id>` |
| **Run validation** | `python run_pipeline.py --step validate --run-id <id>` |
| **Run full test suite** | `python -m pytest -v` |
| **Inspect prioritization JSON** | `Get-Content .mock_blob/drivers-pipeline/runs/EL/2026Q3/<id>/feature_selection/<target>/feature_prioritization.json` |
