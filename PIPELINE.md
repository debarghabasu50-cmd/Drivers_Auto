# Driver Analysis Pipeline

Estimates the causal effect (elasticity) of driver KPIs on a target KPI (e.g. `orders`) with Double Machine Learning (DML), across **combinations of drivers** and **expanding time folds**. Supports multi-division (`div`) and multi-quarter (`quarter`) enterprise execution with automated caching, failure isolation, and a robust consumer serving layer.

**Stack:** Apache Airflow (orchestrator) → Azure Machine Learning (one command job per step) → Azure Blob Storage (all artifacts).  
**Locally:** The same code runs with zero cloud dependencies using mock local storage.

> [!NOTE]
> Orchestration, fold-level parallelization, Stage-1 residual caching, failure isolation, consumer serving publishing, and logging contracts are fully implemented and verified with automated test suites. The statistical science functions (`select_kpis`, `select_confounders`, `fit_stage1_unit`, `fit_stage2`) currently use modular deterministic mocks ready for domain modeling replacement.

---

## 1. Flow at a Glance

```
                trigger (quarterly schedule | manual event)
                                      │
                                      ▼
┌───────────────────────────────────────────────────────────────────────────┐
│ 1. init_run                                                               │
│    Creates: frozen master config, content-addressed data snapshot,        │
│             run manifest (status=RUNNING), scoped discovery index         │
└─────────────────────────────────────┬─────────────────────────────────────┘
                                      │
                                      ▼
┌───────────────────────────────────────────────────────────────────────────┐
│ 2. feature_selection (per target KPI)                                     │
│    Emits: selected lagged feature matrix, summary JSON, Excel workbook    │
│    Toggle "off" → Reuses compatible prior run's artifacts via fingerprint │
└─────────────────────────────────────┬─────────────────────────────────────┘
                                      │
                                      ▼
┌───────────────────────────────────────────────────────────────────────────┐
│ 3. confounder_selection (per target KPI)                                  │
│    Emits: confounder sets per candidate treatment driver                  │
│    Toggle "off" → Reuses compatible prior run's artifacts via fingerprint │
└─────────────────────────────────────┬─────────────────────────────────────┘
                                      │
                 Airflow Dynamic Task Mapping (Target × Fold)
                 Parallel execution across up to 8 worker nodes
                 ┌────────────────────┼────────────────────┐
                 ▼                    ▼                    ▼
          ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
          │ 4. dml_fold  │     │ 4. dml_fold  │     │ 4. dml_fold  │
          │    fold = f1 │     │    fold = f2 │ ... │    fold = fN │
          └──────┬───────┘     └──────┬───────┘     └──────┬───────┘
                 │                    │                    │
                 │   - Enumerate deterministic combinations        │
                 │   - Stage-1 residual cache lookup (.npz + meta) │
                 │   - Stage-2 cross-fitted OLS elasticity         │
                 │   - Emits: dml_result.parquet + stage1_refs     │
                 └────────────────────┼────────────────────┘
                                      │
                                      ▼   trigger_rule = "all_done"
┌───────────────────────────────────────────────────────────────────────────┐
│ 5. consolidate                                                            │
│    - Unions all fold result tables and computes cross-fold aggregations   │
│    - Validates completeness (expected vs present rows)                    │
│    - Renders reviewer Excel sign-off workbook                             │
│    - Publishes to consumer serving layer:                                 │
│        published/{div}/{quarter}/runs/{run_id}/ (immutable)               │
│        published/{div}/{quarter}/latest/        (canonical pointer)       │
│    - Closes run manifest (SUCCEEDED | PARTIAL | FAILED)                   │
└───────────────────────────────────────────────────────────────────────────┘
```

Each step runs as an independent Azure ML command job (`python run_pipeline.py --step <step> ...`). When a step is switched "off" in configuration, Airflow executes it in-process on the worker, writing a `SKIPPED` status without provisioning cloud compute.

---

## 2. Multi-Division and Multi-Quarter Blob Layout

The storage layout provides complete isolation across divisions (e.g., `"EL"` for Electronics, `"HM"` for Home) and financial quarters (e.g., `"2026Q3"`), eliminates cross-talk in Stage-1 caches, guarantees immutable run workspaces, and provides an intuitive consumer serving layer for multiple re-runs.

### 2.1 Storage Layout Tree

```
{container}/
├── snapshots/                                              # Immutable content-addressed input tables
│   └── {snapshot_id}/
│       └── input.parquet                                   # Deduplicated raw data snapshot (write-once)
│
├── cache/                                                  # Shared analytical cache (cross-run, cross-fold)
│   └── dml_stage1/
│       └── {div}/                                          # Division-scoped cache partitioning
│           └── {target}/
│               └── {train_window}/
│                   └── {unit_key}/
│                       ├── residuals.npz                   # Compressed residual arrays (oof & test)
│                       └── metadata.json                   # Provenance & commit marker (written last)
│
├── runs/
│   ├── _index/                                             # Fast catalog lookup
│   │   ├── {run_id}.json                                   # Direct run pointer across system
│   │   └── {div}/
│   │       └── {quarter}/
│   │           └── {run_id}.json                           # Scoped run discovery index
│   └── {div}/
│       └── {quarter}/
│           └── {run_id}/                                   # Completely self-contained run workspace
│               ├── run_manifest.json                       # Single source of truth run contract
│               ├── master_config.json                      # Frozen configuration snapshot
│               ├── status/                                 # Individual task status files
│               │   ├── init_run.json
│               │   ├── feature_selection_{target}.json
│               │   ├── confounder_selection_{target}.json
│               │   ├── dml_fold_{target}_{fold_id}.json
│               │   └── consolidate.json
│               ├── logs/                                   # Durable human-readable log files
│               │   ├── init_run.log                        # Standard Python .log format (not JSON)
│               │   ├── dml_fold_{target}_{fold_id}.log
│               │   └── run.log
│               ├── feature_selection/
│               │   └── {target}/
│               │       ├── data_with_lagged_features.parquet # Enriched dataset with lagged features (used downstream)
│               │       ├── selected_features.parquet
│               │       └── fs_summary.json                 # Contains step_fp for reuse validation
│               ├── confounder_selection/
│               │   └── {target}/
│               │       └── confounder_set.json             # Contains step_fp for reuse validation
│               └── dml/
│                   └── {target}/
│                       └── fold={fold_id}/
│                           ├── dml_result.parquet          # Point estimates, SE, p-values, R2
│                           └── stage1_refs.parquet         # Pointers to Stage-1 unit keys used
│
├── published/                                              # Consumer Serving Layer
│   └── {div}/
│       └── {quarter}/
│           ├── latest/                                     # Always points to the latest successful run
│           │   ├── consolidated_result.parquet             # Canonical endpoint consumed by downstream BI
│           │   └── run_info.json                           # Points to active run_id, timestamp, row counts
│           └── runs/
│               └── {run_id}/                               # Immutable historical run records
│                   ├── consolidated_result.parquet
│                   └── run_info.json
│
└── reports/                                                # Reviewer & Stakeholder Workbooks
    └── {div}/
        └── {quarter}/
            ├── latest/
            │   └── dml_report.xlsx                         # Latest reviewer Excel workbook
            └── runs/
                └── {run_id}/
                    ├── fs_report.xlsx                      # Feature selection diagnostics
                    └── dml_report.xlsx                     # Consolidated DML sign-off workbook
```

### 2.2 Step I/O Contract Matrix

| Step | Consumes | Emits (Blob Paths) |
| :--- | :--- | :--- |
| **init_run** | `config/master_config.json`, input table (`data.input_uri`) | `snapshots/{snapshot_id}/input.parquet`<br>`runs/{div}/{quarter}/{run_id}/master_config.json`<br>`runs/{div}/{quarter}/{run_id}/run_manifest.json`<br>`runs/_index/{div}/{quarter}/{run_id}.json`<br>`runs/_index/{run_id}.json` |
| **feature_selection** | Snapshot input, frozen master config | `runs/{div}/{quarter}/{run_id}/feature_selection/{target}/data_with_lagged_features.parquet`<br>`runs/{div}/{quarter}/{run_id}/feature_selection/{target}/selected_features.parquet`<br>`runs/{div}/{quarter}/{run_id}/feature_selection/{target}/fs_summary.json`<br>`reports/{div}/{quarter}/runs/{run_id}/fs_report.xlsx` |
| **confounder_selection** | `data_with_lagged_features.parquet`, `fs_summary.json`, frozen config | `runs/{div}/{quarter}/{run_id}/confounder_selection/{target}/confounder_set.json` (runs once; common confounder set used for each treatment) |
| **dml** (per fold) | `data_with_lagged_features.parquet`, `confounder_set.json`, frozen DML config, Stage-1 cache | `runs/{div}/{quarter}/{run_id}/dml/{target}/fold={fold_id}/dml_result.parquet`<br>`runs/{div}/{quarter}/{run_id}/dml/{target}/fold={fold_id}/stage1_refs.parquet`<br>New cache units in `cache/dml_stage1/{div}/{target}/{train_window}/{unit_key}/` |
| **consolidate** | Run manifest, all task status JSONs, all fold `dml_result.parquet` files | `published/{div}/{quarter}/runs/{run_id}/consolidated_result.parquet`<br>`published/{div}/{quarter}/runs/{run_id}/run_info.json`<br>`published/{div}/{quarter}/latest/consolidated_result.parquet`<br>`published/{div}/{quarter}/latest/run_info.json`<br>`reports/{div}/{quarter}/runs/{run_id}/dml_report.xlsx`<br>`reports/{div}/{quarter}/latest/dml_report.xlsx`<br>Final updated `run_manifest.json` and index |

---

## 3. Consumer Serving Layer & Multiple Reruns

The consumer serving layer is designed for zero-downtime consumption by business dashboards (Tableau, PowerBI), feature stores, and downstream analytics.

### 3.1 Intuitive Rerun Lifecycle

When running multiple iterations or reruns for the same division and quarter (e.g., parameter adjustments or updated raw data):

1. **Immutable Run History:** Every rerun mints a unique `run_id` (e.g., `dra_20261013T0800_1a2b3c`). Its artifacts are written under `published/{div}/{quarter}/runs/{run_id}/` alongside a `run_info.json` descriptor. Past runs are never overwritten or deleted.
2. **Atomic `latest/` Pointer:** When the run completes with status `SUCCEEDED` (or acceptable `PARTIAL`), the `consolidate` barrier updates `published/{div}/{quarter}/latest/consolidated_result.parquet` and `published/{div}/{quarter}/latest/run_info.json`.
3. **Consumer Stability:** Downstream BI queries can query `published/{div}/{quarter}/latest/` without needing to discover dynamic `run_id` strings. If a run fails, the previous good `latest/` remains active and uncorrupted.
4. **Audit Metadata:** `run_info.json` contains full provenance:
   ```json
   {
     "run_id": "dra_20261013T0800_1a2b3c",
     "div": "EL",
     "quarter": "2026Q3",
     "published_at": "2026-10-13T08:05:22.124Z",
     "status": "SUCCEEDED",
     "present_rows": 60,
     "expected_rows": 60,
     "failed_rows": 0,
     "manifest_uri": "runs/EL/2026Q3/dra_20261013T0800_1a2b3c/run_manifest.json"
   }
   ```

---

## 4. Component Modularity and Clean Seams

Every pipeline component is a pure Python function adhering to the uniform signature:
```python
def run_xxx(ctx: StepContext) -> StepResult:
    ...
```

The runner wrapper (`run_pipeline.py`) handles all orchestration concerns:
- Binding process logging context
- Loading and caching the frozen configuration
- Live status reporting (`RUNNING` → `SUCCEEDED` / `FAILED` / `SKIPPED`)
- Persisting human-readable `.log` files to storage
- Translating exceptions into standardized error categories and process exit codes

```
   run_pipeline.py  (Orchestration Wrapper)
   ┌────────────────────────────────────────────────────────────────────────┐
   │ 1. Bind logging context (run_id, component, task_id, attempt)          │
   │ 2. Load frozen master_config.json & set active scope (div, quarter)    │
   │ 3. Write status: RUNNING                                               │
   │ 4. If step toggle == "off" → validate reuse & write status SKIPPED     │
   │    Else → execute component function: run_xxx(ctx)                     │
   │ 5. Write final status JSON (SUCCEEDED | PARTIAL | FAILED)              │
   │ 6. Write durable task log file: runs/{div}/{quarter}/{r}/logs/{t}.log  │
   │ 7. Emit TASK_SUMMARY log & exit with status code (0, 1, or 2)          │
   └────────────────────────────────────────────────────────────────────────┘
                    │
                    ▼
          Component Implementation (Science Seams)
          - Reads data via ctx.load_snapshot(), ctx.storage, ctx.config
          - Returns StepResult(outputs, counters, warnings, status)
          - Raises PipelineError(ErrorCategory, msg) on failures
```

To plug in production models, replace the functions in these seams:

| Function | File | Contract Responsibility |
| :--- | :--- | :--- |
| `select_kpis` | `components/feature_selection/feature_selection.py` | Selects KPI drivers, assigns tiers (P0/P1), determines lag structure |
| `select_confounders` | `components/confounder_selection/confounder_selection.py` | Selects confounder controls for each treatment driver |
| `fit_stage1_unit` | `components/dml/dml_analysis.py` | Cross-fits nuisance models ($Y \sim W$ or $T \sim W$) and computes residuals |
| `fit_stage2` | `components/dml/dml_analysis.py` | Regresses outcome residuals on treatment residuals to yield elasticity |

---

## 5. Double Machine Learning (DML) & Stage-1 Residual Caching

Stage-1 residual estimation represents >90% of total pipeline compute time. The caching engine provides massive speedups with zero data leakage.

### 5.1 Combinations and Deduplication

1. **Combination Enumeration:** For $N$ selected drivers and combination size $k$ (`drivers_per_combo`), the pipeline enumerates all combinations satisfying `must_include` constraints. If the number of combinations exceeds `max_combinations`, a deterministic selection retains the lowest `combination_id` hashes, ensuring exact reproducibility regardless of input order.
2. **Control Set Resolution:** For each combination $C$ and treatment driver $t \in C$, controls are defined as:
   $$\text{Controls}(t) = (C \setminus \{t\}) \cup \text{Confounders}(t)$$
3. **Unit Deduplication:** Across combinations, many driver sets share identical control sets. The pipeline deduplicates Stage-1 units before fitting. In typical workloads, 1,680 elasticity rows require only 1,050 Stage-1 nuisance model fits.

### 5.2 Content-Addressed Caching Protocol

- **Simplified Unit Key Fingerprint:** The 16-character hexadecimal key `unit_key` deterministically encodes only analytical identity (modelling hyperparameters and inner CV are excluded):
  ```python
  unit_key = fp({
      "target": target,
      "role": "outcome" | "treatment",
      "variable": variable,
      "controls": sorted(controls),
      "source_data_hash": col_hash,      # SHA-256 of training data columns
      "train_window": train_window,
      "test_window": test_window,
      "cache_version": cache_version     # bumped only when methodology code changes
  })
  ```
- **Two-Phase Commit Marker:**
  - `residuals.npz` is written first containing out-of-fold and test residuals.
  - `metadata.json` is written **last** containing verification details.
  - A cache entry is considered a valid **HIT** only when both files are verified. A worker crash during write results in a clean miss, which is safely refitted on retry.
- **Concurrency Safety:** Blob writes use `overwrite=False`. Multiple concurrent workers attempting to write the same shared unit will not corrupt each other; the first writer commits cleanly.
- **Cache Modes:**
  - `read_write` (default): Uses hits and caches new fits.
  - `read_only`: Uses hits but writes no new cache entries (useful for benchmarking).
  - `off`: Ignores the cache entirely.

---

## 6. Three-Level Failure Isolation Matrix

The pipeline provides robust failure isolation across all failure modes:

| Isolation Level | Trigger Condition | System Action | Pipeline Status | Airflow / Recovery Action |
| :--- | :--- | :--- | :--- | :--- |
| **Level 1: Unit / Combo** | ML model convergence failure or numerical error (`MODEL_CONVERGENCE`, `NUMERIC`) on one control set. | Caught inside worker process; recorded in `stage1_refs.parquet`. Combinations utilizing the failed unit receive `status="FAILED"`. | **Pipeline continues**. All unaffected combinations compute normally. | No retry needed. Fold completes with status `PARTIAL`. |
| **Level 2: Fold Level** | Azure ML node preemption, network disconnect, or transient VM fault (`INFRA_TRANSIENT`). | Task fails with exit code `1`. Airflow retries the task (`retries=2`). Downstream `consolidate` uses `trigger_rule="all_done"`. | **Pipeline continues**. Parallel folds proceed uninterrupted. | On retry, completed `.npz` units hit the cache instantly; only incomplete units are fitted. If permanently failed, `consolidate` executes and marks run `PARTIAL`. |
| **Level 3: Reporting** | Openpyxl workbook formatting or schema rendering failure. | Exception caught in `try...except`, logs `REPORT_FAILED` warning, sets `report_status="FAILED"`. | **Pipeline continues**. Data parquet publication is 100% unaffected. | No retry needed. Reporting is presentation-only and non-blocking. |

### Process Exit Codes

- `0`: Task succeeded (`SUCCEEDED`, `PARTIAL`, or `SKIPPED`).
- `1`: Transient failure (`INFRA_TRANSIENT`, `TIMEOUT`). Airflow automatically schedules a retry.
- `2`: Permanent failure (`CONFIG`, `DATA_SCHEMA`, `DATA_QUALITY`, `UPSTREAM_MISSING`, `CODE_BUG`). Airflow raises `AirflowFailException` and aborts retries.

---

## 7. Logging Contract

### 7.1 Durable Python `.log` Files

All task execution logs are persisted to storage as standard Python `.log` text files:
- Path: `runs/{div}/{quarter}/{run_id}/logs/{task_id}.log`
- Format:
  ```
  2026-10-13 08:30:15,124 [INFO] [dml] [dml_fold_orders_f1] 10 combinations, 30 rows; stage-1 cache: 28 hits, 2 misses | event=CACHE_EVAL counters={'units_total': 30, 'cache_hits': 28}
  2026-10-13 08:30:15,892 [INFO] [dml] [dml_fold_orders_f1] stage-1 progress 2/2 | event=STAGE1_PROGRESS counters={'fitted': 2, 'to_fit': 2}
  2026-10-13 08:30:16,045 [INFO] [dml] [dml_fold_orders_f1] dml_fold_orders_f1 -> SUCCEEDED | event=TASK_SUMMARY duration_ms=921.0 status=SUCCEEDED
  ```

### 7.2 Structured Stdout Ingestion

To support automated container log streaming, Azure Monitor, Log Analytics, and test assertions, `sys.stdout` simultaneously outputs structured JSON records:
```json
{
  "timestamp": "2026-10-13T08:30:15.124Z",
  "level": "INFO",
  "event": "CACHE_EVAL",
  "message": "10 combinations, 30 rows; stage-1 cache: 28 hits, 2 misses",
  "component": "dml",
  "task_id": "dml_fold_orders_f1",
  "run_id": "dra_20261013T0830_307678",
  "target": "orders",
  "fold_id": "f1",
  "attempt": 1,
  "image": "local",
  "counters": {"units_total": 30, "cache_hits": 28}
}
```

---

## 8. Master Configuration Specification

The configuration file `config/master_config.json` governs pipeline behavior:

```json
{
  "div": "EL",
  "quarter": "2026Q3",
  "pipeline": {
    "name": "driver_analysis",
    "div": "EL",
    "quarter": "2026Q3"
  },
  "data": {
    "input_uri": "input/EL/2026Q3/kpi_data.parquet",
    "date_col": "date",
    "freq": "M"
  },
  "targets": [
    {
      "target_id": "orders",
      "column": "orders_value"
    }
  ],
  "steps": {
    "feature_selection": "on",
    "confounder_selection": "on",
    "dml": "on",
    "consolidation": "on",
    "reports": "on"
  },
  "source_artifacts": {
    "feature_selection_run_id": null,
    "confounder_selection_run_id": null
  },
  "kpis": {
    "mode": "auto_plus_extra",
    "tiers": ["P0"],
    "explicit": [],
    "extra": [
      "price_index",
      "competitor_index",
      "gdp_growth",
      "lead_time",
      "marketing_spend"
    ],
    "exclude": []
  },
  "confounders": {
    "candidate_pool": [
      "macro_inflation",
      "interest_rate",
      "industry_index"
    ],
    "window": "common_train"
  },
  "combinations": {
    "drivers_per_combo": 3,
    "max_combinations": 100,
    "must_include": []
  },
  "folds": [
    {
      "fold_id": "f1",
      "train_start": "2021-01-01",
      "train_end": "2024-12-31",
      "test_start": "2025-01-01",
      "test_end": "2025-03-31"
    },
    {
      "fold_id": "f2",
      "train_start": "2021-01-01",
      "train_end": "2025-03-31",
      "test_start": "2025-04-01",
      "test_end": "2025-06-30"
    }
  ],
  "cache": {
    "mode": "read_write"
  },
  "execution": {
    "max_parallel_folds": 8,
    "unit_timeout_s": 300,
    "retries": 2,
    "n_jobs": 1
  }
}
```

---

## 9. Local Execution and Test Suite

### 9.1 Running Locally

Run the entire pipeline end-to-end using local mock storage (`.mock_blob/`):
```bash
python run_local.py
```

Run an individual step by hand:
```bash
python run_pipeline.py --step init_run --run-id dra_manual_001
python run_pipeline.py --step feature_selection --run-id dra_manual_001 --target orders
python run_pipeline.py --step confounder_selection --run-id dra_manual_001 --target orders
python run_pipeline.py --step dml --run-id dra_manual_001 --target orders --fold-id f1
python run_pipeline.py --step consolidate --run-id dra_manual_001
```

Inspect the real-time status of a run:
```bash
python -m pipelines.run_status --run-id dra_manual_001
```

### 9.2 Automated Verification Suite

Run all automated unit and end-to-end integration tests:
```bash
python -m pytest -v
```

All 38 test suites cover:
- Content-addressed hashing & reproducibility (`test_data.py`)
- Full pipeline traceability and Stage-1 cache hits (`test_e2e.py`)
- Python `.log` file validation and consumer serving layer rerun isolation (`test_features.py`)
- Cache healing upon file corruption (`test_reuse_and_ops.py`)
- Airflow failure routing and toggle bypass without cloud clusters (`test_reuse_and_ops.py`)
