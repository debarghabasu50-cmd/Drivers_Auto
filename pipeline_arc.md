10/9/26, 2:35 PM                                                        Gmail - simple pipeline arc prompt



                                                                                                   Debargha Basu <rahulbasu500@gmail.com>


  simple pipeline arc prompt

  Debargha Basu <debargha.basu@in.abb.com>                                                                                 Fri, Oct 9, 2026 at 2:12 PM
  To: Debargha Basu <rahulbasu500@gmail.com>


    Driver Analysis Causal Pipeline — Agent Context

    0. Role and working rules

    You are a senior ML platform engineer. You help design and implement an end-to-end, no-touch causal driver-analysis
    pipeline: Airﬂo                                                                  w → Azure ML → Azure Blob.
    Priorities, in order:

         1. Reproducibility.
         2. Traceability.
         3. Simplicity.
         4. Performance.
    Rules:
            Keep it simple. Do not add services, components, abstractions or files beyond this document unless you
            justify why they're needed. Prefer a function over a class, and a module over a package.
            Existing science code (feature selection, DML, confounder method) is reused as-is under `src/methods/`.
            Never invent or alter the methodology.
            Anything marked OPEN (Section 18) must be asked about, not assumed. State every assumption
            explicitly.
            If a request conflicts with an invariant (Section 19), flag the conflict. Do not silently pick one side.
            Out of scope: the Validation component (planned; it will later consume `published/`), creating upstream
            data, and CI internals beyond build/deploy.

    1. Business context

            Goal: estimate the causal effect (elasticity) of driver KPIs on target KPIs (e.g. `orders`) using Double
            Machine Learning (DML).
            Drivers: N KPIs selected and lagged by Feature Selection. They are grouped into tiers P0, P1 and P2.
            Combinations: every set of k drivers out of N (e.g. N=10, k=7 gives 120 combinations), optionally
            capped.
            Folds: time-based validation periods (e.g. 4 quarterly folds), each with an expanding train window and a
            test window.
            Cadence: quarterly refresh, plus manual reruns.
            Outputs: Parquet/JSON for machines. Excel reports for reviewer sign-off; these are rendered
            downstream and are never a data contract.

    2. Platform roles

     Layer              Role                                                                                 Never does
     Airﬂo                    wOrchestration, fan-out, retries, the only writer of `run_manifest.json`       Heavy compute
     Azure ML           Runs one command job per task. One Docker image (tag = git SHA),                     Orchestration
                        one entrypoint `python -m src.main`
     Azure Blob         Reads input; stores all artifacts, cache, logs and the manifest                      —
     Azure              Test → build image → register AML environment → deploy DAG                           Triggering business runs (optional
     Pipelines (CI)                                                                                          manual only)
    Communication rules:
            Tasks communicate only through Blob files under the run, located via the manifest.
            Runs communicate only through `cache/` and the run index.
            XCom carries only `run_id` and `work_items`.


https://mail.google.com/mail/u/1/?ui=2&ik=7d280f7890&view=lg&permmsgid=msg-f:1878560952167301212                                                     1/12

---

10/9/26, 2:35 PM                                                               Gmail - simple pipeline arc prompt
     3. Identiﬁer             s and hashing

     Two hash functions exist, and only two:
             `fp(obj)` = sha256 of canonical JSON (sorted keys), truncated to 16 hex characters.
             `data_fp(df)` = `fp` of the sum of `pandas.util.hash_pandas_object(df, index=True)`.

     ID                     Deﬁnition                                                                Use
     `run_id`               `dra_{YYYYMMDDTHHMM}_{6 hex}`, created by                                Lineage, paths, logs, AML tags
                            Airﬂo                    w `init_run`
     `snapshot_id`          `snap_` + `data_fp(validated input)`                                     Data provenance; content-
                                                                                                     addresses the frozen input
     `conﬁg_v      ersion`  `cfg_` + `fp(resolved merged conﬁg)`                                     Reproducibility
     `image`                Docker tag = git SHA                                                     Code version (recorded, not part of
                                                                                                     cache keys)
     `cache_version`        Integer in `dml.json`, bumped manually when method                       Part of every cache key
                            code changes results
     `step_fp`              Per-step ﬁng                     erprint chain (Section 7)               Step-level reuse (`auto` switch)
     `combination_id`       `cmb_` + `fp(sorted drivers)`                                            DML output key
     `unit_key`             Stage-1 cache key (Section 10)                                           Residual cache
     `task_id`              Airﬂo                    w task id + map key, e.g. `dml_fold[orders      f2]`                                           Logs,
                                                                                                                                                    status ﬁles

     4. Pipeline at a glance

     ```
     TRIGGER: quarterly schedule | manual {overrides, replay_run_id, switches}
     │
     T1 init_run (Airﬂo                                                                                 w + small AML job "prepare")
     │ create run_id + run_manifest.json (RUNNING) + run index entry
     │ merge/validate conﬁg → c      onﬁg_v      ersion; validate input → snapshot_id
     │ resolve step switches and step ﬁng                     erprints → step_plan; work_items = targets × folds
     ▼
     T2 feature_selection (AML, one job, loops targets, per-KPI cache)
     │ → selected_features.parquet + fs_summary.json per target; FS Excel (non-blocking)
     ▼
     T3 confounder_selection (AML, one job, ONCE per target, not per fold; per-treatment cache)
     │ → confounder_set.json per target (all treatments)
     ▼ Airﬂo                       w .expand(work_items), max 8 concurrent, 1 AML job = 1 node
     T4 dml_fold[target|fold] × N
     │ plan units → stage-1 (cache + process pool) → stage-2 → dml_result.parquet + stage1_refs.parquet
     ▼ trigger_rule = all_done
     T5 consolidate (AML small)
     │ union folds, completeness, cross-fold aggregates → consolidated_result.parquet; DML Excel (non-blocking)
     ▼ trigger_rule = all_done
     T6 ﬁnaliz                     e_run (Airﬂo                                                        w only)
     ﬁnal s      tatus SUCCEEDED | PARTIAL | FAILED, end_time, lineage summary → manifest + run index
     ```
     Every Airﬂo                            w task wrapper does the following:
          1. Read `step_plan` from the manifest. If the step is SKIPPED or CACHED, patch the manifest and return
             successfully without submitting a job, so downstream tasks are not skipped.
          2. Otherwise, submit the AML job with args `--step --run-id [--target --fold-id]` and tags `{run_id, step,
             task_id}`, then wait for it to finish (deferrable).
          3. Read `runs/{run_id}/status/{task_id}.json`. If it's missing (hard crash), record FAILED with category
             `INFRA_TRANSIENT` or `UNKNOWN`.
          4. Patch the manifest (Section 8).
          5. Raise according to the retry policy (Section 12).

     5. Task details



https://mail.google.com/mail/u/1/?ui=2&ik=7d280f7890&view=lg&permmsgid=msg-f:1878560952167301212                                                                   2/12

---

10/9/26, 2:35 PM                                                            Gmail - simple pipeline arc prompt
     Task                        Reads                                   Does                   Writes                                 On failure
                                                                         Merge conﬁgs
                                                                         (env < component
                                                                         < main <
                                                                         overrides),
                                                                         validate
                                 Repo conﬁgs (in the imag      e):       (pydantic) →
                                 `main_conﬁg.json`,                      `conﬁg_v      ersion`.
                                 `feature_selection.json`,               Validate input         `runs/R/run_manifest.json`,
                                 `confounder_selection.json`,            (columns, dtypes,      `runs/R/master_conﬁg.json`,            Run stops
     T1 init_run                 `dml.json`, `env/{env}.yaml`;           date continuity,       `snapshots/{snapshot_id}/              (category
                                 Airﬂo                    w params;      nulls, target/KPI      input.parquet`,                        `CONFIG` /
                                 `input/{yyyymm}/`; run index;           presence) →            `runs/_index/R.json`,                  `DATA_SCHEMA`)
                                 if replaying:                           `snapshot_id`;         `status/init_run.json`
                                 `runs/{replay_id}/master_               write the
                                 conﬁg.json` + tha      t snapshot       snapshot only if it
                                                                         doesn't already
                                                                         exist. Compute
                                                                         step ﬁng                     erprints,
                                                                         resolve switches,
                                                                         build work_items
                                                                         Per target, per
                                                                         KPI: cache lookup,
                                                                         run the existing                                              Per target: a
                                                                         FS logic on misses     `feature_selection/R/{target}/         failed target is
                                 master_conﬁg      , snapshot (from      (joblib parallel).     selected_features.parquet`,            marked in
     T2 feature_selection        manifest),                              Build the lagged       `fs_summary.json`;                     fs_summary and
                                 `cache/feature_selection/`              matrix, assign         `reports/feature_selection/R/          its downstream
                                                                         tiers, resolve KPIs    feature_selection_report.xlsx`         work exits
                                                                         (`auto` = P0,                                                 SKIPPED
                                                                         `explicit`, or
                                                                         `auto_plus_extra`,
                                                                         minus `exclude`)
                                                                         Once per target,                                              Per treatment:
                                                                         on the CS window                                              the treatment is
                                 master_conﬁg      , FS artifacts        (Section 6). For                                              marked FAILED,
     T3                          (from manifest),                        each treatment:        `confounder_selection/R/{              and
     confounder_selection        `cache/confounder_selection/`           cache lookup,          target}/confounder_set.json`           combinations
                                                                         then the existing                                             containing it are
                                                                         method on misses                                              skipped in DML
                                                                         (joblib parallel)
                                                                                                                                       Isolated to the
                                 master_conﬁg      , FS and CS                                  `dml/R/{target}/fold={fold_id}         fold; unit failures
     T4 dml_fold                 artifacts (from manifest),              Section 6              /dml_result.parquet`,                  are isolated to
                                 `cache/dml_stage1/`                                            `stage1_refs.parquet`                  the aﬀected
                                                                                                                                       combinations
     T5 consolidate              master_conﬁg      , manifest (DML       Union the results.     `published/R/consolidated_             Retryable on its
                                 artifact paths), all available          Completeness =         result.parquet`,                       own; its inputs
                                 `dml_result.parquet`                    expected rows (Σ       `reports/dml/R/dml_report.             are untouched
                                                                         targets × folds ×      xlsx`
                                                                         combinations × k)
                                                                         vs present rows.
                                                                         Per (target,
                                                                         combination,
                                                                         treatment) across
                                                                         folds: mean, std,
                                                                         CV of elasticity,
                                                                         n_folds. Render
                                                                         the Excel
                                                                         (Combination /
                                                                         Driver / Metric /

https://mail.google.com/mail/u/1/?ui=2&ik=7d280f7890&view=lg&permmsgid=msg-f:1878560952167301212                                                            3/12

---

10/9/26, 2:35 PM                                                          Gmail - simple pipeline arc prompt
                                                                       Fold1..N / average
                                                                       of CV)
                                                                       Final status:
                                                                       SUCCEEDED if all
                                                                       steps and rows
                                                                       are complete,
     T6 ﬁnaliz                     e_runManifest                       PARTIAL if any          Manifest, run index entry             —
                                                                       usable output
                                                                       exists, otherwise
                                                                       FAILED. Aggregate
                                                                       `reused_from`
     Report rendering runs inside T2/T5, is wrapped so that a failure becomes a warning, and is controlled by its own switch.

     6. Confounder selection and DML internals

     CS window (OPEN #3): by default this is `common_train` = data start up to the earliest fold's `train_end`. It is contained
     in every fold's training data, so a single CS run is leakage-free for all folds. The alternative is `full`, which includes test
     periods and carries a leakage risk.
     Inside `dml_fold[target|fold]`, on one node:
         1. Load: the fold window from config. Slice the feature matrix into TRAIN (≤ train_end) and TEST (the test
            window). Load the resolved KPIs and `confounder_set.json`. If the target failed upstream → SKIPPED.
         2. Plan units (in memory): enumerate combinations C(N, k), applying `max_combinations` and
            `must_include`. For each combination and each treatment t in it, the controls are `W = (combination − t)
            ∪ confounders(t)`. That yields:

                    outcome unit: Y ~ W
                    treatment unit: t ~ W

     Deduplicate by `unit_key`; outcome units are shared by every combination with the same W. Illustrative counts: 10 KPIs,
     k=7 → 120 combinations → about 1,511 unique units per fold.
         3. Stage 1:
                    Make one `list_blobs` call on `cache/dml_stage1/{target}/{train_window}/` to get the set of existing
                    keys. These are the hits; everything else is a miss.
                    Misses run in a `joblib` (loky) process pool: n_jobs = cores, 1 thread per worker, seed derived from
                    `unit_key`, per-unit timeout.
                    Each unit cross-fits on TRAIN (inner CV) to produce out-of-fold residuals, plus TEST-window
                    residuals from the models trained on TRAIN, and writes one cache entry.
                    A unit error is recorded with its category and the combinations it affects. It is never raised.
         4. Stage 2 (cheap, always recomputed): for each (combination, treatment), regress the outcome residuals
            on the treatment residuals. This gives elasticity, std_error, p_value, R² and test-window metrics. Rows
            that can't be computed get `status=FAILED` and a reason.
         5. Write: `dml_result.parquet`, `stage1_refs.parquet` and `status/dml_fold[...].json`, with counters:
            combinations, units, hits, fitted, failed, combos_failed, reused_from.

     7. Step switches (on / oﬀ / aut                                                                  o)

     Conﬁg: `s      teps: {feature_selection, confounder_selection, dml, consolidation, reports}`. Each value is `"on" | "oﬀ" |
     "auto"`, plus an optional `source_run_id`.
     Step ﬁng                     erprint chain:
            `fs_fp = fp(snapshot_id, targets, kpis spec, feature_selection cfg, cache_version)`
            `cs_fp = fp(fs_fp, confounder_selection cfg)`
            `dml_fp[target|fold] = fp(cs_fp, fold def, combinations cfg, dml cfg)`
            `cons_fp = fp(all dml_fp, consolidation cfg)`

     Value   Behaviour
     `on`    Run the step. Internal unit caches still apply.
     `auto`  Search the run index for the latest run where this step SUCCEEDED with the same `step_fp`. If found → CACHED,
             and the manifest artifacts point to the source run (no copy). If not found → run.



https://mail.google.com/mail/u/1/?ui=2&ik=7d280f7890&view=lg&permmsgid=msg-f:1878560952167301212                                                         4/12

---

10/9/26, 2:35 PM                                                             Gmail - simple pipeline arc prompt
     `oﬀ`     Don't run. If a downstream step that is on/auto needs its outputs, take them from `source_run_id`, or from the
              latest compatible run. If none exists → fail in init_run (`CONFIG`). If nothing downstream needs it → SKIPPED.
     Rules:

             The fingerprint chain guarantees that a changed upstream invalidates downstream `auto` reuse.
             All switch decisions are made once in init_run and stored in `manifest.step_plan`.
             `reports` never blocks anything.
             Downstream steps always find upstream artifacts through `manifest.steps..artifacts`, never by building
             paths. That is what makes reuse transparent.

     8. Run manifest and run index

     `runs/{run_id}/run_manifest.json` is the single source of truth. It is created by the Airﬂo                                                w `init_run` task before Feature
     Selection starts, and each later task patches its own section when it completes.
             Writer: only Airflow wrappers, through `src/core/manifest.py`.
             Concurrency: read → patch own section → conditional upload with ETag `If-Match`, retried up to 10
             times with jitter. AML jobs never write the manifest; they write `status/{task_id}.json`.
             Any `run_id` lookup gives the exact snapshot, config, image, step statuses and artifact locations, without
             searching logs.

     ```json
     {
     "schema_version": "1.0",
     "run_id": "dra_20261008T0600_a1b2c3",
     "status": "RUNNING",
     "triggered_by": "schedule | manual:<user> | replay:<run_id>",
     "start_time": "2026-10-08T06:00:05Z", "end_time": null,
     "airﬂo                        w": {"dag_id": "driver_analysis", "dag_run_id": "scheduled__2026-10-08T06:00:00"},
     "image": "driveranalysis:3f9c2e1",
     "snapshot": {"snapshot_id": "snap_51c0e8aa90d1f2b3", "source_uri": "input/202609/",
     "uri": "snapshots/snap_51c0e8aa90d1f2b3/input.parquet", "rows": 60, "validation": "PASS"},
     "conﬁg": {"c      onﬁg_v      ersion": "cfg_9f1e44c0a2b7d311", "uri": "runs/dra_.../master_conﬁg.                  json"},
     "step_plan": {"feature_selection": {"mode": "auto", "decision": "RUN", "step_fp": "a1.."},
     "confounder_selection": {"mode": "auto", "decision": "CACHED", "step_fp": "b2..", "source_run_id":
     "dra_20260708T0600_9d8e7f"},
     "dml": {"mode": "on", "decision": "RUN"}, "consolidation": {"mode": "on", "decision": "RUN"},
     "reports": {"mode": "on"}},
     "plan": {"targets": ["orders"], "folds": ["f1","f2","f3","f4"],
     "work_items": ["orders|f1","orders|f2","orders|f3","orders|f4"],
     "resolved_kpis": {"orders": ["k1","k2","k3","k4","k5","k6","k7","k8","k9","k10"]}},
     "steps": {
     "init_run": {"status": "SUCCEEDED", "aml_job_id": "...", "attempt": 1, "started_at": "...", "ended_at": "..."},
     "feature_selection": {"status": "SUCCEEDED", "source_run_id": "dra_20261008T0600_a1b2c3",
     "artifacts": {"orders": {"selected_features": "feature_selection/dra_.../orders/selected_features.parquet",
     "fs_summary": "feature_selection/dra_.../orders/fs_summary.json"}},
     "reports": ["reports/feature_selection/dra_.../feature_selection_report.xlsx"],
     "counters": {"kpis": 10, "cache_hits": 10}, "error": null},
     "confounder_selection": {"status": "CACHED", "source_run_id": "dra_20260708T0600_9d8e7f",
     "artifacts": {"orders": "confounder_selection/dra_20260708T0600_9d8e7f/orders/confounder_set.json"}},
     "dml": {"status": "PARTIAL", "tasks": {
     "orders|f2": {"status": "PARTIAL", "aml_job_id": "...", "attempt": 1,
     "counters": {"combinations": 120, "units": 1511, "hits": 0, "ﬁtt                                                                                                                                                                                          ed": 1510, "failed": 1, "combos_failed": 4},
     "reused_from": {}, "artifacts": {"result": "dml/dra_.../orders/fold=f2/dml_result.parquet",
     "stage1_refs": "dml/dra_.../orders/fold=f2/stage1_refs.parquet"},
     "error": {"category": "MODEL_CONVERGENCE", "message": "1 unit failed"}}}},
     "consolidation": {"status": "SUCCEEDED", "artifacts": {"result": "published/dra_.../consolidated_result.parquet"},
     "reports": ["reports/dml/dra_.../dml_report.xlsx"]}
     },
     "completeness": {"expected_rows": 3360, "present_rows": 3332},
     "lineage": {"reused_from": {"dra_20260708T0600_9d8e7f": {"confounder_selection": 1, "dml_stage1": 0}}}
     }
     ```

https://mail.google.com/mail/u/1/?ui=2&ik=7d280f7890&view=lg&permmsgid=msg-f:1878560952167301212                                                                5/12

---

10/9/26, 2:35 PM                                                              Gmail - simple pipeline arc prompt
     The run index is one small ﬁle per run, `runs/_inde                                                                          x/{run_id}.json`, written at init and at ﬁnaliz             e: `{run_id, status,
     start_time, end_time, snapshot_id, conﬁg_v      ersion, image, step_fp{...}, step_status{...}}`. Discovery = list `runs/_index/`.
     Runs are quarterly and few, so listing is cheap.

     9. Blob layout (single convention)

     ```
     {container}/{div}/
     ├── input/{yyyymm}/... upstream, read-only
     ├── snapshots/{snapshot_id}/input.parquet validated frozen input, content-addressed (deduplicated)
     ├── runs/
     │ ├── _index/{run_id}.json run index (discovery)
     │ └── {run_id}/
     │ ├── run_manifest.json single source of truth
     │ ├── master_conﬁg.json r      esolved, frozen conﬁg
     │ ├── status/{task_id}.json AML job → Airﬂo                                                                                                                                           w handoﬀ
     │ └── logs/{task_id}.jsonl durable log copy
     ├── feature_selection/{run_id}/{target}/selected_features.parquet, fs_summary.json
     ├── confounder_selection/{run_id}/{target}/confounder_set.json
     ├── dml/{run_id}/{target}/fold={fold_id}/dml_result.parquet, stage1_refs.parquet
     ├── published/{run_id}/consolidated_result.parquet
     ├── reports/feature_selection/{run_id}/feature_selection_report.xlsx
     ├── reports/dml/{run_id}/dml_report.xlsx
     └── cache/
     ├── feature_selection/{target}/{key}.parquet
     ├── confounder_selection/{target}/{key}.json
     └── dml_stage1/{target}/{train_window}/{key}.parquet train_window = {train_start}_{train_end}
     ```
     Changes from the initial tree:
             `run_metadata.json` → `run_manifest.json`.
             `dml_result_all.parquet` dropped; `published/` is the single union.
             Confounder sets are per target, not per fold.
             `.npz` → Parquet.
             The stage-1 cache moved out of `dml/{run_id}/` into `cache/`.
             Added `snapshots/`, `status/`, `logs/` and `_index/`.
     Why a separate content-addressed cache:
             A run-scoped cache needs a search across old runs, duplicates residuals, and breaks if old runs are
             deleted.
             Key-addressed lookup is O(1) with no search, and cross-run reuse is automatic.
             One `list_blobs` call per prefix replaces thousands of individual existence checks.
             One Blob lifecycle rule on `cache/` handles retention.
             Lineage is kept in two places: `producer_run_id` in each entry's metadata, and `stage1_refs.parquet` in
             the consuming run's folder, which lists the keys used without copying any data.
             The `train_window` prefix keeps listings small and makes old windows easy to prune.

     10. Caching

     `get_or_compute(kind, key_parts, fn)` works as follows:
          1. `key = fp(key_parts + cache_version)`.
          2. If the key exists → HIT: return the data plus `producer_run_id`.
          3. Otherwise → MISS: compute, then write in a single upload with `overwrite=False`. This is atomic in Blob,
             and a conflict means identical content already exists, so it is ignored.
          4. The entry metadata stores `producer_run_id`, `key_parts`, `created_at`.

     `cache.mode`: `oﬀ | r                                                                                 ead_only | read_write`.
     Kind                         Key parts                                                                             Granularity
     feature_selection            target, `data_fp`(target series), `data_fp`(KPI series), fs cfg                       target × KPI (OPEN #1: if FS is
                                                                                                                        joint, key on the whole input)



https://mail.google.com/mail/u/1/?ui=2&ik=7d280f7890&view=lg&permmsgid=msg-f:1878560952167301212                                                                 6/12

---

10/9/26, 2:35 PM                                                          Gmail - simple pipeline arc prompt
     confounder_selection       target, treatment, `data_fp`(treatment + pool on CS window), CS                   target × treatment
                                window, cs cfg
     dml_stage1                 target, role, variable, sorted controls, `data_fp`(Y or variable +                target × fold window × unit
                                controls on TRAIN), train window, inner CV, learner spec, seed
     Confounders are part of the controls, so a CS change automatically invalidates the stage-1 units that depend on it.
     Lagged column values are hashed, so a lag change also causes a miss. Stage 2 is never cached.
     Eﬀect of each change:
     Change                                          Result
     Same conﬁg r      erun                          100% hits; only stage 2 runs
     KPI added                                       Old units hit; only units involving the new KPI are ﬁtt              ed (illustrative: 11 KPIs →
                                                     330 combinations, 4,053 units, 1,511 reused)
     KPI removed                                     All remaining units hit
     New quarter appended                            Old fold windows hit; only the new fold computes
     One KPI's history revised                       Only units using that column miss
     Learner params changed or                       Full recompute (correct)
     `cache_version` bumped
     Determinism (OPEN #4): seeds are derived from the key, library versions are pinned in the image, and learners use 1
     thread. Before enabling cache reads in prod, run a determinism test: ﬁt the same unit twice and r                equire equal residual
     hashes.

     11. Parallelism

     Level                     Mechanism
     target × fold             Airﬂo                    w `.expand(work_items)`, `max_active_tis_per_dagrun=8`; one AML job per node on `cpu-
                               large` (max_nodes ≥ 8); deferrable wait
     Units inside a fold       `joblib` loky process pool over cache misses only; `OMP_NUM_THREADS=1`
     FS KPIs and CS            `joblib` within their single job
     treatments
     Blob I/O                  SDK `max_concurrency` and one listing per preﬁx. No as      yncio: the work is CPU-bound, and the I/O
                               volume is small

     12. Failure handling

     Statuses: `STARTED, SUCCEEDED, PARTIAL, FAILED, SKIPPED, CACHED`.
     Error categories: `DATA_SCHEMA, DATA_QUALITY, CONFIG, UPSTREAM_MISSING, MODEL_CONVERGENCE, NUMERIC,
     TIMEOUT, INFRA_TRANSIENT, CODE_BUG, UNKNOWN` (OPEN #6).
     Level        Failure                    Eﬀect                                                 Run continues
     Unit         One stage-1 ﬁt             Its combinations are FAILED                           Yes
     Treatment    CS for one treatment       Combinations containing it are skipped                Yes
     Fold         Whole job                  Retried; the retry ﬁts only r              emaining missesYes
     Target       FS for a target            Its folds are SKIPPED                                 Yes
     Report       Excel render               Warning only                                          Yes
     init_run     Conﬁg or da      ta invalidRun stops                                             No

            Job exit codes: 0 for SUCCEEDED, PARTIAL, SKIPPED or CACHED; 1 for FAILED.
            The Airflow wrapper raises `AirflowException` (retry) for `INFRA_TRANSIENT` and `TIMEOUT`, and
            `AirflowFailException` (no retry) for everything else.
            Repair = a new run with the same config. The cache makes it fit only what's missing.

     13. Logging and observability

     There is one shared module, `src/core/logger.py`, used by all components and by the Airﬂo                                                w helpers
     (component=`orchestrator`).
            Context is bound once per job: `run_id, component, task_id, target, fold_id, attempt, aml_job_id`.
            Helper: `log_event(state, fields)` for STARTED, SUCCEEDED, FAILED, SKIPPED or CACHED.
            Event fields: `ts, level, run_id, component, task_id, target, fold_id, event, error_category, duration_ms,
            counters, msg`.
            Sinks:


https://mail.google.com/mail/u/1/?ui=2&ik=7d280f7890&view=lg&permmsgid=msg-f:1878560952167301212                                                        7/12

---

10/9/26, 2:35 PM                                                            Gmail - simple pipeline arc prompt
                 1. JSON lines to stdout, captured by AML job logs and Airflow task logs.
                 2. Azure Monitor OpenTelemetry exporter to Application Insights / Log Analytics, if
                    `APPLICATIONINSIGHTS_CONNECTION_STRING` is set. This is the central view.
                 3. A durable copy uploaded in a `finally` block to `runs/{run_id}/logs/{task_id}.jsonl`.
             Levels: step transitions and summary counters at INFO; progress every N units at INFO; per-unit events
             at DEBUG.
             Views:
                    Pipeline level: filter by run_id, order by time.
                    Component level: filter by component.
                    The AML job id and AML tags link to the job UI.
             Optional: log fold counters as AML job metrics (MLflow, native to AML).

     ```kusto
     AppTraces | where tostring(Properties.run_id) == "dra_20261008T0600_a1b2c3"
     | project TimeGenerated, component=tostring(Properties.component), task=tostring(Properties.task_id),
     event=tostring(Properties.event), cat=tostring(Properties.error_category), Message
     | order by TimeGenerated asc
     AppTraces | where tostring(Properties.event) == "FAILED"
     | summarize n=count() by tostring(Properties.component), tostring(Properties.error_category)
     ```

     14. Repo structure and code contracts

     ```
     driver-analysis/
     ├── airﬂo                                  w/dags/driver_analysis_dag.py DAG + submit_aml_step() + task wrapper
     ├── aml/ environment.yml, compute.yml
     ├── conﬁg      / main_conﬁg.json, f      eature_selection.json, confounder_selection.json, dml.json, env/{dev,prod}.yaml
     ├── src/
     │ ├── main.py single CLI entrypoint
     │ ├── core/ conﬁg.p      y contracts.py storage.py tracking.py cache.py manifest.py logger.py
     │ ├── steps/ prepare.py feature_selection.py confounder_selection.py dml.py consolidate.py
     │ └── methods/ feature_selection_dml.py confounder_methods.py dml_analysis.py reporting.py
     ├── tests/ ﬁx                      tures/ unit/ integration/ integration: local run on ﬁx                                                     tures; rerun = 100% cache hit
     ├── docs/ pipeline_ﬂo                                                                                                 w.md methodology.md
     ├── .archive/ legacy scripts not used by the pipeline
     ├── Dockerﬁle r                equirements.txt azure-pipelines.yml Makeﬁle .amlignor      e .dockerignore .gitignore .env.example
     ```
     Module responsibilities:
             `core/config.py`: merge, validate, `config_version`.
             `contracts.py`: pydantic models for all JSON contracts; schemas are exported with
             `model_json_schema()`.
             `storage.py`: the only module that talks to Blob; handles paths for its own outputs.
             `tracking.py`: `fp`, `data_fp`, run_id, status-file context manager.
             `cache.py`: `get_or_compute`.
             `manifest.py`: ETag patching and the run index (imported by the DAG; the Airflow image installs the repo
             package).

     `src/main.py` ﬂo                  w:
         1. Parse `--step --run-id [--target --fold-id] [--overrides] [--replay-run-id]`.
         2. Bind log context.
         3. Build ctx: env yaml (from `ENV`), then config (the `prepare` step builds it; every other step reads
             `runs/R/master_config.json`, never the repo), storage client, image tag.
         4. Open the status record.
         5. Dispatch to `steps/<step>.run(ctx, status)`.
         6. Always write `status/{task_id}.json` and upload logs, even on a crash.
         7. Exit with the code from Section 12.
     Every step follows the same contract:
         1. Read upstream artifacts via manifest pointers, and the cache.


https://mail.google.com/mail/u/1/?ui=2&ik=7d280f7890&view=lg&permmsgid=msg-f:1878560952167301212                                                             8/12

---

10/9/26, 2:35 PM                                                        Gmail - simple pipeline arc prompt
         2. Compute using `methods/` (pure functions: DataFrames in, DataFrames out; no I/O, config or logging
            context), wrapped in `get_or_compute` where cacheable.
         3. Write its own outputs.
         4. Fill in the status record: outputs, counters, cache stats, `reused_from`, warnings.
    Current repo → target mapping:
            `components//` science code → `src/methods/`; their I/O → `src/steps/`.
            `components/residual_caching` → `core/cache.py`.
            `shared/logging` and `shared/path_builder` → `core/logger.py` and `core/storage.py`.
            `shared/schema/.json` → `core/contracts.py`.
            `config/config_fetcher.py` → `core/config.py`.
            `run_pipeline.py` (both copies), `pipelines/main_pipeline.py`, `src/comp_pipeline.py` → replaced by
            `src/main.py`; the DAG is the pipeline.
            Unrelated scripts (`s&p_data_pipeline.py`, `scenario_planning.py`, `integration_`,
            `correlation_pipeline.py`, `misc_data_driver.py`) → `.archive/` or a separate repo.
            `.amltmp` → delete and add to `.gitignore`.
            `validation/evaluate.py` → parked; it becomes `steps/validate.py` later.

    15. Conﬁgur      ation

    ```json
    {
    "pipeline": {"name": "driver_analysis", "div": "EL"},
    "data": {"input_uri": "input/202609/kpi_data.parquet", "date_col": "date", "freq": "M"},
    "targets": [{"target_id": "orders", "column": "orders_value"}],
    "kpis": {"mode": "auto_plus_extra", "tiers": ["P0"], "explicit": [], "extra": ["k11"], "exclude": []},
    "confounders": {"candidate_pool": ["m1","m2","m3","m4"], "window": "common_train"},
    "combinations": {"drivers_per_combo": 7, "max_combinations": null, "must_include": []},
    "folds": [
    {"fold_id": "f1", "train_start": "2021-01-01", "train_end": "2024-12-31", "test_start": "2025-01-01", "test_end": "2025-
    03-31"},
    {"fold_id": "f2", "train_start": "2021-01-01", "train_end": "2025-03-31", "test_start": "2025-04-01", "test_end": "2025-
    06-30"}
    ],
    "steps": {"feature_selection": "auto", "confounder_selection": "auto", "dml": "on",
    "consolidation": "on", "reports": "on", "source_run_id": null},
    "cache": {"mode": "read_write"},
    "execution": {"max_parallel_folds": 8, "unit_timeout_s": 600, "retries": 2}
    }
    ```
            Component configs hold method parameters: `feature_selection.json`; `confounder_selection.json`
            (method = existing code, params); `dml.json` (`inner_cv_splits`, outcome/treatment learner spec, `seed`,
            `cache_version`).
            `env/.yaml`: storage account, container, AML workspace, compute names, App Insights connection
            secret reference.
            Precedence: env < component < main < run overrides. The merged result is frozen as
            `master_config.json`.

    16. Data contracts

     Artifact                            Content
     `selected_features.parquet`         `date, target, kpi, lag, column (kpi__lagN), value` — the lagged feature matrix, the source
                                         of truth for KPI values; never rebuilt from Excel
     `fs_summary.json`                   `run_id, target, snapshot_id, status, kpis[{kpi, tier, lag, column, included,
                                         exclusion_reason, cache_key, producer_run_id}], resolved_kpis[]`
     `confounder_set.json`               `run_id, target, window, method, params, treatments[{treatment_kpi, status, cache_key,
                                         producer_run_id, confounders[{kpi, lag, reason}], excluded[{kpi, reason}]}]`
     Stage-1 cache entry                 `date, split (train_oof/test), y_true, y_hat, residual`; metadata `producer_run_id,
                                         key_parts, created_at`




https://mail.google.com/mail/u/1/?ui=2&ik=7d280f7890&view=lg&permmsgid=msg-f:1878560952167301212                                                     9/12

---

10/9/26, 2:35 PM                                                            Gmail - simple pipeline arc prompt
     `stage1_refs.parquet`                 `unit_key, role, variable, controls, cache_path, producer_run_id, outcome
                                           (HIT/FITTED/FAILED), error_category`
     `dml_result.parquet`                  `run_id, target, fold_id, combination_id, drivers, treatment_kpi, elasticity, std_error,
                                           p_value, r2, test_rmse, n_obs, status, error_reason`
     `consolidated_result.parquet`         dml_result columns + `mean_elasticity, std_elasticity, cv_elasticity, n_folds`
                                           `run_id, task_id, step, target, fold_id, status, aml_job_id, image, attempt, started_at,
     `status/{task_id}.json`               ended_at, outputs[], counters{}, cache{hits, misses, reused_from{}}, warnings[],
                                           error{category, message}`

     17. Reference scenarios

     Scenario                  Behaviour
     A. Normal                 New snapshot → FS/CS run (unit caches partially hit) → 4 folds in parallel → f2 PARTIAL (1 unit
     quarterly run             failed) → consolidate → run PARTIAL
     B. Repair                 Same conﬁg → all hits e      xcept the failed unit → SUCCEEDED; `reused_from = {A: ...}`
     C. Replay/audit           `replay_run_id=A` → same `master_conﬁg.json` and snap      shot, same image → identical results. Run
                               with `cache.mode=oﬀ` t                                                                          o verify determinism
     D. KPI added              `kpis.extra += k11` → FS: 1 miss; CS: only k11 computed (if the pool is separate, OPEN #2); DML: old
                               units hit
     E. DML-only               New learner params, `feature_selection=auto`, `confounder_selection=auto` → both CACHED from
     change                    the latest matching run; DML recomputes
     F. Step oﬀ                `confounder_selection=oﬀ`, `sour                                                                                                         ce_run_id=X` → manifest points to X's confounder_set; DML runs
     To trace a number in a report: consolidated row (run_id, fold, combination_id, treatment) → `dml_result` →
     `stage1_refs` (cache keys, producer_run_id) → fold status ﬁle and logs (aml_job_id) → the pr                        oducer run's manifest →
     snapshot and conﬁg.

     18. OPEN decisions (ask, do not assume)

         1. Does FS score each KPI independently? This decides per-KPI caching.
         2. Is the confounder pool separate from the driver superset? If not, adding a KPI reruns CS for all
             treatments.
         3. CS window: `common_train` (default) or `full`.
         4. Is DML deterministic? Run the determinism test before enabling cache reads.
         5. Cap on combinations (`max_combinations`), and `must_include` rules.
         6. Final error taxonomy.
         7. Exact Excel columns and aggregates ("average of CV", other existing columns).
         8. Owner of the canonical input schema.
         9. Cache retention period (Blob lifecycle rule on `cache/`).
        10. Whether a staging AML workspace exists.

     19. Invariants (every design must satisfy)

             One image, one entrypoint, one manifest per run; only Airflow writes the manifest.
             No step reads the repo config after init; everything comes from `runs/{run_id}/master_config.json`.
             Upstream artifacts are located via the manifest only.
             Every write is either run-scoped (`{component}/{run_id}/...`) or content-addressed (`cache/`,
             `snapshots/`).
             A failure is isolated to the smallest unit that depends on it; Excel never blocks.
             Every artifact, log line and AML job carries `run_id`; every cache hit records `producer_run_id`.
             Two hash functions only (`fp`, `data_fp`); cache keys include `cache_version`, not the git SHA.
             Excel is presentation only and is never read by the pipeline.



     Current repo structure,
     C:.
     | .amlignore
     | .amlignore.amltmp
     | .dockerignore
     | .env.example
     | .gitignore

https://mail.google.com/mail/u/1/?ui=2&ik=7d280f7890&view=lg&permmsgid=msg-f:1878560952167301212                                                           10/12

---

10/9/26, 2:35 PM                                                             Gmail - simple pipeline arc prompt
     | azure-pipelines-ace.yml
     | Dockerﬁle
     | environment.yml
     | Makeﬁle
     | README.md
     | requirements.txt
     | run_pipeline.py
     |
     +---airﬂo                                    w
     | | __init__.py
     | |
     | ---dags
     | driver_analysis_dag.py
     | __init__.py
     |
     +---components
     | | __init__.py
     | |
     | +---confounder_selection
     | | __init__.py
     | |
     | +---dml
     | | dml_analysis.py
     | | __init__.py
     | |
     | +---feature_selection
     | | .gitkeep
     | | feature_selection_dml.py
     | | __init__.py
     | |
     | +---residual_caching
     | | __init__.py
     | |
     | ---validation
     | evaluate.py
     | __init__.py
     |
     +---conﬁg
     | | .amlignore
     | | .amlignore.amltmp
     | | conﬁg_f      etcher.py
     | | conﬁg_v      al.json
     | | dml_conﬁg.json
     | | feature_selection_conﬁg.json
     | | integration_conﬁg.json
     | | master_conﬁg.json
     | | model_params.yaml
     | |
     | ---env
     | dev_env.yaml
     | prod.yaml
     | staging.yaml
     |
     +---docs
     | methodology.md
     |
     +---pipelines
     | main_pipeline.py
     | run_pipeline.py
     | __init__.py
     |
     +---shared

https://mail.google.com/mail/u/1/?ui=2&ik=7d280f7890&view=lg&permmsgid=msg-f:1878560952167301212                                                              11/12

---

10/9/26, 2:35 PM                                                           Gmail - simple pipeline arc prompt
     | | __init__.py
     | |
     | +---logging
     | | __init__.py
     | |
     | +---path_builder
     | | __init__.py
     | |
     | ---schema
     | cache.metadata.json
     | confounder_set.schema.json
     | feature_selected.schema.json
     | run_manifest.json
     | snapshot.schema.json
     |
     +---src
     | .amlignore
     | .amlignore.amltmp
     | comp_pipeline.py
     | correlation_pipeline.py
     | data.py
     | features.py
     | integration_main.py
     | integration_utils.py
     | misc_data_driver.py
     | models.py
     | pipeline_utils.py
     | reporting.py
     | s&p_data_pipeline.py
     | scenario_planning.py
     | train.py
     | __init__.py
     |
     +---tests
     | test_data.py
     | test_features.py
     | test_models.py
     |
     ---tools
     migrate_repo.py
     migration_mapping.json


































https://mail.google.com/mail/u/1/?ui=2&ik=7d280f7890&view=lg&permmsgid=msg-f:1878560952167301212                                                         12/12
