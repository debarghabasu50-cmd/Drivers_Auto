"""
Airflow DAG: init_run -> feature_selection[target] -> confounder_selection[target] -> dml_fold[target|fold]
-> consolidate (last step; closes the run).

The DAG is a thin dispatcher: every task calls `run_step(...)`, which runs one pipeline step, reads its
status file and raises according to the failure category (non-retryable -> AirflowFailException so Airflow
does not burn retries; transient -> normal exception -> retry). Targets/folds come from the manifest that
init_run wrote, never from DAG constants.
"""
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))  # append, not insert: the local ./airflow package must not shadow real Airflow

import run_pipeline
from shared.logging import logger
from shared.path_builder.paths import get_manifest_path, get_master_config_path, get_task_status_path, make_task_id
from shared.path_builder.storage import get_storage_client
from shared.schema.contracts import is_retryable, new_run_id

try:
    from airflow import DAG
    from airflow.exceptions import AirflowFailException
    from airflow.models.xcom_arg import XComArg
    from airflow.operators.python import PythonOperator
except ImportError:  # lets the module import (and its helpers be tested) without Airflow installed
    class AirflowFailException(Exception): pass

    class XComArg:
        def __init__(self, *a, **k): pass

    class DAG:
        def __init__(self, *a, **k): pass
        def __enter__(self): return self
        def __exit__(self, *a): pass

    class PythonOperator:
        output = None
        def __init__(self, *a, **k): pass
        @classmethod
        def partial(cls, **k): return cls()
        def expand(self, **k): return self
        def __rshift__(self, other): return other


def _secret(name: str) -> str:
    try:
        from airflow.models import Variable
        return Variable.get(name, default_var="")
    except ImportError:
        return os.environ.get(name, "")


def _is_switched_off(step: str, run_id: str) -> bool:
    """Steps switched off in the frozen config only write a SKIPPED status (and validate reuse): no AML cluster
    needed, so they run in-process on the Airflow worker instead of paying for a job."""
    path = get_master_config_path(run_id)
    storage = get_storage_client()
    if not storage.exists(path) and storage.exists(f"runs/{run_id}/master_config.json"):
        path = f"runs/{run_id}/master_config.json"
    toggle = run_pipeline.STEPS[step].toggle
    return bool(toggle) and storage.exists(path) and storage.read_json(path).get("steps", {}).get(toggle) == "off"


def submit_and_wait(argv: list, tags: dict) -> int:
    """AML_MODE=aml -> Azure ML command job (see shared/aml.py); anything else -> in-process (local/dev/tests)."""
    if os.environ.get("AML_MODE", "local") != "aml":
        return run_pipeline.main(argv)
    from shared import aml
    secrets = {k: _secret(k) for k in ("AZURE_STORAGE_CONNECTION_STRING", "APP_INSIGHTS_CONNECTION_STRING")}
    return aml.submit_and_wait(argv, secrets, tags, timeout_s=int(os.environ.get("AML_JOB_TIMEOUT_S", 6 * 3600)))


def run_step(step: str, run_id: str, target: str = None, fold_id: str = None, attempt: int = 1) -> None:
    argv = ["--step", step, "--run-id", run_id, "--attempt", str(attempt)]
    if target:
        argv += ["--target", target]
    if fold_id:
        argv += ["--fold-id", fold_id]
    task_id = make_task_id(step, target, fold_id)
    ctx = dict(run_id=run_id, component="airflow", task_id=task_id, target=target, fold_id=fold_id, attempt=attempt)
    logger.bind_context(**ctx)
    if _is_switched_off(step, run_id):
        exit_code = run_pipeline.main(argv)
    else:
        tags = {"run_id": run_id, "step": step, "task_id": task_id, "attempt": str(attempt)}
        exit_code = submit_and_wait(argv, tags)
    logger.bind_context(**ctx)  # in-process runs re-bind the process-wide context; restore ours

    storage = get_storage_client()
    path = get_task_status_path(run_id, task_id)
    status = storage.read_json(path) if storage.exists(path) else None
    if status is None or status["status"] == "RUNNING":  # job died/was cancelled before writing a final status
        logger.error(f"{task_id}: job ended without a final status (exit code {exit_code})", event="NO_STATUS",
                     error_category="INFRA_TRANSIENT")
        raise RuntimeError(f"{task_id}: job ended without a final status (exit code {exit_code})")  # retry
    if status["status"] == "FAILED":
        category, message = status["error"]["category"], status["error"]["message"]
        if is_retryable(category):
            raise RuntimeError(f"{task_id} failed ({category}): {message}")  # Airflow retries
        raise AirflowFailException(f"{task_id} failed ({category}): {message}")  # no point retrying


def _attempt(context) -> int:
    ti = context.get("ti")
    return getattr(ti, "try_number", 1) if ti else 1


def _run_id(context) -> str:
    return context["ti"].xcom_pull(task_ids="init_run", key="run_id")


def task_init_run(**context):
    run_id = new_run_id(context.get("logical_date"), seed=context.get("run_id"))  # idempotent across retries
    run_step("init_run", run_id, attempt=_attempt(context))
    plan = get_storage_client().read_json(get_manifest_path(run_id))["plan"]
    ti = context["ti"]
    ti.xcom_push(key="run_id", value=run_id)  # XCom carries only run_id + work items
    ti.xcom_push(key="target_items", value=[{"target": t} for t in plan["targets"]])
    return [{"work_item": w} for w in plan["work_items"]]


def task_feature_selection(target: str, **context):
    run_step("feature_selection", _run_id(context), target=target, attempt=_attempt(context))


def task_confounder_selection(target: str, **context):
    run_step("confounder_selection", _run_id(context), target=target, attempt=_attempt(context))


def task_dml_fold(work_item: str, **context):
    target, fold_id = work_item.split("|")
    run_step("dml", _run_id(context), target=target, fold_id=fold_id, attempt=_attempt(context))


def task_consolidate(**context):
    run_step("consolidate", _run_id(context), attempt=_attempt(context))


with DAG(
    dag_id="driver_analysis_causal_pipeline",
    default_args={"owner": "ml-platform", "retries": 2, "retry_delay": timedelta(minutes=2)},
    description="Driver Analysis Causal Pipeline (Airflow -> AML -> Azure Blob)",
    schedule_interval="@quarterly",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["driver_analysis", "causal", "dml"],
) as dag:
    init = PythonOperator(task_id="init_run", python_callable=task_init_run)

    # all_done downstream: a failed upstream task makes dependents fail fast with UPSTREAM_MISSING
    # (non-retryable) instead of silently skipping, so healthy targets still proceed and the failure is visible.
    fs = PythonOperator.partial(task_id="feature_selection", python_callable=task_feature_selection,
                                trigger_rule="all_done").expand(op_kwargs=XComArg(init, key="target_items"))
    cs = PythonOperator.partial(task_id="confounder_selection", python_callable=task_confounder_selection,
                                trigger_rule="all_done").expand(op_kwargs=XComArg(init, key="target_items"))
    dml = PythonOperator.partial(task_id="dml_fold", python_callable=task_dml_fold, trigger_rule="all_done",
                                 max_active_tis_per_dag=8).expand(op_kwargs=init.output)
    consolidate = PythonOperator(task_id="consolidate", python_callable=task_consolidate, trigger_rule="all_done")

    init >> fs >> cs >> dml >> consolidate
