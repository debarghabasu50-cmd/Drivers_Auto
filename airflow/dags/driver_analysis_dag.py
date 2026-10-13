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
    from airflow.models.param import Param
    from airflow.models.xcom_arg import XComArg
    from airflow.operators.python import PythonOperator
except ImportError:  # lets the module import (and its helpers be tested) without Airflow installed
    class AirflowFailException(Exception): pass

    class XComArg:
        def __init__(self, *a, **k): pass

    class Param:
        def __init__(self, default=None, description="", **kwargs):
            self.default = default
            self.description = description

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
        def __rrshift__(self, other): return self


def _resolve_secret_or_param(key_names: list, params: dict, default: str = "") -> str:
    """Resolve a secret or parameter without hardcoding: checks DAG params, Airflow Variables, and env vars."""
    for k in key_names:
        # 1. Check runtime DAG execution params
        if params and k in params and params[k]:
            return str(params[k])
        # 2. Check Airflow Variables / secrets backend
        try:
            from airflow.models import Variable
            val = Variable.get(k, default_var=None)
            if val is not None and val != "":
                return str(val)
        except Exception:
            pass
        # 3. Check environment variables
        val = os.environ.get(k)
        if val is not None and val != "":
            return str(val)
    return default


def _secret(name: str) -> str:
    return _resolve_secret_or_param([name], params={})


def _is_switched_off(step: str, run_id: str) -> bool:
    """Steps switched off in the frozen config only write a SKIPPED status (and validate reuse): no AML cluster
    needed, so they run in-process on the Airflow worker instead of paying for a job."""
    path = get_master_config_path(run_id)
    storage = get_storage_client()
    if not storage.exists(path) and storage.exists(f"runs/{run_id}/master_config.json"):
        path = f"runs/{run_id}/master_config.json"
    toggle = run_pipeline.STEPS[step].toggle
    return bool(toggle) and storage.exists(path) and storage.read_json(path).get("steps", {}).get(toggle) == "off"


def task_load_aml_workspace(**context) -> dict:
    """Dedicated task to dynamically resolve all AML secrets, connection strings, blob & Loki URLs,
    and register the environment YAML if running on Azure ML. Nothing is hardcoded or stored in code.
    Pushes the full workspace configuration dictionary to XCom for downstream steps to pull."""
    params = context.get("params", {}) if context else {}
    aml_env = _resolve_secret_or_param(["aml_env", "AML_ENV"], params, default="dev_env")

    # Load base template as non-secret fallback for cluster/container naming if YAML exists
    base_cfg = {}
    try:
        from shared.aml import load_env_config
        base_cfg = load_env_config(aml_env)
    except Exception:
        pass

    base_aml = base_cfg.get("azure_ml", {})
    base_storage = base_cfg.get("storage", {})
    base_telemetry = base_cfg.get("telemetry", {})

    sub_id = _resolve_secret_or_param(
        ["subscription_id", "aml_subscription_id", "AZURE_SUBSCRIPTION_ID", "AML_SUBSCRIPTION_ID"],
        params, default=base_aml.get("subscription_id", "")
    )
    rg = _resolve_secret_or_param(
        ["resource_group", "aml_resource_group", "AZURE_RESOURCE_GROUP", "AML_RESOURCE_GROUP"],
        params, default=base_aml.get("resource_group", "")
    )
    ws = _resolve_secret_or_param(
        ["workspace_name", "aml_workspace_name", "AZURE_WORKSPACE_NAME", "AML_WORKSPACE_NAME"],
        params, default=base_aml.get("workspace_name", "")
    )
    cluster = _resolve_secret_or_param(
        ["compute_cluster", "aml_compute_cluster", "AML_COMPUTE_CLUSTER"],
        params, default=base_aml.get("compute_cluster", "cpu-large")
    )
    env_name = _resolve_secret_or_param(
        ["environment_name", "aml_environment_name", "AML_ENVIRONMENT_NAME"],
        params, default=base_aml.get("environment_name", "driver-analysis-env:1")
    )
    env_yaml = _resolve_secret_or_param(
        ["environment_yaml", "aml_environment_yaml", "AML_ENVIRONMENT_YAML"],
        params, default="environment.yaml"
    )

    container = _resolve_secret_or_param(
        ["container", "azure_storage_container", "AZURE_STORAGE_CONTAINER"],
        params, default=base_storage.get("container", "driver-analysis-dev")
    )
    blob_url = _resolve_secret_or_param(
        ["blob_url", "azure_blob_url", "AZURE_BLOB_URL"],
        params, default=""
    )
    storage_cs = _resolve_secret_or_param(
        ["storage_connection_string", "AZURE_STORAGE_CONNECTION_STRING"],
        params, default=""
    )

    loki_url = _resolve_secret_or_param(
        ["loki_url", "LOKI_URL"],
        params, default=""
    )
    app_insights_cs = _resolve_secret_or_param(
        ["app_insights_connection_string", "APP_INSIGHTS_CONNECTION_STRING"],
        params, default=""
    )

    aml_config = {
        "azure_ml": {
            "subscription_id": sub_id,
            "resource_group": rg,
            "workspace_name": ws,
            "compute_cluster": cluster,
            "environment_name": env_name,
            "environment_yaml": env_yaml,
        },
        "storage": {
            "container": container,
            "blob_url": blob_url,
            "connection_string": storage_cs,
        },
        "telemetry": {
            "loki_url": loki_url,
            "app_insights_connection_string": app_insights_cs,
        },
    }

    # Register/update the environment YAML in Azure ML if AML mode is active
    if os.environ.get("AML_MODE", "local") == "aml":
        try:
            from shared import aml
            registered_env = aml.register_environment(env_yaml, env_cfg=aml_config)
            aml_config["azure_ml"]["environment_name"] = registered_env
            logger.info(f"Environment registered in AML: {registered_env}", event="AML_ENV_REGISTERED")
        except Exception as exc:
            logger.warning(f"Could not register environment YAML '{env_yaml}': {exc}", event="AML_ENV_REG_WARNING")

    ti = context.get("ti") if context else None
    if ti:
        ti.xcom_push(key="aml_workspace", value=aml_config)
        ti.xcom_push(key="aml_config", value=aml_config)

    logger.info(f"AML workspace configuration resolved: workspace={ws}, cluster={cluster}, env={aml_config['azure_ml']['environment_name']}",
                event="AML_WORKSPACE_RESOLVED")
    return aml_config


def _aml_config(context) -> dict:
    """Pull the resolved workspace & secrets configuration from XCom, falling back gracefully if called standalone."""
    ti = context.get("ti") if context else None
    if ti:
        try:
            cfg = ti.xcom_pull(task_ids="load_aml_workspace", key="aml_workspace")
            if cfg:
                return cfg
        except Exception:
            pass
        try:
            cfg = ti.xcom_pull(task_ids="load_aml_workspace", key="aml_config")
            if cfg:
                return cfg
        except Exception:
            pass
    return task_load_aml_workspace(**(context or {}))


def submit_and_wait(argv: list, tags: dict, aml_config: dict = None) -> int:
    """AML_MODE=aml -> Azure ML command job (see shared/aml.py); anything else -> in-process (local/dev/tests)."""
    if os.environ.get("AML_MODE", "local") != "aml":
        return run_pipeline.main(argv)
    from shared import aml
    aml_config = aml_config or _aml_config(None)
    secrets = {}
    storage_cs = aml_config.get("storage", {}).get("connection_string") or _secret("AZURE_STORAGE_CONNECTION_STRING")
    if storage_cs:
        secrets["AZURE_STORAGE_CONNECTION_STRING"] = storage_cs
    app_insights_cs = aml_config.get("telemetry", {}).get("app_insights_connection_string") or _secret("APP_INSIGHTS_CONNECTION_STRING")
    if app_insights_cs:
        secrets["APP_INSIGHTS_CONNECTION_STRING"] = app_insights_cs
    return aml.submit_and_wait(argv, secrets, tags, env_cfg=aml_config,
                               timeout_s=int(os.environ.get("AML_JOB_TIMEOUT_S", 6 * 3600)))


def run_step(step: str, run_id: str, target: str = None, fold_id: str = None, attempt: int = 1, aml_config: dict = None) -> None:
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
        try:
            exit_code = submit_and_wait(argv, tags, aml_config=aml_config)
        except TypeError:
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
    aml_cfg = _aml_config(context)
    run_step("init_run", run_id, attempt=_attempt(context), aml_config=aml_cfg)
    plan = get_storage_client().read_json(get_manifest_path(run_id))["plan"]
    ti = context["ti"]
    ti.xcom_push(key="run_id", value=run_id)  # XCom carries only run_id + work items
    ti.xcom_push(key="target_items", value=[{"target": t} for t in plan["targets"]])
    return [{"work_item": w} for w in plan["work_items"]]


def task_feature_selection(target: str, **context):
    aml_cfg = _aml_config(context)
    run_step("feature_selection", _run_id(context), target=target, attempt=_attempt(context), aml_config=aml_cfg)


def task_confounder_selection(target: str, **context):
    aml_cfg = _aml_config(context)
    run_step("confounder_selection", _run_id(context), target=target, attempt=_attempt(context), aml_config=aml_cfg)


def task_dml_fold(work_item: str, **context):
    target, fold_id = work_item.split("|")
    aml_cfg = _aml_config(context)
    run_step("dml", _run_id(context), target=target, fold_id=fold_id, attempt=_attempt(context), aml_config=aml_cfg)


def task_consolidate(**context):
    aml_cfg = _aml_config(context)
    run_step("consolidate", _run_id(context), attempt=_attempt(context), aml_config=aml_cfg)


DAG_PARAMS = {
    "aml_env": Param("dev_env", type="string", description="Environment configuration profile name (e.g. dev_env, prod)"),
    "subscription_id": Param("", type="string", description="Azure Subscription ID (optional override; default from secret/env)"),
    "resource_group": Param("", type="string", description="Azure Resource Group name (optional override; default from secret/env)"),
    "workspace_name": Param("", type="string", description="Azure ML Workspace name (optional override; default from secret/env)"),
    "compute_cluster": Param("", type="string", description="Azure ML Compute Cluster name (optional override; default from secret/env)"),
    "environment_name": Param("", type="string", description="Azure ML Environment name (optional override; default from secret/env)"),
    "environment_yaml": Param("environment.yaml", type="string", description="Path to AML environment specification YAML"),
    "container": Param("", type="string", description="Azure Blob Storage container name (optional override; default from secret/env)"),
    "blob_url": Param("", type="string", description="Azure Blob Storage account URL (optional override; default from secret/env)"),
    "loki_url": Param("", type="string", description="Loki/Log ingestion URL (optional override; default from secret/env)"),
}


with DAG(
    dag_id="driver_analysis_causal_pipeline",
    default_args={"owner": "ml-platform", "retries": 2, "retry_delay": timedelta(minutes=2)},
    description="Driver Analysis Causal Pipeline (Airflow -> AML -> Azure Blob)",
    schedule_interval="@quarterly",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["driver_analysis", "causal", "dml"],
    params=DAG_PARAMS,
) as dag:
    load_aml = PythonOperator(task_id="load_aml_workspace", python_callable=task_load_aml_workspace)
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

    load_aml >> init >> fs >> cs >> dml >> consolidate
