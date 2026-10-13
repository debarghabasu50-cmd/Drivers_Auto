"""
Azure ML job submission (SDK v2). One pipeline step = one AML command job running the registered
environment (the Docker image tagged with the git SHA; its WORKDIR must contain this repo).

Used by the Airflow DAG when AML_MODE=aml. Everything that touches Azure is injected or lazily imported,
so the logic is unit-testable with a fake MLClient.
"""
import os
import shlex
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from shared.logging import logger

CONFIG_ENV_DIR = Path(__file__).resolve().parents[1] / "config" / "env"
TERMINAL_STATES = {"Completed", "Failed", "Canceled", "NotResponding"}


def load_env_config(env: Optional[str] = None) -> Dict[str, Any]:
    """config/env/<AML_ENV>.yaml: storage container, AML workspace, compute cluster, environment name."""
    import yaml
    name = env or os.environ.get("AML_ENV", "dev_env")
    return yaml.safe_load((CONFIG_ENV_DIR / f"{name}.yaml").read_text(encoding="utf-8"))


def build_command(argv: list) -> str:
    return "python run_pipeline.py " + " ".join(shlex.quote(a) for a in argv)


def get_ml_client(env_cfg: Dict[str, Any]):
    from azure.ai.ml import MLClient
    from azure.identity import DefaultAzureCredential
    a = env_cfg["azure_ml"]
    return MLClient(DefaultAzureCredential(), a["subscription_id"], a["resource_group"], a["workspace_name"])


def register_environment(env_yaml_path: Optional[str] = None, ml_client=None,
                         env_cfg: Optional[Dict[str, Any]] = None) -> str:
    """Register or update an Azure ML Environment using a YAML specification."""
    from azure.ai.ml import load_environment
    env_cfg = env_cfg or load_env_config()
    ml_client = ml_client or get_ml_client(env_cfg)

    path_str = env_yaml_path or env_cfg.get("azure_ml", {}).get("environment_yaml") or "environment.yaml"
    env_path = Path(path_str)
    if not env_path.is_absolute():
        candidates = [CONFIG_ENV_DIR / path_str, Path(path_str).resolve(), Path(__file__).resolve().parents[1] / path_str]
        for c in candidates:
            if c.exists():
                env_path = c
                break

    if not env_path.exists():
        fallback = env_cfg.get("azure_ml", {}).get("environment_name", "driver-analysis-env:1")
        logger.warning(f"Environment YAML '{path_str}' not found; using fallback {fallback}", event="AML_ENV_NOT_FOUND")
        return fallback

    env_entity = load_environment(source=str(env_path))
    registered = ml_client.environments.create_or_update(env_entity)
    registered_id = f"{registered.name}:{registered.version}"
    logger.info(f"AML environment registered: {registered_id}", event="AML_ENV_REGISTERED", environment=registered_id)
    return registered_id


def submit_and_wait(argv: list, secrets: Dict[str, str], tags: Dict[str, str], ml_client=None,
                    env_cfg: Optional[Dict[str, Any]] = None, poll_s: int = 15, timeout_s: int = 6 * 3600,
                    sleep: Callable[[float], None] = time.sleep) -> int:
    """Submit the step as an AML command job and block until it ends. Returns 0 if the job Completed, else 1.
    The caller decides success/retry from the step's status file, not from this code: a job that ended without
    writing one (preemption, OOM, timeout) is treated as a transient failure."""
    from azure.ai.ml import command
    env_cfg = env_cfg or load_env_config()
    ml_client = ml_client or get_ml_client(env_cfg)
    a = env_cfg["azure_ml"]
    env_vars = {
        **secrets,
        "AZURE_STORAGE_CONTAINER": env_cfg.get("storage", {}).get("container", "driver-analysis-dev"),
        "IMAGE_TAG": os.environ.get("IMAGE_TAG", "unknown"),
    }
    if env_cfg.get("storage", {}).get("blob_url"):
        env_vars["AZURE_BLOB_URL"] = env_cfg["storage"]["blob_url"]
    if env_cfg.get("telemetry", {}).get("loki_url"):
        env_vars["LOKI_URL"] = env_cfg["telemetry"]["loki_url"]
    if env_cfg.get("telemetry", {}).get("app_insights_connection_string"):
        env_vars["APP_INSIGHTS_CONNECTION_STRING"] = env_cfg["telemetry"]["app_insights_connection_string"]

    job = command(
        command=build_command(argv),
        environment=a["environment_name"],
        compute=a["compute_cluster"],
        environment_variables=env_vars,
        tags=tags,
        display_name=tags.get("task_id"),
        experiment_name="driver_analysis",
    )
    submitted = ml_client.jobs.create_or_update(job)
    logger.info(f"AML job submitted: {submitted.name}", event="AML_SUBMITTED", job_name=submitted.name,
                studio_url=getattr(submitted, "studio_url", None))

    deadline = time.time() + timeout_s
    while True:
        state = ml_client.jobs.get(submitted.name).status
        if state in TERMINAL_STATES:
            logger.info(f"AML job {submitted.name} ended: {state}", event="AML_FINISHED", job_name=submitted.name,
                        job_state=state)
            return 0 if state == "Completed" else 1
        if time.time() > deadline:
            ml_client.jobs.begin_cancel(submitted.name)
            logger.error(f"AML job {submitted.name} exceeded {timeout_s}s; cancelled", event="AML_TIMEOUT",
                         error_category="TIMEOUT", job_name=submitted.name)
            return 1
        sleep(poll_s)
