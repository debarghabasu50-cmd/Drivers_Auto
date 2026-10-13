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
    job = command(
        command=build_command(argv),
        environment=a["environment_name"],
        compute=a["compute_cluster"],
        environment_variables={**secrets, "AZURE_STORAGE_CONTAINER": env_cfg["storage"]["container"],
                               "IMAGE_TAG": os.environ.get("IMAGE_TAG", "unknown")},
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
