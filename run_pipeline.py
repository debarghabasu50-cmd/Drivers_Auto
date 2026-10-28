"""
Unified CLI entrypoint (one AML command job = one invocation):

    python run_pipeline.py --step <step> --run-id <run_id> [--target T] [--fold-id F] [--attempt N]

This wrapper is the ONLY writer of runs/{div}/{quarter}/{run_id}/status/{task_id}.json and logs/{task_id}.log.
Binds logging context, times a task, classifies errors and emits TASK_STARTED / TASK_SUMMARY. Components are
plain functions ``run_xxx(ctx) -> StepResult`` registered in STEPS below.

Exit codes: 0 = SUCCEEDED/PARTIAL/SKIPPED, 1 = failed but retryable (transient infra), 2 = failed, do not retry.
"""
import argparse
import os
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from components.confounder_selection.confounder_selection import run_confounder_selection
from components.dml.dml_analysis import run_dml_fold
from components.feature_selection.feature_selection import run_feature_selection
from components.validation.evaluate import run_validation
from pipelines.consolidate import run_consolidation
from pipelines.init_run import load_repo_config, run_init
from shared.context import REUSABLE_STEPS as REUSABLE, StepContext, StepResult
from shared.logging import logger
from shared.path_builder.paths import (
    get_manifest_path,
    get_master_config_path,
    get_task_log_path,
    get_task_status_path,
    make_task_id,
    set_active_scope,
)
from shared.path_builder.storage import get_storage_client
from shared.schema.contracts import ErrorCategory, PipelineError, TaskStatusContract, is_retryable


@dataclass
class StepSpec:
    fn: Callable[[StepContext], StepResult]
    toggle: Optional[str] = None  # key in config["steps"]; "off" => SKIPPED
    needs_target: bool = False
    needs_fold: bool = False


STEPS = {
    "init_run": StepSpec(run_init),
    "feature_selection": StepSpec(run_feature_selection, "feature_selection", needs_target=True),
    "confounder_selection": StepSpec(run_confounder_selection, "confounder_selection", needs_target=True),
    "dml": StepSpec(run_dml_fold, "dml", needs_target=True, needs_fold=True),
    "consolidate": StepSpec(run_consolidation),
    "validate": StepSpec(run_validation),
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _classify(exc: BaseException) -> str:
    if isinstance(exc, PipelineError):
        return exc.category
    if isinstance(exc, TimeoutError):
        return ErrorCategory.TIMEOUT
    if type(exc).__module__.startswith("azure"):
        name = type(exc).__name__
        if name == "ResourceNotFoundError":
            return ErrorCategory.UPSTREAM_MISSING
        if name in ("ClientAuthenticationError", "HttpResponseError") and getattr(exc, "status_code", 0) in (401, 403):
            return ErrorCategory.CONFIG
        return ErrorCategory.INFRA_TRANSIENT
    if isinstance(exc, (ConnectionError, OSError)):
        return ErrorCategory.INFRA_TRANSIENT
    return ErrorCategory.CODE_BUG


def _load_config(step: str, run_id: str, storage, config_path: str) -> dict:
    if step == "init_run":
        cfg = load_repo_config(config_path)
        set_active_scope(cfg.get("div") or cfg.get("pipeline", {}).get("div"),
                         cfg.get("quarter") or cfg.get("pipeline", {}).get("quarter"))
        return cfg

    # Fast O(1) catalog lookup across divisions and quarters
    index_ptr = f"runs/_index/{run_id}.json"
    if storage.exists(index_ptr):
        idx = storage.read_json(index_ptr)
        div = idx.get("div")
        quarter = idx.get("quarter")
        set_active_scope(div, quarter)
        frozen = get_master_config_path(run_id, div=div, quarter=quarter)
    else:
        frozen = get_master_config_path(run_id)

    if not storage.exists(frozen):
        # Fallback only if flat discovery index was not yet present
        matching = [b for b in storage.list_blobs("runs/") if b.replace("\\", "/").endswith(f"/{run_id}/master_config.json")]
        if matching:
            frozen = matching[0]
        else:
            raise PipelineError(ErrorCategory.UPSTREAM_MISSING, f"frozen config missing: {frozen} (did init_run run?)")

    cfg = storage.read_json(frozen)
    set_active_scope(cfg.get("div") or cfg.get("pipeline", {}).get("div"),
                     cfg.get("quarter") or cfg.get("pipeline", {}).get("quarter"))
    return cfg


def main(argv=None, storage=None) -> int:
    parser = argparse.ArgumentParser(description="Driver Analysis Causal Pipeline step runner")
    parser.add_argument("--step", required=True, choices=sorted(STEPS))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--target", default=None)
    parser.add_argument("--fold-id", default=None)
    parser.add_argument("--attempt", type=int, default=int(os.environ.get("ATTEMPT", "1")))
    parser.add_argument("--config", default=None, help="init_run only: alternative master_config.json")
    args = parser.parse_args(argv)

    storage = storage or get_storage_client()
    spec = STEPS[args.step]
    task_id = make_task_id(args.step, args.target, args.fold_id)
    logger.bind_context(run_id=args.run_id, component=args.step, task_id=task_id,
                        target=args.target, fold_id=args.fold_id, attempt=args.attempt)
    status = TaskStatusContract(run_id=args.run_id, task_id=task_id, step=args.step, status="RUNNING",
                                target=args.target, fold_id=args.fold_id, attempt=args.attempt, started_at=_now())
    
    t0 = time.time()
    exit_code = 0
    config = None

    try:
        if (spec.needs_target and not args.target) or (spec.needs_fold and not args.fold_id):
            raise PipelineError(ErrorCategory.CONFIG, f"step '{args.step}' requires --target"
                                + (" and --fold-id" if spec.needs_fold else ""))
        config = _load_config(args.step, args.run_id, storage, args.config)
        if config and args.target:
            for t in config.get("targets", []):
                if args.target == t.get("column") or args.target == t.get("target_id"):
                    args.target = t.get("target_id")
                    break
        task_id = make_task_id(args.step, args.target, args.fold_id)
        status.task_id = task_id
        status.target = args.target
        logger.bind_context(run_id=args.run_id, component=args.step, task_id=task_id,
                            target=args.target, fold_id=args.fold_id, attempt=args.attempt)
        status_path = get_task_status_path(args.run_id, task_id)
        try:
            storage.write_json(status_path, status.to_dict())  # live visibility: task is RUNNING
        except Exception as write_exc:
            msg = f"CRITICAL: Failed to write initial RUNNING status file {status_path}: {write_exc}"
            sys.stderr.write(f"{msg}\n")
            sys.stderr.flush()
            logger.error(msg, event="STATUS_WRITE_FAILED",
                         error_category=ErrorCategory.INFRA_TRANSIENT,
                         status_path=status_path, intended_status="RUNNING",
                         exc=traceback.format_exc())
            raise PipelineError(ErrorCategory.INFRA_TRANSIENT, msg) from write_exc
        sys.stdout.write(
            f"\n\n\n{'=' * 80}\n"
            f"  >>> PROCESS STARTING: [{args.step.upper()}]  task_id: {task_id}\n"
            f"{'=' * 80}\n\n"
        )
        sys.stdout.flush()
        logger.info(f"task started: {task_id}", event="TASK_STARTED")

        ctx = StepContext(run_id=args.run_id, step=args.step, config=config, storage=storage,
                          target=args.target, fold_id=args.fold_id, attempt=args.attempt)
        if spec.toggle and config.get("steps", {}).get(spec.toggle, "on") == "off":
            result = StepResult(status="SKIPPED", warnings=[f"steps.{spec.toggle}=off"])
            if args.step in REUSABLE:
                src = ctx.upstream_run_id(args.step)
                reused_path = REUSABLE[args.step][1](src, args.target, div=ctx.div, quarter=ctx.quarter)
                reused_fp = ctx.step_fp(args.step)
                result.outputs = [reused_path]
                result.counters = {"reused_from": src}
                logger.log_provenance(
                    source_run_id=src,
                    path=reused_path,
                    target=args.target,
                    fingerprint=reused_fp,
                    msg=f"{args.step} reused from run {src}",
                    event="ARTIFACT_REUSED",
                    reused_from=src,
                )
        else:
            result = spec.fn(ctx)
        status.status, status.outputs = result.status, result.outputs
        status.counters, status.warnings = result.counters, result.warnings
    except BaseException as exc:
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        category = _classify(exc)
        status.status = "FAILED"
        status.error = {"category": category, "message": str(exc), "traceback": traceback.format_exc()}
        exit_code = 1 if is_retryable(category) else 2
        logger.error(f"{task_id} failed: {exc}", event="TASK_FAILED", error_category=category,
                     exc=status.error["traceback"])
    finally:
        status.ended_at = _now()
        duration_ms = round((time.time() - t0) * 1000, 1)
        status_path = get_task_status_path(args.run_id, task_id)

        try:
            storage.write_json(status_path, status.to_dict())
        except Exception as write_exc:
            msg = f"CRITICAL: Failed to write final status file {status_path} (intended status={status.status}): {write_exc}"
            sys.stderr.write(f"{msg}\n")
            sys.stderr.flush()
            logger.error(msg, event="STATUS_WRITE_FAILED",
                         error_category=ErrorCategory.INFRA_TRANSIENT,
                         status_path=status_path, intended_status=status.status,
                         exc=traceback.format_exc())
            exit_code = exit_code or 1

        # If consolidate step failed, ensure the run manifest is closed with a terminal status
        if args.step == "consolidate" and status.status == "FAILED":
            try:
                m_path = get_manifest_path(args.run_id)
                if not storage.exists(m_path):
                    index_ptr = f"runs/_index/{args.run_id}.json"
                    if storage.exists(index_ptr):
                        idx = storage.read_json(index_ptr)
                        m_path = get_manifest_path(args.run_id, div=idx.get("div"), quarter=idx.get("quarter"))
                    else:
                        matching = [b for b in storage.list_blobs("runs/") if b.replace("\\", "/").endswith(f"/{args.run_id}/run_manifest.json")]
                        if matching:
                            m_path = matching[0]
                if storage.exists(m_path):
                    m = storage.read_json(m_path)
                    if m.get("status") == "RUNNING":
                        m["status"] = "FAILED"
                        m["end_time"] = status.ended_at
                        storage.write_json(m_path, m)
            except Exception as m_exc:
                logger.error(f"failed to set terminal status on manifest during consolidation failure: {m_exc}",
                             event="MANIFEST_UPDATE_FAILED")

        logger.log_event("TASK_SUMMARY", f"{task_id} -> {status.status}",
                         level="ERROR" if status.status == "FAILED" else "INFO",
                         counters=status.counters, duration_ms=duration_ms,
                         error_category=(status.error or {}).get("category"), status=status.status)
        sys.stdout.write(
            f"\n{'-' * 80}\n"
            f"  <<< PROCESS COMPLETED: [{args.step.upper()}]  task_id: {task_id}  status: {status.status}\n"
            f"{'-' * 80}\n\n\n"
        )
        sys.stdout.flush()

        # Write durable .log file for this task: preserve latest canonical AND attempt-specific retry logs
        try:
            log_content = logger.get_task_log_text()
            canonical_log_path = get_task_log_path(args.run_id, task_id)
            storage.write_text(canonical_log_path, log_content)
            if args.attempt:
                attempt_log_path = get_task_log_path(args.run_id, task_id, attempt=args.attempt)
                if attempt_log_path != canonical_log_path:
                    storage.write_text(attempt_log_path, log_content)
        except Exception as log_exc:
            sys.stderr.write(f"Failed to write task log file: {log_exc}\n")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
