"""
Validation component (reserved). Will consume published/ results after consolidation.
"""
from shared.context import StepContext, StepResult
from shared.logging import logger


def run_validation(ctx: StepContext) -> StepResult:
    logger.info("Validation is reserved for a later phase; nothing to do.")
    return StepResult(status="SKIPPED")
