"""
Driver-combination enumeration (single source of truth, used by DML to plan work and by consolidate to
compute the expected row count).

Every set of `drivers_per_combo` (k) drivers out of the N selected treatments, optionally restricted to
sets containing `must_include`, then deterministically capped at `max_combinations`.
"""
from itertools import combinations
from math import comb
from typing import Any, Dict, List, Tuple

from shared.schema.contracts import ErrorCategory, PipelineError, fp


def combination_id(drivers: Tuple[str, ...]) -> str:
    return "cmb_" + fp(sorted(drivers))


def enumerate_combinations(drivers: List[str], cfg: Dict[str, Any]) -> List[Tuple[str, ...]]:
    """Sorted driver tuples. The cap keeps the combinations with the smallest id hash, so the selection is
    reproducible, independent of input order, and unbiased with respect to driver names."""
    k = int(cfg["drivers_per_combo"])
    cap = cfg.get("max_combinations")
    must = sorted(cfg.get("must_include", []))
    pool = sorted(set(drivers))

    if k < 1 or k > len(pool):
        raise PipelineError(ErrorCategory.CONFIG,
                            f"drivers_per_combo={k} needs 1..{len(pool)} selected drivers, got {len(pool)}")
    unknown = [m for m in must if m not in pool]
    if unknown or len(must) > k:
        raise PipelineError(ErrorCategory.CONFIG,
                            f"must_include invalid: unknown={unknown}, size={len(must)} (k={k})")

    rest = [d for d in pool if d not in must]
    combos = [tuple(sorted(must + list(c))) for c in combinations(rest, k - len(must))]
    if cap and len(combos) > cap:
        combos = sorted(combos, key=combination_id)[:cap]
    return sorted(combos)


def expected_rows(n_drivers: int, cfg: Dict[str, Any]) -> int:
    """Rows per fold = combinations x k (one elasticity per driver per combination), without enumerating."""
    k = int(cfg["drivers_per_combo"])
    must = len(cfg.get("must_include", []))
    total = comb(n_drivers - must, k - must)
    cap = cfg.get("max_combinations")
    return (min(total, cap) if cap else total) * k
