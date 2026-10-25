"""Active-learning metrics.

Normalized AUBC is the trapezoidal integral of the endpoint metric over the
realized labeled-image budget, divided by the evaluated budget span.  The
result therefore remains on the metric scale and can be read as
budget-averaged mAP.  ``al.analysis.metric`` selects mAP50 for VOC-style sets
or mAP50-95 for COCO-style evaluation.
"""
import math

import numpy as np

DEFAULT_METRIC = "test_mAP50_95"


def aubc(rounds: list, field: str = DEFAULT_METRIC) -> float:
    # Prefer the realized number of labeled images. Percentage schedules can
    # require integer rounding, so the stored nominal fractions are not always
    # the exact x coordinates. Dividing by the x span makes counts and exact
    # fractions equivalent representations of the same image-budget curve.
    has_realized_budget = ["n_labeled" in r for r in rounds]
    if has_realized_budget and all(has_realized_budget):
        budget_key = "n_labeled"
    elif any(has_realized_budget):
        raise ValueError(
            "mixed budget schema: n_labeled must be present in every round"
        )
    else:
        budget_key = "frac"
    if rounds and all("round" in r for r in rounds):
        records = sorted(rounds, key=lambda r: int(r["round"]))
        pts = [(float(r[budget_key]), float(r[field])) for r in records]
    else:
        pts = sorted((float(r[budget_key]), float(r[field])) for r in rounds)
    if len(pts) < 2:
        return math.nan
    x = np.array([p[0] for p in pts], dtype=float)
    y = np.array([p[1] for p in pts], dtype=float)
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
        raise ValueError("AUBC inputs must be finite")
    if np.any(np.diff(x) <= 0):
        raise ValueError("AUBC budgets must be unique and strictly increasing")
    span = x[-1] - x[0]
    return float(np.trapezoid(y, x) / span)


def _at_cap(rounds: list, cap_round: int):
    for r in rounds:
        if r["round"] == cap_round:
            return r
    return None


def map_at_cap(rounds: list, cap_round: int,
               field: str = DEFAULT_METRIC) -> float:
    r = _at_cap(rounds, cap_round)
    return float(r[field]) if r else math.nan


def per_seed_metric(results, strategy: str, metric_fn) -> dict:
    out = {}
    for sd in results.seeds:
        key = (strategy, sd)
        if key in results.by:
            out[sd] = metric_fn(results.by[key])
    return out


def paired_diffs(results, strat_a: str, strat_b: str, metric_fn,
                 cap_round: int) -> dict:
    # ragged seeds have different AUBC spans, so they are not paired observations
    ok = (set(seeds_complete(results, strat_a, cap_round)) &
          set(seeds_complete(results, strat_b, cap_round)))
    a = per_seed_metric(results, strat_a, metric_fn)
    b = per_seed_metric(results, strat_b, metric_fn)
    return {sd: a[sd] - b[sd] for sd in sorted(ok)
            if not (math.isnan(a[sd]) or math.isnan(b[sd]))}


def seeds_complete(results, strategy: str, cap_round: int) -> list:
    out = []
    for sd in results.seeds:
        key = (strategy, sd)
        if key in results.by:
            rs = {r["round"] for r in results.by[key]}
            if set(range(cap_round + 1)) <= rs:
                out.append(sd)
    return out
