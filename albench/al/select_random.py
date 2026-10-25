"""Random sampling: a uniform draw without replacement from the unlabeled pool
(Settles, 2009)."""


def select(model, candidates, labeled, budget, cfg, rng):
    if budget <= 0:
        return []
    pool = sorted(set(candidates) - set(labeled))
    return sorted(rng.sample(pool, budget))
