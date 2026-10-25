"""Initial labeled set."""


def random_initial(items, frac, rng):
    """Uniform image-level random initial set, reproducible from ``rng``.

    Class labels are not used, as in PPAL's random initial splits. Choosing the
    initial set by ground-truth class would give every strategy an
    oracle-stratified start.
    """
    keys = sorted(key for key, _ in items)
    if len(keys) != len(set(keys)):
        raise ValueError("initial-pool items must have unique image keys")
    n = min(len(keys), max(1, round(len(keys) * frac)))
    return sorted(rng.sample(keys, n))
