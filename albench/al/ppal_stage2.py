"""PPAL acquisition diagnostics. Not used for selection."""
import numpy as np

NULL_SEED = 20260610          # null-draw seed, separate from the run seeds
N_NULL = 64                   # null subsets per round
_QS = (0.0, 5.0, 25.0, 50.0, 75.0, 95.0, 100.0)


def coverage(D: np.ndarray, subset) -> float:
    # k-Center cost: mean over candidates of nearest-pick distance
    s = np.asarray(list(subset), int)
    return float(D[:, s].min(axis=1).mean())


def spread(D: np.ndarray, subset) -> float:
    # min pairwise distance among picks (lower = picks clumped)
    s = np.asarray(list(subset), int)
    if s.size < 2:
        return 0.0
    sub = D[np.ix_(s, s)].copy()
    np.fill_diagonal(sub, np.inf)
    return float(sub.min())


def acquisition_diag(D: np.ndarray, pick_idx, stage1_scores,
                     pool_n: int, *, n_null: int = N_NULL,
                     null_seed: int = NULL_SEED) -> dict:
    # coverage/spread of the picks vs N_NULL random subsets, packed for savez
    # stage1_scores = whole pool sorted desc, cut at pool_n
    D = np.asarray(D, float)
    pick = np.asarray(list(pick_idx), int)
    n = D.shape[0]
    k = pick.size
    if n == 0 or k == 0:
        nan_null = np.full(n_null, np.nan, float)
        return {
            "coverage_picked": np.float64(np.nan),
            "spread_picked": np.float64(np.nan),
            "coverage_null": nan_null.copy(),
            "spread_null": nan_null.copy(),
            "d_quantiles": np.full(len(_QS), np.nan, float),
            "d_quantile_qs": np.asarray(_QS, float),
            "n_cand": np.int64(n), "k": np.int64(k),
            "n_null": np.int64(n_null), "null_seed": np.int64(null_seed),
            "stage1_scores": np.asarray(stage1_scores, np.float32),
            "pool_n": np.int64(pool_n),
        }
    rng = np.random.RandomState(null_seed)
    cov_null = np.empty(n_null, float)
    spr_null = np.empty(n_null, float)
    for i in range(n_null):
        sub = rng.choice(n, size=min(k, n), replace=False)
        cov_null[i] = coverage(D, sub)
        spr_null[i] = spread(D, sub)
    off = D[~np.eye(n, dtype=bool)] if n > 1 else np.zeros(1)
    return {
        "coverage_picked": np.float64(coverage(D, pick)),
        "spread_picked": np.float64(spread(D, pick)),
        "coverage_null": cov_null,
        "spread_null": spr_null,
        "d_quantiles": np.percentile(off, _QS).astype(float),
        "d_quantile_qs": np.asarray(_QS, float),
        "n_cand": np.int64(n),
        "k": np.int64(k),
        "n_null": np.int64(n_null),
        "null_seed": np.int64(null_seed),
        "stage1_scores": np.asarray(stage1_scores, np.float32),
        "pool_n": np.int64(pool_n),
    }
