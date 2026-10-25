"""PPAL CCMS diversity (Yang et al., CVPR 2024). Pure NumPy."""
import numpy as np


def _probabilities(values: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim != 2:
        raise ValueError("CCMS features must have shape (detections, classes)")
    if not len(values):
        return values
    if np.any(values < 0.0) or not np.all(np.isfinite(values)):
        raise ValueError("CCMS probabilities must be finite and non-negative")
    return values / np.maximum(values.sum(axis=1, keepdims=True), eps)


def _directional_kl(source: np.ndarray, target: np.ndarray,
                    source_scores: np.ndarray, target_scores: np.ndarray,
                    score_thr: float, eps: float = 1e-12) -> float:
    """PPAL directional nearest-match distance using KL(target || source)."""
    source_valid = np.asarray(source_scores, dtype=float) > score_thr
    target_valid = np.asarray(target_scores, dtype=float) > score_thr
    if not source_valid.any() or not target_valid.any():
        return 0.0
    p = np.clip(source[source_valid], eps, None)
    q = np.clip(target[target_valid], eps, None)
    # KL(q || p) = sum(q log q) - log(p) @ q.  This matrix form avoids a
    # (source detections x target detections x classes) temporary allocation.
    q_log_q = np.sum(q * np.log(q), axis=1)
    pairwise = q_log_q[None, :] - np.log(p) @ q.T
    pairwise = np.maximum(pairwise, 0.0)  # remove negative round-off only
    nearest = pairwise.min(axis=1)
    weights = np.asarray(source_scores, dtype=float)[source_valid]
    return float(np.sum(nearest * weights) / (np.sum(weights) + 1e-5))


def image_distance_matrix(feats: list, labels: list, scores: list,
                          score_thr: float) -> np.ndarray:
    """PPAL Eq. (6)-(7), using its SSD/class-posterior KL branch.

    ``labels`` is kept for diagnostics only. PPAL's KL branch matches
    detections across all predicted labels (same_label=False).
    """
    N = len(feats)
    if not (len(labels) == len(scores) == N):
        raise ValueError("CCMS features/labels/scores are not aligned")
    nf = [_probabilities(np.asarray(f, dtype=float)) for f in feats]
    sc = [np.asarray(s, float) for s in scores]
    directional = np.zeros((N, N), dtype=float)
    for i in range(N):
        if len(nf[i]) != len(sc[i]):
            raise ValueError("CCMS features and scores differ in length")
        for j in range(N):
            if i != j and len(nf[i]) and len(nf[j]):
                directional[i, j] = _directional_kl(
                    nf[i], nf[j], sc[i], sc[j], score_thr)
    # The public PPAL code sets KL > 2 to 0. KL is unbounded and values above
    # 2 are valid distances, so they are kept here.
    return 0.5 * (directional + directional.T)


def kcenter_greedy_matrix(D: np.ndarray, K: int, rng) -> list:
    # random first center, then farthest-point greedy on D
    N = D.shape[0]
    centroids = [rng.randrange(N)]
    while len(centroids) < K:
        md = D[:, centroids].min(axis=1)
        md[centroids] = -1.0
        centroids.append(int(np.argmax(md)))
    return centroids


def kmedoids(D: np.ndarray, K: int, n_iter: int, rng) -> list:
    # k-Center greedy init, then n_iter Lloyd-style medoid updates
    N = D.shape[0]
    if D.shape != (N, N):
        raise ValueError(f"kmedoids: expected a square matrix, got {D.shape}")
    if not 0 < K <= N:
        raise ValueError(f"kmedoids: K={K} outside 1..{N}")
    centroids = np.array(kcenter_greedy_matrix(D, K, rng))
    idx = np.arange(N)
    for _ in range(n_iter):
        assign = np.argmin(D[:, centroids], axis=1)
        # Equal distances are common when two images have no category in
        # common.  Keep every medoid in its own cluster so a tie cannot make
        # the other clusters empty (the official implementation asserts).
        assign[centroids] = np.arange(K)
        new = []
        for k in range(K):
            members = idx[assign == k]
            assert len(members) >= 1
            sub = D[np.ix_(members, members)]
            new.append(int(members[np.argmin(sub.sum(axis=1))]))
        centroids = np.array(new)
    return centroids.tolist()
