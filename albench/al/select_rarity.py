"""Rarity-aware stratified active learning (Benhafid and Selouani, Applied
Sciences 2026, 16(3), 1236) for YOLOv8.

Each pool image gets a composite score from five max-normalized terms:
difficulty D, information I, class diversity C, box dispersion G and embedding
diversity E (Eqs. 1-12). A share rho of the query is filled by per-class quotas
that favour classes rare in the labeled set (Eqs. 13-17), and the rest by
score rank.
"""
from dataclasses import dataclass
import warnings

import numpy as np

from .dataset import image_class_counts


@dataclass(frozen=True)
class Prediction:
    """Post-inference outputs for one image, aligned by detection."""

    classes: np.ndarray
    confidence: np.ndarray
    boxes_xyxy: np.ndarray


def bernoulli_entropy(p, eps: float = 1e-12) -> np.ndarray:
    """Paper Eq. (1), using log2 as in the paper's supplementary metadata."""
    p = np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)
    return -(p * np.log2(p) + (1.0 - p) * np.log2(1.0 - p))


def rarity_weights(frequencies, beta: float, alpha_min: float,
                   alpha_max: float, eps: float = 1e-12) -> np.ndarray:
    """Pool-prediction rarity weights from paper Eq. (3)."""
    freq = np.asarray(frequencies, dtype=float)
    if freq.ndim != 1:
        raise ValueError("rarity: class frequencies must be one-dimensional")
    median = float(np.median(freq)) if len(freq) else 0.0
    raw = (median / (freq + eps)) ** beta
    return np.clip(raw, alpha_min, alpha_max)


def _geometric_dispersion(boxes_xyxy: np.ndarray) -> float:
    """Paper Eqs. (8)-(9), including the zero self-pairs in N^2."""
    boxes = np.asarray(boxes_xyxy, dtype=float).reshape(-1, 4)
    if len(boxes) <= 1:
        return 0.0
    centers = 0.5 * (boxes[:, :2] + boxes[:, 2:])
    delta = centers[:, None, :] - centers[None, :, :]
    return float(np.sqrt(np.sum(delta * delta, axis=2)).mean())


def image_indicators(prediction: Prediction, alpha: np.ndarray,
                     gamma: float, nc: int) -> tuple[float, float, float, float]:
    """Raw D, I, C and G indicators from paper Eqs. (4)-(9)."""
    cls = np.asarray(prediction.classes, dtype=int)
    conf = np.asarray(prediction.confidence, dtype=float)
    boxes = np.asarray(prediction.boxes_xyxy, dtype=float).reshape(-1, 4)
    if not (len(cls) == len(conf) == len(boxes)):
        raise ValueError("rarity: classes/confidence/boxes are not aligned")
    if not len(conf):
        return 0.0, 0.0, 0.0, 0.0
    if np.any((cls < 0) | (cls >= nc)):
        raise ValueError("rarity: predicted class outside configured range")

    # Eqs. (2), (4), and (5).
    focal = (1.0 - conf) ** gamma
    weight = alpha[cls]
    difficult = float(np.sum(weight * focal * bernoulli_entropy(conf)))
    information = float(np.sum(weight * focal * conf))

    # Eqs. (6)-(7).
    counts = np.bincount(cls, minlength=nc).astype(float)
    q = counts / counts.sum()
    class_diversity = float(1.0 - np.sum(q * q))
    geometry = _geometric_dispersion(boxes)
    return difficult, information, class_diversity, geometry


def _max_normalize(values) -> np.ndarray:
    """Paper Eq. (11)."""
    values = np.asarray(values, dtype=float)
    maximum = float(np.max(values)) if len(values) else 0.0
    return values / maximum if maximum > 0.0 else np.zeros_like(values)


def embedding_diversity(embeddings: np.ndarray, n_clusters: int,
                        iterations: int, seed: int) -> np.ndarray:
    """Nearest k-means++ centroid distance from paper Eq. (10)."""
    from scipy.cluster.vq import kmeans2
    from scipy.spatial.distance import cdist

    z = np.asarray(embeddings, dtype=float)
    if z.ndim != 2:
        raise ValueError("rarity: embeddings must have shape (images, dims)")
    if not len(z):
        return np.zeros(0, dtype=float)
    if not np.all(np.isfinite(z)):
        raise ValueError("rarity: embeddings contain NaN or infinity")
    norm = np.linalg.norm(z, axis=1, keepdims=True)
    z = np.divide(z, norm, out=np.zeros_like(z), where=norm > 0)
    n_unique = len(np.unique(z, axis=0))
    k = min(max(1, int(n_clusters)), n_unique)
    if k == n_unique:
        return np.zeros(len(z), dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        centers, _ = kmeans2(
            z, k, iter=max(1, int(iterations)), minit="++", missing="warn",
            check_finite=False, seed=int(seed))
    return cdist(z, centers, metric="euclidean").min(axis=1)


def composite_scores(predictions: list[Prediction], embeddings: np.ndarray,
                     nc: int, budget: int, params: dict, seed: int):
    """Paper Eqs. (1)-(12). Returns the scores and their components."""
    frequencies = np.zeros(nc, dtype=float)
    for pred in predictions:
        cls = np.asarray(pred.classes, dtype=int)
        if len(cls):
            frequencies += np.bincount(cls, minlength=nc)[:nc]
    alpha = rarity_weights(
        frequencies, params["beta"], params["alpha_min"],
        params["alpha_max"], params.get("eps", 1e-12))
    raw = np.asarray([
        image_indicators(pred, alpha, params["gamma"], nc)
        for pred in predictions
    ], dtype=float)
    embed = embedding_diversity(
        embeddings, budget, params["kmeans_iter"], seed)
    if len(raw) != len(embed):
        raise ValueError("rarity: prediction and embedding counts differ")
    components = np.column_stack([
        *(_max_normalize(raw[:, i]) for i in range(4)),
        _max_normalize(embed),
    ])
    return components.sum(axis=1), components, frequencies, alpha


def labeled_instance_counts(labeled, nc: int) -> np.ndarray:
    """Ground-truth instance counts in L_t, as required by paper Eq. (14)."""
    counts = np.zeros(nc, dtype=float)
    for image in sorted(set(labeled)):
        counts += np.asarray(image_class_counts(image, nc), dtype=float)
    return counts


def _integer_quotas(weights: np.ndarray, total: int) -> np.ndarray:
    """Eq. (15) plus its stated integer adjustment to an exact total."""
    weights = np.asarray(weights, dtype=float)
    quotas = np.zeros(len(weights), dtype=int)
    if total <= 0 or not len(weights):
        return quotas
    positive = np.flatnonzero(weights > 0)
    if not len(positive):
        return quotas
    ideal = total * weights[positive] / weights[positive].sum()
    q = np.floor(ideal + 0.5).astype(int)
    diff = int(total - q.sum())
    residual = ideal - q
    if diff > 0:
        order = sorted(range(len(q)), key=lambda i: (-residual[i],
                                                      -weights[positive[i]],
                                                      positive[i]))
        for i in range(diff):
            q[order[i % len(order)]] += 1
    elif diff < 0:
        order = sorted(range(len(q)), key=lambda i: (residual[i],
                                                      weights[positive[i]],
                                                      positive[i]))
        need = -diff
        while need:
            changed = False
            for i in order:
                if q[i] > 0:
                    q[i] -= 1
                    need -= 1
                    changed = True
                    if not need:
                        break
            if not changed:
                break
    quotas[positive] = q
    if quotas.sum() != total:
        raise RuntimeError("rarity: integer quota adjustment failed")
    return quotas


def stratified_query(paths: list[str], predictions: list[Prediction],
                     scores: np.ndarray, alpha: np.ndarray,
                     labeled_counts: np.ndarray, budget: int, rho: float,
                     beta: float):
    """Paper Eqs. (13)-(17) and dominant/presence routing in Section 3.3."""
    if not (len(paths) == len(predictions) == len(scores)):
        raise ValueError("rarity: paths/predictions/scores are not aligned")
    if not 0 <= budget <= len(paths):
        raise ValueError(f"rarity: budget={budget} outside 0..{len(paths)}")
    nc = len(alpha)
    k_strat = int(np.floor(rho * budget))
    k_free = budget - k_strat

    median_lab = float(np.median(labeled_counts)) if nc else 0.0
    effective = alpha * (median_lab / (labeled_counts + 1.0)) ** beta
    # Eq. (15) is defined only for classes present in the predicted pool.
    present = np.zeros(nc, dtype=bool)
    for pred in predictions:
        if len(pred.classes):
            present[np.asarray(pred.classes, dtype=int)] = True
    effective = np.where(present, effective, 0.0)
    # A zero median can occur in an extremely small L0.  Preserve the same
    # rarity ordering using alpha so the stated quota remains computable.
    if k_strat and effective.sum() <= 0 and present.any():
        effective = np.where(present, alpha, 0.0)
    quotas = _integer_quotas(effective, k_strat) \
        if effective.sum() > 0 else np.zeros(nc, dtype=int)

    ranked = sorted(range(len(paths)), key=lambda i: (-scores[i], paths[i]))
    class_sets = [set(np.asarray(p.classes, dtype=int).tolist())
                  for p in predictions]
    dominant = []
    for pred in predictions:
        cls = np.asarray(pred.classes, dtype=int)
        dominant.append(int(np.bincount(cls, minlength=nc).argmax())
                        if len(cls) else None)

    selected: list[int] = []
    selected_set: set[int] = set()
    class_order = sorted(np.flatnonzero(quotas),
                         key=lambda c: (-effective[c], int(c)))
    for c in class_order:
        need = int(quotas[c])
        for rule in (lambda i, c=c: dominant[i] == c,
                     lambda i, c=c: c in class_sets[i]):
            for i in ranked:
                if need <= 0:
                    break
                if i not in selected_set and rule(i):
                    selected.append(i)
                    selected_set.add(i)
                    need -= 1
            if need <= 0:
                break

    # Redistribute unavailable quotas toward higher effective weights.
    for c in sorted(np.flatnonzero(effective),
                    key=lambda c: (-effective[c], int(c))):
        if len(selected) >= k_strat:
            break
        for i in ranked:
            if i not in selected_set and c in class_sets[i]:
                selected.append(i)
                selected_set.add(i)
                if len(selected) >= k_strat:
                    break

    routed = len(selected)
    # The supplement has 300 detections/image.  If YOLOv8 cannot route all
    # stratified slots, transfer only the unavailable slots to free TopK.  No
    # unlabeled ground truth is consulted and Q still has the exact budget.
    free_target = budget - routed
    free: list[int] = []
    for i in ranked:
        if len(free) >= free_target:
            break
        if i not in selected_set:
            free.append(i)
            selected_set.add(i)
    final = selected + free
    if len(final) != budget:
        raise RuntimeError(f"rarity: selected {len(final)} of {budget} images")
    diag = {
        "scores": np.asarray(scores, dtype=float),
        "alpha": np.asarray(alpha, dtype=float),
        "effective_weights": np.asarray(effective, dtype=float),
        "quotas": np.asarray(quotas, dtype=int),
        "k_strat": np.int64(k_strat),
        "k_free": np.int64(k_free),
        "k_free_actual": np.int64(free_target),
        "n_unroutable_fill": np.int64(k_strat - routed),
    }
    return sorted(paths[i] for i in final), diag


def _predict_pool(model, paths: list[str], cfg: dict) -> list[Prediction]:
    from .device import device

    al = cfg["al"]
    params = al["rarity_aware"]
    batch = al["acq_batch"]
    output: list[Prediction] = []
    model.predictor = None
    try:
        for start in range(0, len(paths), batch):
            chunk = paths[start:start + batch]
            results = model.predict(
                chunk, conf=params["score_thr"], iou=params["iou"],
                imgsz=al["imgsz"],
                device=device(), rect=False, batch=batch, stream=True,
                max_det=al["max_det"], verbose=False)
            for result in results:
                boxes = result.boxes
                xyxy = boxes.xyxy.detach().cpu().numpy().astype(float)
                # Paper geometry is computed in detector-input coordinates.
                # Ultralytics returns original-image coordinates, so reproduce
                # the square letterbox transform used by rect=False.
                original_h, original_w = result.orig_shape
                size = float(al["imgsz"])
                gain = size / max(float(original_h), float(original_w))
                pad_x = (size - float(original_w) * gain) / 2.0
                pad_y = (size - float(original_h) * gain) / 2.0
                xyxy = xyxy * gain + np.asarray(
                    [pad_x, pad_y, pad_x, pad_y], dtype=float)
                output.append(Prediction(
                    boxes.cls.detach().cpu().numpy().astype(int),
                    boxes.conf.detach().cpu().numpy().astype(float),
                    xyxy))
    finally:
        model.predictor = None
    return output


def _embed_pool(model, paths: list[str], cfg: dict) -> np.ndarray:
    """Use the detector-exposed image embeddings required by paper Eq. (10)."""
    from .device import device

    al = cfg["al"]
    batch = al["acq_batch"]
    output = []
    model.predictor = None
    try:
        for start in range(0, len(paths), batch):
            chunk = paths[start:start + batch]
            tensors = model.embed(
                chunk, imgsz=al["imgsz"], device=device(), rect=False,
                batch=batch, verbose=False)
            output.extend(np.asarray(t.detach().cpu().numpy(), dtype=float)
                          for t in tensors)
    finally:
        model.predictor = None
    if not output:
        return np.zeros((0, 0), dtype=float)
    return np.vstack(output)


def select(model, candidates, labeled, budget, cfg, rng, *,
           prediction_fn=_predict_pool, embedding_fn=_embed_pool):
    """Select ``budget`` images, with the same interface as the other selectors."""
    if budget <= 0 or model is None:
        return []
    pool = sorted(set(candidates) - set(labeled))
    nc = cfg["classes"]["nc"]
    params = cfg["al"]["rarity_aware"]
    predictions = prediction_fn(model, pool, cfg)
    embeddings = embedding_fn(model, pool, cfg)
    cluster_seed = rng.randrange(0, 2**32)
    scores, components, frequencies, alpha = composite_scores(
        predictions, embeddings, nc, budget, params, cluster_seed)
    counts = labeled_instance_counts(labeled, nc)
    picks, diag = stratified_query(
        pool, predictions, scores, alpha, counts, budget,
        params["rho"], params["beta"])
    diag.update({
        "components": components,
        "pool_frequencies": frequencies,
        "labeled_counts": counts,
        "n_no_detection": np.int64(sum(not len(p.classes)
                                        for p in predictions)),
    })
    model._rarity_acq_diag = diag
    if diag["n_no_detection"]:
        warnings.warn(
            f"rarity-aware: {int(diag['n_no_detection'])} pool image(s) "
            "had no detections. Their D/I/C/G components are zero and the "
            "deterministic budget guard may route them by composite score",
            RuntimeWarning, stacklevel=2)
    return picks
