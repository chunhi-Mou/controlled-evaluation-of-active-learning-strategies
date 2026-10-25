"""PPAL (Yang et al., CVPR 2024, arXiv:2211.11612) for YOLOv8.

Stage 1 ranks the pool by DCUS uncertainty and keeps the top delta * budget
images. Stage 2 selects the query from them by k-medoids on CCMS distances.
"""
import numpy as np

from .ppal_difficulty import reweight
from .ccms import image_distance_matrix, kmedoids


def box_entropy(p) -> float:
    """Binary entropy of a box's top class score, in nats.

    YOLOv8 scores each class with an independent sigmoid, so the scores do not
    form a distribution over classes. The binary entropy of the top score is
    the uncertainty of the box's predicted class.
    """
    s = float(np.clip(np.max(np.asarray(p, dtype=float)), 1e-10, 1 - 1e-10))
    return float(-(s * np.log(s) + (1 - s) * np.log(1 - s)))


def _real_feature_fn(model, paths, cfg):
    from .ppal_features import extract_det_features
    return extract_det_features(model, paths, cfg)


def _stage2_with_empty_fallback(candidates, feats, labels, scores, budget,
                                score_thr, kmeans_iter, rng):
    """Run CCMS only where Eq. 7 has a non-empty detection set.

    In DCUS (Eq. 5) an image with no detections has zero uncertainty. CCMS
    (Eq. 7) divides by the sum of detection scores and is undefined for such an
    image. These images keep their Stage-1 score of zero, are left out of CCMS,
    and fill the query by seeded random sampling only if there are fewer valid
    CCMS candidates than the budget.
    """
    n = len(candidates)
    if not (len(feats) == len(labels) == len(scores) == n):
        raise ValueError("ppal: Stage-2 feature outputs must match candidates")
    if not 0 <= budget <= n:
        raise ValueError(f"ppal: budget={budget} outside 0..{n}")

    valid = [i for i, f in enumerate(feats) if len(f) > 0]
    valid_set = set(valid)
    empty = [i for i in range(n) if i not in valid_set]
    vf = [feats[i] for i in valid]
    vl = [labels[i] for i in valid]
    vs = [scores[i] for i in valid]
    D = image_distance_matrix(vf, vl, vs, score_thr)

    n_ccms = min(budget, len(valid))
    if n_ccms == len(valid):
        pick = list(range(n_ccms))
    elif n_ccms:
        pick = kmedoids(D, n_ccms, kmeans_iter, rng)
    else:
        pick = []
    selected = [candidates[valid[i]] for i in pick]

    n_fill = budget - len(selected)
    fallback_pool = sorted(candidates[i] for i in empty)
    if n_fill > len(fallback_pool):
        raise RuntimeError(
            f"ppal: {len(valid)} CCMS candidates + {len(fallback_pool)} "
            f"zero-detection candidates cannot fill budget {budget}")
    selected.extend(rng.sample(fallback_pool, n_fill))
    stats = {"n_ccms_candidates": len(valid),
             "n_no_detection": len(empty),
             "n_ccms_picks": len(pick),
             "n_random_fill": n_fill}
    return sorted(selected), D, pick, stats


def select(model, candidates, labeled, budget, cfg, rng, *,
           feature_fn=_real_feature_fn):
    if budget <= 0 or model is None:
        return []
    from .score_predictor import ScorePredictor
    from .device import device
    pp = cfg["al"]["ppal"]
    imgsz = cfg["al"]["imgsz"]
    bs = cfg["al"]["acq_batch"]
    max_det = pp["uncertainty_max_det"]
    pool = sorted(set(candidates) - set(labeled))
    cq = getattr(model, "_ppal_class_quality", None)
    if cq is None:
        raise RuntimeError("ppal: model has no _ppal_class_quality "
                           "(quality-EMA callback not run)")
    w = reweight(np.asarray(cq, float), pp["alpha"], pp["weight_ub"])

    # Stage 1: DCUS uncertainty, class-reweighted binary entropy summed per image
    model.predictor = None                  # so predict() installs ScorePredictor
    scored = []
    # bound the source list (Ultralytics eagerly converts list sources)
    for start in range(0, len(pool), bs):
        paths = pool[start:start + bs]
        # rect=False => square imgsz => scores batch-invariant
        results = model.predict(
            paths, predictor=ScorePredictor, conf=pp["score_thr"],
            iou=pp["nms_iou"], imgsz=imgsz, device=device(), rect=False,
            batch=bs, stream=True, max_det=max_det, verbose=False)
        for path, res in zip(paths, results):
            tot = 0.0
            for row in res.cls_scores:
                p = np.asarray(row.detach().cpu().numpy(), float)
                tot += box_entropy(p) * float(w[int(p.argmax())])
            scored.append((tot, path))
    scored.sort(key=lambda t: (-t[0], t[1]))
    pool_n = min(pp["delta"] * budget, len(scored))
    cand = sorted(p for _, p in scored[:pool_n])

    # Stage 2: CCMS diversity
    feats, labels, scores = feature_fn(model, cand, cfg)
    selected, D, pick, fallback = _stage2_with_empty_fallback(
        cand, feats, labels, scores, min(budget, len(cand)),
        pp["score_thr"], pp["kmeans_iter"], rng)
    if fallback["n_no_detection"]:
        import warnings
        warnings.warn(
            "PPAL Stage 2 excluded "
            f"{fallback['n_no_detection']} zero-detection candidate(s), "
            f"seeded-random fill={fallback['n_random_fill']}",
            RuntimeWarning, stacklevel=2)
    # Acquisition diagnostics
    from .ppal_stage2 import acquisition_diag
    diag = acquisition_diag(
        D, pick, np.array([t for t, _ in scored], float), pool_n)
    diag.update({k: np.int64(v) for k, v in fallback.items()})
    model._ppal_acq_diag = diag
    return selected
