"""YOLOv8 adapter for PPAL CCMS classification-probability features.

YOLOv8 uses a separate classification branch at each detection scale, so its
neck channel vectors do not share one comparable feature space.  PPAL uses
class-posterior vectors with KL divergence for detectors with that property.
"""
import numpy as np


def conditional_class_distribution(scores, eps: float = 1e-12) -> np.ndarray:
    """Normalize YOLO's independent sigmoid scores for PPAL's KL adapter."""
    values = np.asarray(scores, dtype=float)
    if values.ndim != 2:
        raise ValueError("PPAL class scores must have shape (detections, classes)")
    if not len(values):
        return values.copy()
    if np.any(values < 0.0) or not np.all(np.isfinite(values)):
        raise ValueError("PPAL class scores must be finite and non-negative")
    denominator = values.sum(axis=1, keepdims=True)
    return values / np.maximum(denominator, eps)


def extract_det_features(model, paths: list, cfg: dict):
    """Return per-detection class distributions, labels, and confidence."""
    from .device import device
    from .score_predictor import ScorePredictor

    al = cfg["al"]
    pp = al["ppal"]
    batch = al["acq_batch"]
    max_det = pp["diversity_max_det"]
    features, labels, scores = [], [], []
    model.predictor = None
    try:
        for start in range(0, len(paths), batch):
            chunk = paths[start:start + batch]
            results = model.predict(
                chunk, predictor=ScorePredictor, conf=pp["score_thr"],
                iou=pp["nms_iou"], imgsz=al["imgsz"], device=device(),
                rect=False, batch=batch, stream=True, max_det=max_det,
                verbose=False)
            for result in results:
                raw = np.asarray(
                    result.cls_scores.detach().cpu().numpy(), dtype=float)
                probabilities = conditional_class_distribution(raw)
                boxes = result.boxes
                features.append(probabilities)
                labels.append(boxes.cls.detach().cpu().numpy().astype(int))
                scores.append(boxes.conf.detach().cpu().numpy().astype(float))
    finally:
        model.predictor = None
    return features, labels, scores
