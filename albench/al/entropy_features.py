"""Per-detection feature extraction for Entropy + ENMS + DivProto on YOLOv8.

This module is a YOLOv8 adaptation of Wu, Chen, and Huang (CVPR 2022),
not official author code. The paper requires one detector feature f_{i,k} for
each post-NMS detection, but does not specify an exact YOLO layer or pooling
operation.

Features come from Ultralytics' native object-feature path:

1. A forward pre-hook captures the feature maps entering the final Detect head.
2. Ultralytics NMS returns the raw prediction indices retained for each image.
3. DetectionPredictor.get_obj_feats() converts every FPN position to a common
   feature dimension and indexes the retained positions.
4. The resulting ``result.feats`` rows are aligned one-to-one with
   ``result.boxes`` rows.

This needs no letterbox inversion, ROIAlign, anchor-to-level mapping or
padding of features from different FPN levels. It requires a standard YOLOv8
detection checkpoint and an Ultralytics release whose DetectionPredictor
provides ``get_obj_feats`` (8.4.51 is pinned in requirements.txt).
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Iterator, Mapping, Sequence

import numpy as np
import torch


DEFAULT_IMAGE_SIZE: Final[int] = 640
DEFAULT_SCORE_THRESHOLD: Final[float] = 0.05
DEFAULT_NMS_IOU_THRESHOLD: Final[float] = 0.70
DEFAULT_MAX_DETECTIONS: Final[int] = 300
DEFAULT_BATCH_SIZE: Final[int] = 8
DEFAULT_T_ENMS: Final[float] = 0.5
DEFAULT_RECTANGULAR_INFERENCE: Final[bool] = False
DEFAULT_HALF_PRECISION: Final[bool] = False
DEFAULT_VERBOSE: Final[bool] = False

_FEATURE_HOOK_ATTRIBUTE: Final[str] = "_entropy_object_feature_hook"
_PREDICT_START_EVENT: Final[str] = "on_predict_start"


@dataclass(frozen=True, slots=True)
class ImageDetections:
    """Post-NMS detections and one detector feature vector per detection."""

    path: str
    boxes: np.ndarray          # shape (N, 4), xyxy in original-image pixels
    labels: np.ndarray         # shape (N,)
    confidences: np.ndarray    # shape (N,)
    features: np.ndarray       # shape (N, D)


def _validate_extraction_arguments(
    *,
    imgsz: int | tuple[int, int],
    score_threshold: float,
    nms_iou_threshold: float,
    max_detections: int,
    batch_size: int,
) -> None:
    if isinstance(imgsz, int):
        if imgsz <= 0:
            raise ValueError(f"imgsz must be positive, got {imgsz}")
    else:
        if len(imgsz) != 2 or any(int(value) <= 0 for value in imgsz):
            raise ValueError(
                "imgsz tuple must contain two positive integers, "
                f"got {imgsz}"
            )

    if not 0.0 <= score_threshold <= 1.0:
        raise ValueError(
            "score_threshold must be in [0, 1], "
            f"got {score_threshold}"
        )

    if not 0.0 <= nms_iou_threshold <= 1.0:
        raise ValueError(
            "nms_iou_threshold must be in [0, 1], "
            f"got {nms_iou_threshold}"
        )

    if max_detections <= 0:
        raise ValueError(
            "max_detections must be positive, "
            f"got {max_detections}"
        )

    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")


def _require_native_feature_support() -> None:
    """Fail early when the installed Ultralytics lacks native object features."""

    try:
        from ultralytics.models.yolo.detect import DetectionPredictor
    except ImportError as error:
        raise RuntimeError(
            "Ultralytics is not installed (see requirements.txt)"
        ) from error

    if not callable(getattr(DetectionPredictor, "get_obj_feats", None)):
        raise RuntimeError(
            "The installed Ultralytics DetectionPredictor does not provide "
            "get_obj_feats(). Install the version in requirements.txt."
        )


def _resolve_detect_head(predictor: Any) -> torch.nn.Module:
    """Return the final standard Detect head from an Ultralytics predictor."""

    try:
        from ultralytics.nn.modules.head import Detect
    except ImportError as error:
        raise RuntimeError("Unable to import Ultralytics Detect head") from error

    # Same path as Ultralytics' tracking code:
    # predictor.model (AutoBackend) -> .model (DetectionModel) -> .model[-1].
    try:
        detect_head = predictor.model.model.model[-1]
    except (AttributeError, IndexError, TypeError) as error:
        raise RuntimeError(
            "Could not locate the final Ultralytics Detect head. "
            "This extractor targets standard PyTorch YOLO detection models."
        ) from error

    if not isinstance(detect_head, Detect):
        raise TypeError(
            "The final model module must be an Ultralytics Detect head, "
            f"got {type(detect_head).__name__}."
        )

    if bool(getattr(detect_head, "end2end", False)):
        raise ValueError(
            "End-to-end/NMS-free Detect heads are not supported because the "
            "Entropy paper applies ENMS after ordinary detector NMS."
        )

    return detect_head


def _enable_native_object_features(predictor: Any) -> None:
    """Attach the Detect-input hook consumed by DetectionPredictor.postprocess."""

    _require_native_feature_support()

    previous_handle = getattr(
        predictor,
        _FEATURE_HOOK_ATTRIBUTE,
        None,
    )
    if previous_handle is not None:
        previous_handle.remove()

    detect_head = _resolve_detect_head(predictor)
    predictor._feats = None

    def pre_hook(_module: torch.nn.Module, inputs: tuple[Any, ...]) -> None:
        if not inputs:
            raise RuntimeError("Detect pre-hook received no input")

        feature_maps = inputs[0]
        if not isinstance(feature_maps, (list, tuple)):
            raise TypeError(
                "Detect input must be a list/tuple of FPN feature maps, "
                f"got {type(feature_maps).__name__}."
            )

        if not feature_maps or not all(
            torch.is_tensor(feature_map)
            for feature_map in feature_maps
        ):
            raise TypeError(
                "Every Detect input feature map must be a torch.Tensor"
            )

        # Create a new list because some model forward paths mutate lists.
        predictor._feats = list(feature_maps)

    handle = detect_head.register_forward_pre_hook(pre_hook)
    setattr(predictor, _FEATURE_HOOK_ATTRIBUTE, handle)


@contextlib.contextmanager
def _temporary_native_feature_capture(model: Any) -> Iterator[None]:
    """Install and reliably remove the temporary prediction callback/hook."""

    callbacks = getattr(model, "callbacks", None)
    if not isinstance(callbacks, dict):
        raise TypeError(
            "model must be an Ultralytics YOLO object exposing callbacks"
        )

    event_callbacks = callbacks.setdefault(_PREDICT_START_EVENT, [])
    event_callbacks.append(_enable_native_object_features)

    try:
        yield
    finally:
        # Remove only the callback installed by this module.
        try:
            event_callbacks.remove(_enable_native_object_features)
        except ValueError:
            pass

        predictor = getattr(model, "predictor", None)
        if predictor is not None:
            handle = getattr(
                predictor,
                _FEATURE_HOOK_ATTRIBUTE,
                None,
            )
            if handle is not None:
                handle.remove()
                delattr(predictor, _FEATURE_HOOK_ATTRIBUTE)

            # Prevent a later normal prediction from reusing stale feature maps.
            predictor._feats = None


def _features_to_numpy(raw_features: Any, detection_count: int) -> np.ndarray:
    """Convert ``result.feats`` to a finite float32 matrix of shape (N, D)."""

    if detection_count == 0:
        # Current Ultralytics returns [] for an empty NMS result.
        return np.zeros((0, 0), dtype=np.float32)

    if raw_features is None:
        raise RuntimeError(
            "result.feats is missing although detections exist. The installed "
            "Ultralytics feature hook/postprocess path is incompatible."
        )

    if torch.is_tensor(raw_features):
        features = raw_features.detach().float().cpu().numpy()
    else:
        features = np.asarray(raw_features, dtype=np.float32)

    if features.ndim == 1 and detection_count == 1:
        features = features.reshape(1, -1)

    features = np.asarray(features, dtype=np.float32)

    if features.ndim != 2 or features.shape[0] != detection_count:
        raise RuntimeError(
            "Feature/detection alignment mismatch: expected "
            f"({detection_count}, D), got {features.shape}."
        )

    if features.shape[1] == 0:
        raise RuntimeError("Non-empty detections produced zero-dimensional features")

    if not np.isfinite(features).all():
        raise RuntimeError("Object features contain NaN or infinity")

    return features


def extract_detection_features(
    model: Any,
    paths: Sequence[str | Path],
    *,
    imgsz: int | tuple[int, int] = DEFAULT_IMAGE_SIZE,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    nms_iou_threshold: float = DEFAULT_NMS_IOU_THRESHOLD,
    max_detections: int = DEFAULT_MAX_DETECTIONS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    rect: bool = DEFAULT_RECTANGULAR_INFERENCE,
    half: bool = DEFAULT_HALF_PRECISION,
    device: Any = None,
    verbose: bool = DEFAULT_VERBOSE,
) -> list[ImageDetections]:
    """Extract one native YOLO detector feature vector for each final box.

    The returned rows are aligned by index:

    ``labels[k]``, ``confidences[k]``, ``boxes[k]``, and ``features[k]`` all
    describe the same post-NMS detection.

    Features are not L2-normalized. ``entropy_core.enms_indices`` normalizes
    rows before the cosine, and Eq. (3) of the paper builds prototypes from the
    unnormalized vectors.
    """

    _require_native_feature_support()
    _validate_extraction_arguments(
        imgsz=imgsz,
        score_threshold=score_threshold,
        nms_iou_threshold=nms_iou_threshold,
        max_detections=max_detections,
        batch_size=batch_size,
    )

    image_paths = [str(path) for path in paths]
    if not image_paths:
        return []

    # Build a fresh DetectionPredictor and restore the previous one afterwards.
    previous_predictor = getattr(model, "predictor", None)
    model.predictor = None

    outputs: list[ImageDetections] = []

    try:
        with _temporary_native_feature_capture(model):
            for start in range(0, len(image_paths), batch_size):
                chunk_paths = image_paths[start:start + batch_size]
                results = model.predict(
                    # A list source is eagerly loaded by Ultralytics, so the
                    # explicit chunk is the actual acquisition batch bound.
                    source=chunk_paths,
                    stream=True,
                    imgsz=imgsz,
                    conf=score_threshold,
                    iou=nms_iou_threshold,
                    max_det=max_detections,
                    batch=len(chunk_paths),
                    rect=rect,
                    half=half,
                    device=device,
                    verbose=verbose,
                    save=False,
                )

                chunk_result_count = 0
                for result in results:
                    if chunk_result_count >= len(chunk_paths):
                        raise RuntimeError(
                            "Ultralytics returned more results than paths in "
                            f"acquisition chunk starting at index {start}"
                        )

                    # List sources are converted to PIL internally, where
                    # Result.path may become image0.jpg. The result order is
                    # retained, so preserve the corresponding AL pool key.
                    expected_path = chunk_paths[chunk_result_count]
                    chunk_result_count += 1

                    if result.boxes is None:
                        raise RuntimeError(
                            f"Detection result has no Boxes object: "
                            f"{expected_path}"
                        )

                    boxes = (
                        result.boxes.xyxy
                        .detach()
                        .float()
                        .cpu()
                        .numpy()
                        .astype(np.float32, copy=False)
                    )
                    labels = (
                        result.boxes.cls
                        .detach()
                        .cpu()
                        .numpy()
                        .astype(np.int64, copy=False)
                    )
                    confidences = (
                        result.boxes.conf
                        .detach()
                        .float()
                        .cpu()
                        .numpy()
                        .astype(np.float32, copy=False)
                    )

                    detection_count = len(labels)
                    features = _features_to_numpy(
                        raw_features=getattr(result, "feats", None),
                        detection_count=detection_count,
                    )

                    if not (
                        len(boxes)
                        == len(labels)
                        == len(confidences)
                        == len(features)
                    ):
                        raise RuntimeError(
                            f"Detection array mismatch for {expected_path}: "
                            f"boxes={len(boxes)}, labels={len(labels)}, "
                            f"confidences={len(confidences)}, "
                            f"features={len(features)}"
                        )

                    if not np.isfinite(confidences).all():
                        raise RuntimeError(
                            f"Non-finite confidence found for {expected_path}"
                        )

                    outputs.append(
                        ImageDetections(
                            path=expected_path,
                            boxes=boxes,
                            labels=labels,
                            confidences=confidences,
                            features=features,
                        )
                    )

                if chunk_result_count != len(chunk_paths):
                    raise RuntimeError(
                        "Ultralytics returned a different number of results "
                        f"for acquisition chunk starting at index {start}: "
                        f"expected {len(chunk_paths)}, "
                        f"received {chunk_result_count}"
                    )
    finally:
        # The context manager cleans the newly created predictor's hook first.
        model.predictor = previous_predictor

    nonempty_dimensions = {
        int(item.features.shape[1])
        for item in outputs
        if len(item.features) > 0
    }
    if len(nonempty_dimensions) > 1:
        raise RuntimeError(
            "Object feature dimensions differ across images: "
            f"{sorted(nonempty_dimensions)}"
        )

    return outputs


def build_entropy_candidates(
    detections: Sequence[ImageDetections],
    *,
    t_enms: float = DEFAULT_T_ENMS,
):
    """Turn extracted detections into EntropyCandidate objects."""

    from .entropy_core import make_entropy_candidate

    return [
        make_entropy_candidate(
            path=item.path,
            labels=item.labels,
            confidences=item.confidences,
            features=item.features,
            t_enms=t_enms,
        )
        for item in detections
    ]


def extract_entropy_candidates(
    model: Any,
    paths: Sequence[str | Path],
    *,
    t_enms: float = DEFAULT_T_ENMS,
    imgsz: int | tuple[int, int] = DEFAULT_IMAGE_SIZE,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    nms_iou_threshold: float = DEFAULT_NMS_IOU_THRESHOLD,
    max_detections: int = DEFAULT_MAX_DETECTIONS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    rect: bool = DEFAULT_RECTANGULAR_INFERENCE,
    half: bool = DEFAULT_HALF_PRECISION,
    device: Any = None,
    verbose: bool = DEFAULT_VERBOSE,
):
    """Run YOLO inference, extract features and build EntropyCandidate objects."""

    detections = extract_detection_features(
        model=model,
        paths=paths,
        imgsz=imgsz,
        score_threshold=score_threshold,
        nms_iou_threshold=nms_iou_threshold,
        max_detections=max_detections,
        batch_size=batch_size,
        rect=rect,
        half=half,
        device=device,
        verbose=verbose,
    )

    return build_entropy_candidates(
        detections,
        t_enms=t_enms,
    )


def extract_entropy_candidates_from_config(
    model: Any,
    paths: Sequence[str | Path],
    cfg: Mapping[str, Any],
):
    """Extract candidates with the settings in ``cfg["al"]["entropy"]``.

    ``al.imgsz`` and ``al.acq_batch`` set the input size and batch. ``rect``,
    ``half`` and ``verbose`` may also be set under ``al.entropy`` and default
    to the constants at the top of this module.
    """

    try:
        al_cfg = cfg["al"]
        entropy_cfg = al_cfg["entropy"]
        imgsz = al_cfg["imgsz"]
    except (KeyError, TypeError) as error:
        raise ValueError(
            "cfg must contain al.imgsz and an al.entropy mapping"
        ) from error

    if not isinstance(al_cfg, Mapping):
        raise TypeError("cfg['al'] must be a mapping")

    if not isinstance(entropy_cfg, Mapping):
        raise TypeError("cfg['al']['entropy'] must be a mapping")

    from .device import device as resolve_device

    return extract_entropy_candidates(
        model=model,
        paths=paths,
        t_enms=float(
            entropy_cfg.get("t_enms", DEFAULT_T_ENMS)
        ),
        imgsz=imgsz,
        score_threshold=float(
            entropy_cfg.get("score_thr", DEFAULT_SCORE_THRESHOLD)
        ),
        nms_iou_threshold=float(
            entropy_cfg.get("nms_iou", DEFAULT_NMS_IOU_THRESHOLD)
        ),
        max_detections=int(
            entropy_cfg.get("max_det", DEFAULT_MAX_DETECTIONS)
        ),
        batch_size=int(
            entropy_cfg.get(
                "batch_size",
                al_cfg.get("acq_batch", DEFAULT_BATCH_SIZE),
            )
        ),
        rect=bool(
            entropy_cfg.get("rect", DEFAULT_RECTANGULAR_INFERENCE)
        ),
        half=bool(
            entropy_cfg.get("half", DEFAULT_HALF_PRECISION)
        ),
        device=resolve_device(),
        verbose=bool(
            entropy_cfg.get("verbose", DEFAULT_VERBOSE)
        ),
    )
