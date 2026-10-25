"""Entropy-ENMS-DivProto acquisition (Wu et al., CVPR 2022) for YOLOv8."""

from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Final

import numpy as np

from albench.data.labels import label_for

from .entropy_core import EntropyCandidate, divproto_select


YOLO_LABEL_FIELD_COUNT: Final[int] = 5


def labeled_instance_counts(
    labeled: Iterable[str | Path],
    nc: int,
) -> np.ndarray:
    """Count boxes per class using ground truth from the labeled set only."""

    if nc <= 0:
        raise ValueError(f"nc must be positive, got {nc}")

    counts = np.zeros(nc, dtype=np.int64)

    for image_path in sorted({str(path) for path in labeled}):
        label_path = label_for(image_path)

        # YOLO permits background-only images without a label file.
        if not label_path.exists():
            continue

        for line_number, raw_line in enumerate(
            label_path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            line = raw_line.strip()
            if not line:
                continue

            fields = line.split()
            if len(fields) != YOLO_LABEL_FIELD_COUNT:
                raise ValueError(
                    f"{label_path}:{line_number}: expected "
                    f"{YOLO_LABEL_FIELD_COUNT} YOLO fields, "
                    f"got {len(fields)}"
                )

            try:
                class_value = float(fields[0])
            except ValueError as error:
                raise ValueError(
                    f"{label_path}:{line_number}: invalid class id "
                    f"{fields[0]!r}"
                ) from error

            if not np.isfinite(class_value) or not class_value.is_integer():
                raise ValueError(
                    f"{label_path}:{line_number}: class id must be a finite "
                    f"integer, got {fields[0]!r}"
                )

            class_id = int(class_value)
            if not 0 <= class_id < nc:
                raise ValueError(
                    f"{label_path}:{line_number}: class id {class_id} is "
                    f"outside [0, {nc})"
                )

            counts[class_id] += 1

    return counts


def _real_candidate_fn(
    model,
    paths: Sequence[str],
    cfg: dict,
) -> list[EntropyCandidate]:
    _validate_model_classes(model, cfg)
    from .entropy_features import extract_entropy_candidates_from_config

    return extract_entropy_candidates_from_config(
        model=model,
        paths=paths,
        cfg=cfg,
    )


def _validate_model_classes(model, cfg: dict) -> None:
    expected = list(cfg["classes"]["names"])
    raw_names = getattr(model, "names", None)
    if isinstance(raw_names, dict):
        try:
            actual = [raw_names[index] for index in range(len(raw_names))]
        except KeyError as error:
            raise RuntimeError("Checkpoint class ids are not contiguous") from error
    elif isinstance(raw_names, (list, tuple)):
        actual = list(raw_names)
    else:
        raise RuntimeError("Entropy selector cannot read checkpoint class names")
    if actual != expected:
        raise RuntimeError(
            "Entropy checkpoint classes do not match the benchmark: "
            f"checkpoint={actual}, expected={expected}"
        )


def _validate_candidate_summaries(
    pool: Sequence[str],
    summaries: Sequence[EntropyCandidate],
    nc: int,
) -> list[EntropyCandidate]:
    expected_paths = set(pool)
    actual_paths = [item.path for item in summaries]

    if len(actual_paths) != len(set(actual_paths)):
        raise RuntimeError(
            "Entropy feature extraction returned duplicate image paths"
        )

    actual_path_set = set(actual_paths)
    missing = sorted(expected_paths - actual_path_set)
    extra = sorted(actual_path_set - expected_paths)

    if missing or extra:
        raise RuntimeError(
            "Entropy feature extraction did not preserve the unlabeled pool: "
            f"missing={missing[:3]}, extra={extra[:3]}"
        )

    for item in summaries:
        if not np.isfinite(item.score) or item.score < 0:
            raise RuntimeError(
                f"Entropy candidate {item.path} has invalid score {item.score}"
            )
        prototype_classes = set(item.prototypes)
        presence_classes = set(item.presence)
        if prototype_classes != presence_classes:
            raise RuntimeError(
                f"Entropy candidate {item.path} has inconsistent prototype "
                "and presence classes"
            )
        invalid_classes = sorted(
            repr(class_id)
            for class_id in prototype_classes
            if not isinstance(class_id, (int, np.integer))
            or not 0 <= int(class_id) < nc
        )
        if invalid_classes:
            raise RuntimeError(
                f"Entropy candidate {item.path} contains class ids outside "
                f"[0, {nc}): {invalid_classes}"
            )
        for class_id, prototype in item.prototypes.items():
            value = np.asarray(prototype, dtype=float)
            if value.ndim != 1 or value.size == 0 or not np.isfinite(value).all():
                raise RuntimeError(
                    f"Entropy candidate {item.path} class {class_id} has an "
                    "invalid prototype"
                )
            if np.linalg.norm(value) <= 0:
                raise RuntimeError(
                    f"Entropy candidate {item.path} class {class_id} has a "
                    "zero-norm prototype"
                )
        if any(
            not np.isfinite(confidence) or not 0 <= confidence <= 1
            for confidence in item.presence.values()
        ):
            raise RuntimeError(
                f"Entropy candidate {item.path} has invalid class presence"
            )
        if not (
            0 <= item.representative_count <= item.detection_count
        ):
            raise RuntimeError(
                f"Entropy candidate {item.path} has invalid detection counts"
            )

    return list(summaries)


def _acquisition_diagnostics(
    summaries: Sequence[EntropyCandidate],
    picks: Sequence[str],
    counts: np.ndarray,
    nc: int,
    details: dict,
) -> dict[str, np.ndarray]:
    paths = [item.path for item in summaries]
    picked_rank = {path: rank for rank, path in enumerate(picks)}
    balanced = details["balanced_paths"]
    presence = np.zeros((len(summaries), nc), dtype=np.float32)
    prototype_mask = np.zeros((len(summaries), nc), dtype=np.uint8)
    for row, item in enumerate(summaries):
        for class_id, confidence in item.presence.items():
            presence[row, class_id] = confidence
            prototype_mask[row, class_id] = 1

    initial_quota = np.zeros(nc, dtype=np.float64)
    remaining_quota = np.zeros(nc, dtype=np.float64)
    minority_mask = np.zeros(nc, dtype=np.uint8)
    for class_id, value in details["initial_quotas"].items():
        minority_mask[class_id] = 1
        initial_quota[class_id] = value
        remaining_quota[class_id] = details["remaining_quotas"][class_id]

    return {
        "paths": np.asarray(paths),
        "scores": np.asarray([item.score for item in summaries], dtype=np.float64),
        "detection_counts": np.asarray(
            [item.detection_count for item in summaries], dtype=np.int32
        ),
        "representative_counts": np.asarray(
            [item.representative_count for item in summaries], dtype=np.int32
        ),
        "presence": presence,
        "prototype_mask": prototype_mask,
        "selected_rank": np.asarray(
            [picked_rank.get(path, -1) for path in paths], dtype=np.int32
        ),
        "selected_phase": np.asarray(
            [1 if path in balanced else (2 if path in picked_rank else 0)
             for path in paths],
            dtype=np.int8,
        ),
        "labeled_instance_counts": np.asarray(counts, dtype=np.int64),
        "minority_mask": minority_mask,
        "initial_quota": initial_quota,
        "remaining_quota": remaining_quota,
        "intra_rejected": np.asarray(details["intra_rejected"], dtype=np.int64),
        "inter_rejected": np.asarray(details["inter_rejected"], dtype=np.int64),
    }


def select(
    model,
    candidates,
    labeled,
    budget,
    cfg,
    rng,
    *,
    candidate_fn: Callable = _real_candidate_fn,
) -> list[str]:
    """Select a query batch with ENMS uncertainty and DivProto diversity."""

    del rng  # deterministic, ties are broken by image path

    if budget <= 0:
        return []

    if model is None:
        raise RuntimeError("Entropy selection requires a trained YOLO model")

    pool = sorted(set(candidates) - set(labeled))
    if budget > len(pool):
        raise ValueError(
            f"budget={budget} exceeds unlabeled pool size={len(pool)}"
        )

    try:
        nc = int(cfg["classes"]["nc"])
        entropy_cfg = cfg["al"]["entropy"]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "cfg must contain classes.nc and an al.entropy mapping"
        ) from error

    summaries = _validate_candidate_summaries(
        pool=pool,
        summaries=candidate_fn(model, pool, cfg),
        nc=nc,
    )

    counts = labeled_instance_counts(
        labeled=labeled,
        nc=nc,
    )

    picks, details = divproto_select(
        candidates=summaries,
        labeled_counts=counts,
        budget=budget,
        nc=nc,
        t_intra=float(entropy_cfg["t_intra"]),
        t_inter=float(entropy_cfg["t_inter"]),
        alpha=float(entropy_cfg["alpha"]),
        beta=float(entropy_cfg["beta"]),
        return_diagnostics=True,
    )
    diagnostics = _acquisition_diagnostics(
        summaries=summaries,
        picks=picks,
        counts=counts,
        nc=nc,
        details=details,
    )
    model._entropy_acq_diag = diagnostics
    return picks
