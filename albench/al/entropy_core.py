"""Entropy, ENMS and DivProto (Wu, Chen and Huang, CVPR 2022, arXiv:2204.07965).

Detector-independent part of the Entropy-ENMS-DivProto selector: binary
entropy per detection, entropy-based NMS for the image score, entropy-weighted
class prototypes, and the DivProto batch selection with minority-class quotas.
"""
from dataclasses import dataclass
import math

import numpy as np
from typing import Final

# Paper defaults.
PAPER_T_ENMS: Final[float] = 0.5
PAPER_T_INTRA: Final[float] = 0.7
PAPER_T_INTER: Final[float] = 0.3
PAPER_ALPHA: Final[float] = 0.5
PAPER_BETA: Final[float] = 0.75

# Conventions for cases the paper does not specify.
INTEGER_TOLERANCE: Final[float] = 1e-12
RELATIVE_TOLERANCE: Final[float] = 0.0
NO_COMPARABLE_SIMILARITY: Final[float] = -1.0
ABSENT_CLASS_CONFIDENCE: Final[float] = 0.0
EMPTY_IMAGE_SCORE: Final[float] = 0.0
QUOTA_DECREMENT: Final[float] = 1.0
QUOTA_EXHAUSTED: Final[float] = 0.0

@dataclass
class EntropyCandidate:
    path: str
    score: float
    prototypes: dict[int, np.ndarray]
    presence: dict[int, float]
    detection_count: int = 0
    representative_count: int = 0


def binary_entropy(confidence) -> np.ndarray:
    p = np.asarray(confidence, dtype=float)
    if not np.isfinite(p).all() or (p < 0.0).any() or (p > 1.0).any():
        raise ValueError("confidence values must be finite probabilities in [0, 1]")
    result = np.zeros_like(p)
    interior = (p > 0.0) & (p < 1.0)
    q = p[interior]
    result[interior] = -(q * np.log(q) + (1.0 - q) * np.log1p(-q))
    return result


def _normalize_rows(features: np.ndarray) -> np.ndarray:
    x = np.asarray(features, dtype=float)
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    if (norm <= 0).any():
        raise ValueError("cosine similarity is undefined for zero-norm features")
    return x / norm


def enms_indices(
    labels: np.ndarray,
    features: np.ndarray,
    entropies: np.ndarray,
    threshold: float = PAPER_T_ENMS,
) -> tuple[float, np.ndarray]:
    labels = np.asarray(labels, dtype=int)
    entropies = np.asarray(entropies, dtype=float)

    if len(labels) == 0:
        return EMPTY_IMAGE_SCORE, np.zeros(0, dtype=int)

    normalized = _normalize_rows(features)
    alive = np.ones(len(labels), dtype=bool)
    selected = []

    while alive.any():
        remaining = np.flatnonzero(alive)

        current = remaining[np.argmax(entropies[remaining])]
        selected.append(int(current))
        alive[current] = False

        same_class = np.flatnonzero(alive & (labels == labels[current]))
        if len(same_class) == 0:
            continue

        similarities = normalized[same_class] @ normalized[current]
        redundant = same_class[similarities > threshold]
        alive[redundant] = False

    selected = np.asarray(selected, dtype=int)
    return float(entropies[selected].sum()), selected


def build_prototypes(
    labels: np.ndarray,
    features: np.ndarray,
    entropies: np.ndarray,
) -> dict[int, np.ndarray]:
    labels = np.asarray(labels, dtype=int)
    features = np.asarray(features, dtype=float)
    entropies = np.asarray(entropies, dtype=float)

    prototypes = {}

    for class_id in sorted(set(labels.tolist())):
        mask = labels == class_id
        weights = entropies[mask]
        denominator = float(weights.sum())
        if denominator <= 0:
            raise ValueError(
                f"class {class_id} has zero entropy mass, so Eq. (3) is undefined"
            )

        prototypes[class_id] = (weights[:, None] * features[mask]).sum(axis=0) / denominator

    return prototypes

def class_presence(labels: np.ndarray, confidence: np.ndarray) -> dict[int, float]:
    labels = np.asarray(labels, dtype=int)
    confidence = np.asarray(confidence, dtype=float)

    return {
        class_id: float(confidence[labels == class_id].max())
        for class_id in sorted(set(labels.tolist()))
    }

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator == 0:
        raise ValueError("cosine similarity is undefined for zero-norm features")

    return float(np.dot(a, b) / denominator)


def intra_similarity(
    candidate: EntropyCandidate,
    selected: list[EntropyCandidate],
) -> float:
    if not selected or not candidate.prototypes:
        return NO_COMPARABLE_SIMILARITY

    per_class_max = []

    for class_id, prototype in candidate.prototypes.items():
        previous = [
            item.prototypes[class_id]
            for item in selected
            if class_id in item.prototypes
        ]

        if not previous:
            return NO_COMPARABLE_SIMILARITY

        per_class_max.append(
            max(cosine(prototype, other) for other in previous)
        )

    return (
        min(per_class_max)
        if per_class_max
        else NO_COMPARABLE_SIMILARITY
    )


def build_minority_quotas(
    labeled_counts: np.ndarray,
    budget: int,
    nc: int,
    alpha: float = PAPER_ALPHA,
    beta: float = PAPER_BETA,
) -> dict[int, float]:
    counts = np.asarray(labeled_counts, dtype=float)

    if counts.ndim != 1 or len(counts) != nc:
        raise ValueError(
            f"labeled_counts must have shape ({nc},), got {counts.shape}"
        )

    if not np.isfinite(counts).all() or (counts < 0).any():
        raise ValueError("labeled_counts must be finite and non-negative")

    if nc <= 0:
        raise ValueError(f"nc must be positive, got {nc}")

    if budget < 0:
        raise ValueError(f"budget must be non-negative, got {budget}")

    if not (0.0 < alpha < beta < 1.0):
        raise ValueError(
            "paper requires 0 < alpha < beta < 1, "
            f"got alpha={alpha}, beta={beta}"
        )

    raw_minority_n = alpha * nc
    nearest_integer = round(raw_minority_n)

    if not math.isclose(
        raw_minority_n,
        nearest_integer,
        rel_tol=RELATIVE_TOLERANCE,
        abs_tol=INTEGER_TOLERANCE,
    ):
        raise ValueError(
            f"alpha * nc must be an integer for DivProto, "
            f"got {alpha} * {nc} = {raw_minority_n}"
        )

    minority_n = int(nearest_integer)

    minority_classes = sorted(
        range(nc),
        key=lambda class_id: (counts[class_id], class_id),
    )[:minority_n]

    quota_value = beta / (alpha * nc) * budget

    return {
        class_id: float(quota_value)
        for class_id in minority_classes
    }


def divproto_select(
    candidates: list[EntropyCandidate],
    labeled_counts: np.ndarray,
    budget: int,
    nc: int,
    t_intra: float = PAPER_T_INTRA,
    t_inter: float = PAPER_T_INTER,
    alpha: float = PAPER_ALPHA,
    beta: float = PAPER_BETA,
    *,
    return_diagnostics: bool = False,
) -> list[str] | tuple[list[str], dict]:
    if budget < 0:
        raise ValueError(f"budget must be non-negative, got {budget}")

    if budget == 0:
        return []

    if budget > len(candidates):
        raise ValueError(
            f"budget={budget} exceeds candidate count={len(candidates)}"
        )

    paths = [candidate.path for candidate in candidates]
    if len(paths) != len(set(paths)):
        raise ValueError("candidate.path values must be unique")

    ranked = sorted(
        candidates,
        key=lambda candidate: (-candidate.score, candidate.path),
    )

    quotas = build_minority_quotas(
        labeled_counts=labeled_counts,
        budget=budget,
        nc=nc,
        alpha=alpha,
        beta=beta,
    )
    initial_quotas = dict(quotas)
    active_minority = set(quotas)

    selected: list[EntropyCandidate] = []
    selected_paths: set[str] = set()
    balanced_paths: set[str] = set()
    intra_rejected = 0
    inter_rejected = 0

    for candidate in ranked:
        if len(selected) >= budget or not active_minority:
            break

        mg = intra_similarity(candidate, selected)
        if mg >= t_intra:
            intra_rejected += 1
            continue

        present_minority = [
            class_id
            for class_id in sorted(active_minority)
            if candidate.presence.get(
                class_id,
                ABSENT_CLASS_CONFIDENCE,
            ) > t_inter
        ]

        if not present_minority:
            inter_rejected += 1
            continue

        selected.append(candidate)
        selected_paths.add(candidate.path)
        balanced_paths.add(candidate.path)

        for class_id in present_minority:
            quotas[class_id] -= QUOTA_DECREMENT

            if quotas[class_id] <= QUOTA_EXHAUSTED:
                active_minority.remove(class_id)

    for candidate in ranked:
        if len(selected) >= budget:
            break

        if candidate.path in selected_paths:
            continue

        selected.append(candidate)
        selected_paths.add(candidate.path)

    if len(selected) != budget:
        raise RuntimeError(
            f"DivProto requested {budget} images but selected "
            f"{len(selected)}"
        )

    result = [candidate.path for candidate in selected]
    if not return_diagnostics:
        return result
    return result, {
        "initial_quotas": initial_quotas,
        "remaining_quotas": quotas,
        "balanced_paths": balanced_paths,
        "intra_rejected": intra_rejected,
        "inter_rejected": inter_rejected,
    }


def validate_detection_arrays(
    labels: np.ndarray,
    confidences: np.ndarray,
    features: np.ndarray,
) -> None:
    labels = np.asarray(labels)
    confidences = np.asarray(confidences, dtype=float)
    features = np.asarray(features, dtype=float)

    if labels.ndim != 1:
        raise ValueError(f"labels must have shape (N,), got {labels.shape}")
    if confidences.ndim != 1:
        raise ValueError(
            f"confidences must have shape (N,), got {confidences.shape}"
        )
    if not np.isfinite(labels).all() or not np.equal(labels, np.floor(labels)).all():
        raise ValueError("labels must contain finite integer class ids")
    if (labels < 0).any():
        raise ValueError("labels must contain non-negative class ids")
    if not np.isfinite(confidences).all() or (confidences < 0).any() or (
        confidences > 1
    ).any():
        raise ValueError("confidences must be finite probabilities in [0, 1]")
    if not np.isfinite(features).all():
        raise ValueError("features must contain only finite values")

    detection_count = len(labels)

    if len(confidences) != detection_count:
        raise ValueError(
            f"labels/confidences length mismatch: "
            f"{detection_count} != {len(confidences)}"
        )

    if features.ndim != 2 or len(features) != detection_count:
        raise ValueError(
            f"features must have shape (N, D), got {features.shape}"
        )
    if detection_count and (np.linalg.norm(features, axis=1) <= 0).any():
        raise ValueError("detection features must have non-zero norm")


def make_entropy_candidate(
    path: str,
    labels: np.ndarray,
    confidences: np.ndarray,
    features: np.ndarray,
    t_enms: float = PAPER_T_ENMS,
) -> EntropyCandidate:
    labels = np.asarray(labels)
    confidences = np.asarray(confidences, dtype=float)
    features = np.asarray(features, dtype=float)

    validate_detection_arrays(labels, confidences, features)
    labels = labels.astype(np.int64, copy=False)

    entropies = binary_entropy(confidences)

    image_score, representative_indices = enms_indices(
        labels=labels,
        features=features,
        entropies=entropies,
        threshold=t_enms,
    )

    return EntropyCandidate(
        path=path,
        score=image_score,
        prototypes=build_prototypes(
            labels=labels,
            features=features,
            entropies=entropies,
        ),
        presence=class_presence(
            labels=labels,
            confidence=confidences,
        ),
        detection_count=len(labels),
        representative_count=len(representative_indices),
    )
