"""Audit YOLOv8 post-NMS box/confidence/feature row alignment for Entropy."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from albench.al.dataset import pool_images
from albench.al.entropy_features import extract_detection_features
from albench.config import load_config


def _manual_object_features(feature_maps, indices):
    feature_dim = min(int(feature.shape[1]) for feature in feature_maps)
    levels = []
    for feature in feature_maps:
        batch, channels, height, width = feature.shape
        if channels % feature_dim:
            raise RuntimeError(
                f"Detect feature channels {channels} are not divisible by "
                f"the common dimension {feature_dim}"
            )
        spatial = feature.reshape(batch, channels, height * width).permute(0, 2, 1)
        levels.append(
            spatial.reshape(batch, height * width, feature_dim,
                            channels // feature_dim).mean(dim=-1)
        )
    flattened = torch.cat(levels, dim=1)
    rows = [
        flattened[batch_index].index_select(0, index.long().reshape(-1))
        if index.numel()
        else flattened.new_empty((0, feature_dim))
        for batch_index, index in enumerate(indices)
    ]
    return rows, feature_dim, [tuple(feature.shape) for feature in feature_maps]


def audit(
    config: str,
    checkpoint: str,
    limit: int,
    batch_size: int,
    score_threshold: float | None = None,
) -> dict:
    from ultralytics import YOLO
    from ultralytics.models.yolo.detect import DetectionPredictor
    from ultralytics.models.yolo.detect import predict as detect_predict

    cfg = load_config(config)
    entropy = cfg["al"]["entropy"]
    score_threshold = (
        entropy["score_thr"] if score_threshold is None else score_threshold
    )
    paths = pool_images(cfg["data_yaml"])[:limit]
    if not paths:
        raise RuntimeError("No pool images were found")

    nms_records = []
    feature_records = []
    original_nms = detect_predict.nms.non_max_suppression
    original_get = DetectionPredictor.get_obj_feats

    def audited_nms(prediction, *args, **kwargs):
        if not kwargs.get("return_idxs", False):
            return original_nms(prediction, *args, **kwargs)
        raw = prediction[0] if isinstance(prediction, (list, tuple)) else prediction
        nc = int(raw.shape[1] - 4)
        raw_scores = raw[:, 4:4 + nc].detach().float().cpu().clone()
        result = original_nms(prediction, *args, **kwargs)
        if not isinstance(result, tuple):
            raise RuntimeError("NMS return_idxs=True did not return indices")
        detections, indices = result
        nms_records.append({
            "raw_scores": raw_scores,
            "detections": [value.detach().float().cpu().clone()
                           for value in detections],
            "indices": [value.detach().long().cpu().clone()
                        for value in indices],
        })
        return result

    def audited_get(feature_maps, indices):
        native = original_get(feature_maps, indices)
        manual, feature_dim, shapes = _manual_object_features(
            feature_maps, indices
        )
        errors = []
        stored = []
        for native_rows, manual_rows in zip(native, manual):
            native_tensor = (
                native_rows if torch.is_tensor(native_rows)
                else torch.empty((0, feature_dim), device=manual_rows.device)
            )
            errors.append(
                float((native_tensor - manual_rows).abs().max().item())
                if manual_rows.numel() else 0.0
            )
            stored.append(manual_rows.detach().float().cpu().clone())
        feature_records.append({
            "rows": stored,
            "feature_dim": feature_dim,
            "shapes": shapes,
            "manual_native_max_error": max(errors, default=0.0),
        })
        return native

    detect_predict.nms.non_max_suppression = audited_nms
    DetectionPredictor.get_obj_feats = staticmethod(audited_get)
    try:
        model = YOLO(checkpoint)
        extracted = extract_detection_features(
            model=model,
            paths=paths,
            imgsz=cfg["al"]["imgsz"],
            score_threshold=score_threshold,
            nms_iou_threshold=entropy["nms_iou"],
            max_detections=entropy["max_det"],
            batch_size=batch_size,
            rect=False,
            half=False,
            device=0,
            verbose=False,
        )
    finally:
        detect_predict.nms.non_max_suppression = original_nms
        DetectionPredictor.get_obj_feats = staticmethod(original_get)

    expected_batches = (len(paths) + batch_size - 1) // batch_size
    if len(nms_records) != expected_batches or len(feature_records) != expected_batches:
        raise RuntimeError(
            f"Expected {expected_batches} inference batches, got "
            f"NMS={len(nms_records)} and features={len(feature_records)}"
        )

    confidence_errors = []
    class_errors = 0
    native_manual_feature_errors = []
    result_manual_feature_errors = []
    raw_confidence_errors = []
    raw_class_errors = 0
    zero_norms = 0
    invalid_feature_indices = 0
    feature_level_counts = None
    cursor = 0
    feature_dims = set()
    feature_shapes = []
    for nms_record, feature_record in zip(nms_records, feature_records):
        feature_dims.add(feature_record["feature_dim"])
        feature_shapes.append(feature_record["shapes"])
        level_sizes = [shape[2] * shape[3] for shape in feature_record["shapes"]]
        if feature_level_counts is None:
            feature_level_counts = [0] * len(level_sizes)
        native_manual_feature_errors.append(
            feature_record["manual_native_max_error"]
        )
        for detections, indices, manual_rows in zip(
            nms_record["detections"],
            nms_record["indices"],
            feature_record["rows"],
        ):
            item = extracted[cursor]
            cursor += 1
            if len(item.labels) != len(detections) or len(indices) != len(detections):
                raise RuntimeError("Detection, NMS-index and extracted-row counts differ")
            confidence_errors.append(
                float(np.max(np.abs(item.confidences - detections[:, 4].numpy())))
                if len(detections) else 0.0
            )
            class_errors += int(np.count_nonzero(
                item.labels != detections[:, 5].numpy().astype(np.int64)
            ))
            result_manual_feature_errors.append(
                float(np.max(np.abs(item.features - manual_rows.numpy())))
                if len(detections) else 0.0
            )
            zero_norms += int(np.count_nonzero(
                np.linalg.norm(item.features, axis=1) <= 0
            ))

        for batch_index, (detections, indices) in enumerate(zip(
            nms_record["detections"], nms_record["indices"]
        )):
            if not len(indices):
                continue
            boundaries = np.cumsum(level_sizes)
            raw_indices = indices.numpy()
            invalid_feature_indices += int(np.count_nonzero(
                (raw_indices < 0) | (raw_indices >= boundaries[-1])
            ))
            levels = np.searchsorted(boundaries, raw_indices, side="right")
            for level in range(len(level_sizes)):
                feature_level_counts[level] += int(np.count_nonzero(levels == level))
            selected_scores = nms_record["raw_scores"][batch_index, :, indices]
            raw_confidence_errors.append(float((
                selected_scores.max(dim=0).values - detections[:, 4]
            ).abs().max().item()))
            raw_class_errors += int((
                selected_scores.argmax(dim=0) != detections[:, 5].long()
            ).sum().item())

    if cursor != len(extracted):
        raise RuntimeError("Not every extracted image was audited")

    return {
        "images": len(extracted),
        "batches": expected_batches,
        "detections": int(sum(len(item.labels) for item in extracted)),
        "empty_images": int(sum(len(item.labels) == 0 for item in extracted)),
        "feature_dimensions": sorted(feature_dims),
        "feature_shapes_by_batch": feature_shapes,
        "max_native_manual_feature_error": max(
            native_manual_feature_errors, default=0.0
        ),
        "max_result_manual_feature_error": max(
            result_manual_feature_errors, default=0.0
        ),
        "max_result_nms_confidence_error": max(confidence_errors, default=0.0),
        "result_nms_class_mismatches": class_errors,
        "max_raw_nms_confidence_error": max(raw_confidence_errors, default=0.0),
        "raw_nms_class_mismatches": raw_class_errors,
        "zero_norm_feature_rows": zero_norms,
        "feature_level_counts": feature_level_counts or [],
        "invalid_feature_indices": invalid_feature_indices,
        "path_order_preserved": [item.path for item in extracted] == paths,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="sortwaste")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--score-thr", type=float, default=None)
    args = parser.parse_args()
    result = audit(
        args.config,
        args.checkpoint,
        args.limit,
        args.batch,
        args.score_thr,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
