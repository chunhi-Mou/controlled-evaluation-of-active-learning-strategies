"""PPAL DCUS class difficulty for the YOLOv8 training assigner.

PPAL computes quality from every positive prediction used by the detector's
training target assigner.  It does not use post-NMS inference detections.
"""
import math

import numpy as np


class QualityEMA:
    """Class-wise PPAL quality EMA (paper Eqs. 1-3)."""

    def __init__(self, nc: int, base_momentum: float):
        self.class_quality = np.zeros(nc, dtype=float)
        self.class_momentum = np.full(nc, base_momentum, dtype=float)
        self.base_momentum = float(base_momentum)

    def update(self, avg_q: np.ndarray) -> None:
        """Update from one distributed batch's class-wise mean qualities."""
        avg_q = np.asarray(avg_q, dtype=float)
        if avg_q.shape != self.class_quality.shape:
            raise ValueError(
                f"PPAL quality shape {avg_q.shape} != {self.class_quality.shape}")
        momentum = self.class_momentum
        self.class_quality = (
            momentum * self.class_quality + (1.0 - momentum) * avg_q)
        # As in the official PPAL code, a class with no positive-quality
        # observation in a batch gets a progressively smaller momentum.
        self.class_momentum = np.where(
            avg_q > 0.0, self.base_momentum,
            self.class_momentum * self.base_momentum)

    def update_sums(self, quality_sums: np.ndarray,
                    counts: np.ndarray) -> None:
        quality_sums = np.asarray(quality_sums, dtype=float)
        counts = np.asarray(counts, dtype=float)
        self.update(np.divide(
            quality_sums, counts + 1e-5,
            out=np.zeros_like(quality_sums), where=counts >= 0))

    @property
    def value(self) -> np.ndarray:
        return self.class_quality.copy()


def aligned_iou_xyxy(pred_boxes, target_boxes):
    """Ordinary aligned IoU used by PPAL Eq. (1), not CIoU/GIoU."""
    import torch

    pred = pred_boxes.float()
    target = target_boxes.float()
    top_left = torch.maximum(pred[:, :2], target[:, :2])
    bottom_right = torch.minimum(pred[:, 2:], target[:, 2:])
    intersection_wh = (bottom_right - top_left).clamp_min(0.0)
    intersection = intersection_wh[:, 0] * intersection_wh[:, 1]
    pred_wh = (pred[:, 2:] - pred[:, :2]).clamp_min(0.0)
    target_wh = (target[:, 2:] - target[:, :2]).clamp_min(0.0)
    union = (pred_wh[:, 0] * pred_wh[:, 1]
             + target_wh[:, 0] * target_wh[:, 1] - intersection)
    return torch.where(union > 0.0, intersection / union, torch.zeros_like(union))


def quality_sums_from_assignments(pd_scores, pd_bboxes, target_labels,
                                  target_bboxes, fg_mask, nc: int,
                                  xi: float):
    """Sum PPAL quality for all YOLO positive assignments in one batch."""
    import torch

    sums = torch.zeros(nc, dtype=torch.float32, device=pd_scores.device)
    counts = torch.zeros_like(sums)
    foreground = fg_mask.bool()
    if not foreground.any():
        return sums, counts

    labels = target_labels[foreground].long()
    if torch.any((labels < 0) | (labels >= nc)):
        raise RuntimeError("PPAL assigner returned an invalid foreground class")
    probabilities = pd_scores[foreground].float().gather(
        1, labels[:, None]).squeeze(1).clamp(0.0, 1.0)
    iou = aligned_iou_xyxy(
        pd_bboxes[foreground], target_bboxes[foreground]).clamp(0.0, 1.0)
    quality = probabilities.pow(float(xi)) * iou.pow(1.0 - float(xi))
    sums.scatter_add_(0, labels, quality)
    counts.scatter_add_(0, labels, torch.ones_like(quality))
    return sums, counts


class QualityAssignerProxy:
    """Observe TaskAlignedAssigner outputs without changing its return values."""

    def __init__(self, assigner, state: QualityEMA, xi: float):
        self.assigner = assigner
        self.state = state
        self.xi = float(xi)
        self.nc = state.class_quality.shape[0]

    def __getattr__(self, name):
        return getattr(self.assigner, name)

    def __call__(self, pd_scores, pd_bboxes, anc_points, gt_labels,
                 gt_bboxes, mask_gt):
        import torch

        output = self.assigner(
            pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt)
        target_labels, target_bboxes, _, fg_mask, _ = output
        with torch.no_grad():
            sums, counts = quality_sums_from_assignments(
                pd_scores, pd_bboxes, target_labels, target_bboxes,
                fg_mask, self.nc, self.xi)
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.all_reduce(sums)
                torch.distributed.all_reduce(counts)
            self.state.update_sums(
                sums.detach().cpu().numpy(), counts.detach().cpu().numpy())
        return output


def make_batch_quality_callbacks(state: QualityEMA, xi: float) -> dict:
    """Install a read-only observer on YOLOv8's training target assigner."""

    def _on_train_start(trainer) -> None:
        from ultralytics.utils.torch_utils import unwrap_model

        network = unwrap_model(trainer.model)
        criterion = getattr(network, "criterion", None)
        if criterion is None:
            criterion = network.init_criterion()
            network.criterion = criterion
        assigner = getattr(criterion, "assigner", None)
        if assigner is None:
            raise RuntimeError(
                "PPAL DCUS requires a YOLO detection criterion with an assigner")
        if isinstance(assigner, QualityAssignerProxy):
            raise RuntimeError("PPAL quality assigner was installed twice")
        criterion.assigner = QualityAssignerProxy(assigner, state, xi)

    return {"on_train_start": _on_train_start}


def reweight(class_quality: np.ndarray, alpha: float, ub: float) -> np.ndarray:
    """PPAL Eq. (4): lower-quality classes receive higher uncertainty weight."""
    difficulty = 1.0 - np.asarray(class_quality, dtype=float)
    b = math.exp(1.0 / alpha) - 1.0
    return 1.0 + alpha * np.log(b * difficulty + 1.0) * ub
