"""Predictor exposing the full per-class score vector for each kept box via
non_max_suppression(return_idxs=True). Inference-only."""
from ultralytics.models.yolo.detect import DetectionPredictor
from ultralytics.utils import nms


class ScorePredictor(DetectionPredictor):
    def postprocess(self, preds, img, orig_imgs, **kwargs):
        raw = preds[0] if isinstance(preds, (list, tuple)) else preds
        out, keepi = nms.non_max_suppression(
            raw, self.args.conf, self.args.iou, self.args.classes,
            self.args.agnostic_nms, max_det=self.args.max_det,
            nc=0, return_idxs=True)
        scores = raw.transpose(-1, -2)[..., 4:]          # (B, anchors, nc)
        results = self.construct_results(out, img, orig_imgs)
        for r, kept, sc in zip(results, keepi, scores):
            r.cls_scores = sc[kept.long()] if kept.numel() else sc[:0]
        return results
