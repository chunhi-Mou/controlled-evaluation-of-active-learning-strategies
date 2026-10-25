"""Evaluation settings shared by the baseline, AL rounds and checkpoint checks."""


def evaluation_kwargs(cfg: dict, run: dict) -> dict:
    ev = cfg["evaluation"]
    return {
        "imgsz": run["imgsz"],
        "batch": run["batch"],
        "workers": run.get("workers", 8),
        "half": ev["half"],
        "augment": ev["augment"],
        "conf": ev["conf"],
        "iou": ev["iou"],
        "rect": run["rect"],
        "max_det": run["max_det"],
    }
