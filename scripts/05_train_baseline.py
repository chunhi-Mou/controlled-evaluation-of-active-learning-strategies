"""Train the YOLOv8n baseline on a fraction of the training split (--frac 1.0 for all of it)."""
import argparse
import csv
import os
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from albench.al.device import device
from albench.al.health import health_baseline
from albench.al.cleanup import release
from albench.al.protocol import assert_fixed_batch, enforce_fixed_batch
from albench.config import load_config, load_seeds
from albench.data.labels import IMG_EXT
from albench.repro import set_seed

DEFAULT_CONFIG = "sortwaste"


def _val_map50_curve(results_csv: Path):
    if not results_csv.exists():
        return [], []
    rows = [{k.strip(): v.strip() for k, v in r.items()}
            for r in csv.DictReader(results_csv.open())]
    if not rows:
        return [], []
    key = next((k for k in rows[0] if "mAP50" in k and "95" not in k), None)
    if not key:
        return [], []
    return ([int(r["epoch"]) for r in rows],
            [float(r[key]) for r in rows])


def _per_class_ap_lines(names, per_class: dict) -> str:
    out = []
    for i, n in enumerate(names):
        v = per_class.get(i)
        out.append(f"- **{n}**: {v:.4f}" if v is not None
                   else f"- **{n}**: n/a")
    return "\n".join(out)


def run(config, seeds, data=None, frac=1.0) -> None:
    cfg = load_config(Path(config))
    seed = load_seeds(seeds)[cfg["split"]["seed_index"]]
    set_seed(seed)

    data_yaml_path = Path(data or cfg["data_yaml"])
    d = yaml.safe_load(data_yaml_path.read_text(encoding="utf-8"))
    base = Path(d.get("path", str(data_yaml_path.parent)))

    d["val"] = str((base / d["val"]).resolve())
    d["test"] = str((base / d["test"]).resolve())
    d.pop("path", None)

    train_imgs = sorted(str(p) for p in (base / d["train"]).iterdir()
                        if p.suffix.lower() in IMG_EXT)

    out_dir = Path(cfg["splits_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    if frac < 1.0:
        n_train = max(1, int(len(train_imgs) * frac))
        sub = random.Random(seed).sample(train_imgs, n_train)
        sub_txt = out_dir / f"train_frac{frac}.txt"
        sub_txt.write_text("\n".join(sub) + "\n", encoding="utf-8")
        d["train"] = str(sub_txt.resolve())
        data_yaml_path = out_dir / f"data_frac{frac}.yaml"
    else:
        n_train = len(train_imgs)
        d["train"] = str((base / d["train"]).resolve())
        data_yaml_path = out_dir / "data_full_baseline.yaml"
    data_yaml_path.write_text(yaml.safe_dump(d), encoding="utf-8")

    from ultralytics import YOLO
    b = cfg["baseline"]
    model = YOLO(b["model"])
    enforce_fixed_batch(
        model, b["batch"], expected_epochs=b["epochs"],
        expected_warmup=b["warmup_epochs"],
        expected_close_mosaic=b["close_mosaic"])
    name = f"baseline_frac{frac}_seed{seed}"
    out = Path(b["project"])
    out.mkdir(parents=True, exist_ok=True)

    eval_model = None
    try:
        model.train(data=str(data_yaml_path), epochs=b["epochs"], imgsz=b["imgsz"],
                    batch=b["batch"], cache=b.get("cache", False),
                    workers=b.get("workers", 8),
                    device=device(), project=str(out), name=name,
                    pretrained=b.get("pretrained", True),
                    warmup_epochs=b["warmup_epochs"], resume=False,
                    close_mosaic=b["close_mosaic"],
                    patience=b["patience"], val=b.get("val", True),
                    rect=b["rect"], max_det=b["max_det"],
                    seed=seed, plots=True, exist_ok=True)
        assert_fixed_batch(model, b["batch"])
        run_dir = Path(model.trainer.save_dir)
    except Exception:
        release(model)
        raise
    results_csv = run_dir / "results.csv"
    reports_out = Path(cfg["reports_dir"]) / "baseline" / name
    reports_out.mkdir(parents=True, exist_ok=True)

    # model.train() leaves best.pt loaded regardless of the knob
    ckpt = run_dir / "weights" / f"{b.get('eval_checkpoint', 'last')}.pt"
    if not ckpt.is_file():
        release(model)
        raise FileNotFoundError(f"baseline checkpoint was not written: {ckpt}")
    try:
        release(model)
        model = None
        eval_model = YOLO(str(ckpt))
        from albench.evaluation import evaluation_kwargs
        metrics = eval_model.val(data=str(data_yaml_path), split="test",
                                 device=device(),
                                 **evaluation_kwargs(cfg, b))
    finally:
        release(model, eval_model)
    test_map50 = float(metrics.box.map50)
    test_map = float(metrics.box.map)      # COCO reports the ceiling at .5:.95
    per_class = {int(i): float(v)
                 for i, v in zip(metrics.box.ap_class_index, metrics.box.ap50)}
    names = list(d.get("names", []))

    ep, mp = _val_map50_curve(results_csv)
    best_map50, best_epoch = 0.0, 0
    if mp:
        best_i = max(range(len(mp)), key=lambda i: mp[i])
        best_map50, best_epoch = mp[best_i], ep[best_i]

    if ep:
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(ep, mp, marker=".", label=f"frac={frac}")
        ax.set(xlabel="Epoch", ylabel="val mAP50",
               title=f"Baseline Convergence (frac={frac}, imgsz={b['imgsz']})")
        ax.legend()
        ax.grid(True, alpha=.3)
        fig.tight_layout()
        fig.savefig(reports_out / "convergence_curve.png", dpi=120)
        plt.close(fig)

    md = [
        f"# Baseline: {name}",
        "",
        f"- **Fraction**: {frac}",
        f"- **Images**: {n_train}",
        f"- **Model**: {b['model']}",
        f"- **Epochs (cap)**: {b['epochs']}",
        f"- **Patience**: {b['patience']}",
        f"- **Evaluation checkpoint**: {ckpt.name}",
        f"- **Best Val mAP50**: {best_map50:.4f} (@epoch {best_epoch})",
        f"- **Stopped @**: {ep[-1] if ep else 'N/A'}",
        f"- **Test mAP50**: {test_map50:.4f}",
        f"- **Test mAP50-95**: {test_map:.4f}",
        "",
        "## Full-data baseline" if frac == 1.0 else "## Subsampled baseline",
        f"Trained on {frac*100:.1f}% of the train split.",
        "",
        "## Per-class AP@50 (test)",
        _per_class_ap_lines(names, per_class),
    ]
    (reports_out / "SUMMARY.md").write_text("\n".join(md) + "\n",
                                            encoding="utf-8")
    health_baseline(results_csv, str(reports_out))

    print(f"\n{name} summary:")
    print(f"  Best val mAP50: {best_map50:.4f} @epoch {best_epoch}")
    print(f"  Test mAP50:     {test_map50:.4f}")
    print(f"  Test mAP50-95:  {test_map:.4f}")
    print(f"  Weights/plots:  {run_dir}")
    print(f"  Report:         {reports_out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--seeds", default=None)
    ap.add_argument("--data", default=None,
                    help="dataset yaml, defaults to config data_yaml")
    ap.add_argument("--frac", type=float, default=1.0,
                    help="fraction of train split (1.0=full, 0.05=probe)")
    ap.add_argument("--device", default=None,
                    help="pin this process to a GPU ('0', '1')")
    a = ap.parse_args()
    if a.device:
        os.environ["ALBENCH_DEVICE"] = a.device
    run(a.config, a.seeds, a.data, a.frac)


if __name__ == "__main__":
    main()
