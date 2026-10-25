"""Training-stability / no-overfit figures from per-epoch Ultralytics
results.csv and aggregated Results."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .charts import STRATEGY_STYLE, _save


def epoch_curve(results_csv) -> dict:
    """Parse one Ultralytics per-epoch results.csv into loss/metric lists."""
    p = Path(results_csv)
    out = {"epoch": [], "train_box": [], "train_cls": [],
           "val_box": [], "val_cls": [], "map50": []}
    if not p.exists():
        return out
    rows = [{k.strip(): v.strip() for k, v in r.items()}
            for r in csv.DictReader(p.open())]
    if not rows:
        return out
    mk = next((k for k in rows[0] if "mAP50" in k and "95" not in k), None)
    for r in rows:
        out["epoch"].append(int(float(r["epoch"])))
        out["train_box"].append(float(r["train/box_loss"]))
        out["train_cls"].append(float(r["train/cls_loss"]))
        out["val_box"].append(float(r["val/box_loss"]))
        out["val_cls"].append(float(r["val/cls_loss"]))
        out["map50"].append(float(r[mk]) if mk else float("nan"))
    return out


def seed_variance(results, field="val_mAP50") -> dict:
    """Per-strategy mean/sample-SD of `field`, aligned by round."""
    out = {}
    for s in results.strategies:
        rounds = sorted({r["round"] for sd in results.seeds
                         if (s, sd) in results.by
                         for r in results.by[(s, sd)]})
        mean, std = [], []
        for rd in rounds:
            vals = []
            for sd in results.seeds:
                if (s, sd) in results.by:
                    hit = [r[field] for r in results.by[(s, sd)]
                           if r["round"] == rd]
                    if hit:
                        vals.append(hit[0])
            mean.append(float(np.mean(vals)) if vals else float("nan"))
            std.append(float(np.std(vals, ddof=1)) if len(vals) > 1
                       else float("nan"))
        out[s] = {"rounds": rounds, "mean": mean, "std": std}
    return out


def converge_grid(results, field="val_mAP50"):
    """(row_labels, rounds, matrix) of `field` per (strategy,seed) x round."""
    rows = [(s, sd) for s in results.strategies for sd in results.seeds
            if (s, sd) in results.by]
    rounds = sorted({r["round"] for k in rows for r in results.by[k]})
    mat = np.full((len(rows), len(rounds)), np.nan)
    for i, k in enumerate(rows):
        for r in results.by[k]:
            if r["round"] in rounds:
                mat[i, rounds.index(r["round"])] = r[field]
    labels = [f"{s}·s{sd}" for s, sd in rows]
    return labels, rounds, mat


def health_baseline(baseline_csv, out_dir) -> Path:
    """Baseline training health: train vs val loss + mAP50 (dual axis)."""
    c = epoch_curve(baseline_csv)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(c["epoch"], c["train_box"], color="#4C72B0", label="train box")
    ax.plot(c["epoch"], c["val_box"], color="#4C72B0", ls="--",
            label="val box")
    ax.plot(c["epoch"], c["train_cls"], color="#C44E52", label="train cls")
    ax.plot(c["epoch"], c["val_cls"], color="#C44E52", ls="--",
            label="val cls")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax2 = ax.twinx()
    ax2.plot(c["epoch"], c["map50"], color="#55A868", marker=".",
             label="val mAP50")
    ax2.set_ylabel("val mAP50")
    ax.set_title("Baseline training health (train vs val, no divergence)")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="center right", fontsize=8)
    return _save(fig, out_dir, "health_baseline.png")


def health_gap_overview(results, runs_dir, out_dir) -> Path:
    """Per-round (val − train) box-loss gap, one line per strategy (first
    available seed). A small, stable gap means no overfitting."""
    fig, ax = plt.subplots(figsize=(7, 5))
    for s in results.strategies:
        sd = next((d for d in results.seeds if (s, d) in results.by), None)
        if sd is None:
            continue
        st = STRATEGY_STYLE.get(s, {"color": "k", "marker": "x", "label": s})
        xs, gaps = [], []
        for r in results.by[(s, sd)]:
            c = epoch_curve(Path(runs_dir) / f"{s}_s{sd}_r{r['round']}"
                            / "results.csv")
            if c["epoch"]:
                xs.append(r["round"])
                gaps.append(c["val_box"][-1] - c["train_box"][-1])
        if xs:
            ax.plot(xs, gaps, color=st["color"], marker=st["marker"],
                    label=st["label"])
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_xlabel("AL round")
    ax.set_ylabel("final-epoch (val − train) box loss")
    ax.set_title("Train/val loss gap per round (small, stable = no overfit)")
    if ax.get_legend_handles_labels()[0]:
        ax.legend()
    return _save(fig, out_dir, "health_gap_overview.png")


def health_seed_variance(results, out_dir) -> Path:
    """Across-seed sample SD of val_mAP50 per AL round."""
    sv = seed_variance(results, "val_mAP50")
    fig, ax = plt.subplots(figsize=(7, 5))
    for s in results.strategies:
        st = STRATEGY_STYLE.get(s, {"color": "k", "marker": "x", "label": s})
        d = sv[s]
        ax.plot(d["rounds"], d["std"], color=st["color"], marker=st["marker"],
                label=st["label"])
    ax.set_xlabel("AL round")
    ax.set_ylabel("sample SD of val mAP50 across seeds")
    ax.set_title("Across-seed sample SD (lower = more stable / reproducible)")
    ax.legend()
    return _save(fig, out_dir, "health_seed_variance.png")


def health_converge_heatmap(results, out_dir) -> Path:
    """Heatmap of val_mAP50 per (strategy·seed) × round."""
    labels, rounds, mat = converge_grid(results, "val_mAP50")
    fig, ax = plt.subplots(figsize=(1.5 + 0.6 * len(rounds),
                                    1.2 + 0.4 * len(labels)))
    im = ax.imshow(mat, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(rounds)))
    ax.set_xticklabels(rounds)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("AL round")
    ax.set_ylabel("strategy · seed")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if not np.isnan(mat[i, j]):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                        color="w", fontsize=7)
    ax.set_title("val mAP50 per run × round (blank = missing / failed)")
    fig.colorbar(im, ax=ax, fraction=0.046)
    return _save(fig, out_dir, "health_converge_heatmap.png")
