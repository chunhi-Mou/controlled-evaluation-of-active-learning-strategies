"""AL benchmark figures: 300-dpi PNG, fixed per-strategy style."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .metrics import DEFAULT_METRIC, aubc, seeds_complete

STRATEGY_STYLE = {
    "random": {"color": "#4C72B0", "marker": "o", "label": "Random"},
    "ppal":   {"color": "#8172B3", "marker": "D", "label": "PPAL"},
    "rarity_aware": {"color": "#C44E52", "marker": "s",
                     "label": "Rarity-aware"},
    "entropy": {"color": "#55A868", "marker": "^",
                "label": "Entropy-ENMS-DivProto"},
}
_DPI = 300


def _save(fig, out_dir, name):
    p = Path(out_dir) / name
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(p, dpi=_DPI)
    plt.close(fig)
    return p


def _mean_std(curves):
    L = min(len(c) for c in curves)
    a = np.array([c[:L] for c in curves], dtype=float)
    std = (a.std(axis=0, ddof=1) if a.shape[0] > 1
           else np.full(a.shape[1], np.nan, dtype=float))
    return a.mean(axis=0), std


def curve_metric(results, field, out_dir, cap_round):
    fig, ax = plt.subplots(figsize=(7, 5))
    for s in results.strategies:
        st = STRATEGY_STYLE.get(s, {"color": "k", "marker": "x",
                                    "label": s})
        seedcurves, xs = [], None
        for sd in results.seeds:
            rs = results.by.get((s, sd))
            if not rs:
                continue
            xs = [r["n_labeled"] for r in rs]
            y = [r[field] for r in rs]
            seedcurves.append(y)
            ax.plot(xs, y, color=st["color"], alpha=0.15, linewidth=0.8)
        if seedcurves:
            mean, std = _mean_std(seedcurves)
            x = xs[:len(mean)]
            ax.plot(x, mean, color=st["color"], marker=st["marker"],
                    label=st["label"])
            ax.fill_between(x, mean - std, mean + std,
                            color=st["color"], alpha=0.2)
    ax.set_xlabel("# labeled images")
    ax.set_ylabel(field.replace("_", " "))
    ax.set_title(
        f"{field} vs labeling budget (mean ± sample SD, faint per-seed)")
    ax.legend()
    return _save(fig, out_dir, f"curve_{field}.png")


def aubc_chart(results, out_dir, cap_round, field=DEFAULT_METRIC,
               strategies=None):
    fig, ax = plt.subplots(figsize=(7, 5))
    labels, means, stds, colors, n_seeds = [], [], [], [], []
    short_labels = {
        "random": "Random",
        "ppal": "PPAL",
        "rarity_aware": "Rarity-aware",
        "entropy": "Entropy",
    }
    for s in strategies or results.strategies:
        sd_ok = seeds_complete(results, s, cap_round)
        vals = [aubc([r for r in results.by[(s, sd)]
                      if 0 <= int(r["round"]) <= cap_round], field)
                for sd in sd_ok]
        st = STRATEGY_STYLE.get(s, {"color": "k", "label": s})
        labels.append(short_labels.get(s, st["label"]))
        means.append(np.mean(vals) if vals else np.nan)
        stds.append(np.std(vals, ddof=1) if len(vals) > 1 else np.nan)
        colors.append(st["color"])
        n_seeds.append(len(vals))
    bars = ax.bar(labels, means, yerr=stds, color=colors, capsize=4)
    ax.bar_label(bars, labels=[f"{value:.3f}" for value in means],
                 padding=4)
    metric_label = ("test mAP50:95" if field == "test_mAP50_95"
                    else field.replace("_", " "))
    ax.set_ylabel(f"Normalized AUBC ({metric_label})")
    ax.set_title(
        "Budget-averaged performance over 5-20% labeled images\n"
        f"mean ± sample SD, n={max(n_seeds, default=0)} seeds")
    return _save(fig, out_dir, "AUBC.png")


def per_class_ap_chart(results, out_dir, cap_round):
    names = results.names
    x = np.arange(len(names))
    w = 0.8 / max(1, len(results.strategies))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, s in enumerate(results.strategies):
        sd_ok = seeds_complete(results, s, cap_round)
        rows = []
        for sd in sd_ok:
            cap = [r for r in results.by[(s, sd)] if r["round"] == cap_round]
            if cap:
                rows.append(cap[0]["ap"])
        m = np.mean(rows, axis=0) if rows else np.full(len(names), np.nan)
        st = STRATEGY_STYLE.get(s, {"color": "k", "label": s})
        ax.bar(x + (i - (len(results.strategies) - 1) / 2) * w, m, w,
               label=st["label"], color=st["color"])
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("AP@cap")
    ax.set_title("Per-class AP at the budget cap")
    ax.legend()
    return _save(fig, out_dir, "per_class_AP_cap.png")


def class_balance_chart(results, out_dir):
    strategies = list(results.strategies)
    n = max(1, len(strategies))
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), sharey=True,
                             squeeze=False)
    cmap = plt.cm.tab10
    names = results.names
    for ax, s in zip(axes[0], strategies):
        st = STRATEGY_STYLE.get(s, {"label": s})
        ax.set_title(st["label"])
        ax.set_xlabel("AL round")
        sd = next((d for d in results.seeds if (s, d) in results.by), None)
        if sd is None:
            continue
        rs = results.by[(s, sd)]
        share = np.array([[c / (sum(r["nlab"]) or 1) for c in r["nlab"]]
                          for r in rs])
        rounds = [r["round"] for r in rs]
        for c in range(share.shape[1]):
            ax.plot(rounds, share[:, c], color=cmap(c % 10), marker=".",
                    label=names[c])
    axes[0][0].set_ylabel("per-class labeled-instance share\n(first seed per strategy)")
    for ax in axes[0]:               # legend on the first subplot that has data
        if ax.get_lines():
            ax.legend(fontsize=8)
            break
    return _save(fig, out_dir, "class_balance_evo.png")


def confusion_chart(matrix, names, strategy, out_dir, used_seeds=None):
    labels = list(names) + ["background"]
    m = np.asarray(matrix, dtype=float)
    col = m.sum(axis=0, keepdims=True)
    norm = np.divide(m, col, out=np.zeros_like(m), where=col > 0)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("True")
    ax.set_ylabel("Predicted")
    seed_note = (f" (seeds {', '.join(map(str, used_seeds))})"
                 if used_seeds else "")
    ax.set_title(f"Confusion (col-normalized): {strategy}{seed_note}")
    fig.colorbar(im, ax=ax, fraction=0.046)
    return _save(fig, out_dir, f"confusion_{strategy}.png")
