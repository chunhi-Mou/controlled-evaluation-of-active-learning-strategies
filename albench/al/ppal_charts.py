"""PPAL adaptation diagnostics (Yang et al., CVPR 2024, arXiv:2211.11612).
Reads ppal_quality_*.npy + ppal_acq_*.npz + results.csv."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from .charts import STRATEGY_STYLE, _save
from .metrics import seeds_complete


def _quality_paths(reports_dir, seed):
    return list(Path(reports_dir).glob(f"ppal_quality_ppal_s{seed}_r*.npy"))


def has_quality(reports_dir, seeds) -> bool:
    return any(_quality_paths(reports_dir, sd) for sd in seeds)


def load_difficulty(reports_dir, seeds) -> dict:
    """Per seed: (rounds sorted, (len(rounds), nc) array of 1 - quality EMA).
    Seeds with no npy are skipped (ragged-safe)."""
    out = {}
    for sd in seeds:
        rounds, rows = [], []
        for p in _quality_paths(reports_dir, sd):
            rounds.append(int(p.stem.split("_r")[-1]))
            rows.append(1.0 - np.asarray(np.load(p), float))
        if rounds:
            order = np.argsort(rounds)
            out[sd] = ([rounds[i] for i in order],
                       np.array([rows[i] for i in order], float))
    return out


def selection_share_at_cap(results, strategies, cap_round) -> dict:
    """Per strategy: (nc,) mean per-class labeled-instance share at cap_round,
    over the strategy's AUBC-complete seeds (same filter as per_class_ap_chart)."""
    out = {}
    for s in strategies:
        if s not in results.strategies:
            continue
        rows = []
        for sd in seeds_complete(results, s, cap_round):
            for rec in results.by[(s, sd)]:
                if rec["round"] == cap_round:
                    tot = sum(rec["nlab"]) or 1
                    rows.append([n / tot for n in rec["nlab"]])
        if rows:
            out[s] = np.mean(rows, axis=0)
    return out


def difficulty_ap_points(results, reports_dir, seeds, nc) -> list:
    """(class, round, mean difficulty, mean AP) joined over the seeds where
    both the difficulty npy round and the PPAL AP row exist (ragged-safe)."""
    diff = load_difficulty(reports_dir, seeds)
    idx = {sd: {r: i for i, r in enumerate(rr)} for sd, (rr, _) in diff.items()}
    ap = {}
    for sd in seeds:
        for rec in results.by.get(("ppal", sd), []):
            ap[(sd, rec["round"])] = rec["ap"]
    pts = []
    rounds = sorted({r for rr, _ in diff.values() for r in rr})
    for rd in rounds:
        for c in range(nc):
            ds, aps = [], []
            for sd, (_, arr) in diff.items():
                if rd in idx[sd] and (sd, rd) in ap:
                    ds.append(float(arr[idx[sd][rd], c]))
                    aps.append(float(ap[(sd, rd)][c]))
            if ds:
                pts.append((c, rd, float(np.mean(ds)), float(np.mean(aps))))
    return pts


def ppal_difficulty_evo(reports_dir, seeds, names, out_dir, cap_round):
    """A. Per-class difficulty d = 1 - quality EMA vs AL round, with the mean and
    sample SD over seeds, with faint per-seed lines."""
    diff = load_difficulty(reports_dir, seeds)
    if not diff:
        return None
    common = sorted(set.intersection(*[set(rr) for rr, _ in diff.values()])) \
        if len(diff) > 1 else sorted(next(iter(diff.values()))[0])
    common = [r for r in common if r <= cap_round]
    if not common:
        return None
    # each seed's rows for the common rounds, sliced once (reused per class)
    subs = []
    for rr, arr in diff.values():
        pos = {r: i for i, r in enumerate(rr)}
        subs.append(arr[[pos[r] for r in common]])      # (len(common), nc)
    cmap = plt.cm.tab10
    fig, ax = plt.subplots(figsize=(7, 5))
    for c in range(len(names)):
        col = cmap(c % 10)
        curves = [sub[:, c] for sub in subs]
        for y in curves:
            ax.plot(common, y, color=col, alpha=0.15, linewidth=0.8)
        a = np.array(curves, float)
        mean = a.mean(axis=0)
        ax.plot(common, mean, color=col, marker="o", label=names[c])
        if a.shape[0] > 1:
            std = a.std(axis=0, ddof=1)
            ax.fill_between(common, mean - std, mean + std,
                            color=col, alpha=0.2)
    ax.set_xlabel("AL round")
    ax.set_ylabel("class difficulty  d = 1 - quality EMA")
    ax.set_title("DCUS per-class difficulty (per-round final EMA)")
    ax.legend()
    return _save(fig, out_dir, "ppal_difficulty_evo.png")


def ppal_difficulty_vs_ap(results, reports_dir, seeds, names, out_dir,
                          cap_round):
    """B. Scatter per-class AP (x) vs difficulty (y), one point per
    (class, round), coloured per class. The title shows Spearman's rho."""
    pts = [p for p in difficulty_ap_points(results, reports_dir, seeds,
                                            len(names)) if p[1] <= cap_round]
    if not pts:
        return None
    cmap = plt.cm.tab10
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for c in range(len(names)):
        xs = [ap for (cc, _, _, ap) in pts if cc == c]
        ys = [d for (cc, _, d, _) in pts if cc == c]
        if xs:
            ax.scatter(xs, ys, color=cmap(c % 10), label=names[c], s=30)
    allx = [ap for (_, _, _, ap) in pts]
    ally = [d for (_, _, d, _) in pts]
    rho = spearmanr(allx, ally).correlation if len(allx) > 1 else float("nan")
    ax.set_xlabel("per-class AP")
    ax.set_ylabel("class difficulty  d = 1 - quality EMA")
    ax.set_title(f"DCUS calibration: difficulty vs AP  (Spearman rho={rho:.2f})")
    ax.legend()
    return _save(fig, out_dir, "ppal_difficulty_vs_ap.png")


def ppal_class_selection(results, out_dir, cap_round):
    """C. Per-class labeled-instance share at cap: PPAL vs Random grouped bars
    (mean over complete seeds). Random ~ pool-natural distribution (caption)."""
    share = selection_share_at_cap(results, ["ppal", "random"], cap_round)
    if "ppal" not in share:
        return None
    names = results.names
    x = np.arange(len(names))
    order = [s for s in ("ppal", "random") if s in share]
    w = 0.8 / len(order)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, s in enumerate(order):
        st = STRATEGY_STYLE.get(s, {"color": "k", "label": s})
        ax.bar(x + (i - (len(order) - 1) / 2) * w, share[s], w,
               color=st["color"], label=st["label"])
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("labeled-instance share @cap")
    note = "" if "random" in share else "  (Random baseline absent)"
    ax.set_title("CCMS class selection vs Random" + note)
    ax.legend()
    return _save(fig, out_dir, "ppal_class_selection.png")


# Stage-1/Stage-2 acquisition evidence

def _acq_paths(reports_dir, seed):
    return list(Path(reports_dir).glob(f"ppal_acq_ppal_s{seed}_r*.npz"))


def has_acq(reports_dir, seeds) -> bool:
    return any(_acq_paths(reports_dir, sd) for sd in seeds)


def load_acq(reports_dir, seeds) -> dict:
    """Per seed: (rounds sorted asc, [npz-dict per round]). Ragged-safe."""
    out = {}
    for sd in seeds:
        rounds, recs = [], []
        for p in _acq_paths(reports_dir, sd):
            rounds.append(int(p.stem.split("_r")[-1]))
            with np.load(p) as z:
                recs.append({k: z[k] for k in z.files})
        if rounds:
            order = np.argsort(rounds)
            out[sd] = ([rounds[i] for i in order],
                       [recs[i] for i in order])
    return out


def ppal_stage2_coverage(reports_dir, seeds, out_dir, cap_round):
    """D. Stage-2 medoid coverage (lower = better) and spread (higher = better)
    per round vs a same-size random-subset null."""
    acq = load_acq(reports_dir, seeds)
    if not acq:
        return None
    rounds = sorted({r for rr, _ in acq.values() for r in rr
                     if r <= cap_round})
    if not rounds:
        return None

    def _per_round(key_pick, key_null):
        pick_mean, pick_seed, null_mu, null_sd = [], [], [], []
        for rd in rounds:
            pv, nv = [], []
            for sd, (rr, recs) in acq.items():
                if rd in rr:
                    rec = recs[rr.index(rd)]
                    pv.append(float(rec[key_pick]))
                    nv.append(np.asarray(rec[key_null], float))
            pick_seed.append(pv)
            pick_mean.append(np.mean(pv))
            allnull = np.concatenate(nv)
            null_mu.append(allnull.mean())
            null_sd.append(allnull.std())
        return (np.array(pick_mean), pick_seed,
                np.array(null_mu), np.array(null_sd))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    panels = [("coverage_picked", "coverage_null",
               "coverage = mean min-dist pool->set  (lower = better)",
               "Stage-2 coverage vs random-null"),
              ("spread_picked", "spread_null",
               "spread = min within-set pairwise dist  (higher = better)",
               "Stage-2 spread vs random-null")]
    for ax, (kp, kn, ylab, title) in zip(axes, panels):
        pm, ps, nm, nsd = _per_round(kp, kn)
        ax.fill_between(rounds, nm - nsd, nm + nsd, color="0.7", alpha=0.5,
                        label="random subsets (null, ±σ)")
        ax.plot(rounds, nm, color="0.4", linestyle="--", linewidth=1)
        for i, rd in enumerate(rounds):           # faint per-seed points
            ax.scatter([rd] * len(ps[i]), ps[i],
                       color=STRATEGY_STYLE["ppal"]["color"], alpha=0.35,
                       s=18)
        ax.plot(rounds, pm, color=STRATEGY_STYLE["ppal"]["color"],
                marker="o", label="CCMS picks (mean over seeds)")
        ax.set_xlabel("AL round (model that scored/picked)")
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.legend(fontsize=8)
    fig.suptitle("CCMS Stage-2 selection vs same-pool random null",
                 fontsize=10)
    return _save(fig, out_dir, "ppal_stage2_coverage.png")


def ppal_stage1_cutoff(reports_dir, seeds, out_dir, cap_round):
    """E. Per round: Stage-1 pool-uncertainty distribution (median + percentile
    bands) and the mean score at the δ·b cutoff."""
    acq = load_acq(reports_dir, seeds)
    if not acq:
        return None
    rounds = sorted({r for rr, _ in acq.values() for r in rr
                     if r <= cap_round})
    if not rounds:
        return None
    qs = {q: [] for q in (5, 25, 50, 75, 95)}
    cut_scores = []
    for rd in rounds:
        vals, cuts = [], []
        for sd, (rr, recs) in acq.items():
            if rd in rr:
                rec = recs[rr.index(rd)]
                s = np.asarray(rec["stage1_scores"], float)
                vals.append(s)
                n_cut = int(rec["pool_n"])
                if 0 < n_cut <= s.size:
                    cuts.append(float(s[n_cut - 1]))
        allv = np.concatenate(vals)
        for q in qs:
            qs[q].append(np.percentile(allv, q))
        cut_scores.append(np.mean(cuts) if cuts else np.nan)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.fill_between(rounds, qs[5], qs[95], color="tab:blue", alpha=0.15,
                    label="pool uncertainty 5-95%")
    ax.fill_between(rounds, qs[25], qs[75], color="tab:blue", alpha=0.3,
                    label="pool uncertainty 25-75%")
    ax.plot(rounds, qs[50], color="tab:blue", label="pool median")
    ax.plot(rounds, cut_scores, color=STRATEGY_STYLE["ppal"]["color"],
            marker="o", label="score at δ·b cutoff")
    ax.set_xlabel("AL round (model that scored)")
    ax.set_ylabel("DCUS image uncertainty  U(I)")
    ax.set_title("DCUS Stage-1: pool uncertainty distribution + candidate "
                 "cutoff")
    ax.legend(fontsize=8)
    return _save(fig, out_dir, "ppal_stage1_cutoff.png")
