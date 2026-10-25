"""results.csv + confusion .npy -> figures, tables, PROVENANCE."""
import argparse
import itertools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from albench.al import charts, health, metrics, ppal_charts, tables
from albench.al.report_io import (load_confusion, load_results,
                                   schedule_fracs)
from albench.config import load_config

DEFAULT_CONFIG = "sortwaste"


def _endpoint(name: str, field: str, cap_round: int):
    """al.analysis.primary_endpoint as a one-arg callable over a seed's rounds."""
    if name == "aubc":
        def aubc_to_cap(rs):
            selected = [r for r in rs if 0 <= int(r["round"]) <= cap_round]
            round_ids = [int(r["round"]) for r in selected]
            if len(round_ids) != len(set(round_ids)):
                raise ValueError(
                    f"duplicate rounds in normalized AUBC input: {round_ids}"
                )
            if set(round_ids) != set(range(cap_round + 1)):
                return float("nan")
            return metrics.aubc(selected, field)
        return aubc_to_cap
    return lambda rs: metrics.map_at_cap(rs, cap_round, field)


def _pairs(strategies: list) -> list:
    """Every unordered strategy pair, later-listed one first, so the sign of
    meanΔ reads as "the newer strategy minus the older one"."""
    return [(b, a) for a, b in itertools.combinations(strategies, 2)]


def _provenance(cfg, res, used_seeds) -> str:
    import numpy
    al = cfg["al"]
    realized_budgets = sorted({
        int(row["n_labeled"])
        for rounds in res.by.values()
        for row in rounds
        if "n_labeled" in row
    })
    cap_frac = float(al["schedule"]["cap_frac"])
    n_train = (round(realized_budgets[-1] / cap_frac)
               if realized_budgets else "unknown")
    lines = [
        "AL benchmark provenance", "",
        f"strategies: {res.strategies}",
        f"seeds present: {res.seeds}  (complete to cap: {used_seeds})",
        f"schedule: {al['schedule']}",
        f"analysis: {al['analysis']}",
        "metric summary: normalized_AUBC_5_20, budget_axis=n_labeled, "
        f"evaluated_interval=[{al['schedule']['init_frac']}, "
        f"{al['schedule']['cap_frac']}], dispersion=sample_SD",
        f"realized_budget_range_images="
        f"[{realized_budgets[0]}, {realized_budgets[-1]}], N_train={n_train}"
        if realized_budgets else "realized_budget_range_images=unknown",
        "detector protocol: "
        f"model={al['model']} epochs={al['epochs']} patience={al['patience']} "
        f"batch={al['batch']} rect={al['rect']} max_det={al['max_det']} "
        f"checkpoint={al['eval_checkpoint']}.pt",
        f"PPAL selector: {al['ppal']}",
        f"Rarity-aware selector: {al['rarity_aware']}",
        f"Entropy selector: {al['entropy']}",
        f"numpy {numpy.__version__}",
        "Sources: PPAL arXiv:2211.11612. Rarity-aware stratified AL, "
        "Applied Sciences 2026, 16(3), 1236, DOI:10.3390/app16031236. "
        "Entropy + ENMS + DivProto arXiv:2204.07965. Random: Settles 2009, "
        "TR1648. Metric: trapezoidal AUBC (Zhan et al., IJCAI 2021) "
        "divided by the budget span (as in BAOD, CVPRW 2021).", ""]
    if len(res.seeds) < 3:
        lines[4:4] = [
            "SCOPE WARNING: fewer than three seeds are present, so the mean, "
            "SD and strategy order are not cross-seed summaries.",
            "",
        ]
    return "\n".join(lines)


def run(cfg: dict, csv_path: str, confusion_dir: str, out_dir: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    an = cfg["al"]["analysis"]
    metric = an["metric"]
    fracs = schedule_fracs(cfg["al"]["schedule"])
    R = len(fracs) - 1
    endpoint = _endpoint(an["primary_endpoint"], metric, R)
    res = load_results(csv_path)
    strategy_order = [s for s in cfg["al"]["strategies"]
                      if s in res.strategies]
    strategy_order += [s for s in res.strategies
                       if s not in strategy_order]

    # Coverage banner
    full = {s: metrics.seeds_complete(res, s, R) for s in strategy_order}
    is_partial = any(len(full[s]) < len(res.seeds) for s in strategy_order) \
        or not strategy_order
    cover_rows = [[c["strategy"], c["seed"], c["n_rounds"],
                   "yes" if c["seed"] in full.get(c["strategy"], []) else "no"]
                  for c in res.coverage]
    cover_md = tables.md_table(
        ["strategy", "seed", "rounds_done", "complete_to_cap"], cover_rows)

    # Endpoint per strategy, mean and sample SD over seeds. With one seed the
    # sample SD is undefined and reported as nan.
    aubc_ps = {s: metrics.per_seed_metric(res, s, endpoint)
               for s in strategy_order}
    summary_rows = []
    summary_numeric = []
    for s in strategy_order:
        vals = [aubc_ps[s][sd] for sd in full[s]]
        mcap = [metrics.map_at_cap(res.by[(s, sd)], R, metric)
                for sd in full[s]]
        mean_aubc = np.mean(vals) if vals else float("nan")
        sd_aubc = (np.std(vals, ddof=1) if len(vals) > 1
                   else float("nan"))
        mean_cap = np.mean(mcap) if mcap else float("nan")
        sd_cap = (np.std(mcap, ddof=1) if len(mcap) > 1
                  else float("nan"))
        label = charts.STRATEGY_STYLE.get(s, {"label": s})["label"]
        summary_rows.append([
            label,
            tables.fmt_pm(mean_aubc, sd_aubc),
            tables.fmt_pm(mean_cap, sd_cap),
            len(full[s])])
        summary_numeric.append([
            s, mean_aubc, sd_aubc, mean_cap, sd_cap, len(full[s])])

    # Per-seed values and paired gaps between strategies.
    seed_rows = [[sd] + [tables.fmt(aubc_ps[s].get(sd, float("nan")), 4)
                         for s in strategy_order] for sd in res.seeds]
    gap_lines = []
    for a, b in _pairs(cfg["al"]["strategies"]):
        if a not in res.strategies or b not in res.strategies:
            continue
        d = metrics.paired_diffs(res, a, b, endpoint, R)
        v = np.array(list(d.values()), dtype=float)
        gap_lines.append(
            f"{a} - {b}: n={v.size}  meanΔ="
            f"{v.mean() if v.size else float('nan'):+.4f}  "
            f"sampleSD={v.std(ddof=1) if v.size > 1 else float('nan'):.4f}  "
            + " ".join(f"s{sd}={d[sd]:+.4f}" for sd in sorted(d)))

    # Figures
    for field in dict.fromkeys([metric, "test_mAP50", "val_mAP50"]):
        charts.curve_metric(res, field, str(out), R)
    charts.aubc_chart(res, str(out), R, metric, strategy_order)
    charts.per_class_ap_chart(res, str(out), R)
    charts.class_balance_chart(res, str(out))
    for s in res.strategies:
        cm, used = load_confusion(confusion_dir, s, res.seeds)
        if cm is not None:
            charts.confusion_chart(cm, res.names, s, str(out), used)

    # PPAL adaptation diagnostics
    ppal_figs = []
    if ("ppal" in res.strategies
            and ppal_charts.has_quality(confusion_dir, res.seeds)):
        # Each draw returns None on a partial run, so keep only written PNGs.
        ppal_figs = [p for p in (
            ppal_charts.ppal_difficulty_evo(confusion_dir, res.seeds,
                                            res.names, str(out), R),
            ppal_charts.ppal_difficulty_vs_ap(res, confusion_dir, res.seeds,
                                              res.names, str(out), R),
            ppal_charts.ppal_class_selection(res, str(out), R)) if p]
    # Acquisition files are written at a different point of the loop.
    if "ppal" in res.strategies and ppal_charts.has_acq(confusion_dir,
                                                        res.seeds):
        ppal_figs += [p for p in (
            ppal_charts.ppal_stage2_coverage(confusion_dir, res.seeds,
                                             str(out), R),
            ppal_charts.ppal_stage1_cutoff(confusion_dir, res.seeds,
                                           str(out), R)) if p]

    # Stability / no-overfit figures
    health.health_seed_variance(res, str(out))
    health.health_converge_heatmap(res, str(out))
    health.health_gap_overview(res, cfg["al"]["project"], str(out))
    base_glob = sorted(Path(cfg["baseline"]["project"]).glob(
        "baseline_frac1.0_*/results.csv"))
    if base_glob:
        health.health_baseline(base_glob[0], str(out))

    import csv as _csv
    aubc_tbl = [r[:3] for r in summary_rows]
    tex = tables.latex_table(
        ["Strategy", "Normalized AUBC, 5--20%", "mAP@20%"],
        aubc_tbl,
        caption=f"Normalized AUBC over 5--20\\% labeled images and "
                f"20\\% endpoint on {metric} (mean $\\pm$ sample SD)",
        label="tbl:aubc")
    (out / "tbl_aubc.tex").write_text(tex, encoding="utf-8")
    with (out / "tbl_aubc.csv").open("w", newline="", encoding="utf-8") as f:
        w = _csv.writer(f)
        w.writerow(["strategy", "nAUBC_mean", "nAUBC_sample_sd",
                    "endpoint_mean", "endpoint_sample_sd", "n_seeds"])
        w.writerows(summary_numeric)

    (out / "PROVENANCE").write_text(
        _provenance(cfg, res, {s: full[s] for s in strategy_order}),
        encoding="utf-8")
    endpoint_label = ("normalized_AUBC_5_20"
                      if an["primary_endpoint"] == "aubc"
                      else an["primary_endpoint"])
    (out / "endpoint_by_seed.md").write_text(
        f"{endpoint_label} on {metric}, one row per seed\n\n"
        + tables.md_table(["seed"] + strategy_order, seed_rows)
        + "\n" + "\n".join(gap_lines) + "\n", encoding="utf-8")
    scope_note = ("\nCoverage is per present seed. With fewer than three "
                  "seeds the summaries are not cross-seed results.\n"
                  if len(res.seeds) < 3 else "\n")
    (out / "coverage.txt").write_text(cover_md + scope_note,
                                       encoding="utf-8")
    print(f"report -> {out}/  partial={is_partial}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--out", default=None,
                    help="output dir, defaults to <reports_dir>/al")
    a = ap.parse_args()
    cfg = load_config(Path(a.config))
    rd = Path(cfg["reports_dir"]) / "al"
    run(cfg, str(rd / "results.csv"), str(rd), a.out or str(rd))


if __name__ == "__main__":
    main()
