"""Train one random L0 and audit convergence using validation only."""
import argparse
import csv
import hashlib
import json
import math
import random
import sys
from copy import deepcopy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from albench.al.cleanup import release
from albench.al.dataset import write_round_data_yaml
from albench.al.init_set import random_initial
from albench.al.loop import (_assert_initial_coverage, _pool_items,
                             _train_round)
from albench.al.report_io import schedule_fracs
from albench.config import load_config, load_seeds
from albench.evaluation import evaluation_kwargs


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_val(checkpoint: Path, data_yaml: Path, run_dir: Path,
                    name: str, cfg: dict) -> tuple[float, float]:
    from ultralytics import YOLO
    from albench.al.device import device
    model = YOLO(str(checkpoint))
    try:
        metrics = model.val(
            data=str(data_yaml), split="val", device=device(),
            **evaluation_kwargs(cfg, cfg["al"]), project=str(run_dir),
            name=name, plots=False, exist_ok=True, verbose=False)
        return float(metrics.box.map50), float(metrics.box.map)
    finally:
        release(model)


def _verify_existing(run_dir: Path, cfg: dict) -> None:
    provenance_path = run_dir / "protocol_provenance.json"
    if not provenance_path.is_file():
        raise FileNotFoundError(f"missing provenance: {provenance_path}")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    expected = {
        "protocol_id": cfg["protocol_id"],
        "completed_epochs": 100,
        "warmup_epochs": 0.0,
        "close_mosaic": 0,
        "actual_batch": 64,
        "resume": False,
    }
    mismatches = {key: (provenance.get(key), value)
                  for key, value in expected.items()
                  if provenance.get(key) != value}
    if mismatches:
        raise RuntimeError(f"existing run protocol mismatch: {mismatches}")
    last = run_dir / "weights" / "last.pt"
    if provenance.get("checkpoint_sha256") != _sha256(last):
        raise RuntimeError("last.pt SHA-256 does not match provenance")


def _val_map_curve(path: Path) -> list[float]:
    rows = [{key.strip(): value.strip() for key, value in row.items()}
            for row in csv.DictReader(path.open(encoding="utf-8"))]
    if not rows:
        raise RuntimeError(f"empty training results: {path}")
    key = next((name for name in rows[0]
                if "mAP50-95" in name), None)
    if key is None:
        raise RuntimeError(f"mAP50-95 column is missing from {path}")
    values = [float(row[key]) for row in rows]
    if not all(math.isfinite(value) for value in values):
        raise RuntimeError("validation mAP50-95 contains NaN/Inf")
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="sortwaste")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--check-existing", action="store_true",
        help="audit the exact existing preflight checkpoints without training")
    args = parser.parse_args()

    cfg = load_config(args.config)
    allowed = load_seeds()[:cfg["al"]["seeds_n"]]
    if args.seed not in allowed:
        raise ValueError(f"seed must be one of {allowed}")

    al = cfg["al"]
    items = _pool_items(cfg["data_yaml"])
    frac = schedule_fracs(al["schedule"])[0]
    labeled = set(random_initial(items, frac, random.Random(args.seed)))
    _assert_initial_coverage(
        items, labeled, cfg["classes"]["names"],
        al["initial_min_images_per_class"],
        minimum_boxes=al["initial_min_boxes_per_class"],
        minimum_total_boxes=al["initial_min_total_boxes"],
        maximum_background_frac=al["initial_max_background_frac"])

    split_dir = Path(cfg["splits_dir"]) / "preflight"
    split_dir.mkdir(parents=True, exist_ok=True)
    manifest = split_dir / f"l0_s{args.seed}.txt"
    manifest.write_text("\n".join(sorted(labeled)) + "\n", encoding="utf-8")
    round_yaml = split_dir / f"data_l0_s{args.seed}.yaml"
    write_round_data_yaml(cfg["data_yaml"], str(manifest), str(round_yaml))

    preflight = deepcopy(cfg)
    preflight["al"]["project"] = str(Path(al["project"]) / "preflight")
    metrics = None
    try:
        if args.check_existing:
            run_dir = (Path(preflight["al"]["project"])
                       / f"preflight_s{args.seed}_r0")
            _verify_existing(run_dir, cfg)
        else:
            metrics = _train_round(
                "preflight", args.seed, 0, str(round_yaml), preflight,
                evaluate_test=False)
            run_dir = Path(metrics["run_dir"])
            release(metrics.pop("model", None))

        curve = _val_map_curve(run_dir / "results.csv")
        tail = curve[-min(10, len(curve)):]
        slope = ((tail[-1] - tail[0]) / (len(tail) - 1)
                 if len(tail) > 1 else 0.0)
        terminal_gap = max(curve) - curve[-1]
        best_checkpoint = run_dir / "weights" / "best.pt"
        last_checkpoint = run_dir / "weights" / "last.pt"
        if not best_checkpoint.is_file() or not last_checkpoint.is_file():
            raise FileNotFoundError("preflight best.pt/last.pt is missing")
        best_map50, best_map = _checkpoint_val(
            best_checkpoint, round_yaml, run_dir,
            "audit_best_same_validator", cfg)
        last_map50, last_map = _checkpoint_val(
            last_checkpoint, round_yaml, run_dir,
            "audit_last_same_validator", cfg)
        saved_best_gap = best_map - last_map
        audit_config = {
            "protocol_id": cfg["protocol_id"],
            "split": "val",
            "evaluation": evaluation_kwargs(cfg, cfg["al"]),
            "best_sha256": _sha256(best_checkpoint),
            "last_sha256": _sha256(last_checkpoint),
        }
        fingerprint = hashlib.sha256(json.dumps(
            audit_config, sort_keys=True).encode("utf-8")).hexdigest()
        audit = {**audit_config, "fingerprint": fingerprint,
                 "best_mAP50": best_map50, "best_mAP50_95": best_map,
                 "last_mAP50": last_map50, "last_mAP50_95": last_map,
                 "saved_best_reference_gap": saved_best_gap}
        audit_path = run_dir / "preflight_checkpoint_audit.json"
        if audit_path.is_file():
            previous = json.loads(audit_path.read_text(encoding="utf-8"))
            if previous.get("fingerprint") != fingerprint:
                raise RuntimeError("existing checkpoint audit fingerprint differs")
        audit_path.write_text(json.dumps(audit, indent=2) + "\n",
                              encoding="utf-8")
        print("\nL0 validation-only preflight")
        print(f"  epochs: {len(curve)}")
        print(f"  terminal val mAP50-95: {curve[-1]:.6f}")
        print(f"  CSV best-terminal gap (auxiliary): {terminal_gap:.6f}")
        print(f"  last-{len(tail)}-epoch slope: {slope:+.6f}/epoch")
        print(f"  reloaded best val mAP50-95: {best_map:.6f}")
        print(f"  reloaded last val mAP50-95: {last_map:.6f}")
        print(f"  saved-best reference gap: {saved_best_gap:.6f}")
        print(f"  run: {run_dir}")
        failed = saved_best_gap > 0.02 or abs(slope) > 0.001
        if failed:
            print("  status: REVIEW convergence before confirmatory AL")
        else:
            print("  status: PASS provisional convergence gate")
        print("  test split: NOT EVALUATED")
        if failed:
            raise SystemExit(2)
    finally:
        if metrics is not None:
            release(metrics.get("model"))


if __name__ == "__main__":
    main()
