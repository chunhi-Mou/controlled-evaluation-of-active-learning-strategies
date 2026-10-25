"""One (strategy, seed) AL run."""
import csv
import hashlib
import json
import random
from contextlib import contextmanager
from pathlib import Path

from . import state
from .cleanup import release
from .dataset import (pool_images, image_classes, image_class_counts,
                      write_round_data_yaml)
from .device import device
from .init_set import random_initial
from .report_io import csv_columns, schedule_fracs
from .select_entropy import select as _sel_entropy
from .select_random import select as _sel_random
from .select_ppal import select as _sel_ppal
from .select_rarity import select as _sel_rarity
from .protocol import assert_fixed_batch, enforce_fixed_batch

SELECTORS = {
    "random": _sel_random,
    "ppal": _sel_ppal,
    "rarity_aware": _sel_rarity,
    "entropy": _sel_entropy,
}


def _is_final_round(r, al):
    """True when test evaluation must collect the report confusion matrix."""
    return r == len(schedule_fracs(al["schedule"])) - 1


def _validated_confusion(matrix, nc, strategy, seed):
    """Reject missing/empty Ultralytics confusion artifacts before saving."""
    import numpy as np

    value = np.asarray(matrix)
    expected = (nc + 1, nc + 1)
    if value.shape != expected:
        raise RuntimeError(
            f"{strategy} s{seed}: confusion shape {value.shape}, expected "
            f"{expected}")
    if not np.isfinite(value).all():
        raise RuntimeError(f"{strategy} s{seed}: confusion contains NaN/Inf")
    if value.sum() <= 0:
        raise RuntimeError(
            f"{strategy} s{seed}: confusion is empty. Ultralytics only "
            "populates it when test validation runs with plots=True")
    return value


def _pool_items(data_yaml):
    return [(p, image_classes(p)) for p in pool_images(data_yaml)]


@contextmanager
def _csv_lock(csv_path: str):
    """OS-backed cross-process lock for the shared results.csv."""
    lock = Path(str(csv_path) + ".lock")
    lock.parent.mkdir(parents=True, exist_ok=True)
    with open(lock, "a+b") as stream:
        if __import__("os").name == "nt":
            import msvcrt
            if stream.seek(0, 2) == 0:
                stream.write(b"0")
                stream.flush()
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_LOCK, 1)
            try:
                yield
            finally:
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _csv_has_row_unlocked(csv_path: str, strategy: str, seed: int,
                          r: int) -> bool:
    p = Path(csv_path)
    if not p.exists():
        return False
    with open(p, newline="") as f:
        for row in csv.DictReader(f):
            if (row.get("strategy") == strategy
                    and row.get("seed") == str(seed)
                    and row.get("round") == str(r)):
                return True
    return False


def _csv_has_row(csv_path: str, strategy: str, seed: int, r: int) -> bool:
    """True if results.csv already has this row, safe across GPU processes."""
    with _csv_lock(csv_path):
        return _csv_has_row_unlocked(csv_path, strategy, seed, r)


def _csv_rounds(csv_path: str, strategy: str, seed: int) -> set[int]:
    with _csv_lock(csv_path):
        path = Path(csv_path)
        if not path.exists():
            return set()
        with path.open(newline="") as stream:
            return {int(row["round"]) for row in csv.DictReader(stream)
                    if row.get("strategy") == strategy
                    and row.get("seed") == str(seed)}


def _run_fingerprint(cfg: dict, data_yaml: str, keys: list[str],
                     strategy: str) -> str:
    """Reject resume after protocol, code, model or dataset identity changes."""
    from albench.data.labels import label_for

    digest = hashlib.sha256()
    al = dict(cfg["al"])
    # ``workers`` stays in the fingerprint: on spawn-based platforms it can
    # change which worker RNG stream performs a stochastic augmentation.
    for key in ("project", "state_dir", "cache"):
        al.pop(key, None)
    payload = {"strategy": strategy, "classes": cfg["classes"], "al": al}
    digest.update(json.dumps(payload, sort_keys=True, default=str).encode())
    digest.update(Path(data_yaml).read_bytes())
    for source in sorted(Path(__file__).resolve().parents[1].rglob("*.py")):
        digest.update(source.relative_to(Path(__file__).resolve().parents[1]).as_posix().encode())
        digest.update(source.read_bytes())
    model_path = Path(str(al.get("model", "")))
    if model_path.is_file():
        digest.update(model_path.read_bytes())
    for image in keys:
        image_path = Path(image)
        label = label_for(image_path)
        digest.update(image_path.name.encode())
        digest.update(str(image_path.stat().st_size).encode())
        digest.update(label.name.encode())
        digest.update(label.read_bytes() if label.exists() else b"<missing>")
    return digest.hexdigest()


def _sha(paths):
    h = hashlib.sha256()
    for p in sorted(paths):
        h.update(p.encode())
        h.update(b"\n")
    return h.hexdigest()[:16]


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _assert_initial_coverage(items, labeled, names, minimum, *,
                             minimum_boxes=None, minimum_total_boxes=None,
                             maximum_background_frac=None):
    """Audit a random L0 without using labels to redraw or modify it."""
    classes = dict(items)
    counts = [sum(c in classes[path] for path in labeled)
              for c in range(len(names))]
    summary = ", ".join(f"{name}={count}"
                        for name, count in zip(names, counts))
    print(f"L0 images per class: {summary}")
    failed = [name for name, count in zip(names, counts)
              if count < minimum]
    if failed:
        raise RuntimeError(
            f"random L0 has fewer than {minimum} images for classes "
            f"{failed}. Keep this seed and report the protocol failure, "
            "do not redraw it")
    if minimum_boxes is not None or minimum_total_boxes is not None:
        box_counts = [0] * len(names)
        for path in labeled:
            per_class = image_class_counts(path, len(names))
            box_counts = [a + b for a, b in zip(box_counts, per_class)]
        box_summary = ", ".join(
            f"{name}={count}" for name, count in zip(names, box_counts))
        print(f"L0 boxes per class: {box_summary}, total={sum(box_counts)}")
        if (minimum_boxes is not None
                and any(count < minimum_boxes for count in box_counts)):
            raise RuntimeError(
                f"random L0 has a class with fewer than {minimum_boxes} "
                "boxes. Keep this seed and report the protocol failure, "
                "do not redraw it")
        if (minimum_total_boxes is not None
                and sum(box_counts) < minimum_total_boxes):
            raise RuntimeError(
                f"random L0 has fewer than {minimum_total_boxes} total "
                "boxes. Keep this seed and report the protocol failure, "
                "do not redraw it")
    if maximum_background_frac is not None:
        background = sum(not classes[path] for path in labeled)
        fraction = background / len(labeled) if labeled else 1.0
        print(f"L0 background-only: {background}/{len(labeled)} ({fraction:.2%})")
        if fraction > maximum_background_frac:
            raise RuntimeError(
                f"random L0 background fraction {fraction:.2%} exceeds "
                f"{maximum_background_frac:.2%}. Keep this seed and report "
                "the protocol failure, do not redraw it")
    return counts


def _train_round(strategy, seed, r, data_yaml, cfg, *, evaluate_test=True):
    import time
    from ultralytics import YOLO
    from albench.repro import set_seed
    al = cfg["al"]
    nc = cfg["classes"]["nc"]
    collect_confusion = evaluate_test and _is_final_round(r, al)
    set_seed(seed)
    t0 = time.time()
    model = YOLO(al["model"])
    enforce_fixed_batch(
        model, al["batch"], expected_epochs=al["epochs"],
        expected_warmup=al["warmup_epochs"],
        expected_close_mosaic=al["close_mosaic"])
    name = f"{strategy}_s{seed}_r{r}"

    _ema = None
    if strategy == "ppal":
        from .ppal_difficulty import (QualityEMA, make_batch_quality_callbacks)
        pp = al["ppal"]
        _ema = QualityEMA(nc, pp["ema_momentum"])
        for event, fn in make_batch_quality_callbacks(
                _ema, pp["xi"]).items():
            model.add_callback(event, fn)

    proj = str(Path(al["project"]).resolve())
    model_val = None
    try:
        model.train(data=data_yaml, epochs=al["epochs"], imgsz=al["imgsz"],
                    batch=al["batch"], cache=al.get("cache", False),
                    workers=al.get("workers", 8),
                    patience=al["patience"], device=device(),
                    warmup_epochs=al["warmup_epochs"], resume=False,
                    close_mosaic=al["close_mosaic"],
                    rect=al["rect"], max_det=al["max_det"],
                    project=proj, name=name,
                    pretrained=al.get("pretrained", True), seed=seed,
                    plots=True, val=al.get("val", True), exist_ok=True)
        assert_fixed_batch(model, al["batch"])
        run_dir = Path(model.trainer.save_dir)
        ckpt = run_dir / "weights" / f"{al.get('eval_checkpoint', 'last')}.pt"
        if not ckpt.is_file():
            raise FileNotFoundError(f"round checkpoint was not written: {ckpt}")
        init_path = Path(str(al["model"]))
        provenance = {
            "protocol_id": cfg["protocol_id"],
            "strategy": strategy,
            "seed": seed,
            "round": r,
            "initialization": str(al["model"]),
            "initialization_sha256": (
                _file_sha256(init_path) if init_path.is_file() else None),
            "resume": bool(getattr(model.trainer.args, "resume", False)),
            "configured_epochs": int(al["epochs"]),
            "completed_epochs": int(model.trainer.epoch) + 1,
            "warmup_epochs": float(model.trainer.args.warmup_epochs),
            "close_mosaic": int(model.trainer.args.close_mosaic),
            "actual_batch": int(model.trainer.batch_size),
            "batches_per_epoch": len(model.trainer.train_loader),
            "evaluation_and_acquisition_checkpoint": str(ckpt.resolve()),
            "checkpoint_sha256": _file_sha256(ckpt),
            "evaluation": cfg["evaluation"],
        }
        (run_dir / "protocol_provenance.json").write_text(
            json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
        print("Round protocol: " + json.dumps(provenance, sort_keys=True))
        # Avoid two generations of dataloader workers at once.
        release(model)
        model = None
        model_val = YOLO(str(ckpt))
        from albench.evaluation import evaluation_kwargs
        eval_args = evaluation_kwargs(cfg, al)
        em = model_val.val(data=data_yaml, split="val", device=device(),
                           **eval_args,
                           project=proj, name=f"{name}_val", plots=False,
                           exist_ok=True, verbose=False)
        tm = None
        if evaluate_test:
            tm = model_val.val(data=data_yaml, split="test", device=device(),
                               **eval_args,
                               project=proj, name=f"{name}_test",
                               plots=collect_confusion,
                               exist_ok=True, verbose=False)
        ap = ([float(x) for x in tm.box.maps[:nc]] if tm is not None
              else [float("nan")] * nc)
        if _ema is not None:
            model_val._ppal_class_quality = _ema.value
        return {"val_map50": float(em.box.map50),
                "val_map50_95": float(em.box.map),
                "test_map50": (float(tm.box.map50) if tm is not None
                               else float("nan")),
                "test_map50_95": (float(tm.box.map) if tm is not None
                                  else float("nan")),
                "ap": ap, "train_s": time.time() - t0,
                "confusion": (tm.confusion_matrix.matrix
                              if tm is not None and collect_confusion else None),
                "ppal_quality": (_ema.value if _ema is not None else None),
                "model": model_val, "run_dir": str(run_dir)}
    except Exception:
        release(model_val)
        raise
    finally:
        release(model)


def run(strategy, seed, cfg, data_yaml=None, *, csv_path=None,
        train_fn=_train_round, select_fns=SELECTORS, stop_after_round=None):
    al = cfg["al"]
    names = cfg["classes"]["names"]
    data_yaml = data_yaml or cfg["data_yaml"]
    csv_path = csv_path or str(Path(cfg["reports_dir"]) / "al" / "results.csv")
    splits = Path(cfg["splits_dir"])
    items = _pool_items(data_yaml)
    keys = [k for k, _ in items]
    fingerprint = _run_fingerprint(cfg, data_yaml, keys, strategy)
    fracs = schedule_fracs(al["schedule"])
    R = len(fracs) - 1
    if stop_after_round is not None and not 0 <= stop_after_round <= R:
        raise ValueError(
            f"stop_after_round={stop_after_round} outside 0..{R}")

    rng = random.Random(seed)
    # Every strategy receives the same uniform-random L0 for a given seed.
    # Labels are not used to build it.
    labeled = set(random_initial(items, fracs[0], rng))
    _assert_initial_coverage(
        items, labeled, names, al["initial_min_images_per_class"],
        minimum_boxes=al["initial_min_boxes_per_class"],
        minimum_total_boxes=al["initial_min_total_boxes"],
        maximum_background_frac=al["initial_max_background_frac"])

    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    with _csv_lock(csv_path):
        if not Path(csv_path).exists():
            with open(csv_path, "w", newline="") as f:
                csv.writer(f).writerow(csv_columns(names))

    # every done round must have a CSV row
    _missing = [r for r in range(R + 1)
                if state.is_done(al["state_dir"], strategy, seed, r)
                and not _csv_has_row(csv_path, strategy, seed, r)]
    if _missing:
        raise RuntimeError(
            f"CSV/state inconsistency for {strategy} s{seed}: state says "
            f"rounds {_missing} are done but results.csv has no row for "
            f"them. Delete the orphaned state file(s) and re-run.")
    _orphan_csv = sorted(
        r for r in _csv_rounds(csv_path, strategy, seed)
        if not state.is_done(al["state_dir"], strategy, seed, r))
    if _orphan_csv:
        raise RuntimeError(
            f"CSV/state inconsistency for {strategy} s{seed}: results.csv "
            f"contains rounds {_orphan_csv} but state is missing. Remove the "
            "orphan CSV row(s) before re-running, so that old metrics are not "
            "silently kept after retraining.")

    for r in range(R + 1):
        if state.is_done(al["state_dir"], strategy, seed, r):
            d = state.load(al["state_dir"], strategy, seed, r)
            if d.get("fingerprint") != fingerprint:
                raise RuntimeError(
                    f"resume fingerprint mismatch for {strategy} s{seed} r{r}: "
                    "config, code, model or dataset changed")
            labeled = set(d["next_labeled"])
            if "rng_state" in d:
                rng.setstate(state.decode_rng_state(d["rng_state"]))
            if stop_after_round is not None and r >= stop_after_round:
                break
            continue
        splits.mkdir(parents=True, exist_ok=True)
        txt = splits / f"al_{strategy}_s{seed}_r{r}.txt"
        txt.write_text("\n".join(sorted(labeled)) + "\n")
        ry = splits / f"data_al_{strategy}_s{seed}_r{r}.yaml"
        write_round_data_yaml(data_yaml, str(txt), str(ry))
        m = train_fn(strategy, seed, r, str(ry), cfg)
        if r == R:
            import numpy as np
            cm_path = Path(csv_path).parent / f"confusion_{strategy}_s{seed}.npy"
            np.save(cm_path, _validated_confusion(
                m.get("confusion"), len(names), strategy, seed))
        if strategy == "ppal" and m.get("ppal_quality") is not None:
            import numpy as np
            np.save(Path(csv_path).parent /
                    f"ppal_quality_{strategy}_s{seed}_r{r}.npy",
                    np.asarray(m["ppal_quality"]))
        picks = []
        if r < R:
            cands = sorted(set(keys) - labeled)
            nxt = max(1, round(len(keys) * fracs[r + 1]))
            want = nxt - len(labeled)
            picks = select_fns[strategy](
                m.get("model"), cands, labeled, want, cfg, rng)
            assert len(set(picks)) == len(picks), (
                f"{strategy} s{seed} r{r}: query has "
                f"{len(picks) - len(set(picks))} duplicate(s)")
            assert not (set(picks) & labeled), (
                f"{strategy} s{seed} r{r}: query re-selects "
                f"{len(set(picks) & labeled)} already-labeled image(s)")
            assert len(picks) == want, (
                f"{strategy} s{seed} r{r}: asked for {want} images, "
                f"selector returned {len(picks)}")
            next_labeled = sorted(labeled | set(picks))
            _diag = getattr(m.get("model"), "_ppal_acq_diag", None)
            if strategy == "ppal" and _diag is not None:
                import numpy as np
                np.savez_compressed(
                    Path(csv_path).parent /
                    f"ppal_acq_{strategy}_s{seed}_r{r}.npz", **_diag)
            _rarity_diag = getattr(m.get("model"), "_rarity_acq_diag", None)
            if strategy == "rarity_aware" and _rarity_diag is not None:
                import numpy as np
                np.savez_compressed(
                    Path(csv_path).parent /
                    f"rarity_acq_{strategy}_s{seed}_r{r}.npz",
                    **_rarity_diag)
            _entropy_diag = getattr(m.get("model"), "_entropy_acq_diag", None)
            if strategy == "entropy" and _entropy_diag is not None:
                import numpy as np
                np.savez_compressed(
                    Path(csv_path).parent /
                    f"entropy_acq_{strategy}_s{seed}_r{r}.npz",
                    **_entropy_diag)
        else:
            next_labeled = sorted(labeled)
        # Last use of the eval model. Without this its workers outlive the round.
        release(m.pop("model", None))
        ap = m.get("ap") or [0.0] * len(names)
        nlab = _class_counts(labeled, len(names))
        # write the CSV row before marking state done
        with _csv_lock(csv_path):
            if not _csv_has_row_unlocked(csv_path, strategy, seed, r):
                with open(csv_path, "a", newline="") as f:
                    csv.writer(f).writerow([
                        strategy, seed, r, len(labeled), fracs[r],
                        m["val_map50"], m["test_map50"],
                        m.get("test_map50_95", 0.0),
                        *ap, *nlab, _sha(labeled), _sha(picks),
                        _sha(next_labeled), m.get("train_s", 0.0)])
                    f.flush()
                    import os as _os
                    _os.fsync(f.fileno())
        state.write_atomic(al["state_dir"], strategy, seed, r, {
            "labeled": sorted(labeled), "next_labeled": next_labeled,
            "rng_state": state.encode_rng_state(rng.getstate()),
            "fingerprint": fingerprint,
            "labeled_sha": _sha(labeled), "query_sha": _sha(picks),
            "next_labeled_sha": _sha(next_labeled),
            "val_map50": m["val_map50"], "test_map50": m["test_map50"]})
        labeled = set(next_labeled)
        if stop_after_round is not None and r >= stop_after_round:
            break


def _class_counts(labeled, nc):
    cnt = [0] * nc
    for k in labeled:
        for c, value in enumerate(image_class_counts(k, nc)):
            cnt[c] += value
    return cnt
