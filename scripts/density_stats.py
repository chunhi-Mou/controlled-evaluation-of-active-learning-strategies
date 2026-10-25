"""Object density of a YOLO dataset: boxes and classes per image, class counts.

On a dataset where almost every image holds one box of one class, choosing
images is the same as choosing classes, so image-level acquisition strategies
can hardly differ. This diagnostic is not part of the numbered pipeline.

    python scripts/density_stats.py --config sortwaste --split train
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from albench.config import load_config
from albench.data.labels import IMG_EXT, labels_path

DEFAULT_CONFIG = "sortwaste"


def _pct(n: int, d: int) -> str:
    return f"{100 * n / d:5.1f}%" if d else "  n/a"


def _hist(counts: Counter, total: int, label: str, top: int = 8) -> None:
    print(f"\n{label}")
    for k in sorted(counts)[:top]:
        n = counts[k]
        print(f"  {k:>3} | {n:7} {_pct(n, total)} {'#' * int(40 * n / total)}")
    rest = sum(n for k, n in counts.items() if k > sorted(counts)[top - 1]) \
        if len(counts) > top else 0
    if rest:
        print(f"  >{sorted(counts)[top - 1]} | {rest:7} {_pct(rest, total)}")


def scan(images_dir: Path):
    """(boxes_per_image, classes_per_image, class_instances, areas)."""
    lbl_dir = labels_path(images_dir)
    boxes, classes, inst, areas = Counter(), Counter(), Counter(), []
    n = 0
    for img in sorted(images_dir.iterdir()):
        if img.suffix.lower() not in IMG_EXT:
            continue
        n += 1
        p = (lbl_dir / img.name).with_suffix(".txt")
        if not p.exists():
            boxes[0] += 1
            classes[0] += 1
            continue
        cls, nb = set(), 0
        for ln in p.read_text().splitlines():
            f = ln.split()
            if len(f) != 5:
                continue
            nb += 1
            cls.add(int(float(f[0])))
            inst[int(float(f[0]))] += 1
            areas.append(float(f[3]) * float(f[4]))
        boxes[nb] += 1
        classes[len(cls)] += 1
    return n, boxes, classes, inst, areas


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--split", default="train",
                    help="train | val | test (of the export tree)")
    a = ap.parse_args()

    cfg = load_config(Path(a.config))
    images = Path(cfg["data_yaml"]).parent / a.split / "images"
    if not images.is_dir():                      # fall back to the raw root
        d = cfg["dataset"]
        images = Path(d["root"]) / d["images_dir"]
    if not images.is_dir():
        raise SystemExit(f"no images dir found (tried {images})")

    n, boxes, classes, inst, areas = scan(images)
    if not n:
        raise SystemExit(f"{images} has no images")
    total_boxes = sum(inst.values())
    names = cfg["classes"]["names"]

    print(f"{cfg['tag']} / {a.split}  <- {images}")
    print(f"images {n}   boxes {total_boxes}   "
          f"mean {total_boxes / n:.2f} boxes/image")

    _hist(boxes, n, "boxes per image")
    _hist(classes, n, "distinct classes per image")

    print("\nclass instances")
    for c in sorted(inst, key=lambda k: -inst[k]):
        nm = names[c] if c < len(names) else f"?{c}"
        print(f"  {nm:<16} {inst[c]:7} {_pct(inst[c], total_boxes)}")
    if inst:
        hi, lo = max(inst.values()), min(inst.values())
        print(f"  imbalance ratio (max/min): {hi / lo:.1f}x")

    if areas:
        areas.sort()
        q = [areas[int(p * (len(areas) - 1))] for p in (.05, .25, .5, .75, .95)]
        print("\nbox area (fraction of image)")
        print("  p05 %.4f  p25 %.4f  p50 %.4f  p75 %.4f  p95 %.4f" % tuple(q))
        tiny = sum(1 for x in areas if x < 0.01)
        print(f"  < 1% of image: {tiny} ({_pct(tiny, len(areas))})")

    multi = n - boxes.get(0, 0) - boxes.get(1, 0)
    multi_cls = sum(v for k, v in classes.items() if k > 1)
    print(f"\nVERDICT  multi-object images {multi} ({_pct(multi, n)}), "
          f"multi-class images {multi_cls} ({_pct(multi_cls, n)})")
    if multi_cls / n < 0.2:
        print("  <20% multi-class: class-balanced acquisition degenerates to "
              "picking a class per image, which stratified random already does. "
              "Expect a narrow gap between strategies.")


if __name__ == "__main__":
    main()
