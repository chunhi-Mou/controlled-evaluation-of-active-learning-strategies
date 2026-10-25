"""Dataset distribution charts (images/class counts per split) -> <reports_dir>/charts/."""
import argparse
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from albench.config import load_config
from albench.data.labels import labels_path, scan_dataset

DEFAULT_CONFIG = "sortwaste"
SPLITS = ("train", "val", "test")


def _scan_split(base: Path, rel_images: str):
    img_dir = base / rel_images
    return scan_dataset(img_dir, labels_path(img_dir))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--data", default=None,
                    help="dataset yaml, defaults to config data_yaml")
    ap.add_argument("--out", default=None,
                    help="output dir, defaults to <reports_dir>/charts")
    a = ap.parse_args()

    cfg = load_config(Path(a.config))
    data = Path(a.data or cfg["data_yaml"])
    d = yaml.safe_load(data.read_text(encoding="utf-8"))
    base = Path(d.get("path", str(data.parent)))
    names = d["names"]
    nc = d["nc"]
    out = Path(a.out) if a.out else Path(cfg["reports_dir"]) / "charts"
    out.mkdir(parents=True, exist_ok=True)

    n_imgs = {}                      # split -> #images
    imgs_per_class = {}              # split -> {cls: #images}
    boxes_per_class = {}             # split -> {cls: #boxes}
    for s in SPLITS:
        ds = _scan_split(base, d[s])
        n_imgs[s] = len(ds.items)
        ipc = Counter()
        for it in ds.items:
            for c in it.classes:
                ipc[c] += 1
        imgs_per_class[s] = ipc
        boxes_per_class[s] = ds.box_counts

    tot_imgs = {c: sum(imgs_per_class[s].get(c, 0) for s in SPLITS)
                for c in range(nc)}
    tot_boxes = {c: sum(boxes_per_class[s].get(c, 0) for s in SPLITS)
                 for c in range(nc)}

    # 01: images per split
    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(SPLITS, [n_imgs[s] for s in SPLITS],
                  color=["#4C72B0", "#55A868", "#C44E52"])
    ax.bar_label(bars, padding=3)
    ax.set_ylabel("Images")
    ax.set_title(f"Images per split (total {sum(n_imgs.values()):,})")
    fig.tight_layout()
    fig.savefig(out / "01_images_per_split.png", dpi=120)
    plt.close(fig)

    # 02: boxes and images per class (whole dataset)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for ax, counts, title, color in (
        (axes[0], tot_boxes, "Boxes per class", "steelblue"),
        (axes[1], tot_imgs, "Images per class", "coral")):
        vals = [counts[c] for c in range(nc)]
        b = ax.barh(names[::-1], vals[::-1], color=color)
        ax.bar_label(b, padding=3, fontsize=9)
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out / "02_class_overall.png", dpi=120)
    plt.close(fig)

    # 03: images per class, grouped by split (raw counts)
    import numpy as np
    x = np.arange(nc)
    w = 0.26
    fig, ax = plt.subplots(figsize=(12, 4.5))
    for i, s in enumerate(SPLITS):
        ax.bar(x + (i - 1) * w, [imgs_per_class[s].get(c, 0) for c in range(nc)],
               w, label=s)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("Images")
    ax.set_yscale("log")
    ax.set_title("Images per class, per split (log scale)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "03_images_per_class_per_split.png", dpi=120)
    plt.close(fig)

    # 04: per-class share within each split
    fig, ax = plt.subplots(figsize=(12, 4.5))
    for i, s in enumerate(SPLITS):
        tot = sum(imgs_per_class[s].values()) or 1
        ax.bar(x + (i - 1) * w,
               [100 * imgs_per_class[s].get(c, 0) / tot for c in range(nc)],
               w, label=s)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("% of split's images")
    ax.set_title("Per-class share within each split")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "04_class_proportion_per_split.png", dpi=120)
    plt.close(fig)

    # DISTRIBUTION.md
    md = [f"# Dataset distribution (from {base})", "",
          "## Images per split", "",
          "| split | images |", "|---|---|"]
    for s in SPLITS:
        md.append(f"| {s} | {n_imgs[s]:,} |")
    md.append(f"| **total** | **{sum(n_imgs.values()):,}** |")
    md += ["", "## Per class (whole dataset)", "",
           "| id | class | boxes | images |", "|---|---|---|---|"]
    for c in range(nc):
        md.append(f"| {c} | {names[c]} | {tot_boxes[c]:,} | {tot_imgs[c]:,} |")
    md += ["", "## Images per class, per split", "",
           "| class | " + " | ".join(SPLITS) + " |",
           "|---|" + "---|" * len(SPLITS)]
    for c in range(nc):
        row = " | ".join(str(imgs_per_class[s].get(c, 0)) for s in SPLITS)
        md.append(f"| {names[c]} | {row} |")
    (out / "DISTRIBUTION.md").write_text("\n".join(md) + "\n",
                                         encoding="utf-8")

    print(f"charts -> {out}/  (4 png + DISTRIBUTION.md)")
    print("images per split: " +
          " ".join(f"{s}={n_imgs[s]}" for s in SPLITS))


if __name__ == "__main__":
    main()
