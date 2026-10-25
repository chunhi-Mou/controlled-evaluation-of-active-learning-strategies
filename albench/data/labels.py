"""Scan a YOLO dataset and reconcile image/label pairs by stem."""
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

IMG_EXT = {".jpg", ".jpeg", ".png"}


def labels_path(p) -> Path:
    """YOLO layout sibling: .../images/... -> .../labels/...

    Splits on path components, not on the "/images/" substring, which never
    matches a Windows path and silently yields an unlabeled dataset.
    """
    parts = list(Path(p).parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            break
    else:
        raise ValueError(f"no 'images' component in {p!r}")
    return Path(*parts)


def label_for(image) -> Path:
    """Label .txt of an image under the YOLO images/labels layout."""
    return labels_path(image).with_suffix(".txt")


@dataclass
class Item:
    image: Path
    label: Path
    classes: set[int]


@dataclass
class Dataset:
    items: list[Item]
    images_without_labels: list[str]
    labels_without_images: list[str]
    box_counts: dict[int, int] = field(default_factory=dict)


def _parse_label(p: Path) -> tuple[set[int], Counter]:
    classes, boxes = set(), Counter()
    for ln in p.read_text().splitlines():
        ln = ln.strip()
        if not ln:
            continue
        parts = ln.split()
        if len(parts) != 5:
            raise ValueError(f"{p}: line has {len(parts)} fields (expect 5): {ln!r}")
        c = int(float(parts[0]))
        classes.add(c)
        boxes[c] += 1
    return classes, boxes


def scan_dataset(images_dir: Path, labels_dir: Path) -> Dataset:
    images_dir, labels_dir = Path(images_dir), Path(labels_dir)
    imgs = {p.stem: p for p in images_dir.iterdir()
            if p.suffix.lower() in IMG_EXT and not p.name.startswith(".")}
    lbls = {p.stem: p for p in labels_dir.iterdir()
            if p.suffix == ".txt" and not p.name.startswith(".")}
    items, total = [], Counter()
    for stem in sorted(set(imgs) & set(lbls)):
        cls, boxes = _parse_label(lbls[stem])
        total.update(boxes)
        items.append(Item(imgs[stem], lbls[stem], cls))
    return Dataset(
        items=items,
        images_without_labels=sorted(imgs[s].name for s in set(imgs) - set(lbls)),
        labels_without_images=sorted(lbls[s].name for s in set(lbls) - set(imgs)),
        box_counts=dict(sorted(total.items())),
    )
