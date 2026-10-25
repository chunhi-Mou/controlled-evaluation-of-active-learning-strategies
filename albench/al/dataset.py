"""Pool listing, per-image class set, and per-round data.yaml writing."""
from pathlib import Path
import yaml

from albench.data.labels import IMG_EXT, label_for


def _base_of(data_yaml: str, d: dict) -> Path:
    return Path(d.get("path", str(Path(data_yaml).parent)))


def pool_images(data_yaml: str) -> list[str]:
    d = yaml.safe_load(Path(data_yaml).read_text(encoding="utf-8"))
    train_dir = _base_of(data_yaml, d) / d["train"]
    return sorted(str(p.resolve()) for p in train_dir.iterdir()
                  if p.suffix.lower() in IMG_EXT)


def image_classes(img_path: str) -> frozenset[int]:
    lbl = label_for(img_path)
    if not lbl.exists():
        return frozenset()
    ids = set()
    for line in lbl.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            ids.add(int(float(line.split()[0])))
    return frozenset(ids)


def image_class_counts(img_path: str, nc: int) -> list[int]:
    """Per-class ground-truth instance counts for one labeled image."""
    counts = [0] * nc
    label = label_for(img_path)
    if not label.exists():
        return counts
    for line in label.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if fields:
            cls = int(float(fields[0]))
            if 0 <= cls < nc:
                counts[cls] += 1
    return counts


def write_round_data_yaml(base_data_yaml: str, train_txt: str,
                          out_yaml: str) -> None:
    d = yaml.safe_load(Path(base_data_yaml).read_text(encoding="utf-8"))
    base = _base_of(base_data_yaml, d)
    d["train"] = str(Path(train_txt).resolve())
    d["val"] = str((base / d["val"]).resolve())
    d["test"] = str((base / d["test"]).resolve())
    d.pop("path", None)                # train/val/test now absolute, no base
    Path(out_yaml).write_text(yaml.safe_dump(d), encoding="utf-8")
