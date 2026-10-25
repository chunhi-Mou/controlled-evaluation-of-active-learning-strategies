"""Dataset statistics for the data-quality report."""
from collections import Counter
from albench.data.labels import Dataset


def summarize(ds: Dataset, names: list[str]) -> dict:
    per_class = Counter()
    multi = 0
    for it in ds.items:
        for c in it.classes:
            per_class[c] += 1
        if len(it.classes) >= 2:
            multi += 1
    n = len(ds.items)
    return {
        "n_paired": n,
        "images_without_labels": len(ds.images_without_labels),
        "labels_without_images": len(ds.labels_without_images),
        "box_counts": ds.box_counts,
        "images_per_class": dict(sorted(per_class.items())),
        "multi_object_images": multi,
        "single_object_images": n - multi,
        "names": names,
    }


def render_markdown(summary: dict) -> str:
    lines = ["# Dataset audit", ""]
    lines.append(f"- paired images: {summary['n_paired']}")
    lines.append(f"- images without labels: {summary['images_without_labels']}")
    lines.append(f"- labels without images: {summary['labels_without_images']}")
    n = summary["n_paired"] or 1
    mo = summary["multi_object_images"]
    lines.append(f"- multi-object images: {mo} ({100*mo/n:.1f}%)")
    lines.append("")
    lines.append("| id | class | boxes | images |")
    lines.append("|---|---|---|---|")
    for i, nm in enumerate(summary["names"]):
        lines.append(f"| {i} | {nm} | {summary['box_counts'].get(i,0)} | "
                     f"{summary['images_per_class'].get(i,0)} |")
    return "\n".join(lines) + "\n"
