"""Build a portable YOLO tree from the official SortWaste package.

The official archive stores train/val images under ``splited_all_dataset`` and
their YOLO labels under ``splited_all_dataset_yolo``.  This script joins those
two trees without changing the published split.  Files are copied by default
so the exported dataset is independent of the extracted staging tree.
"""
import argparse
import json
import os
import shutil
from pathlib import Path

SPLITS = ("train", "val", "test")
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
ROOT = Path(__file__).resolve().parents[1]
NAMES = ["PET", "HDPE", "MixedSoftPlastic", "ECAL", "Metal", "Cardboard",
         "MixedRigidPlastic", "PETOil"]


def _files(folder: Path, extensions: set[str]) -> dict[str, Path]:
    if not folder.is_dir():
        raise FileNotFoundError(folder)
    return {p.stem: p for p in sorted(folder.iterdir())
            if p.is_file() and p.suffix.lower() in extensions}


def _place(src: Path, dst: Path, copy_only: bool) -> str:
    if dst.exists():
        if dst.stat().st_size != src.stat().st_size:
            raise FileExistsError(
                f"refusing to overwrite different-size file: {dst}")
        if copy_only and os.path.samefile(src, dst):
            # Break an existing hard link one file at a time. os.replace is
            # atomic, so an interrupted conversion can safely be re-run.
            tmp = dst.with_name(dst.name + ".copying")
            shutil.copy2(src, tmp)
            os.replace(tmp, dst)
            return "copied"
        return "existing"
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not copy_only:
        try:
            os.link(src, dst)
            return "linked"
        except OSError:
            pass
    shutil.copy2(src, dst)
    return "copied"


def prepare(source: Path, out: Path, names: list[str], nc: int,
            copy_only: bool = True) -> None:
    counts = {"linked": 0, "copied": 0, "existing": 0}
    for split in SPLITS:
        images = _files(source / "splited_all_dataset" / split / "images",
                        IMAGE_EXT)
        labels = _files(source / "splited_all_dataset_yolo" / split / "labels",
                        {".txt"})
        missing_labels = sorted(set(images) - set(labels))
        missing_images = sorted(set(labels) - set(images))
        if missing_labels or missing_images:
            raise RuntimeError(
                f"{split}: stem mismatch, images without labels="
                f"{missing_labels[:5]}, labels without images="
                f"{missing_images[:5]}")

        for stem in sorted(images):
            status = _place(images[stem],
                            out / split / "images" / images[stem].name,
                            copy_only)
            counts[status] += 1
            status = _place(labels[stem],
                            out / split / "labels" / labels[stem].name,
                            copy_only)
            counts[status] += 1
        out_images = _files(out / split / "images", IMAGE_EXT)
        out_labels = _files(out / split / "labels", {".txt"})
        if set(out_images) != set(images) or set(out_labels) != set(labels):
            raise RuntimeError(
                f"{split}: destination contains missing or unexpected stems")
        print(f"{split}: {len(images)} image-label pairs")

    data_yaml = "\n".join([
        f"path: {json.dumps(str(out.resolve()))}",
        "train: train/images", "val: val/images", "test: test/images",
        f"nc: {nc}", f"names: {json.dumps(names)}", ""])
    (out / "data.yaml").write_text(data_yaml, encoding="utf-8")
    print(f"files: {counts}")
    print(f"dataset: {out.resolve()}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--source",
        default=str(ROOT / "datasets/sortwaste_official_staging/dataset"),
        help="folder containing splited_all_dataset[_yolo]")
    ap.add_argument(
        "--out", default=None,
        help="output folder, defaults to <DATA_ROOT>/export_sortwaste")
    ap.add_argument(
        "--hardlink", action="store_true",
        help="opt in to NTFS hard links instead of independent copies")
    a = ap.parse_args()

    data_root = Path(os.environ.get("DATA_ROOT") or ROOT / "datasets")
    out = Path(a.out).resolve() if a.out else data_root / "export_sortwaste"
    prepare(Path(a.source).resolve(), out, NAMES, len(NAMES),
            copy_only=not a.hardlink)


if __name__ == "__main__":
    main()
