"""Audit the configured dataset -> <reports_dir>/dataset_audit.md."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from albench.config import load_config
from albench.data.labels import scan_dataset
from albench.data.audit import summarize, render_markdown

DEFAULT_CONFIG = "sortwaste"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    a = ap.parse_args()

    cfg = load_config(Path(a.config))
    root = Path(cfg["dataset"]["root"])
    ds = scan_dataset(root / cfg["dataset"]["images_dir"],
                      root / cfg["dataset"]["labels_dir"])
    summary = summarize(ds, cfg["classes"]["names"])
    out = Path(cfg["reports_dir"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "dataset_audit.md").write_text(render_markdown(summary),
                                          encoding="utf-8")
    print(f"n_paired={summary['n_paired']} "
          f"orphans(img/lbl)={summary['images_without_labels']}/"
          f"{summary['labels_without_images']} "
          f"box_counts={summary['box_counts']}")


if __name__ == "__main__":
    main()
