"""Build self-contained, locally verified Kaggle code/data archives."""
import argparse
import hashlib
import json
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from albench.config import ROOT, load_config

DEFAULT_CONFIG = "sortwaste"
SPLITS = ("train", "val", "test")
CODE = (
    "albench", "configs", "scripts", "kaggle", "tests",
    "requirements.txt", "requirements.lock.txt", "pyproject.toml",
    "README.md", "yolov8n.pt",
)
SKIP_DIRS = {"__pycache__", ".ipynb_checkpoints"}
SKIP_SUFFIXES = {".pyc", ".pyo", ".cache"}


def _gb(n: int) -> str:
    return f"{n / 2**30:.2f} GB"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def walk(root: Path, base: Path) -> list[tuple[Path, str, int]]:
    """Return sorted (source, archive name, bytes), supporting files or dirs."""
    if not root.exists():
        raise FileNotFoundError(f"required packaging input is missing: {root}")
    paths = [root] if root.is_file() else sorted(root.rglob("*"))
    output = []
    for path in paths:
        if (path.is_file() and not SKIP_DIRS.intersection(path.parts)
                and path.suffix.lower() not in SKIP_SUFFIXES):
            output.append((path, path.relative_to(base).as_posix(),
                           path.stat().st_size))
    return output


def plan(items: list, cap: int) -> list[list]:
    """Greedy, order-preserving parts, each no larger than cap bytes."""
    if cap <= 0:
        return [items]
    oversized = [item for item in items if item[2] > cap]
    if oversized:
        raise ValueError(f"single file exceeds part cap: {oversized[0][1]}")
    parts, current, size = [], [], 0
    for item in items:
        if current and size + item[2] > cap:
            parts.append(current)
            current, size = [], 0
        current.append(item)
        size += item[2]
    if current:
        parts.append(current)
    return parts


def _zip_plan_matches(dest: Path, items: list) -> bool:
    """Safe resume check: member names, sizes and source mtimes must match."""
    if not dest.exists():
        return False
    try:
        with zipfile.ZipFile(dest) as archive:
            infos = archive.infolist()
            if len(infos) != len(items):
                return False
            for info, (source, arcname, size) in zip(infos, items):
                source_time = datetime.fromtimestamp(source.stat().st_mtime)
                # ZIP timestamps have a two-second resolution.
                expected_time = source_time.timetuple()[:6]
                if (info.filename != arcname or info.file_size != size
                        or info.date_time[:5] != expected_time[:5]
                        or abs(info.date_time[5] - expected_time[5]) > 2):
                    return False
        return True
    except (OSError, zipfile.BadZipFile):
        return False


def write_zip(dest: Path, items: list, force: bool = False) -> None:
    """Write atomically, reusing an existing archive only if it matches this plan."""
    if dest.exists() and not force:
        if _zip_plan_matches(dest, items):
            print(f"[reuse verified plan] {dest.name}")
            return
        raise RuntimeError(
            f"stale archive does not match current inputs: {dest}. "
            "Re-run with --force to replace the exact planned archives.")
    size = sum(item[2] for item in items)
    print(f"[pack] {dest.name}: {len(items)} files, {_gb(size)}", flush=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    tmp.unlink(missing_ok=True)
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED, allowZip64=True) as archive:
        for n, (source, arcname, _) in enumerate(items, 1):
            archive.write(source, arcname)
            if n % 2000 == 0:
                print(f"    {n}/{len(items)}", flush=True)
    tmp.replace(dest)


def _verify_union(archives: list[Path], expected: list, crc: bool) -> None:
    expected_map = {arcname: size for _, arcname, size in expected}
    actual = {}
    for path in archives:
        print(f"[verify] {path.name}", flush=True)
        with zipfile.ZipFile(path) as archive:
            if crc:
                bad = archive.testzip()
                if bad is not None:
                    raise RuntimeError(f"CRC failure in {path.name}: {bad}")
            for info in archive.infolist():
                if info.filename in actual:
                    raise RuntimeError(f"duplicate archive member: {info.filename}")
                actual[info.filename] = info.file_size
    if actual != expected_map:
        missing = sorted(set(expected_map) - set(actual))[:5]
        extra = sorted(set(actual) - set(expected_map))[:5]
        wrong = sorted(name for name in set(actual) & set(expected_map)
                       if actual[name] != expected_map[name])[:5]
        raise RuntimeError(
            f"archive union differs from source, missing={missing}, "
            f"extra={extra}, wrong_size={wrong}")


def _split_counts(export: Path) -> dict:
    counts = {}
    for split in SPLITS:
        images = [p for p in (export / split / "images").iterdir()
                  if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        labels = [p for p in (export / split / "labels").iterdir()
                  if p.is_file() and p.suffix.lower() == ".txt"]
        if {p.stem for p in images} != {p.stem for p in labels}:
            raise RuntimeError(f"{split}: image/label stems do not match")
        counts[split] = {"images": len(images), "labels": len(labels)}
    return counts


def _write_json(path: Path, value: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    tmp.replace(path)


def _git_info() -> dict:
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
            capture_output=True, text=True).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain"], cwd=ROOT, check=True,
            capture_output=True, text=True).stdout.strip())
        return {"head": head, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"head": None, "dirty": None}


def _dataset_items(export: Path) -> list[tuple[Path, str, int]]:
    """Payload allowlist that leaves out local caches and staging files."""
    items = walk(export / "data.yaml", export.parent)
    image_suffixes = {".jpg", ".jpeg", ".png"}
    for split in SPLITS:
        for path in sorted((export / split / "images").iterdir()):
            if path.is_file() and path.suffix.lower() in image_suffixes:
                items.append((path, path.relative_to(export.parent).as_posix(),
                              path.stat().st_size))
        for path in sorted((export / split / "labels").iterdir()):
            if path.is_file() and path.suffix.lower() == ".txt":
                items.append((path, path.relative_to(export.parent).as_posix(),
                              path.stat().st_size))
    return items


def _validate_portable_data_yaml(export: Path) -> None:
    """Reject a dataset archive tied to one machine's absolute path."""
    data = yaml.safe_load((export / "data.yaml").read_text(encoding="utf-8"))
    configured_path = data.get("path") if isinstance(data, dict) else None
    if configured_path and Path(str(configured_path)).is_absolute():
        raise RuntimeError(
            "dataset data.yaml contains a non-portable absolute path: "
            f"{configured_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--out", default="dist/kaggle",
                        help="archive parent (relative paths use the repo root)")
    parser.add_argument("--part-gb", type=float, default=2.0,
                        help="maximum source payload per data ZIP, 0 for one ZIP")
    parser.add_argument("--code-only", action="store_true")
    parser.add_argument("--force", action="store_true",
                        help="replace planned ZIPs instead of safe reuse")
    parser.add_argument("--skip-crc", action="store_true",
                        help="skip the full post-write CRC read (not recommended)")
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    tag = cfg["tag"]
    export = Path(cfg["data_yaml"]).parent
    output = Path(args.out)
    output = (output if output.is_absolute() else ROOT / output) / tag
    output.mkdir(parents=True, exist_ok=True)

    code_items = [item for name in CODE for item in walk(ROOT / name, ROOT)]
    code_zip = output / "albench-code.zip"
    write_zip(code_zip, code_items, args.force)
    _verify_union([code_zip], code_items, not args.skip_crc)
    _write_json(output / "albench-code-manifest.json", {
        "format": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "archive": {"name": code_zip.name, "bytes": code_zip.stat().st_size,
                    "sha256": _sha256(code_zip)},
        "entries": len(code_items),
        "model": cfg["al"]["model"],
        "git": _git_info(),
    })
    if args.code_only:
        print(f"code archive -> {code_zip}")
        return

    if not export.is_dir():
        raise SystemExit(f"no export at {export}, run scripts/00_prepare_sortwaste.py first")
    _validate_portable_data_yaml(export)
    split_counts = _split_counts(export)
    data_items = _dataset_items(export)
    parts = plan(data_items, int(args.part_gb * 2**30))
    planned_names = {
        (f"{tag}-data.zip" if len(parts) == 1
         else f"{tag}-data-{index:02d}.zip")
        for index in range(1, len(parts) + 1)
    }
    obsolete = sorted(path for path in output.glob(f"{tag}-data*.zip")
                      if path.name not in planned_names)
    if obsolete and not args.force:
        raise RuntimeError(
            "obsolete data ZIPs from another part plan remain: "
            f"{[p.name for p in obsolete]}, use --force to remove them")
    for path in obsolete:
        path.unlink()
        print(f"[remove obsolete] {path.name}")
    data_archives = []
    for index, part in enumerate(parts, 1):
        name = (f"{tag}-data.zip" if len(parts) == 1
                else f"{tag}-data-{index:02d}.zip")
        path = output / name
        write_zip(path, part, args.force)
        data_archives.append(path)
    _verify_union(data_archives, data_items, not args.skip_crc)

    manifest = {
        "format": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "tag": tag,
        "export_dir": export.name,
        "payload_bytes": sum(item[2] for item in data_items),
        "split_counts": split_counts,
        "git": _git_info(),
        "archives": [
            {"name": path.name, "bytes": path.stat().st_size,
             "sha256": _sha256(path)} for path in data_archives
        ],
        "files": [{"path": arcname, "bytes": size}
                  for _, arcname, size in data_items],
    }
    manifest_path = output / f"{tag}-data-manifest.json"
    _write_json(manifest_path, manifest)
    total = code_zip.stat().st_size + sum(p.stat().st_size for p in data_archives)
    print(f"\nverified {len(data_archives) + 1} ZIPs, {_gb(total)} -> {output}")
    print(f"Code Kaggle Dataset: {code_zip.name} + albench-code-manifest.json")
    print(f"Data Kaggle Dataset: {', '.join(p.name for p in data_archives)} + "
          f"{manifest_path.name}")
    print("Kaggle preserves directory structure while unpacking the parts.")


if __name__ == "__main__":
    main()
