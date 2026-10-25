import ast
import importlib.util
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[1]


def _load_packer():
    path = ROOT / "scripts" / "08_pack_kaggle.py"
    spec = importlib.util.spec_from_file_location("pack_kaggle", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PackagingTests(unittest.TestCase):
    def test_file_inputs_and_dataset_allowlist(self):
        packer = _load_packer()
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            single = root / "README.md"
            single.write_text("ok", encoding="utf-8")
            self.assertEqual(packer.walk(single, root)[0][1], "README.md")

            export = root / "export_sortwaste"
            (export / "data.yaml").parent.mkdir(parents=True)
            (export / "data.yaml").write_text("nc: 1", encoding="utf-8")
            packer._validate_portable_data_yaml(export)
            for split in packer.SPLITS:
                images = export / split / "images"
                labels = export / split / "labels"
                images.mkdir(parents=True)
                labels.mkdir(parents=True)
                (images / f"{split}.png").write_bytes(b"image")
                (labels / f"{split}.txt").write_text(
                    "0 0.5 0.5 1 1", encoding="utf-8")
                (labels / "labels.cache").write_bytes(b"local paths")
            items = packer._dataset_items(export)
            names = [item[1] for item in items]
            self.assertEqual(len(names), 7)
            self.assertFalse(any(name.endswith(".cache") for name in names))
            cap = max(item[2] for item in items)
            self.assertEqual(sum(len(part) for part in packer.plan(items, cap)),
                             len(items))
            (export / "data.yaml").write_text(
                "path: C:/machine-specific/data\n", encoding="utf-8")
            with self.assertRaises(RuntimeError):
                packer._validate_portable_data_yaml(export)

    def test_notebooks_are_valid_and_protocol_fixed(self):
        for name in ("kaggle_al.ipynb", "kaggle_baseline.ipynb"):
            notebook = json.loads(
                (ROOT / "kaggle" / name).read_text(encoding="utf-8"))
            self.assertEqual(notebook["nbformat_minor"], 4)
            code = "\n".join(
                "".join(cell["source"]) for cell in notebook["cells"]
                if cell["cell_type"] == "code"
                and not "".join(cell["source"]).startswith("!"))
            ast.parse(code)
            self.assertNotIn("BATCH =", code)
            self.assertIn("batch\"] == 64", code)
            self.assertIn("data-manifest.json", code)
            self.assertIn("yolov8n.pt", code)
        al_text = (ROOT / "kaggle" / "kaggle_al.ipynb").read_text(
            encoding="utf-8")
        self.assertIn("rarity_aware", al_text)
        self.assertIn('cfg[\\"al\\"][\\"workers\\"] = 2', al_text)


if __name__ == "__main__":
    unittest.main()
