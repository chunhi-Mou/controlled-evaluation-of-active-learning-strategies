import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from albench.evaluation import evaluation_kwargs
from types import SimpleNamespace

from albench.al.loop import _assert_initial_coverage, _run_fingerprint
from albench.al.protocol import enforce_fixed_batch


class _FloatModel:
    def __init__(self):
        self.float_calls = 0

    def float(self):
        self.float_calls += 1
        return self


class _Wrapper:
    def __init__(self):
        self.callbacks = {}
        self.trainer = None

    def add_callback(self, event, callback):
        self.callbacks[event] = callback


class ProtocolTests(unittest.TestCase):
    def test_locked_evaluation_kwargs_are_separate_from_selectors(self):
        cfg = {"evaluation": {"conf": 0.001, "iou": 0.7,
                              "half": False, "augment": False}}
        run = {"imgsz": 640, "batch": 64, "workers": 2,
               "rect": False, "max_det": 300}
        self.assertEqual(evaluation_kwargs(cfg, run), {
            "imgsz": 640, "batch": 64, "workers": 2, "half": False,
            "augment": False, "conf": 0.001, "iou": 0.7,
            "rect": False, "max_det": 300})
    def test_batch_guard_and_validation_fp32(self):
        wrapper = _Wrapper()
        enforce_fixed_batch(
            wrapper, 64, expected_epochs=100, expected_warmup=0.0,
            expected_close_mosaic=0)
        train_model = _FloatModel()
        wrapper.trainer = SimpleNamespace(
            batch_size=64,
            args=SimpleNamespace(batch=64, compile=False, epochs=100,
                                 warmup_epochs=0.0, close_mosaic=0,
                                 resume=False),
            ema=SimpleNamespace(ema=train_model),
            model=_FloatModel(),
            _oom_retries=0,
        )
        wrapper.callbacks["on_train_epoch_start"](wrapper.trainer)
        self.assertEqual(wrapper.trainer._oom_retries, 3)

        validator = SimpleNamespace(
            training=True, args=SimpleNamespace(half=True))
        wrapper.callbacks["on_val_start"](validator)
        self.assertFalse(validator.args.half)
        self.assertEqual(train_model.float_calls, 1)

        wrapper.trainer.args.resume = True
        with self.assertRaisesRegex(RuntimeError, "resume=False"):
            wrapper.callbacks["on_train_epoch_start"](wrapper.trainer)
        wrapper.trainer.args.resume = False
        wrapper.trainer.args.warmup_epochs = 3.0
        with self.assertRaisesRegex(RuntimeError, "warmup_epochs=0.0"):
            wrapper.callbacks["on_train_epoch_start"](wrapper.trainer)

    def test_random_l0_coverage_is_a_guard_not_a_redraw(self):
        items = [("a.jpg", {0, 1}), ("b.jpg", {0}), ("c.jpg", {1})]
        self.assertEqual(
            _assert_initial_coverage(
                items, {"a.jpg", "b.jpg", "c.jpg"}, ["A", "B"], 2),
            [2, 2])
        with self.assertRaisesRegex(RuntimeError, "do not redraw"):
            _assert_initial_coverage(items, {"b.jpg"}, ["A", "B"], 1)

    def test_workers_are_part_of_resume_fingerprint(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / "images" / "a.jpg"
            label = root / "labels" / "a.txt"
            image.parent.mkdir()
            label.parent.mkdir()
            image.write_bytes(b"image")
            label.write_text("0 0.5 0.5 1 1\n", encoding="utf-8")
            data = root / "data.yaml"
            data.write_text("nc: 1\n", encoding="utf-8")
            cfg = {
                "classes": {"nc": 1, "names": ["A"]},
                "al": {"workers": 2, "cache": False,
                       "project": "ignored", "state_dir": "ignored"},
            }
            first = _run_fingerprint(cfg, str(data), [str(image)], "random")
            cfg["al"]["workers"] = 1
            second = _run_fingerprint(cfg, str(data), [str(image)], "random")
        self.assertNotEqual(first, second)


if __name__ == "__main__":
    unittest.main()
