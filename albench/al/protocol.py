"""Guards for benchmark training invariants shared by every strategy."""


def enforce_fixed_batch(model, expected: int, *, expected_epochs: int = None,
                        expected_warmup: float = None,
                        expected_close_mosaic: int = None) -> None:
    """Keep the fixed train schedule and FP32 in-training validation."""
    def _guard(trainer) -> None:
        actual = int(trainer.batch_size)
        if actual != expected or int(trainer.args.batch) != expected:
            raise RuntimeError(
                f"protocol requires batch={expected}, but trainer uses "
                f"batch={actual}")
        if (expected_epochs is not None
                and int(trainer.args.epochs) != expected_epochs):
            raise RuntimeError(
                f"protocol requires epochs={expected_epochs}, but trainer "
                f"uses epochs={trainer.args.epochs}")
        if (expected_warmup is not None
                and float(trainer.args.warmup_epochs) != expected_warmup):
            raise RuntimeError(
                f"protocol requires warmup_epochs={expected_warmup}, but "
                f"trainer uses warmup_epochs={trainer.args.warmup_epochs}")
        if bool(getattr(trainer.args, "resume", False)):
            raise RuntimeError(
                "protocol requires resume=False: every round must start from "
                "the same pretrained yolov8n.pt")
        if (expected_close_mosaic is not None
                and int(trainer.args.close_mosaic) != expected_close_mosaic):
            raise RuntimeError(
                f"protocol requires close_mosaic={expected_close_mosaic}, "
                f"but trainer uses close_mosaic={trainer.args.close_mosaic}")
        # BaseTrainer catches OOM and halves a single-GPU batch while this is
        # below 3.  Setting it here makes OOM explicit instead of silently
        # turning one strategy/seed into a different experiment.
        trainer._oom_retries = 3

    def _validation_fp32(validator) -> None:
        # Ultralytics 8.4.51 forces half precision for in-training validation
        # whenever AMP training is enabled. With batch=64, summing the dense
        # YOLOv8 classification BCE can overflow float16 even though training
        # loss and detection metrics are finite. Keep AMP for training, but
        # evaluate the validation loss in float32.
        if not getattr(validator, "training", False):
            return
        validator.args.half = False
        trainer = getattr(model, "trainer", None)
        if trainer is None:
            raise RuntimeError("validation started without an attached trainer")
        ema = getattr(trainer, "ema", None)
        val_model = getattr(ema, "ema", None) or trainer.model
        if trainer.args.compile and hasattr(val_model, "_orig_mod"):
            val_model = val_model._orig_mod
        val_model.float()

    model.add_callback("on_train_epoch_start", _guard)
    model.add_callback("on_val_start", _validation_fp32)


def assert_fixed_batch(model, expected: int) -> None:
    trainer = getattr(model, "trainer", None)
    actual = getattr(trainer, "batch_size", None)
    if actual is None or int(actual) != expected:
        raise RuntimeError(
            f"protocol requires batch={expected}, but the completed trainer reports "
            f"batch={actual!r}")
