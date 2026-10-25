"""Terminate dataloader worker processes held by a finished YOLO object."""
import gc


def _shutdown(loader) -> None:
    it = getattr(loader, "iterator", None)
    if it is None:
        it = getattr(loader, "_iterator", None)
    workers = getattr(it, "_workers", None)
    if not workers:
        return
    # Mirrors InfiniteDataLoader.__del__: terminate, then reap.
    for w in workers:
        if w.is_alive():
            w.terminate()
    try:
        it._shutdown_workers()
    except Exception:
        pass


def release(*models) -> None:
    """Free workers and CUDA cache for models that will not be used again.

    Ultralytics kills workers only in InfiniteDataLoader.__del__, and
    trainer/validator/dataloader form a reference cycle, so refcounting never
    fires it. Under spawn (Windows) every surviving worker keeps its own torch
    import, ~700 MB, and generations stack up across runs.
    """
    for model in models:
        if model is None:
            continue
        for owner in (getattr(model, "trainer", None),
                      getattr(model, "validator", None)):
            if owner is None:
                continue
            for attr in ("train_loader", "test_loader", "dataloader"):
                _shutdown(getattr(owner, attr, None))
        model.trainer = None
        model.validator = None
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
