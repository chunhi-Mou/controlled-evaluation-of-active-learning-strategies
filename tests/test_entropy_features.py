import contextlib

import numpy as np
import pytest


torch = pytest.importorskip("torch")

from albench.al import entropy_features


class _Boxes:
    def __init__(self):
        self.xyxy = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
        self.cls = torch.tensor([0.0])
        self.conf = torch.tensor([0.5])


class _Result:
    def __init__(self, reported_path):
        self.path = reported_path
        self.boxes = _Boxes()
        self.feats = torch.tensor([[1.0, 2.0, 3.0]])


class _FakeModel:
    def __init__(self):
        self.predictor = None
        self.calls = []

    def predict(self, *, source, **_kwargs):
        self.calls.append(list(source))
        return iter(
            _Result(f"image{index}.jpg")
            for index in range(len(source))
        )


def test_extractor_chunks_list_sources_and_preserves_pool_paths(monkeypatch):
    monkeypatch.setattr(
        entropy_features,
        "_require_native_feature_support",
        lambda: None,
    )
    monkeypatch.setattr(
        entropy_features,
        "_temporary_native_feature_capture",
        lambda _model: contextlib.nullcontext(),
    )

    model = _FakeModel()
    paths = [f"pool_{index}.jpg" for index in range(5)]

    outputs = entropy_features.extract_detection_features(
        model=model,
        paths=paths,
        batch_size=2,
    )

    assert model.calls == [paths[:2], paths[2:4], paths[4:]]
    assert [item.path for item in outputs] == paths
    assert all(item.features.shape == (1, 3) for item in outputs)
    assert all(np.isfinite(item.features).all() for item in outputs)
