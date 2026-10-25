import random

import numpy as np
import pytest

from albench.al.entropy_core import EntropyCandidate
from albench.al.select_entropy import (
    _validate_model_classes,
    labeled_instance_counts,
    select,
)
from albench.config import load_config


def test_labeled_instance_counts_counts_boxes_not_images(tmp_path):
    image_dir = tmp_path / "images"
    label_dir = tmp_path / "labels"
    image_dir.mkdir()
    label_dir.mkdir()

    first = image_dir / "first.jpg"
    second = image_dir / "second.jpg"
    first.touch()
    second.touch()

    (label_dir / "first.txt").write_text(
        "0 0.5 0.5 0.2 0.2\n"
        "0 0.6 0.6 0.2 0.2\n"
        "1 0.4 0.4 0.1 0.1\n",
        encoding="utf-8",
    )
    (label_dir / "second.txt").write_text(
        "1 0.5 0.5 0.2 0.2\n",
        encoding="utf-8",
    )

    counts = labeled_instance_counts([str(first), str(second)], nc=3)

    assert counts.tolist() == [2, 2, 0]


def test_select_runs_full_core_flow_with_candidate_adapter(tmp_path):
    image_dir = tmp_path / "images"
    label_dir = tmp_path / "labels"
    image_dir.mkdir()
    label_dir.mkdir()

    labeled_image = image_dir / "labeled.jpg"
    labeled_image.touch()
    (label_dir / "labeled.txt").write_text(
        "0 0.5 0.5 0.2 0.2\n"
        "0 0.6 0.6 0.2 0.2\n"
        "1 0.4 0.4 0.1 0.1\n",
        encoding="utf-8",
    )

    pool = [str(image_dir / f"pool_{index}.jpg") for index in range(3)]

    def fake_candidate_fn(_model, paths, _cfg):
        return [
            EntropyCandidate(
                path=paths[0],
                score=3.0,
                prototypes={2: np.array([1.0, 0.0])},
                presence={2: 0.8},
            ),
            EntropyCandidate(
                path=paths[1],
                score=2.0,
                prototypes={3: np.array([0.0, 1.0])},
                presence={3: 0.8},
            ),
            EntropyCandidate(
                path=paths[2],
                score=1.0,
                prototypes={0: np.array([1.0, 1.0])},
                presence={0: 0.8},
            ),
        ]

    cfg = {
        "classes": {"nc": 4, "names": ["a", "b", "c", "d"]},
        "al": {
            "entropy": {
                "t_intra": 0.7,
                "t_inter": 0.3,
                "alpha": 0.5,
                "beta": 0.75,
            }
        },
    }

    class Model:
        pass

    model = Model()
    picks = select(
        model=model,
        candidates=pool,
        labeled={str(labeled_image)},
        budget=2,
        cfg=cfg,
        rng=random.Random(13),
        candidate_fn=fake_candidate_fn,
    )

    assert picks == pool[:2]
    assert model._entropy_acq_diag["selected_rank"].tolist() == [0, 1, -1]
    assert model._entropy_acq_diag["selected_phase"].tolist() == [1, 1, 0]


def test_select_rejects_prediction_class_outside_detector_nc(tmp_path):
    pool = [str(tmp_path / "pool.jpg")]

    def fake_candidate_fn(_model, paths, _cfg):
        return [
            EntropyCandidate(
                path=paths[0],
                score=1.0,
                prototypes={99: np.array([1.0, 0.0])},
                presence={99: 0.5},
                detection_count=1,
                representative_count=1,
            )
        ]

    cfg = {
        "classes": {"nc": 4, "names": ["a", "b", "c", "d"]},
        "al": {
            "entropy": {
                "t_intra": 0.7,
                "t_inter": 0.3,
                "alpha": 0.5,
                "beta": 0.75,
            }
        },
    }
    with pytest.raises(RuntimeError, match="outside"):
        select(
            model=object(),
            candidates=pool,
            labeled=set(),
            budget=1,
            cfg=cfg,
            rng=random.Random(13),
            candidate_fn=fake_candidate_fn,
        )


def test_sortwaste_config_enables_entropy_with_realizable_alpha():
    sortwaste = load_config("sortwaste")

    assert "entropy" in sortwaste["al"]["strategies"]
    assert sortwaste["al"]["entropy"]["alpha"] == 0.5


def test_entropy_checkpoint_class_order_must_match_config():
    cfg = {
        "classes": {"nc": 2, "names": ["PET", "Metal"]},
    }

    class Model:
        names = {0: "Metal", 1: "PET"}

    with pytest.raises(RuntimeError, match="do not match"):
        _validate_model_classes(Model(), cfg)
