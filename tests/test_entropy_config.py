from copy import deepcopy

import pytest

from albench.config import _validate, load_config


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("score_thr", float("nan"), "score_thr"),
        ("nms_iou", 1.5, "nms_iou"),
        ("max_det", 0, "max_det"),
        ("t_enms", float("nan"), "t_enms"),
        ("t_intra", 2.0, "t_intra"),
        ("t_inter", -0.1, "t_inter"),
    ],
)
def test_invalid_entropy_adapter_config_is_rejected(key, value, message):
    cfg = deepcopy(load_config("sortwaste"))
    cfg["al"]["entropy"][key] = value
    with pytest.raises(ValueError, match=message):
        _validate(cfg)


def test_entropy_alpha_beta_and_realizable_minority_count_are_validated():
    cfg = deepcopy(load_config("sortwaste"))
    cfg["al"]["entropy"].update(alpha=0.3, beta=0.2)
    with pytest.raises(ValueError, match="alpha < beta"):
        _validate(cfg)

    cfg = deepcopy(load_config("sortwaste"))
    cfg["al"]["entropy"]["alpha"] = 0.3
    with pytest.raises(ValueError, match="integer"):
        _validate(cfg)
