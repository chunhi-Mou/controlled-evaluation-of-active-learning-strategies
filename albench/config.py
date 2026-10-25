"""Load and validate the central benchmark config."""
import os
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs"
BASE_CONFIG = CONFIG_DIR / "base.yaml"
SEEDS_CONFIG = CONFIG_DIR / "seeds.yaml"
DATASET_DIR = CONFIG_DIR / "datasets"

STRATEGIES = ("random", "ppal", "rarity_aware", "entropy")
ENDPOINTS = ("aubc", "map_at_cap")
METRICS = ("val_mAP50", "test_mAP50", "test_mAP50_95")   # results.csv columns

_DERIVED = {
    # data.yaml sits beside the train/val/test folders written by
    # 00_prepare_sortwaste.py. {{}} escapes ${DATA_ROOT}.
    "data_yaml": "${{DATA_ROOT}}/export_{tag}/data.yaml",
    "reports_dir": "reports/{tag}/{protocol}",
    "splits_dir": "splits/{tag}/{protocol}",
    "baseline.project": "runs/{tag}/{protocol}/baseline",
    "al.project": "runs/{tag}/{protocol}/al",
    "al.state_dir": "state/{tag}/{protocol}/al",
}
_PATH_KEYS = tuple(_DERIVED) + ("dataset.root",)


def work_root() -> Path:
    return Path(os.environ.get("ALBENCH_WORK", ROOT))


def data_root() -> Path:
    """Folder that ${DATA_ROOT} points to, datasets/ by default."""
    return Path(os.environ.get("DATA_ROOT") or work_root() / "datasets")


def resolve(p) -> Path:
    p = Path(os.path.expandvars(str(p).replace("${DATA_ROOT}",
                                               str(data_root()))))
    return p if p.is_absolute() else work_root() / p


def _get(cfg: dict, dotted: str):
    node = cfg
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def _set(cfg: dict, dotted: str, value) -> None:
    parts = dotted.split(".")
    node = cfg
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value


def _merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        out[k] = (_merge(out[k], v)
                  if isinstance(v, dict) and isinstance(out.get(k), dict)
                  else v)
    return out


def config_path(spec) -> Path:
    """A bare name means configs/datasets/<name>.yaml, anything else is a path from ROOT."""
    s = str(spec)
    if not any(c in s for c in "/\\") and not s.endswith((".yaml", ".yml")):
        return DATASET_DIR / f"{s}.yaml"
    p = Path(s)
    return p if p.is_absolute() else ROOT / p


def _validate(cfg: dict) -> None:
    ratios = cfg.get("split", {}).get("ratios")
    if not (isinstance(ratios, list) and len(ratios) == 3):
        raise ValueError(f"split.ratios must be a 3-list, got {ratios!r}")
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"split.ratios must sum to 1.0, got {sum(ratios)}")
    cls = cfg.get("classes", {})
    if len(cls.get("names", [])) != cls.get("nc"):
        raise ValueError(f"classes.nc={cls.get('nc')} but "
                         f"{len(cls.get('names', []))} names listed")
    for sec in ("baseline", "al"):
        run = cfg.get(sec, {})
        ck = run.get("eval_checkpoint", "last")
        if ck != "last":
            raise ValueError(f"{sec}.eval_checkpoint must be 'last' for the "
                             f"fixed detection-AL protocol, got {ck!r}")
        if run.get("patience") != 0:
            raise ValueError(f"{sec}.patience must be 0 (fixed epochs, no "
                             f"early stopping), got {run.get('patience')!r}")
        if run.get("epochs") != 100:
            raise ValueError(f"{sec}.epochs must be 100 for the fixed common "
                             f"protocol, got {run.get('epochs')!r}")
        if run.get("warmup_epochs") != 0.0:
            raise ValueError(
                f"{sec}.warmup_epochs must be 0.0: with batch=64 the L0 "
                "round has only three batches/epoch and Ultralytics otherwise "
                f"enforces a 100-batch warmup, got {run.get('warmup_epochs')!r}")
        if run.get("close_mosaic") != 0:
            raise ValueError(
                f"{sec}.close_mosaic must be 0: L0 has only three batches "
                "per epoch, so the default final phase has too few updates, "
                f"got {run.get('close_mosaic')!r}")
        if Path(str(run.get("model", ""))).name.lower() != "yolov8n.pt":
            raise ValueError(f"{sec}.model must be yolov8n.pt, got "
                             f"{run.get('model')!r}")
        if run.get("pretrained") is not True:
            raise ValueError(f"{sec}.pretrained must be true for the common "
                             f"protocol, got {run.get('pretrained')!r}")
        if run.get("batch") != 64:
            raise ValueError(f"{sec}.batch must be 64 for this benchmark, "
                             f"got {run.get('batch')!r}")
        if run.get("rect") is not False:
            raise ValueError(f"{sec}.rect must be false, got "
                             f"{run.get('rect')!r}")
        max_det = run.get("max_det")
        if not isinstance(max_det, int) or max_det <= 0:
            raise ValueError(f"{sec}.max_det must be a positive int, got "
                             f"{max_det!r}")
    ev = cfg.get("evaluation", {})
    if not isinstance(ev.get("conf"), (int, float)) or not (
            0 < ev["conf"] < 1):
        raise ValueError("evaluation.conf must be in (0, 1)")
    if not isinstance(ev.get("iou"), (int, float)) or not (
            0 < ev["iou"] < 1):
        raise ValueError("evaluation.iou must be in (0, 1)")
    for key in ("half", "augment"):
        if ev.get(key) is not False:
            raise ValueError(f"evaluation.{key} must be false")
    # seeds_n slices load_seeds(), so 0 would run nothing and a negative value
    # would silently drop seeds.
    n, pool = cfg.get("al", {}).get("seeds_n"), len(load_seeds())
    if not isinstance(n, int) or not 1 <= n <= pool:
        raise ValueError(f"al.seeds_n={n!r}, expected an int in 1..{pool} "
                         f"(seeds available in {SEEDS_CONFIG.name})")
    init = cfg.get("al", {}).get("initial_pool", "random")
    if init != "random":
        raise ValueError("al.initial_pool must be 'random' for the detection-AL "
                         f"benchmark protocol, got {init!r}")
    minimum = cfg.get("al", {}).get("initial_min_images_per_class")
    if not isinstance(minimum, int) or minimum < 1:
        raise ValueError("al.initial_min_images_per_class must be an integer >= 1")
    for key in ("initial_min_boxes_per_class", "initial_min_total_boxes"):
        value = cfg.get("al", {}).get(key)
        if not isinstance(value, int) or value < 1:
            raise ValueError(f"al.{key} must be an integer >= 1")
    background = cfg.get("al", {}).get("initial_max_background_frac")
    if not isinstance(background, (int, float)) or not 0 <= background <= 1:
        raise ValueError("al.initial_max_background_frac must be in [0, 1]")
    unknown = set(cfg.get("al", {}).get("strategies", [])) - set(STRATEGIES)
    if unknown:
        raise ValueError(f"al.strategies: unknown {sorted(unknown)}, "
                         f"expected a subset of {list(STRATEGIES)}")
    schedule = cfg.get("al", {}).get("schedule", {})
    init = schedule.get("init_frac")
    step = schedule.get("step_frac")
    cap = schedule.get("cap_frac")
    if not all(isinstance(x, (int, float)) for x in (init, step, cap)):
        raise ValueError("al.schedule fractions must be numeric")
    if not (0 < init < cap <= 1 and step > 0):
        raise ValueError("al.schedule must satisfy 0 < init_frac < cap_frac "
                         "<= 1 and step_frac > 0")
    pp = cfg.get("al", {}).get("ppal", {})
    if not isinstance(pp.get("delta"), int) or pp["delta"] < 1:
        raise ValueError(f"al.ppal.delta must be an int >= 1, got "
                         f"{pp.get('delta')!r}")
    if not isinstance(pp.get("score_thr"), (int, float)) or not (
            0 < pp["score_thr"] < 1):
        raise ValueError(f"al.ppal.score_thr must be in (0, 1), got "
                         f"{pp.get('score_thr')!r}")
    if pp.get("ema_momentum") != 0.99:
        raise ValueError("al.ppal.ema_momentum must be 0.99 for PPAL")
    if not isinstance(pp.get("nms_iou"), (int, float)) or not (
            0 < pp["nms_iou"] < 1):
        raise ValueError(f"al.ppal.nms_iou must be in (0, 1), got "
                         f"{pp.get('nms_iou')!r}")
    for key in ("uncertainty_max_det", "diversity_max_det"):
        if not isinstance(pp.get(key), int) or pp[key] < 1:
            raise ValueError(f"al.ppal.{key} must be a positive int, got "
                             f"{pp.get(key)!r}")
    entropy = cfg.get("al", {}).get("entropy", {})
    if not isinstance(entropy, dict):
        raise ValueError("al.entropy must be a mapping")
    if not isinstance(entropy.get("score_thr"), (int, float)) or not (
            0 < entropy["score_thr"] < 1):
        raise ValueError("al.entropy.score_thr must be in (0, 1)")
    if not isinstance(entropy.get("nms_iou"), (int, float)) or not (
            0 < entropy["nms_iou"] < 1):
        raise ValueError("al.entropy.nms_iou must be in (0, 1)")
    if not isinstance(entropy.get("max_det"), int) or entropy["max_det"] < 1:
        raise ValueError("al.entropy.max_det must be a positive int")
    for key in ("t_enms", "t_intra"):
        value = entropy.get(key)
        if not isinstance(value, (int, float)) or not -1 <= value <= 1:
            raise ValueError(f"al.entropy.{key} must be in [-1, 1]")
    t_inter = entropy.get("t_inter")
    if not isinstance(t_inter, (int, float)) or not 0 <= t_inter <= 1:
        raise ValueError("al.entropy.t_inter must be in [0, 1]")
    alpha, beta = entropy.get("alpha"), entropy.get("beta")
    if not all(isinstance(value, (int, float)) for value in (alpha, beta)) or (
            not 0 < alpha < beta < 1):
        raise ValueError("al.entropy must satisfy 0 < alpha < beta < 1")
    minority_count = alpha * cls["nc"]
    if abs(minority_count - round(minority_count)) > 1e-12:
        raise ValueError(
            "al.entropy.alpha * classes.nc must be an integer for DivProto"
        )
    if "batch_size" in entropy and (
            not isinstance(entropy["batch_size"], int)
            or entropy["batch_size"] < 1):
        raise ValueError("al.entropy.batch_size must be a positive int")
    for key in ("rect", "half"):
        if key in entropy and entropy[key] is not False:
            raise ValueError(f"al.entropy.{key} must be false")
    ra = cfg.get("al", {}).get("rarity_aware", {})
    for key in ("gamma", "beta"):
        if not isinstance(ra.get(key), (int, float)) or ra[key] <= 0:
            raise ValueError(f"al.rarity_aware.{key} must be > 0, got "
                             f"{ra.get(key)!r}")
    for key in ("alpha_min", "alpha_max"):
        if not isinstance(ra.get(key), (int, float)) or ra[key] <= 0:
            raise ValueError(f"al.rarity_aware.{key} must be > 0, got "
                             f"{ra.get(key)!r}")
    if ra["alpha_min"] > ra["alpha_max"]:
        raise ValueError("al.rarity_aware.alpha_min must be <= alpha_max")
    if not isinstance(ra.get("rho"), (int, float)) or not 0 <= ra["rho"] <= 1:
        raise ValueError(f"al.rarity_aware.rho must be in [0, 1], got "
                         f"{ra.get('rho')!r}")
    if not isinstance(ra.get("score_thr"), (int, float)) or not (
            0 < ra["score_thr"] < 1):
        raise ValueError(f"al.rarity_aware.score_thr must be in (0, 1), got "
                         f"{ra.get('score_thr')!r}")
    if not isinstance(ra.get("iou"), (int, float)) or not 0 < ra["iou"] < 1:
        raise ValueError(f"al.rarity_aware.iou must be in (0, 1), got "
                         f"{ra.get('iou')!r}")
    if not isinstance(ra.get("eps"), (int, float)) or ra["eps"] <= 0:
        raise ValueError(f"al.rarity_aware.eps must be > 0, got "
                         f"{ra.get('eps')!r}")
    if not isinstance(ra.get("kmeans_iter"), int) or ra["kmeans_iter"] < 1:
        raise ValueError("al.rarity_aware.kmeans_iter must be an int >= 1, "
                         f"got {ra.get('kmeans_iter')!r}")
    # 07_al_report.py dispatches on these values, so reject typos here.
    an = cfg.get("al", {}).get("analysis", {})
    for key, allowed in (("primary_endpoint", ENDPOINTS), ("metric", METRICS)):
        if an.get(key) not in allowed:
            raise ValueError(f"al.analysis.{key}={an.get(key)!r}, expected "
                             f"one of {list(allowed)}")


def load_config(path) -> dict:
    p = config_path(path)
    cfg = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    if p.parent == DATASET_DIR:
        base = yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8")) or {}
        cfg = _merge(base, cfg)
    _validate(cfg)
    tag = cfg.get("tag")
    protocol = cfg.get("protocol_id")
    if not isinstance(protocol, str) or not protocol.strip():
        raise ValueError("config needs a non-empty protocol_id so incompatible "
                         "runs cannot share outputs or resume state")
    for dotted, template in _DERIVED.items():
        if _get(cfg, dotted) is None:
            if not tag:
                raise ValueError(f"config needs `tag:` to derive {dotted}")
            _set(cfg, dotted, template.format(tag=tag, protocol=protocol))
    for dotted in _PATH_KEYS:
        if _get(cfg, dotted) is not None:
            _set(cfg, dotted, str(resolve(_get(cfg, dotted))))
    return cfg


def load_seeds(path=None) -> list[int]:
    p = SEEDS_CONFIG if path is None else Path(path)
    if not p.is_absolute():
        p = ROOT / p
    return list(yaml.safe_load(p.read_text(encoding="utf-8"))["seeds"])
