"""Per-(strategy, seed, round) resume state with atomic write, marked done only
when the required keys are present."""
import json
import os
from pathlib import Path

_REQUIRED = ("labeled", "next_labeled", "val_map50", "test_map50")


def state_path(state_dir: str, strategy: str, seed: int, r: int) -> Path:
    return Path(state_dir) / f"{strategy}_s{seed}" / f"r{r}.json"


def is_done(state_dir: str, strategy: str, seed: int, r: int) -> bool:
    p = state_path(state_dir, strategy, seed, r)
    if not p.exists():
        return False
    try:
        d = json.loads(p.read_text())
    except (ValueError, OSError):
        return False
    return all(k in d for k in _REQUIRED)


def write_atomic(state_dir: str, strategy: str, seed: int, r: int,
                 payload: dict) -> None:
    p = state_path(state_dir, strategy, seed, r)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload))
    os.replace(tmp, p)


def load(state_dir: str, strategy: str, seed: int, r: int) -> dict:
    return json.loads(state_path(state_dir, strategy, seed, r).read_text())


def encode_rng_state(s: tuple) -> list:
    """python random.getstate() -> JSON-safe list (tuple/inner-tuple -> list)."""
    return [s[0], list(s[1]), s[2]]


def decode_rng_state(j: list) -> tuple:
    """Inverse of encode_rng_state for random.setstate()."""
    return (j[0], tuple(j[1]), j[2])
