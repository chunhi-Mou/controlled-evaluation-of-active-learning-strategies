"""Run AL strategies across seeds."""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from albench.config import STRATEGIES, load_config, load_seeds
from albench.al import loop

DEFAULT_CONFIG = "sortwaste"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--seeds", default=None)
    ap.add_argument("--data", default=None,
                    help="dataset yaml, defaults to config data_yaml")
    ap.add_argument("--strategy", default=None, choices=STRATEGIES,
                    help="one strategy, defaults to all of al.strategies")
    ap.add_argument("--seed", default=None, type=int, nargs="+",
                    help="one or more seeds, defaults to the first seeds_n")
    ap.add_argument("--device", default=None,
                    help="pin this process to a GPU ('0', '1'). Use one "
                         "process per GPU, since DDP drops PPAL's callbacks")
    ap.add_argument("--stop-after-round", type=int, default=None,
                    help="stop after this completed round, rerun to resume")
    a = ap.parse_args()

    if a.device:
        os.environ["ALBENCH_DEVICE"] = a.device
    cfg = load_config(Path(a.config))
    al = cfg["al"]
    data = a.data or cfg["data_yaml"]
    strategies = [a.strategy] if a.strategy else al["strategies"]
    seed_list = (a.seed if a.seed is not None
                 else load_seeds(a.seeds)[: al["seeds_n"]])

    failures = []
    for strategy in strategies:
        for seed in seed_list:
            print(f"{strategy} seed={seed}", flush=True)
            try:
                loop.run(strategy, seed, cfg, data,
                         stop_after_round=a.stop_after_round)
            except Exception as e:
                print(f"FAILED: {strategy} seed={seed}: {e}", flush=True)
                failures.append((strategy, seed))
    if failures:
        for s, sd in failures:
            print(f"  failed: {s} seed={sd}", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
