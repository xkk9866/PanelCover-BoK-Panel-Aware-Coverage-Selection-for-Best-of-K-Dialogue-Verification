"""CLI entry point for PsyState training.

Example:

    python -m psystate.run_train --config configs/psystate_full.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from .losses import LossConfig
from .trainer import TrainArgs, train


def _apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    for kv in overrides:
        if "=" not in kv:
            continue
        k, v = kv.split("=", 1)
        # Try yaml parse of value so we can pass ints/floats/bools cleanly.
        try:
            v = yaml.safe_load(v)
        except Exception:
            pass
        cfg[k] = v
    return cfg


def build_args_from_cfg(cfg: dict) -> TrainArgs:
    loss_keys = {k for k in LossConfig.__dataclass_fields__}
    loss_cfg_kwargs = {k: cfg[k] for k in list(cfg.keys()) if k in loss_keys}
    for k in list(cfg.keys()):
        if k in loss_keys and not k.startswith("use_"):
            cfg.pop(k)
    # The use_* toggles belong to both TrainArgs and LossConfig — leave them
    # in cfg for TrainArgs and let LossConfig read them via propagation.
    train_args = TrainArgs(**{k: v for k, v in cfg.items() if k in TrainArgs.__dataclass_fields__})
    # Propagate any loss weight overrides.
    for k, v in loss_cfg_kwargs.items():
        setattr(train_args.loss_cfg, k, v)
    return train_args


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    cfg = _apply_overrides(cfg, args.overrides)
    train_args = build_args_from_cfg(cfg)
    print("[config]\n" + json.dumps({k: (str(v) if not isinstance(v, (int, float, bool, str, type(None))) else v) for k, v in vars(train_args).items()}, indent=2, ensure_ascii=False))
    train(train_args)


if __name__ == "__main__":
    main()
