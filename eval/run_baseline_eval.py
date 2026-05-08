r"""End-to-end evaluation of one trained baseline (MISC / MultiESC /...).

Pipeline:
  1. Run the planner over ``v10_eval_contexts.jsonl`` to get top-K
     strategies per context (``baseline_topk_from_ckpt``).
  2. Run the fair Best-of-N selector under the canonical BT verifier
     shared by every system (``v12_best_of_n_fair``).
  3. Re-fit the BT proxy on the union of all panel verdicts and
     compute BT-projected winrates of the new system against every
     other cached fair-BoN system (``bt_winrate_proxy``).
  4. Bootstrap CIs and one-sided sign-flip p-values for the current
     PsyState-SCT focal system vs every cached baseline.

Usage::

    python -m eval.run_baseline_eval --ckpt runs/baseline_misc/ckpt-400 \
        --system_name misc

The output paths are::

    data/fair_bon_v12/<system>_topk.jsonl
    data/fair_bon_v12/<system>_bon_responses.jsonl
    results/bt_winrate_sweep.json       (refreshed)
    results/sctbok_significance.json    (refreshed)
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str], *, env: dict | None = None) -> None:
    print(f"\n[run] {' '.join(cmd)}\n", flush=True)
    full_env = os.environ.copy()
    if env:
        full_env.update(env)
    proc = subprocess.run(cmd, env=full_env)
    if proc.returncode != 0:
        raise SystemExit(f"[run] {' '.join(cmd)} -> exit {proc.returncode}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--system_name", required=True,
                    help="Short identifier (e.g. ``misc``, ``multiesc``)."
                    "  The fair-BoN responses will be written under "
                    "system_id=<system_name>_bon_v12.")
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--K", type=int, default=3)
    ap.add_argument("--skip_topk", action="store_true",
                    help="Use existing topk (e.g. when re-scoring only).")
    ap.add_argument("--skip_bon", action="store_true",
                    help="Use existing _bon_responses.jsonl.")
    ap.add_argument("--skip_sweep", action="store_true",
                    help="Skip BT sweep refresh.")
    args = ap.parse_args()

    env = {"CUDA_VISIBLE_DEVICES": args.gpu, "KMP_DUPLICATE_LIB_OK": "TRUE"}

    topk_path = Path("data/fair_bon_v12") / f"{args.system_name}_topk.jsonl"
    if not args.skip_topk:
        _run([
            sys.executable, "-m", "eval.baseline_topk_from_ckpt",
            "--ckpt", args.ckpt,
            "--out_topk", str(topk_path),
            "--system_name", args.system_name,
            "--K", str(args.K),
        ], env=env)

    if not args.skip_bon:
        _run([
            sys.executable, "-m", "eval.v12_best_of_n_fair",
            "--systems", args.system_name,
            "--out_responses_jsonl", "NUL",
            "--out_dir", "data/fair_bon_v12",
            "--topk_dir", "data/fair_bon_v12",
            "--verifier", "bt-overall",
        ], env=env)

    if not args.skip_sweep:
        _run([sys.executable, "-m", "eval.bt_winrate_proxy"])
        _run([sys.executable, "-m", "eval.bsbok_significance"])


if __name__ == "__main__":
    main()
