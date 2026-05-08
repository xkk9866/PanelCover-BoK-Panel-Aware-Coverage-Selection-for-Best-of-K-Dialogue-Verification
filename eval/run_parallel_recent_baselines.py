"""Train recent ESC baselines on GPU0/GPU1 in parallel.

This launcher is intentionally small: it runs repository-native configs
through ``psystate.run_train`` and assigns each process to one visible GPU.
After training, pass the resulting checkpoints to ``eval.run_dual_gpu_refresh``
to build top-K plans, materialise BoN responses, and refresh metrics.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


DEFAULT_JOBS = [
    ("kemi", "configs/baseline_kemi.yaml"),
    ("rag", "configs/baseline_rag.yaml"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", nargs="+", default=["0", "1"])
    ap.add_argument("--jobs", nargs="*", default=[
        f"{name}:{cfg}" for name, cfg in DEFAULT_JOBS
    ], help="name:config.yaml entries")
    ap.add_argument("--steps", type=int, default=None,
                    help="Optional override for num_train_steps, useful for smoke tests.")
    args = ap.parse_args()

    procs: list[tuple[str, subprocess.Popen]] = []
    for i, item in enumerate(args.jobs):
        name, cfg = item.split(":", 1)
        gpu = args.gpus[i % len(args.gpus)]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu
        env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        log_path = Path("logs") / f"train_{name}_gpu{gpu}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, "-m", "psystate.run_train", "--config", cfg]
        if args.steps is not None:
            cmd.append(f"num_train_steps={args.steps}")
        print(f"[parallel-baseline] gpu={gpu} {name}: {' '.join(cmd)} -> {log_path}",
              flush=True)
        f = log_path.open("w", encoding="utf-8")
        procs.append((name, subprocess.Popen(
            cmd, env=env, stdout=f, stderr=subprocess.STDOUT, text=True)))

    failed = []
    for name, proc in procs:
        rc = proc.wait()
        print(f"[parallel-baseline] {name} exit={rc}", flush=True)
        if rc != 0:
            failed.append((name, rc))
    if failed:
        raise SystemExit("failed jobs: " + ", ".join(f"{n}:{rc}" for n, rc in failed))


if __name__ == "__main__":
    main()
