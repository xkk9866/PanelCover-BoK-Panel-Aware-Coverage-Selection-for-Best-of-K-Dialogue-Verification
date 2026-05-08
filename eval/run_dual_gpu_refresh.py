"""Dual-GPU refresh helper for expensive baseline materialization.

Most paper metrics in ``eval.run_full_experiments`` are CPU-only because
they consume cached top-K plans and response files.  The GPU-expensive
part is rebuilding baseline top-K files from checkpoints.  This helper
lets a user provide any available checkpoints and runs those baseline
refreshes on GPU0/GPU1 in parallel before launching the CPU metric
refresh.

Example
-------

python -m eval.run_dual_gpu_refresh \
  --misc_ckpt runs/baseline_misc/ckpt-400 \
  --multiesc_ckpt runs/baseline_multiesc/ckpt-400 \
  --transesc_ckpt runs/baseline_transesc/ckpt-400
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def _launch(cmd: list[str], *, gpu: str, log_path: Path) -> subprocess.Popen:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    f = log_path.open("w", encoding="utf-8")
    print(f"[dual-gpu] gpu={gpu} log={log_path} :: {' '.join(cmd)}",
          flush=True)
    return subprocess.Popen(
        cmd,
        env=env,
        stdout=f,
        stderr=subprocess.STDOUT,
        text=True,
    )


def _wait_all(procs: list[tuple[str, subprocess.Popen]]) -> None:
    failed = []
    for name, proc in procs:
        rc = proc.wait()
        print(f"[dual-gpu] {name} exit={rc}", flush=True)
        if rc != 0:
            failed.append((name, rc))
    if failed:
        msg = ", ".join(f"{n}:{rc}" for n, rc in failed)
        raise SystemExit(f"[dual-gpu] failed jobs: {msg}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--misc_ckpt")
    ap.add_argument("--multiesc_ckpt")
    ap.add_argument("--transesc_ckpt")
    ap.add_argument("--gpus", nargs="+", default=["0", "1"])
    ap.add_argument("--K", type=int, default=3)
    ap.add_argument("--skip_cpu_refresh", action="store_true")
    args = ap.parse_args()

    jobs = [
        ("misc", args.misc_ckpt),
        ("multiesc", args.multiesc_ckpt),
        ("transesc", args.transesc_ckpt),
    ]
    jobs = [(name, ckpt) for name, ckpt in jobs if ckpt]
    if not jobs:
        print("[dual-gpu] no checkpoints supplied; only CPU refresh will run",
              flush=True)

    procs: list[tuple[str, subprocess.Popen]] = []
    for i, (name, ckpt) in enumerate(jobs):
        gpu = args.gpus[i % len(args.gpus)]
        cmd = [
            sys.executable, "-m", "eval.run_baseline_eval",
            "--ckpt", ckpt,
            "--system_name", name,
            "--gpu", gpu,
            "--K", str(args.K),
            "--skip_sweep",
        ]
        procs.append((
            name,
            _launch(cmd, gpu=gpu,
                    log_path=Path("logs") / f"refresh_{name}_gpu{gpu}.log"),
        ))
    _wait_all(procs)

    if not args.skip_cpu_refresh:
        print("[dual-gpu] refreshing CPU metrics", flush=True)
        rc = subprocess.run([sys.executable, "-m", "eval.run_full_experiments"])
        if rc.returncode != 0:
            raise SystemExit(rc.returncode)


if __name__ == "__main__":
    main()
