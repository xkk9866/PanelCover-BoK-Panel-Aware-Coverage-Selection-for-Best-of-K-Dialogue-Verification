"""Run SCT-BoK internal ablations and score against real baselines.

Each variant only emits its top-K plan; final response materialisation
goes through ``eval.v12_best_of_n_fair`` so all variants and all
baselines share the canonical BT verifier.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np


DIMS = (
    "overall",
    "empathy",
    "specificity",
    "actionability",
    "safety",
    "appropriateness",
    "helpfulness",
)


def _state_weights(ctx: dict) -> np.ndarray:
    s = (ctx.get("posterior_state") or {})
    distress = float(s.get("distress", 0.5))
    readiness = float(s.get("readiness", 0.5))
    alliance = float(s.get("alliance", 0.5))
    clarity = float(s.get("clarity", 0.5))
    risk_lvl = str(ctx.get("risk_level") or "none")
    risk_any = bool(ctx.get("risk_any") or risk_lvl in {"mild", "severe", "imminent"})
    severe = risk_lvl in {"severe", "imminent"}
    base = np.asarray(
        # overall, emp, spec, act, safe, app, help
        [0.18, 0.15, 0.14, 0.12, 0.10, 0.14, 0.17], dtype=np.float64,
    )
    base[1] += 0.16 * max(distress - 0.5, 0.0)
    base[4] += 0.18 if severe else (0.09 if risk_any else 0.0)
    base[2] += 0.12 * max(0.65 - clarity, 0.0)
    base[3] += 0.12 * max(readiness - 0.55, 0.0)
    base[5] += 0.10 * max(0.55 - alliance, 0.0)
    return base / base.sum()


def _norm(s: str | None) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def _bt_winrate(text_a: str, text_b: str, bt: dict[str, float]) -> float:
    if not text_a or not text_b or text_a == text_b:
        return 0.5
    sa = bt.get(text_a, 0.0)
    sb = bt.get(text_b, 0.0)
    return float(1.0 / (1.0 + math.exp(-(sa - sb))))


def _bootstrap_ci(values: np.ndarray, *, B: int = 2000, seed: int = 20260502) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    if len(values) == 0:
        return float("nan"), float("nan")
    boot = np.empty(B)
    for i in range(B):
        idx = rng.integers(0, len(values), size=len(values))
        boot[i] = values[idx].mean()
    boot.sort()
    return float(boot[int(0.025 * B)]), float(boot[int(0.975 * B)])


def _signflip_pvalue(values: np.ndarray, *, B: int = 5000, seed: int = 20260502) -> float:
    rng = np.random.default_rng(seed)
    centred = values - 0.5
    obs = float(centred.mean())
    if obs <= 0:
        return 1.0
    count = 0
    for _ in range(B):
        signs = rng.choice([-1.0, 1.0], size=len(centred))
        if float((centred * signs).mean()) >= obs:
            count += 1
    return float((count + 1) / (B + 1))


def _run_variant(name: str, *, K: int, use_lcb: bool, no_state: bool, no_constraints: bool) -> str:
    cmd = [
        sys.executable,
        "-m",
        "eval.robust_therapeutic_bok",
        "--K",
        str(K),
        "--system_id",
        name,
        "--out_topk",
        f"data/fair_bon_v12/{name}_topk.jsonl",
    ]
    if use_lcb:
        cmd.append("--use_robust_lcb")
    if no_state:
        cmd.append("--no_state_scenarios")
    if no_constraints:
        cmd.append("--no_clinical_constraints")
    print(f"\n[sctbok-ablate] plan: {' '.join(cmd)}", flush=True)
    env = os.environ.copy()
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    proc = subprocess.run(cmd, env=env)
    if proc.returncode != 0:
        raise SystemExit(f"[sctbok-ablate] {name} plan failed ({proc.returncode})")
    bon_cmd = [
        sys.executable,
        "-m",
        "eval.v12_best_of_n_fair",
        "--systems", name,
        "--topk_dir", "data/fair_bon_v12",
        "--out_dir", "data/fair_bon_v12",
        "--out_responses_jsonl", "NUL",
        "--system_suffix", "_bon_v12",
        "--verifier", "bt-overall",
    ]
    print(f"[sctbok-ablate] verify: {' '.join(bon_cmd)}", flush=True)
    proc = subprocess.run(bon_cmd, env=env)
    if proc.returncode != 0:
        raise SystemExit(f"[sctbok-ablate] {name} verify failed ({proc.returncode})")
    return f"{name}_bon_v12"


def _score_pair(idx_text: dict[tuple[str, str], str],
                  bt: dict[str, dict[str, float]],
                  ctx_idx: dict[str, dict],
                  focal: str, base: str) -> dict:
    sids = sorted({sid for sid, sys_id in idx_text if sys_id == focal} & {sid for sid, sys_id in idx_text if sys_id == base})
    rec: dict = {"n": len(sids)}
    if not sids:
        for dim in DIMS:
            rec[dim] = float("nan")
        rec["state"] = float("nan")
        return rec
    for dim in DIMS:
        values = np.asarray([
            _bt_winrate(idx_text[(sid, focal)], idx_text[(sid, base)], bt[dim])
            for sid in sids
        ])
        rec[dim] = float(values.mean())
        if dim == "overall":
            lo, hi = _bootstrap_ci(values)
            rec["overall_ci_lo"] = lo
            rec["overall_ci_hi"] = hi
            rec["overall_p"] = _signflip_pvalue(values)
    state_arr = []
    for sid in sids:
        ctx = ctx_idx.get(sid, {})
        w = _state_weights(ctx)
        ta = idx_text[(sid, focal)]
        tb = idx_text[(sid, base)]
        sa = sum(float(w[j]) * float(bt[d].get(ta, 0.0)) for j, d in enumerate(DIMS))
        sb = sum(float(w[j]) * float(bt[d].get(tb, 0.0)) for j, d in enumerate(DIMS))
        state_arr.append(float(1.0 / (1.0 + math.exp(-(sa - sb)))))
    state_np = np.asarray(state_arr)
    rec["state"] = float(state_np.mean())
    s_lo, s_hi = _bootstrap_ci(state_np)
    rec["state_ci_lo"] = s_lo
    rec["state_ci_hi"] = s_hi
    rec["state_p"] = _signflip_pvalue(state_np)
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip_run", action="store_true")
    ap.add_argument("--out", default="results/rcbok_ablation_sweep.json")
    ap.add_argument("--baselines", nargs="+", default=[
        "misc_bon_v12",
        "multiesc_bon_v12",
        "transesc_bon_v12",
        "lexicon_bon_v12",
        "majority_bon_v12",
    ])
    args = ap.parse_args()

    variants: list[tuple[str, dict]] = [
        ("psystate_sctbok",       dict(K=3, use_lcb=False, no_state=False, no_constraints=False)),
        ("sctbok_with_lcb",       dict(K=3, use_lcb=True,  no_state=False, no_constraints=False)),
        ("sctbok_no_state",       dict(K=3, use_lcb=False, no_state=True,  no_constraints=False)),
        ("sctbok_no_constraints", dict(K=3, use_lcb=False, no_state=False, no_constraints=True)),
        ("sctbok_plain_topk",     dict(K=3, use_lcb=False, no_state=True,  no_constraints=True)),
        ("sctbok_K1",             dict(K=1, use_lcb=False, no_state=False, no_constraints=False)),
        ("sctbok_K2",             dict(K=2, use_lcb=False, no_state=False, no_constraints=False)),
        ("sctbok_K4",             dict(K=4, use_lcb=False, no_state=False, no_constraints=False)),
        ("sctbok_K5",             dict(K=5, use_lcb=False, no_state=False, no_constraints=False)),
    ]

    if not args.skip_run:
        for name, kw in variants:
            _run_variant(name, **kw)

    from eval.bt_winrate_proxy import _build_response_text_index, _collect_dim_games, _fit_bt

    idx_text = _build_response_text_index()
    games = _collect_dim_games(idx_text)
    bt = {dim: _fit_bt(games.get(dim, [])) for dim in DIMS}

    eval_rows: list[dict] = []
    p_eval = Path("data/judge_eval_v10/v10_eval_contexts.jsonl")
    if p_eval.exists():
        for line in p_eval.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                eval_rows.append(json.loads(line))
    ctx_idx = {str(r.get("sample_id")): r for r in eval_rows}

    out: dict = {"variants": {}}
    for name, kw in variants:
        focal = f"{name}_bon_v12"
        out["variants"][name] = {"config": kw, "vs": {}}
        for base in args.baselines:
            out["variants"][name]["vs"][base] = _score_pair(
                idx_text, bt, ctx_idx, focal, base
            )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    short = lambda b: b.replace("_bon_v12", "")
    print("\n=== Overall BT-projected winrate ===")
    header = f"{'Variant':<24s} " + " ".join(f"{short(b):>10s}" for b in args.baselines)
    print(header); print("-" * len(header))
    for name, rec in out["variants"].items():
        cells = []
        for base in args.baselines:
            v = rec["vs"][base].get("overall", float("nan"))
            cells.append(f"{v:>10.3f}" if not math.isnan(v) else f"{'-':>10s}")
        print(f"{name:<24s} " + " ".join(cells))

    print("\n=== State-conditioned BT winrate ===")
    print(header); print("-" * len(header))
    for name, rec in out["variants"].items():
        cells = []
        for base in args.baselines:
            v = rec["vs"][base].get("state", float("nan"))
            cells.append(f"{v:>10.3f}" if not math.isnan(v) else f"{'-':>10s}")
        print(f"{name:<24s} " + " ".join(cells))

    print(f"\n[rcbok-ablate] wrote {args.out}")


if __name__ == "__main__":
    main()
