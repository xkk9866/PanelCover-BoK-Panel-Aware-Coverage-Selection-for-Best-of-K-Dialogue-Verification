r"""Bootstrap CIs and sign-flip tests for a fair-BoN focal system.

Reports two BT-projected winrates side by side for every baseline:

* ``overall`` -- canonical L2-fit BT-overall logit
* ``state``   -- state-conditioned linear combination of per-dimension
  BT logits, weights mirror the SCT-BoK planner / state-aware verifier.

Each value is sigmoid(score(focal) - score(baseline)) averaged over
contexts; bootstrap 95% CIs and one-sided sign-flip p-values for the
overall winrate are reported too.

The historical filename is kept for compatibility with older scripts.
The default focal system is PsyState-SCT under the canonical fair-BoN
verifier.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np

DIMS = ("overall", "empathy", "specificity", "actionability",
        "safety", "appropriateness", "helpfulness")


def _norm(s: str | None) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def _bt_winrate(text_a: str, text_b: str,
                  bt: dict[str, float]) -> float:
    if not text_a or not text_b:
        return 0.5
    if text_a == text_b:
        return 0.5
    sa = bt.get(text_a, 0.0)
    sb = bt.get(text_b, 0.0)
    return float(1.0 / (1.0 + math.exp(-(sa - sb))))


def _bootstrap_ci(values: np.ndarray, *, B: int = 2000,
                   alpha: float = 0.05, seed: int = 20260502
                   ) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(values)
    if n == 0:
        return float("nan"), float("nan")
    boots = np.empty(B)
    for i in range(B):
        idx = rng.integers(0, n, size=n)
        boots[i] = values[idx].mean()
    boots.sort()
    lo = boots[int(alpha / 2 * B)]
    hi = boots[int((1 - alpha / 2) * B)]
    return float(lo), float(hi)


def _signflip_pvalue(values: np.ndarray, *, B: int = 5000,
                      seed: int = 20260502) -> float:
    """One-sided sign-flip permutation test for H0: mean = 0.5
    against H1: mean > 0.5.  Statistic is the observed mean."""
    rng = np.random.default_rng(seed)
    centred = values - 0.5
    obs = float(centred.mean())
    if obs <= 0:
        return 1.0
    n = len(centred)
    count = 0
    for _ in range(B):
        signs = rng.choice([-1.0, 1.0], size=n)
        if (centred * signs).mean() >= obs:
            count += 1
    return float((count + 1) / (B + 1))


def _load_jsonl(path: Path) -> list[dict]:
    out: list[dict] = []
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


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
        [0.18, 0.15, 0.14, 0.12, 0.10, 0.14, 0.17],  # overall, emp, spec, act, safe, app, help
        dtype=np.float64,
    )
    base[1] += 0.16 * max(distress - 0.5, 0.0)            # empathy
    base[4] += 0.18 if severe else (0.09 if risk_any else 0.0)  # safety
    base[2] += 0.12 * max(0.65 - clarity, 0.0)            # specificity
    base[3] += 0.12 * max(readiness - 0.55, 0.0)          # actionability
    base[5] += 0.10 * max(0.55 - alliance, 0.0)           # appropriateness
    return base / base.sum()


def _state_score(text: str, bt: dict[str, dict[str, float]],
                  ctx: dict) -> float:
    w = _state_weights(ctx)
    s = 0.0
    for j, d in enumerate(DIMS):
        s += float(w[j]) * float(bt.get(d, {}).get(text, 0.0))
    return float(s)


def main() -> None:
    import argparse
    from eval.bt_winrate_proxy import (
        _build_response_text_index, _collect_dim_games, _fit_bt
    )

    ap = argparse.ArgumentParser()
    ap.add_argument("--focal", default="psystate_sctbok_bon_v12")
    ap.add_argument("--baselines", nargs="+", default=[
        "lexicon_bon_v12", "majority_bon_v12",
        "misc_bon_v12",
        "multiesc_bon_v12",
        "transesc_bon_v12",
        "btgreedy_bon_v12",
        "oracle_bon_v12",
    ])
    ap.add_argument("--eval_set",
                    default="data/judge_eval_v10/v10_eval_contexts.jsonl")
    ap.add_argument("--out", default="results/sctbok_significance.json")
    args = ap.parse_args()

    print("[bsbok-sig] indexing responses ...")
    idx_text = _build_response_text_index()
    games = _collect_dim_games(idx_text)
    bt: dict[str, dict[str, float]] = {
        d: _fit_bt(games.get(d, [])) for d in DIMS
    }

    eval_rows = _load_jsonl(Path(args.eval_set))
    ctx_index: dict[str, dict] = {str(r.get("sample_id")): r for r in eval_rows}

    focal = args.focal
    baselines = args.baselines

    out: dict = {"focal": focal, "baselines": {}}
    print(f"\n{'Baseline':<24s} {'n':>4s} "
           f"{'over.':>7s}{'95% CI':>16s}{'p':>8s} "
           f"{'state':>7s}{'95% CI':>16s}{'p':>8s} "
           f"{'emp.':>6s}{'spec':>6s}{'act.':>6s}{'safe':>6s}")
    print("-" * 130)
    for base in baselines:
        if base == focal:
            continue
        sids = sorted({s for (s, sys) in idx_text if sys == focal}
                       & {s for (s, sys) in idx_text if sys == base})
        if not sids:
            print(f"{base:<24s}   --   (no cached responses; "
                   f"run the BoN bridge first)")
            continue
        per_dim_arr: dict[str, np.ndarray] = {}
        for d in DIMS:
            arr = []
            for sid in sids:
                ta = idx_text[(sid, focal)]
                tb = idx_text[(sid, base)]
                arr.append(_bt_winrate(ta, tb, bt[d]))
            per_dim_arr[d] = np.asarray(arr)
        # state-conditioned winrate uses a per-context BT-aggregated
        # logit so the verifier and the planner share the same scoring.
        state_arr: list[float] = []
        for sid in sids:
            ctx = ctx_index.get(sid, {})
            ta = idx_text[(sid, focal)]
            tb = idx_text[(sid, base)]
            sa = _state_score(ta, bt, ctx)
            sb = _state_score(tb, bt, ctx)
            state_arr.append(float(1.0 / (1.0 + math.exp(-(sa - sb)))))
        state_np = np.asarray(state_arr)
        n = len(sids)
        ovr = per_dim_arr["overall"]
        mean = float(ovr.mean()) if n else float("nan")
        ci_lo, ci_hi = _bootstrap_ci(ovr)
        p_val = _signflip_pvalue(ovr)
        s_mean = float(state_np.mean()) if n else float("nan")
        s_ci_lo, s_ci_hi = _bootstrap_ci(state_np)
        s_p = _signflip_pvalue(state_np)
        rec = {"n": n,
               "mean": mean, "ci_lo": ci_lo, "ci_hi": ci_hi,
               "p_one_sided": p_val,
               "state_mean": s_mean, "state_ci_lo": s_ci_lo,
               "state_ci_hi": s_ci_hi, "state_p_one_sided": s_p,
               "per_dim_mean": {d: float(per_dim_arr[d].mean())
                                  for d in DIMS}}
        out["baselines"][base] = rec
        ci_str = f"[{ci_lo:.3f},{ci_hi:.3f}]"
        s_ci_str = f"[{s_ci_lo:.3f},{s_ci_hi:.3f}]"
        emp = per_dim_arr["empathy"].mean()
        spec = per_dim_arr["specificity"].mean()
        act = per_dim_arr["actionability"].mean()
        sfe = per_dim_arr["safety"].mean()
        print(f"{base:<24s} {n:>4d} "
               f"{mean:>7.3f}{ci_str:>16s}{p_val:>8.4f} "
               f"{s_mean:>7.3f}{s_ci_str:>16s}{s_p:>8.4f} "
               f"{emp:>6.3f}{spec:>6.3f}{act:>6.3f}{sfe:>6.3f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\n[bsbok-sig] wrote {args.out}")


if __name__ == "__main__":
    main()
