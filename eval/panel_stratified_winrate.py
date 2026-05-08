"""Stratified panel-aware winrate by inter-expert disagreement.

The key conditional SOTA claim of Panel-Aware Best-of-$K$ is:

  * On contexts where the panel agrees (low inter-expert variance over
    the seven candidate responses), all sensible planners give similar
    results and the algorithmic edge is bounded.
  * On contexts where the panel disagrees substantially (high
    inter-expert variance), panel-aware coverage matters and the gap
    of PA-SCT over BT-Greedy / SCT-BoK / recent ESC planners should
    grow.

For each context $x$, we measure the panel disagreement on the
seven cached candidate responses as the maximum across experts of

    var_e (BT_e^overall(text(x, a))) for a in K-set,

and stratify contexts into high vs. low disagreement halves
(median-split).  We then report PA-SCT-vs-baseline panel-aware
winrate per stratum.

A consistent gap that grows in the high-disagreement stratum is
strong evidence that the panel-aware verifier paradigm matters
exactly in the clinically interesting regime.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np


EXPERT_IDS = ("E1_supervisor", "E2_client_experience", "E3_safety_reviewer")
DIMS = ("overall", "helpfulness", "empathy", "specificity",
        "actionability", "appropriateness", "safety")
STRATEGIES = (
    "question", "reflection", "empathy", "reframe",
    "summarization", "action_suggestion", "safety_referral",
)


def _norm(s):
    return re.sub(r"\s+", " ", (s or "").strip())


def _sigmoid(x):
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)


def _load_jsonl(p):
    if not Path(p).exists():
        return []
    out = []
    for line in Path(p).read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


def _resp_idx(p):
    idx = {}
    for r in _load_jsonl(p):
        sid = str(r.get("sample_id", ""))
        strat = str(r.get("selected_strategy", ""))
        text = _norm(r.get("response"))
        backend = (r.get("generation_config") or {}).get("backend")
        if sid and strat and text and backend != "safety_template":
            idx.setdefault((sid, strat), text)
    return idx


def _load_panel_bt(p):
    raw = json.loads(Path(p).read_text(encoding="utf-8"))
    return {
        e: {d: {k: float(v) for k, v in raw.get(e, {}).get(d, {}).items()}
             for d in DIMS}
        for e in EXPERT_IDS
    }


def _topk(p):
    out = {}
    for r in _load_jsonl(p):
        sid = str(r.get("sample_id"))
        strats = list(r.get("candidate_strategies") or [])
        if r.get("decision") == "safety_hard":
            strats = ["safety_referral"]
        out[sid] = strats
    return out


def _kset_max(strats, sid, resp_idx, bt):
    best = -np.inf
    for s in strats:
        text = resp_idx.get((sid, s))
        if text is None:
            score = 0.0
        else:
            score = float(bt.get(text, 0.0))
        if score > best:
            best = score
    return float(best) if math.isfinite(best) else 0.0


def _ctx_disagreement(sid, resp_idx, panel_bt):
    """Compute inter-expert variance of BT-overall logits over the
    seven cached candidate responses for context ``sid``.  Higher means
    the experts will diverge on which candidate is best."""
    avail_texts = []
    for s in STRATEGIES:
        t = resp_idx.get((sid, s))
        if t is not None:
            avail_texts.append(t)
    if len(avail_texts) < 2:
        return 0.0
    expert_vec = []
    for e in EXPERT_IDS:
        scores = [float(panel_bt[e]["overall"].get(t, 0.0)) for t in avail_texts]
        expert_vec.append(np.asarray(scores, dtype=np.float64))
    expert_vec = np.stack(expert_vec, axis=0)        # (E, n_resp)
    # For each response, variance across experts:
    per_resp_var = np.var(expert_vec, axis=0)
    # Aggregate as max variance over the candidate pool: this is the
    # most disagreement the verifier could exploit.
    return float(per_resp_var.max())


def _bootstrap_ci(arr, B=5000, seed=20260502):
    arr = np.asarray(arr, dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boot = rng.choice(arr, size=(B, arr.size), replace=True).mean(axis=1)
    lo, hi = float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))
    diffs = arr - 0.5
    obs = float(diffs.mean())
    n_extreme = 0
    for _ in range(B):
        signs = rng.choice([1, -1], size=arr.size, replace=True)
        if (diffs * signs).mean() >= obs:
            n_extreme += 1
    p = (n_extreme + 1) / (B + 1)
    return lo, hi, float(p)


def _winrate_pair(focal_K, base_K, sids, resp_idx, panel_bt):
    wrs = []
    gaps = []
    for sid in sids:
        per_e = []
        per_g = []
        for e in EXPERT_IDS:
            fmax = _kset_max(focal_K[sid], sid, resp_idx, panel_bt[e]["overall"])
            bmax = _kset_max(base_K[sid], sid, resp_idx, panel_bt[e]["overall"])
            per_e.append(_sigmoid(fmax - bmax))
            per_g.append(fmax - bmax)
        wrs.append(np.mean(per_e))
        gaps.append(np.mean(per_g))
    return np.asarray(wrs), np.asarray(gaps)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--topk_dir", default="data/fair_bon_v12")
    ap.add_argument("--responses",
                    default="data/judge_eval_v10/v10_responses.jsonl")
    ap.add_argument("--panel_bt", default="results/panel_bt.json")
    ap.add_argument("--focal", default="psystate_pasct")
    ap.add_argument("--baselines", nargs="+",
                    default=["btgreedy", "psystate_sctbok",
                             "misc", "multiesc", "transesc"])
    ap.add_argument("--out_json",
                    default="results/panel_aware_stratified.json")
    args = ap.parse_args()

    panel_bt = _load_panel_bt(Path(args.panel_bt))
    resp_idx = _resp_idx(Path(args.responses))
    focal_K = _topk(Path(args.topk_dir) / f"{args.focal}_topk.jsonl")

    # Compute disagreement per context
    disagreement = {sid: _ctx_disagreement(sid, resp_idx, panel_bt)
                    for sid in focal_K}
    vals = sorted(disagreement.values())
    median = vals[len(vals) // 2]
    print(f"[stratify] disagreement median={median:.4f} max={vals[-1]:.4f}")

    # Tertiles for finer stratification
    n = len(vals)
    t1 = vals[n // 3]
    t2 = vals[2 * n // 3]
    print(f"[stratify] tertile thresholds: {t1:.4f}, {t2:.4f}")

    # Stratify contexts; safety-hard contexts are excluded from the
    # disagreement analysis since their K-set is forced.
    safety_hard = {sid for sid, K in focal_K.items() if K == ["safety_referral"]}
    tertile_sids = {"low": [], "mid": [], "high": []}
    for sid, v in disagreement.items():
        if sid in safety_hard:
            continue
        if v <= t1:
            tertile_sids["low"].append(sid)
        elif v <= t2:
            tertile_sids["mid"].append(sid)
        else:
            tertile_sids["high"].append(sid)
    for k, v in tertile_sids.items():
        print(f"[stratify] {k:>4s}: {len(v)} contexts")

    record = {"focal": args.focal, "tertiles": {k: {} for k in tertile_sids}}
    print(f"\n{'Stratum':<8s}{'Baseline':<22s}{'WR':>8s}"
          f"{'95% CI':>20s}{'p':>8s}{'gap':>10s}")
    print("-" * 80)
    for stratum, sids in tertile_sids.items():
        for base in args.baselines:
            base_K = _topk(Path(args.topk_dir) / f"{base}_topk.jsonl")
            common_sids = sorted(set(sids) & set(focal_K) & set(base_K))
            if not common_sids:
                continue
            wrs, gaps = _winrate_pair(focal_K, base_K, common_sids,
                                      resp_idx, panel_bt)
            wr = float(wrs.mean())
            gap = float(gaps.mean())
            lo, hi, p = _bootstrap_ci(wrs.tolist(), B=2000)
            record["tertiles"][stratum][base] = {
                "n": len(common_sids),
                "winrate": wr,
                "winrate_ci_lo": lo,
                "winrate_ci_hi": hi,
                "p_value": p,
                "utility_gap": gap,
            }
            ci_str = f"[{lo:.3f}, {hi:.3f}]"
            print(f"{stratum:<8s}{base:<22s}{wr:>8.3f}{ci_str:>20s}"
                  f"{p:>8.3f}{gap:>+10.4f}")
        print()

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(
        json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[stratify] wrote {args.out_json}")


if __name__ == "__main__":
    main()
