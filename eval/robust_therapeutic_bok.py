"""State-Conditioned Therapeutic Coverage Best-of-K planner (SCT-BoK).

Top-K by an average verifier score is suboptimal under state
uncertainty: in mental-health dialogue, the same context is consistent
with multiple plausible counsellor priorities (high distress favours
empathy and validation; low clarity favours specificity; readiness
favours actionability; risk favours safety).  SCT-BoK therefore plans
a small candidate set that *covers* a handful of state-conditioned
clinical scenarios, instead of optimising the average preference only.

Algorithm (one pass per context x):

  1. Build a per-dimension response profile psi_d(x, a) from L2-fit
     panel BT logits, optionally shrunk by a lower-confidence bound
     based on the amount of panel evidence available for the response.
  2. Build a small set of clinical scenarios W(x) parametrised by the
     posterior counselling state (distress, readiness, alliance,
     clarity) and the risk indicator.  Each scenario is a non-negative
     weight vector over the seven BT dimensions.
  3. Greedily pick a K-set S_K to maximise the monotone submodular
     coverage objective F(S | x) = (1/|W|) sum_{w in W(x)} max_{a in S} w . psi(a).
  4. Honour clinical constraints: severe / imminent risk forces
     ``safety_referral`` into the set; high distress or low alliance
     forces an empathy / reflection candidate; very low clarity forces
     a question / summarisation candidate.  Constraints are added
     before the greedy round and never replace a strategy required by
     a stricter rule.

The script writes only the K-set and per-strategy profile diagnostics
to ``<out_topk>``; final response materialisation goes through the
shared fair-BoN verifier in ``eval/v12_best_of_n_fair.py`` so every
system, including SCT-BoK, is selected by exactly the same canonical
BT verifier.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np

from psystate.constants import STRATEGIES


DIMS = (
    "overall",
    "helpfulness",
    "empathy",
    "specificity",
    "actionability",
    "appropriateness",
    "safety",
)


SAFETY_TEMPLATE = (
    "听到你说这些我有些担心你的安全。如果你正经历强烈的伤害自己的念头或处于"
    "情绪非常艰难的时刻，请考虑联系全国 24 小时心理援助热线 400-161-9995，"
    "或告诉一个你信任的人，让 ta 此刻陪着你。"
    "你愿意先告诉我一些此刻你身边能联系到的支持吗？"
)


def _load_jsonl(path: Path) -> list[dict]:
    out: list[dict] = []
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def _norm(s: str | None) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-float(x)))


def _state(ctx: dict, key: str, default: float = 0.5) -> float:
    return float((ctx.get("posterior_state") or {}).get(key, default))


def _risk_level(ctx: dict) -> str:
    return str(ctx.get("risk_level") or "none")


def _is_risk(ctx: dict) -> bool:
    return bool(ctx.get("risk_any") or _risk_level(ctx) in {"mild", "severe", "imminent"})


def _response_index(paths: list[Path]) -> dict[tuple[str, str], dict]:
    idx: dict[tuple[str, str], dict] = {}
    for path in paths:
        for rec in _load_jsonl(path):
            sid = str(rec.get("sample_id", ""))
            strat = str(rec.get("selected_strategy", ""))
            text = _norm(rec.get("response"))
            backend = (rec.get("generation_config") or {}).get("backend")
            if sid and strat and text and backend != "safety_template":
                idx.setdefault((sid, strat), rec)
    return idx


def _bt_models_and_counts() -> tuple[dict[str, dict[str, float]], dict[str, dict[str, int]]]:
    from eval.bt_winrate_proxy import _build_response_text_index, _collect_dim_games, _fit_bt

    idx_text = _build_response_text_index()
    games = _collect_dim_games(idx_text)
    bt: dict[str, dict[str, float]] = {}
    counts: dict[str, dict[str, int]] = {}
    for dim in DIMS:
        dim_games = games.get(dim, [])
        bt[dim] = _fit_bt(dim_games)
        c: dict[str, int] = {}
        for a, b, _ in dim_games:
            c[a] = c.get(a, 0) + 1
            c[b] = c.get(b, 0) + 1
        counts[dim] = c
    return bt, counts


def _profiles_for_context(
    sid: str,
    resp_idx: dict[tuple[str, str], dict],
    bt: dict[str, dict[str, float]],
    counts: dict[str, dict[str, int]],
    *,
    robust_lcb: bool,
    beta: float,
) -> tuple[dict[str, np.ndarray], dict[str, str], dict[str, np.ndarray]]:
    mean: dict[str, np.ndarray] = {}
    lcb: dict[str, np.ndarray] = {}
    text: dict[str, str] = {}
    for strat in STRATEGIES:
        rec = resp_idx.get((sid, strat))
        if rec is None:
            continue
        t = _norm(rec.get("response"))
        if not t:
            continue
        vals = []
        lows = []
        for dim in DIMS:
            p = _sigmoid(bt.get(dim, {}).get(t, 0.0))
            n = counts.get(dim, {}).get(t, 0)
            # Conservative posterior standard error proxy.  It is not a
            # calibrated posterior interval, but it makes unsupported
            # responses compete on lower-confidence quality.
            se = math.sqrt(max(p * (1.0 - p), 1e-4) / max(n + 2, 2))
            vals.append(p)
            lows.append(max(0.0, p - beta * se) if robust_lcb else p)
        mean[strat] = np.asarray(vals, dtype=np.float64)
        lcb[strat] = np.asarray(lows, dtype=np.float64)
        text[strat] = t
    return lcb, text, mean


def _weights(ctx: dict, *, state_scenarios: bool) -> np.ndarray:
    if not state_scenarios:
        return np.ones((1, len(DIMS)), dtype=np.float64) / len(DIMS)

    distress = _state(ctx, "distress")
    readiness = _state(ctx, "readiness")
    alliance = _state(ctx, "alliance")
    clarity = _state(ctx, "clarity")
    risk = _is_risk(ctx)
    severe = _risk_level(ctx) in {"severe", "imminent"}

    base = np.asarray([0.18, 0.17, 0.15, 0.14, 0.12, 0.14, 0.10], dtype=np.float64)
    base[DIMS.index("empathy")] += 0.16 * max(distress - 0.5, 0.0)
    base[DIMS.index("safety")] += (0.18 if severe else 0.09 if risk else 0.0)
    base[DIMS.index("specificity")] += 0.12 * max(0.65 - clarity, 0.0)
    base[DIMS.index("actionability")] += 0.12 * max(readiness - 0.55, 0.0)
    base[DIMS.index("appropriateness")] += 0.10 * max(0.55 - alliance, 0.0)
    base = base / base.sum()

    scenarios = [base]
    for dim, mass in (
        ("empathy", 0.18),
        ("safety", 0.18 if risk else 0.08),
        ("specificity", 0.12),
        ("actionability", 0.12),
        ("appropriateness", 0.10),
    ):
        w = base.copy()
        w[DIMS.index(dim)] += mass
        scenarios.append(w / w.sum())
    return np.asarray(scenarios, dtype=np.float64)


def _utility(profile: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return weights @ profile


def _best_required(group: set[str], profiles: dict[str, np.ndarray], weights: np.ndarray) -> str | None:
    avail = [s for s in group if s in profiles]
    if not avail:
        return None
    return max(avail, key=lambda s: float(_utility(profiles[s], weights).mean()))


def _required_strategies(ctx: dict, profiles: dict[str, np.ndarray], weights: np.ndarray) -> list[str]:
    required: list[str] = []
    severe = _risk_level(ctx) in {"severe", "imminent"}
    distress = _state(ctx, "distress")
    alliance = _state(ctx, "alliance")
    clarity = _state(ctx, "clarity")

    if severe and "safety_referral" in profiles:
        required.append("safety_referral")
    if distress >= 0.65 or alliance <= 0.35:
        s = _best_required({"empathy", "reflection"}, profiles, weights)
        if s and s not in required:
            required.append(s)
    if clarity <= 0.35:
        s = _best_required({"question", "summarization"}, profiles, weights)
        if s and s not in required:
            required.append(s)
    return required


def _greedy(
    profiles: dict[str, np.ndarray],
    weights: np.ndarray,
    K: int,
    *,
    required: list[str],
) -> list[str]:
    selected: list[str] = []
    current = np.full(weights.shape[0], -np.inf, dtype=np.float64)

    for strat in required:
        if strat in profiles and strat not in selected and len(selected) < K:
            selected.append(strat)
            current = np.maximum(current, _utility(profiles[strat], weights))

    while len(selected) < K:
        best_s = None
        best_gain = -np.inf
        for strat, prof in profiles.items():
            if strat in selected:
                continue
            score = _utility(prof, weights)
            gain = float((np.maximum(current, score) - current).mean())
            if gain > best_gain:
                best_gain = gain
                best_s = strat
        if best_s is None:
            break
        selected.append(best_s)
        current = np.maximum(current, _utility(profiles[best_s], weights))
    return selected


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_set", default="data/judge_eval_v10/v10_eval_contexts.jsonl")
    ap.add_argument("--responses", default="data/judge_eval_v10/v10_responses.jsonl")
    ap.add_argument("--safety_overrides", default="data/judge_eval_v10/v12_safety_overrides.jsonl")
    ap.add_argument("--out_topk", default="data/fair_bon_v12/psystate_sctbok_topk.jsonl")
    ap.add_argument("--system_id", default="psystate_sctbok")
    ap.add_argument("--K", type=int, default=3)
    ap.add_argument("--beta", type=float, default=0.75)
    ap.add_argument("--use_robust_lcb", action="store_true",
                    help="Ablation: use lower-confidence BT profiles "
                         "instead of mean BT logits.")
    ap.add_argument("--no_state_scenarios", action="store_true",
                    help="Ablation: drop state-conditioned scenarios.")
    ap.add_argument("--no_clinical_constraints", action="store_true",
                    help="Ablation: drop clinical coverage constraints.")
    args = ap.parse_args()

    eval_rows = _load_jsonl(Path(args.eval_set))
    overrides = {
        str(r.get("sample_id")): r
        for r in _load_jsonl(Path(args.safety_overrides))
        if r.get("shield_fired") or r.get("shield_fired_hard") or r.get("decision") == "hard"
    }
    resp_idx = _response_index([Path(args.responses)])
    bt, counts = _bt_models_and_counts()

    Path(args.out_topk).parent.mkdir(parents=True, exist_ok=True)

    n_total = 0
    n_safety = 0
    n_missing = 0

    with open(args.out_topk, "w", encoding="utf-8") as ftop:
        for ctx in eval_rows:
            sid = str(ctx["sample_id"])
            n_total += 1
            if sid in overrides:
                ftop.write(json.dumps({
                    "sample_id": sid,
                    "planner": args.system_id,
                    "candidate_strategies": ["safety_referral"],
                    "required_strategies": ["safety_referral"],
                    "decision": "safety_hard",
                    "K": args.K,
                }, ensure_ascii=False) + "\n")
                n_safety += 1
                continue

            profiles, text_table, mean_profiles = _profiles_for_context(
                sid,
                resp_idx,
                bt,
                counts,
                robust_lcb=bool(args.use_robust_lcb),
                beta=args.beta,
            )
            if not profiles:
                n_missing += 1
                continue

            weights = _weights(ctx, state_scenarios=not args.no_state_scenarios)
            required = [] if args.no_clinical_constraints else _required_strategies(ctx, profiles, weights)
            topk = _greedy(profiles, weights, args.K, required=required)
            if not topk:
                n_missing += 1
                continue

            ftop.write(json.dumps({
                "sample_id": sid,
                "planner": args.system_id,
                "candidate_strategies": topk,
                "required_strategies": required,
                "weights": weights.tolist(),
                "mean_profiles": {k: v.tolist() for k, v in mean_profiles.items()},
                "robust_profiles": {k: v.tolist() for k, v in profiles.items()},
                "K": args.K,
                "beta": args.beta,
            }, ensure_ascii=False) + "\n")

    print(f"[sctbok] total={n_total} safety_hard={n_safety} missing={n_missing}")
    print(f"[sctbok] wrote {args.out_topk}")


if __name__ == "__main__":
    main()
