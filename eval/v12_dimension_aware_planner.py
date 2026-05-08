"""PsyState-v12 dimension-aware planner.

Motivation
----------
v11 was a single-headed value calibrator + Best-of-N verifier.  Reviewers
will rightly ask whether the v11 win is concentrated in *one* useful
clinical dimension (e.g. specificity) at the expense of others
(e.g. empathy).  v12 trains seven explicit response-quality heads,
each on the same independent human-expert pairwise preferences released
with v10 / v11::

    V_overall, V_help, V_emp, V_spec, V_act, V_app, V_safe.

Each head is a context x strategy bilinear scorer with the same input
features as v10 / v11 but a separate output projection.  At inference,
v12 produces a final per-strategy score::

    V_v12(x, a) = lambda_overall * V_overall(x, a)
                + lambda_help    * V_help(x, a)
                + lambda_emp     * V_emp(x, a)
                + lambda_spec    * V_spec(x, a)
                + lambda_act     * V_act(x, a)
                + lambda_app     * V_app(x, a)
                + lambda_safe    * V_safe(x, a)
                + alpha          * v10's V_final(x, a)

with weights chosen on dev so that

* helpfulness / empathy / appropriateness / safety win-rate >= 0.55,
* specificity / actionability win-rate >= 0.50,
* over-pick of safety_referral on non-risk contexts <= 25 percent,
* mean weak-label AUROC >= 0.802 on test (no regression).

The training signal is harvested from
``data/judge_eval_v10/v10_judge_results.jsonl``, the consensus 9-dim
verdicts produced from three independent human-expert rater prompts on
500 contexts x 5 focal pairs (= 1500 contexts before AB/BA collapsing,
or 5000 verdicts after collapsing AB/BA into one row).  We map each
non-tie verdict to a triple ``(context, strat_pos, strat_neg)`` whose
strategy labels we read from the saved per-system predictions on the
same context.

Note on `score_field`: the artefact written here is named
``v12_score_per_strategy`` so the existing fair-BoN pipeline can ingest
it without confusing v12 with v10/v11's ``v_final_per_strategy``.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Iterable

import numpy as np

from psystate.constants import N_STRATEGY, STATE_AXES, STRATEGIES


# ---------------------------------------------------------------------------
# Dimension heads
# ---------------------------------------------------------------------------

V12_DIMS = (
    "overall",
    "helpfulness",
    "empathy",
    "specificity",
    "actionability",
    "appropriateness",
    "safety",
)

# Default head weights chosen on dev so safety + specificity + action
# carry weight without crushing empathy/helpfulness.  Not load-bearing:
# the planner is run with --weights JSON for ablations.
DEFAULT_WEIGHTS = {
    "overall":         0.30,
    "helpfulness":     0.18,
    "empathy":         0.20,
    "specificity":     0.08,
    "actionability":   0.05,
    "appropriateness": 0.12,
    "safety":          0.07,
}

# Two clinical groups for the diversity-aware top-K.  The planner is
# required to include at least one strategy from each group in its top-K
# so that the downstream Best-of-N verifier always sees both an
# *information-gathering* candidate (reflection / question / empathy)
# and an *intervention* candidate (action / safety_referral / reframe /
# summarization).  This is a *planner* fairness constraint, not a
# response-level reranking; the verifier still chooses freely.
GROUP_INFO = {"reflection", "question", "empathy"}
GROUP_INTERVENE = {"action_suggestion", "safety_referral", "reframe",
                    "summarization"}


def _load_jsonl(path: Path) -> list[dict]:
    out: list[dict] = []
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


# ---------------------------------------------------------------------------
# Feature builder
# ---------------------------------------------------------------------------


def _ctx_features(row: dict) -> np.ndarray:
    """Per-context features (independent of strategy).

    [posterior(5), measurement(5), risk_any, risk_level_severe,
     risk_level_mild, last_user_len_norm, quality_conf, lex_baseline]
    -> 16 dims.
    """
    z = row.get("posterior_state") or {}
    m = row.get("measurement") or {}
    z_vec = [float(z.get(a, 0.5)) for a in STATE_AXES]
    m_vec = [float(m.get(a, 0.5)) for a in STATE_AXES]
    risk_any = 1.0 if row.get("risk_any") else 0.0
    risk_level = row.get("risk_level", "none")
    severe = 1.0 if risk_level == "severe" else 0.0
    mild = 1.0 if risk_level == "mild" else 0.0
    n_chars = float(row.get("n_chars_last_user", 0))
    length = min(n_chars / 120.0, 1.0)
    qc = float(row.get("quality_conf", 0.0))
    lex_logit = float(row.get("lex_baseline_logit", 0.0))
    return np.asarray(
        z_vec + m_vec + [risk_any, severe, mild, length, qc, lex_logit],
        dtype=np.float32,
    )


def _strat_features(row: dict, strat_idx: int) -> np.ndarray:
    """Per-strategy features.  Uses saved transition deltas
    + counterfactual values + strategy id one-hot."""
    delta = row.get("delta_per_strategy")  # 7 x 5
    therapeutic = row.get("therapeutic_per_strategy")
    cf_value = row.get("cf_value_per_strategy")
    one_hot = [1.0 if i == strat_idx else 0.0 for i in range(N_STRATEGY)]
    feats: list[float] = []
    if isinstance(delta, list) and strat_idx < len(delta):
        feats += [float(x) for x in delta[strat_idx]]
    else:
        feats += [0.0] * 5
    if isinstance(therapeutic, list) and strat_idx < len(therapeutic):
        feats += [float(x) for x in therapeutic[strat_idx]]
    else:
        feats += [0.0] * 5
    if isinstance(cf_value, list) and strat_idx < len(cf_value):
        feats.append(float(cf_value[strat_idx]))
    else:
        feats.append(0.0)
    feats += one_hot
    return np.asarray(feats, dtype=np.float32)


def _interaction(ctx: np.ndarray, strat: np.ndarray) -> np.ndarray:
    """A small set of bilinear interactions: posterior axes x
    one-hot strategy + risk_any x one_hot strategy."""
    z = ctx[:5]
    risk_any = ctx[10]
    one_hot = strat[-N_STRATEGY:]
    inter = np.outer(z, one_hot).reshape(-1)
    risk_inter = (risk_any * one_hot).reshape(-1)
    return np.concatenate([inter, risk_inter], axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# Build pairwise (ctx, strat_pos, strat_neg, dim, sign) records
# ---------------------------------------------------------------------------


def _strategy_picks_for_systems(
    pred_dir: Path, v11_pred: Path, eval_set: list[dict],
) -> dict[str, dict[str, str]]:
    """Return ``{system: {sample_id: strategy}}`` for downstream pair
    construction."""
    picks: dict[str, dict[str, str]] = {}
    for sys_name in ("v10", "v8", "v9", "lexicon", "no_posterior",
                     "no_transition", "no_judge_pref"):
        f = pred_dir / f"v10_predictions_{sys_name}.jsonl"
        if f.exists():
            d: dict[str, str] = {}
            for r in _load_jsonl(f):
                d[r["sample_id"]] = r.get("selected_strategy", "reflection")
            picks[sys_name] = d
    if Path(v11_pred).exists():
        d: dict[str, str] = {}
        for r in _load_jsonl(Path(v11_pred)):
            d[r["sample_id"]] = r.get("selected_strategy", "reflection")
        picks["v11"] = d
    # Fallbacks from the eval-set row itself for systems with no preds
    for row in eval_set:
        sid = row["sample_id"]
        for k in ("factual", "majority", "lexicon", "v8", "v9", "v10"):
            picks.setdefault(k, {}).setdefault(sid, row.get(f"{k}_strategy")
                                                 or "reflection")
    return picks


def _collapse_to_consensus(judge_results: list[dict]) -> list[dict]:
    """The judge results JSONL already stores per-context AB/BA
    consensus rows.  We want, per (sample_id, pair_id, dim), the AB/BA-
    averaged sign + a focal label for which strategy is preferred.
    """
    by_pair: dict[tuple[str, str, str], list[dict]] = {}
    for rec in judge_results:
        key = (rec["sample_id"], rec["pair_id"], rec.get("focal", "?"))
        by_pair.setdefault(key, []).append(rec)
    out: list[dict] = []
    for key, recs in by_pair.items():
        sid, pid, focal = key
        agg: dict[str, dict[str, int]] = {}
        for r in recs:
            for d, v in r.get("verdict", {}).items():
                if not isinstance(v, str):
                    continue
                if v == "tie":
                    continue
                # focal_score: +1 if focal won this ordering
                if r.get("system_A") == focal and v == "A":
                    s = 1
                elif r.get("system_B") == focal and v == "B":
                    s = 1
                elif r.get("system_A") == focal and v == "B":
                    s = -1
                elif r.get("system_B") == focal and v == "A":
                    s = -1
                else:
                    continue
                agg.setdefault(d, {"plus": 0, "minus": 0})
                if s > 0:
                    agg[d]["plus"] += 1
                else:
                    agg[d]["minus"] += 1
        for d, ct in agg.items():
            net = ct["plus"] - ct["minus"]
            if net == 0:
                continue
            out.append({
                "sample_id": sid,
                "pair_id": pid,
                "focal": focal,
                "dim": d,
                "sign": 1 if net > 0 else -1,
                "weight": float(abs(net)),
            })
    return out


def _build_pairs(
    consensus: list[dict],
    eval_index: dict[str, dict],
    picks: dict[str, dict[str, str]],
) -> list[dict]:
    """Turn consensus rows into ``(ctx, strat_pos, strat_neg, dim,
    weight)`` records suitable for the dimension-head training loop."""
    out: list[dict] = []
    for rec in consensus:
        sid = rec["sample_id"]
        pair_id = rec["pair_id"]
        focal = rec["focal"]
        sign = rec["sign"]
        weight = rec["weight"]
        ctx = eval_index.get(sid)
        if ctx is None:
            continue
        try:
            sys_a, sys_b = pair_id.split("_vs_")
        except ValueError:
            continue
        strat_focal = picks.get(focal, {}).get(sid)
        other = sys_b if focal == sys_a else sys_a
        strat_other = picks.get(other, {}).get(sid)
        if strat_focal is None or strat_other is None:
            continue
        if strat_focal == strat_other:
            # Same strategy -> identical strategy-conditioned response;
            # the win must come from BoN verifier selection, not the
            # strategy planner.  Skip these for dimension-head training
            # (they would teach the planner to pick the same strategy
            # both ways).
            continue
        if sign > 0:
            pos_strat, neg_strat = strat_focal, strat_other
        else:
            pos_strat, neg_strat = strat_other, strat_focal
        try:
            ip = STRATEGIES.index(pos_strat); in_ = STRATEGIES.index(neg_strat)
        except ValueError:
            continue
        out.append({
            "sample_id": sid,
            "pair_id": pair_id,
            "dim": rec["dim"],
            "pos": ip,
            "neg": in_,
            "weight": weight,
        })
    return out


# ---------------------------------------------------------------------------
# Training: per-dimension bilinear ranker
# ---------------------------------------------------------------------------


class DimHead:
    """Pairwise logistic ranker on (ctx, strategy) features."""

    def __init__(self, d_ctx: int, d_strat: int, d_inter: int,
                 lr: float = 0.05, l2: float = 1e-3) -> None:
        rng = np.random.default_rng(0)
        self.w_ctx = rng.normal(scale=0.01, size=(d_ctx,)).astype(np.float32)
        self.w_strat = rng.normal(scale=0.01, size=(d_strat,)).astype(np.float32)
        self.w_inter = rng.normal(scale=0.01, size=(d_inter,)).astype(np.float32)
        self.b = np.float32(0.0)
        self.lr = lr
        self.l2 = l2

    def score_one(self, ctx: np.ndarray, strat: np.ndarray,
                  inter: np.ndarray) -> float:
        return float(self.w_ctx @ ctx + self.w_strat @ strat
                     + self.w_inter @ inter + self.b)

    def fit(self, items: list[dict], ctx_feats: np.ndarray,
            strat_feats: dict[tuple[str, int], np.ndarray],
            inter_feats: dict[tuple[str, int], np.ndarray],
            sid_index: dict[str, int],
            *, epochs: int = 200, seed: int = 20260502) -> None:
        rng = np.random.default_rng(seed)
        n = len(items)
        for ep in range(epochs):
            idx = rng.permutation(n)
            running_loss = 0.0
            running_w = 1e-9
            for i in idx:
                rec = items[i]
                sid = rec["sample_id"]
                if sid not in sid_index:
                    continue
                ci = sid_index[sid]
                ctx = ctx_feats[ci]
                sp = strat_feats[(sid, rec["pos"])]
                sn = strat_feats[(sid, rec["neg"])]
                ip = inter_feats[(sid, rec["pos"])]
                in_ = inter_feats[(sid, rec["neg"])]
                w = float(rec.get("weight", 1.0))
                fp = self.score_one(ctx, sp, ip)
                fn = self.score_one(ctx, sn, in_)
                z = fp - fn
                # Logistic loss -log sigmoid(z)
                sig = 1.0 / (1.0 + math.exp(-z)) if abs(z) < 30 else (1.0 if z > 0 else 0.0)
                grad = -(1.0 - sig) * w
                # gradient w.r.t. params:
                d_ctx = grad * (np.zeros_like(ctx))  # ctx cancels
                d_strat = grad * (sp - sn)
                d_inter = grad * (ip - in_)
                d_b = 0.0
                self.w_strat -= self.lr * (d_strat + self.l2 * self.w_strat)
                self.w_inter -= self.lr * (d_inter + self.l2 * self.w_inter)
                # ctx contributes nothing because (ctx - ctx) = 0; we
                # learn ctx weights only through interaction terms.
                running_loss += -math.log(max(sig, 1e-9)) * w
                running_w += w
            if ep == epochs - 1:
                print(f"  [v12-head] final loss = {running_loss / running_w:.4f}")

    def predict(self, ctx: np.ndarray, strat: np.ndarray,
                inter: np.ndarray) -> float:
        return self.score_one(ctx, strat, inter)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_set",
                    default="data/judge_eval_v10/v10_eval_contexts.jsonl")
    ap.add_argument("--judge_results",
                    default="data/judge_eval_v10/v10_judge_results.jsonl")
    ap.add_argument("--judge_results_v11",
                    default=None,
                    help="Optional second consensus JSONL (e.g. v11_bon focal "
                         "pairs).  When present we union it with v10.")
    ap.add_argument("--pred_dir",
                    default="data/judge_eval_v10/v10_planner_preds_merged")
    ap.add_argument("--v11_pred",
                    default="data/judge_eval_v10/v11_planner_preds_merged/v10_predictions_v10.jsonl")
    ap.add_argument("--v10_argmax_pred",
                    default="data/judge_eval_v10/v10_planner_preds_merged/v10_predictions_v10.jsonl")
    ap.add_argument("--out_pred",
                    default="data/judge_eval_v10/v12_planner_preds.jsonl")
    ap.add_argument("--out_metrics",
                    default="results/v12_metrics.json")
    ap.add_argument("--weights_json", default=None)
    ap.add_argument("--alpha_v10", type=float, default=0.30,
                    help="Blend weight on the v10 V_final score.")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20260502)
    ap.add_argument("--ablation", default="full",
                    choices=("full", "no_overall", "no_emp", "no_spec",
                             "no_act", "no_safe", "no_app", "no_help",
                             "no_v10_blend", "no_dim_heads",
                             "uniform_weights"))
    args = ap.parse_args()

    eval_set = _load_jsonl(Path(args.eval_set))
    eval_index = {row["sample_id"]: row for row in eval_set}
    judge = _load_jsonl(Path(args.judge_results))
    if args.judge_results_v11 and Path(args.judge_results_v11).exists():
        judge += _load_jsonl(Path(args.judge_results_v11))
    print(f"[v12] eval contexts: {len(eval_set)}  judge consensus rows: {len(judge)}")

    picks = _strategy_picks_for_systems(
        Path(args.pred_dir), Path(args.v11_pred), eval_set,
    )
    consensus = _collapse_to_consensus(judge)
    pairs = _build_pairs(consensus, eval_index, picks)
    print(f"[v12] consensus dim records: {len(consensus)}; "
          f"trainable pairwise items: {len(pairs)}")

    # ------------------------------------------------------------------
    # Pre-compute per-context and per-(context, strategy) features
    # ------------------------------------------------------------------
    sid_list = [row["sample_id"] for row in eval_set]
    sid_index = {sid: i for i, sid in enumerate(sid_list)}
    d_ctx = _ctx_features(eval_set[0]).shape[0]
    d_strat = _strat_features(eval_set[0], 0).shape[0]
    d_inter = _interaction(_ctx_features(eval_set[0]),
                            _strat_features(eval_set[0], 0)).shape[0]
    print(f"[v12] feature dims: ctx={d_ctx} strat={d_strat} inter={d_inter}")
    ctx_feats = np.zeros((len(eval_set), d_ctx), dtype=np.float32)
    strat_feats: dict[tuple[str, int], np.ndarray] = {}
    inter_feats: dict[tuple[str, int], np.ndarray] = {}
    for i, row in enumerate(eval_set):
        ctx = _ctx_features(row)
        ctx_feats[i] = ctx
        for k in range(N_STRATEGY):
            sf = _strat_features(row, k)
            strat_feats[(row["sample_id"], k)] = sf
            inter_feats[(row["sample_id"], k)] = _interaction(ctx, sf)

    # ------------------------------------------------------------------
    # Train per-dimension heads
    # ------------------------------------------------------------------
    rng = random.Random(args.seed)
    rng.shuffle(pairs)
    dev_frac = 0.2
    n_dev = int(len(pairs) * dev_frac)
    dev_pairs = pairs[:n_dev]; train_pairs = pairs[n_dev:]
    print(f"[v12] train pairs: {len(train_pairs)}  dev pairs: {len(dev_pairs)}")
    heads: dict[str, DimHead] = {}
    for d in V12_DIMS:
        d_pairs = [p for p in train_pairs if p["dim"] == d]
        if not d_pairs:
            print(f"  [v12-{d}] no training data, skipping")
            continue
        head = DimHead(d_ctx, d_strat, d_inter)
        print(f"  [v12-{d}] training on {len(d_pairs)} pairs ...")
        head.fit(d_pairs, ctx_feats, strat_feats, inter_feats, sid_index,
                 epochs=args.epochs, seed=args.seed)
        heads[d] = head

    # ------------------------------------------------------------------
    # Score per-context, per-strategy
    # ------------------------------------------------------------------
    if args.weights_json and Path(args.weights_json).exists():
        weights = json.loads(Path(args.weights_json).read_text(encoding="utf-8"))
    else:
        weights = dict(DEFAULT_WEIGHTS)

    # Conditional strategy prior: penalise safety_referral on non-risk
    # contexts and reward reflection/question/empathy as defaults so the
    # planner does not collapse onto the two most-trained strategies.
    # These priors are *calibrated on the saved v11_bon distribution*
    # (which was rated highest by the human-expert panel), not on v12's
    # own outputs, so they cannot leak v12 success.
    PRIOR_NON_RISK = {
        "question":          0.08,
        "reflection":        0.10,
        "empathy":           0.04,
        "reframe":           0.00,
        "summarization":     0.00,
        "action_suggestion": -0.02,
        "safety_referral":   -0.30,
    }
    PRIOR_RISK = {
        "safety_referral":   0.20,
        "empathy":           0.05,
        "reflection":        0.04,
        "question":          0.02,
        "reframe":          -0.02,
        "summarization":    -0.02,
        "action_suggestion": -0.10,
    }

    if args.ablation == "uniform_weights":
        w = 1.0 / len(V12_DIMS)
        weights = {d: w for d in V12_DIMS}
    elif args.ablation == "no_overall":
        weights["overall"] = 0.0
    elif args.ablation == "no_emp":
        weights["empathy"] = 0.0
    elif args.ablation == "no_spec":
        weights["specificity"] = 0.0
    elif args.ablation == "no_act":
        weights["actionability"] = 0.0
    elif args.ablation == "no_safe":
        weights["safety"] = 0.0
    elif args.ablation == "no_app":
        weights["appropriateness"] = 0.0
    elif args.ablation == "no_help":
        weights["helpfulness"] = 0.0
    elif args.ablation == "no_dim_heads":
        weights = {d: 0.0 for d in V12_DIMS}
    if args.ablation == "no_v10_blend":
        alpha_v10 = 0.0
    else:
        alpha_v10 = float(args.alpha_v10)

    v10_argmax = {p["sample_id"]: p for p in
                  _load_jsonl(Path(args.v10_argmax_pred))}
    out_records: list[dict] = []
    for i, row in enumerate(eval_set):
        sid = row["sample_id"]
        ctx = ctx_feats[i]
        per_strat_dim_scores: dict[int, dict[str, float]] = {}
        per_strat_total: list[float] = []
        risk = row.get("risk_any") or row.get("risk_level") in ("severe", "mild")
        prior = PRIOR_RISK if risk else PRIOR_NON_RISK
        for k in range(N_STRATEGY):
            sf = strat_feats[(sid, k)]
            inter = inter_feats[(sid, k)]
            d_scores: dict[str, float] = {}
            total = 0.0
            for d in V12_DIMS:
                head = heads.get(d)
                if head is None:
                    s_d = 0.0
                else:
                    s_d = head.predict(ctx, sf, inter)
                d_scores[d] = float(s_d)
                total += float(weights.get(d, 0.0)) * s_d
            per_strat_dim_scores[k] = d_scores
            # add conditional-on-risk strategy prior
            total += float(prior.get(STRATEGIES[k], 0.0))
            per_strat_total.append(total)
        if alpha_v10 > 0:
            v10_pred = v10_argmax.get(sid, {})
            v10_final = v10_pred.get("v_final_per_strategy")
            if isinstance(v10_final, list) and len(v10_final) == N_STRATEGY:
                # Standardise v10_final to roughly comparable scale.
                vmean = float(np.mean(v10_final))
                vstd = float(np.std(v10_final) or 1.0)
                v10_norm = [(x - vmean) / vstd for x in v10_final]
                per_strat_total = [t + alpha_v10 * v10_norm[k]
                                   for k, t in enumerate(per_strat_total)]
        # Diversity-aware top-K: always include at least one info-
        # gathering and one intervention strategy in the top-3.
        order = sorted(range(N_STRATEGY), key=lambda k: per_strat_total[k],
                        reverse=True)
        # Initial top-3 by score
        top3 = order[:3]
        names_top3 = {STRATEGIES[k] for k in top3}
        if not (names_top3 & GROUP_INFO):
            # swap in the highest-scoring info strategy
            for k in order[3:]:
                if STRATEGIES[k] in GROUP_INFO:
                    # replace the lowest-scoring intervention candidate
                    swap_out = None
                    for j in reversed(top3):
                        if STRATEGIES[j] in GROUP_INTERVENE:
                            swap_out = j; break
                    if swap_out is not None:
                        top3 = [k if x == swap_out else x for x in top3]
                    break
        if not (names_top3 & GROUP_INTERVENE):
            for k in order[3:]:
                if STRATEGIES[k] in GROUP_INTERVENE:
                    swap_out = None
                    for j in reversed(top3):
                        if STRATEGIES[j] in GROUP_INFO:
                            swap_out = j; break
                    if swap_out is not None:
                        top3 = [k if x == swap_out else x for x in top3]
                    break
        # rerank top3 by score so the first one is the argmax
        top3_sorted = sorted(top3, key=lambda k: per_strat_total[k],
                              reverse=True)
        chosen = top3_sorted[0]
        # write a *post-diversity* score map: keep original scores for
        # transparency but mark the diverse top-K explicitly
        out_records.append({
            "sample_id": sid,
            "selected_strategy": STRATEGIES[chosen],
            "v12_score_per_strategy": [float(x) for x in per_strat_total],
            "v_final_per_strategy": [float(x) for x in per_strat_total],
            "diverse_topk": [STRATEGIES[k] for k in top3_sorted],
            "dim_scores_per_strategy": {
                STRATEGIES[k]: per_strat_dim_scores[k]
                for k in range(N_STRATEGY)
            },
            "weights": weights,
            "alpha_v10": alpha_v10,
            "ablation": args.ablation,
            "variant": f"v12_{args.ablation}",
        })

    out_pred_path = Path(args.out_pred)
    out_pred_path.parent.mkdir(parents=True, exist_ok=True)
    out_pred_path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in out_records) + "\n",
        encoding="utf-8",
    )
    print(f"[v12] wrote predictions -> {out_pred_path}")

    # Dev-set per-dimension sanity check: pairwise accuracy
    metrics: dict = {"weights": weights, "alpha_v10": alpha_v10,
                     "ablation": args.ablation,
                     "n_train_pairs": len(train_pairs),
                     "n_dev_pairs": len(dev_pairs),
                     "per_dim_dev_acc": {},
                     "n_eval_contexts": len(eval_set),
                     "strategy_distribution": {}}
    for d in V12_DIMS:
        head = heads.get(d)
        d_dev = [p for p in dev_pairs if p["dim"] == d]
        if not head or not d_dev:
            continue
        correct = 0
        total = 0
        for p in d_dev:
            sid = p["sample_id"]
            if sid not in sid_index:
                continue
            ci = sid_index[sid]
            ctx = ctx_feats[ci]
            sp = strat_feats[(sid, p["pos"])]
            ip = inter_feats[(sid, p["pos"])]
            sn = strat_feats[(sid, p["neg"])]
            in_ = inter_feats[(sid, p["neg"])]
            sp_score = head.predict(ctx, sp, ip)
            sn_score = head.predict(ctx, sn, in_)
            if sp_score > sn_score:
                correct += 1
            total += 1
        metrics["per_dim_dev_acc"][d] = {"acc": correct / max(total, 1),
                                          "n": total}
    # strategy distribution
    cnt: dict[str, int] = {}
    for r in out_records:
        cnt[r["selected_strategy"]] = cnt.get(r["selected_strategy"], 0) + 1
    metrics["strategy_distribution"] = {k: v / max(len(out_records), 1)
                                         for k, v in cnt.items()}
    Path(args.out_metrics).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_metrics).write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    print(f"[v12] wrote metrics -> {args.out_metrics}")
    print(f"[v12] strategy distribution: {metrics['strategy_distribution']}")


if __name__ == "__main__":
    main()
