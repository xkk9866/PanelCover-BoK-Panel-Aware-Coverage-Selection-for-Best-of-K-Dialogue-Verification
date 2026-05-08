"""Panel-Aware State-Conditioned Coverage Best-of-K planner (PA-SCT).

Under the canonical *single-rater* BT-overall verifier, the optimal Best-
of-K planner is *trivially* top-K-by-verifier (BT-Greedy): the verifier
returns the global argmax over the candidate set, so any K-set that
contains the global argmax materialises the same final response.  Real
expert panels, however, *disagree*: on our dataset, three independent
qualified counsellor experts (E1 supervisor, E2 client experience,
E3 safety reviewer) have pairwise Spearman rank correlations on the
overall BT logit of only 0.74-0.86, and they pick a *different* top-1
candidate on 43.4% of randomly drawn K=7 candidate sets.

We therefore propose a **panel-aware stochastic verifier** which, at
deployment time, samples one expert ``e`` uniformly from the panel and
returns the argmax over the K-set under that expert's BT logit.  The
expected utility of a K-set ``S`` for context ``x`` becomes

  F(S | x) = (1/|E|) * sum_{e in E} max_{a in S} BT_e^overall(r(x, a)),

which is monotone submodular in ``S`` (each inner ``max`` is a coverage
function).  Greedy maximisation gives a (1 - 1/e) ~ 63 % approximation
guarantee.

PA-SCT therefore:

  (1) reads per-expert response-text BT logits from
      ``results/panel_bt.json`` (built by ``eval.panel_bt``);
  (2) optionally adds a *state-conditioned dimension lift* on top of the
      overall logit, so that high-distress contexts up-weight empathy,
      severe-risk contexts up-weight safety, etc., across all experts;
  (3) optionally honours **clinical coverage constraints** (severe risk
      forces ``safety_referral``; high distress / low alliance forces an
      empathy / reflection candidate; very low clarity forces a
      question / summarisation candidate); and
  (4) runs greedy submodular maximisation of ``F(S | x)`` over the up
      to seven cached strategies, with the panel-state-conditioned
      lift baked into each expert's score.

The materialisation step (final response per system_id per sample_id)
is shared across systems: PA-SCT only writes the K-set + per-strategy
panel diagnostics to ``data/fair_bon_v12/psystate_pasct_topk.jsonl``;
the shared verifier in ``eval.v12_best_of_n_fair`` (with
``--verifier panel-stochastic``) materialises one response per system
under the panel-stochastic verifier so the comparison is fully fair.
"""
from __future__ import annotations

import argparse
import itertools
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
EXPERT_IDS = ("E1_supervisor", "E2_client_experience", "E3_safety_reviewer")


def _norm(s: str | None) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def _load_panel_bt(path: Path) -> dict[str, dict[str, dict[str, float]]]:
    """Return ``{expert_id: {dim: {text: bt_logit}}}``."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, dict[str, dict[str, float]]] = {}
    for e in EXPERT_IDS:
        out[e] = {}
        for d in DIMS:
            inner = raw.get(e, {}).get(d, {})
            out[e][d] = {k: float(v) for k, v in inner.items()}
    return out


def _state(ctx: dict, key: str, default: float = 0.5) -> float:
    return float((ctx.get("posterior_state") or {}).get(key, default))


def _risk_level(ctx: dict) -> str:
    return str(ctx.get("risk_level") or "none")


def _is_risk(ctx: dict) -> bool:
    return bool(ctx.get("risk_any") or
                _risk_level(ctx) in {"mild", "severe", "imminent"})


def _state_dim_lift(ctx: dict, *, state_lift: bool) -> np.ndarray:
    """Return a per-dimension multiplicative lift mass to add to the
    pure ``overall`` logit.  When ``state_lift=False`` the lift is zero
    for every dim except ``overall`` (i.e. plain overall scoring)."""
    if not state_lift:
        w = np.zeros(len(DIMS), dtype=np.float64)
        w[DIMS.index("overall")] = 1.0
        return w

    distress = _state(ctx, "distress")
    readiness = _state(ctx, "readiness")
    alliance = _state(ctx, "alliance")
    clarity = _state(ctx, "clarity")
    risk = _is_risk(ctx)
    severe = _risk_level(ctx) in {"severe", "imminent"}

    base = np.zeros(len(DIMS), dtype=np.float64)
    base[DIMS.index("overall")] = 1.0
    base[DIMS.index("empathy")] += 0.50 * max(distress - 0.5, 0.0)
    base[DIMS.index("safety")] += 0.80 if severe else (0.40 if risk else 0.0)
    base[DIMS.index("specificity")] += 0.40 * max(0.65 - clarity, 0.0)
    base[DIMS.index("actionability")] += 0.40 * max(readiness - 0.55, 0.0)
    # Non-crisis actionability lift: if the context is not risk-bearing and
    # the user has at least baseline clarity, prefer concrete actionable
    # candidates.  This counteracts the structural under-selection of
    # action_suggestion observed under the panel-stochastic verifier on
    # non-crisis ESC contexts.
    if (not risk) and clarity >= 0.45:
        base[DIMS.index("actionability")] += 0.35
        base[DIMS.index("specificity")] += 0.20
    base[DIMS.index("appropriateness")] += 0.30 * max(0.55 - alliance, 0.0)
    return base


def _expert_score(text: str, expert_bt_per_dim: dict[str, dict[str, float]],
                  *, dim_lift: np.ndarray) -> float:
    """Score ``text`` for one expert as a state-conditioned linear
    combination of per-dimension BT logits.  OOD texts (not seen by the
    panel) return 0.0 across all dims, i.e. neutral."""
    s = 0.0
    for j, d in enumerate(DIMS):
        s += float(dim_lift[j]) * float(expert_bt_per_dim.get(d, {}).get(text, 0.0))
    return s


def _candidate_table(sid: str,
                     resp_idx: dict[tuple[str, str], dict],
                     panel_bt: dict[str, dict[str, dict[str, float]]],
                     *, dim_lift: np.ndarray,
                     multidim_scenarios: bool = False,
                     ) -> tuple[dict[str, str], dict[str, np.ndarray]]:
    """For the K-set planner, build per-strategy scenario score vectors.

    Two modes:
      * standard panel-aware mode (3 scenarios; one per expert), where
        each expert's score is a state-conditioned linear combination of
        per-dim BT logits (default).
      * multi-dim scenarios (3 * 7 = 21 scenarios; one per
        ``(expert, dim)`` pair, weighted by ``dim_lift`` to keep
        clinically less-relevant dimensions from dominating coverage).
    """
    text_table: dict[str, str] = {}
    expert_scores: dict[str, np.ndarray] = {}
    if multidim_scenarios:
        weights = np.asarray(dim_lift, dtype=np.float64)
        weights = weights / max(weights.sum(), 1e-9)
        for strat in STRATEGIES:
            rec = resp_idx.get((sid, strat))
            if rec is None:
                continue
            t = _norm(rec.get("response"))
            if not t:
                continue
            row = []
            for e in EXPERT_IDS:
                for j, d in enumerate(DIMS):
                    bt_text = float(panel_bt[e][d].get(t, 0.0))
                    row.append(bt_text * float(weights[j]) * len(DIMS))
            text_table[strat] = t
            expert_scores[strat] = np.array(row, dtype=np.float64)
        return text_table, expert_scores

    for strat in STRATEGIES:
        rec = resp_idx.get((sid, strat))
        if rec is None:
            continue
        t = _norm(rec.get("response"))
        if not t:
            continue
        scores = np.array([
            _expert_score(t, panel_bt[e], dim_lift=dim_lift)
            for e in EXPERT_IDS
        ], dtype=np.float64)
        text_table[strat] = t
        expert_scores[strat] = scores
    return text_table, expert_scores


def _required_strategies(ctx: dict, profiles: dict[str, np.ndarray]) -> list[str]:
    """Clinical coverage constraints on the K-set."""
    req: list[str] = []
    severe = _risk_level(ctx) in {"severe", "imminent"}
    distress = _state(ctx, "distress")
    alliance = _state(ctx, "alliance")
    clarity = _state(ctx, "clarity")

    def _best(group: set[str]) -> str | None:
        avail = [s for s in group if s in profiles]
        if not avail:
            return None
        return max(avail, key=lambda s: float(profiles[s].mean()))

    if severe and "safety_referral" in profiles:
        req.append("safety_referral")
    if distress >= 0.65 or alliance <= 0.35:
        s = _best({"empathy", "reflection"})
        if s and s not in req:
            req.append(s)
    if clarity <= 0.35:
        s = _best({"question", "summarization"})
        if s and s not in req:
            req.append(s)
    return req


def _greedy_panel_aware(profiles: dict[str, np.ndarray],
                        K: int,
                        *, required: list[str]) -> list[str]:
    """Greedy maximisation of F(S) = (1/|E|) sum_e max_{a in S} score_e(a)."""
    selected: list[str] = []
    n_experts = next(iter(profiles.values())).shape[0] if profiles else 1
    current = np.full(n_experts, -np.inf, dtype=np.float64)

    def _F(curr: np.ndarray) -> float:
        finite = np.where(np.isfinite(curr), curr, 0.0)
        return float(finite.mean())

    for strat in required:
        if strat in profiles and strat not in selected and len(selected) < K:
            selected.append(strat)
            current = np.maximum(current, profiles[strat])

    while len(selected) < K:
        best_s = None
        best_gain = -np.inf
        prev_F = _F(current)
        for strat, prof in profiles.items():
            if strat in selected:
                continue
            new_curr = np.maximum(current, prof)
            gain = _F(new_curr) - prev_F
            if gain > best_gain:
                best_gain = gain
                best_s = strat
        if best_s is None:
            break
        selected.append(best_s)
        current = np.maximum(current, profiles[best_s])
    return selected


def _expand_virtual_experts(
    profiles: dict[str, np.ndarray],
    *,
    n_interp: int = 3,
) -> dict[str, np.ndarray]:
    """Expand the observed expert set with convex-interpolated virtual experts.

    Given ``|E|`` observed experts, we create ``|E| * (|E|-1) / 2 * n_interp``
    additional virtual expert score vectors by linearly interpolating between
    each pair at ``n_interp`` equally-spaced mixture weights.  The expanded
    profile matrix covers a finer discretisation of the convex hull of
    observed expert preferences, so planning under the expanded set is a
    tighter approximation to optimising over the *distribution* of counsellor
    types rather than just the three observed roles.

    This is the practical implementation of the Bayesian expert distribution
    idea without requiring a full hierarchical model: the interpolated experts
    represent the ``plausible counsellor`` manifold spanned by E1, E2, E3.
    """
    if not profiles:
        return profiles
    strategies = list(profiles)
    n_e = next(iter(profiles.values())).shape[0]  # original expert count
    if n_e < 2:
        return profiles

    expanded: dict[str, np.ndarray] = {s: profiles[s].copy() for s in strategies}
    e_indices = list(range(n_e))
    lambdas = [k / (n_interp + 1) for k in range(1, n_interp + 1)]

    for i, j in itertools.combinations(e_indices, 2):
        for lam in lambdas:
            # virtual expert = λ * e_i + (1-λ) * e_j, appended as new column
            for s in strategies:
                orig = profiles[s]
                if s not in expanded:
                    expanded[s] = orig.copy()
                virtual_score = lam * orig[i] + (1.0 - lam) * orig[j]
                expanded[s] = np.append(expanded[s], virtual_score)
    return expanded


def _cvar_score(covered: np.ndarray, alpha: float, tau: float) -> float:
    """Compute the CVaR-augmented panel utility.

    The objective is:

      F_CVaR(S) = (1 - alpha) * E_e[max_a u_e,a]
                + alpha * CVaR_tau(max_a u_e,a)

    where CVaR_tau is the expected value of the bottom-tau fraction of
    per-expert covered utilities.  For alpha=0 this reduces to the plain
    mean; for tau=1/|E| and alpha=1 it reduces to worst-expert (min).

    With |E|=3 and tau=1/3, CVaR_tau = min_e, so the plain mean_min
    objective is a special case.  With tau=0.5, CVaR covers the bottom
    50% of experts, giving a softer robustness criterion.
    """
    n = len(covered)
    mean = float(np.mean(covered))
    if alpha < 1e-9:
        return mean
    sorted_cov = np.sort(covered)
    k = max(1, int(math.ceil(tau * n)))
    cvar = float(np.mean(sorted_cov[:k]))
    return (1.0 - alpha) * mean + alpha * cvar


def _panel_conflict(profiles: dict[str, np.ndarray],
                    anchor: str | None) -> dict[str, float]:
    """Measure how badly the pooled-BT top-1 conflicts with the panel.

    ``anchor`` is the pooled-BT top-1 strategy (i.e. BT-Greedy's materialised
    response under the canonical verifier).  For each expert scenario we
    compute the utility loss from forcing that anchor instead of the expert's
    own favourite candidate:

        regret_e = max_a u_e(a) - u_e(anchor).

    Large positive values identify exactly the contexts where BT-Greedy is
    vulnerable: at least one expert has a much better candidate than the
    pooled top-1.  These diagnostics drive both conditional anchoring and
    disagreement-gated planning.
    """
    if not profiles or anchor is None or anchor not in profiles:
        return {"max": 0.0, "mean": 0.0, "var": 0.0}
    mat = np.stack(list(profiles.values()), axis=0)
    expert_best = mat.max(axis=0)
    regret = expert_best - profiles[anchor]
    return {
        "max": float(np.max(regret)),
        "mean": float(np.mean(np.maximum(regret, 0.0))),
        "var": float(np.var([int(np.argmax(mat[:, j])) for j in range(mat.shape[1])])),
    }


def _exact_panel_aware(
    profiles: dict[str, np.ndarray],
    K: int,
    *,
    required: list[str],
    objective: str = "mean",
    robust_alpha: float = 0.0,
    cvar_tau: float = 0.33,
    expand_virtual: bool = False,
    n_virtual_interp: int = 3,
    regret_baseline: np.ndarray | None = None,
    overreferral_penalty: float = 0.0,
    risk_high: bool = False,
) -> list[str]:
    """Exact subset search for the small PsyState strategy space.

    The original PA-SCT uses greedy submodular maximisation because that
    generalises to larger candidate pools.  In this repository the action
    space is only seven counselling strategies, so we can remove the
    approximation gap entirely and evaluate the best feasible K-set.

    Objectives
    ----------
    * ``mean``: exact panel-stochastic utility.
    * ``mean_min``: adds a worst-expert robustness term (equivalent to
      CVaR at tau = 1/|E|).
    * ``cvar``: CVaR at ``cvar_tau`` fraction of the expert distribution,
      with weight ``robust_alpha`` on the tail term.  This is a strictly
      more general DRO objective than mean-min.
    * ``regret``: directly optimise improvement over BT-Greedy's pooled
      top-1, i.e. ``max_a u_e(a) - u_e(a_BT)``.  This targets the failure
      mode where ordinary coverage selects responses that experts like but
      that are not meaningfully better than BT-Greedy.

    Virtual expert expansion
    ------------------------
    When ``expand_virtual=True``, the profile vectors are first expanded
    by ``_expand_virtual_experts`` which adds linearly interpolated virtual
    counsellors between each observed expert pair.  Planning under the
    expanded panel is distributionally robust to unseen experts on the
    convex hull of the observed panel.
    """
    if not profiles:
        return []

    if expand_virtual:
        profiles = _expand_virtual_experts(profiles, n_interp=n_virtual_interp)

    req = [s for s in required if s in profiles]
    if len(req) >= K:
        return req[:K]
    pool = [s for s in profiles if s not in req]
    need = K - len(req)
    best_combo: tuple[str, ...] | None = None
    best_score = -np.inf

    def _score(combo: tuple[str, ...]) -> float:
        selected = list(req) + list(combo)
        mat = np.stack([profiles[s] for s in selected], axis=0)
        covered = mat.max(axis=0)
        if objective == "mean_min":
            mean = float(np.mean(covered))
            score = (1.0 - robust_alpha) * mean + robust_alpha * float(np.min(covered))
        elif objective == "cvar":
            score = _cvar_score(covered, alpha=robust_alpha, tau=cvar_tau)
        elif objective == "regret":
            baseline = (np.zeros_like(covered) if regret_baseline is None
                        else np.asarray(regret_baseline, dtype=np.float64))
            regret = covered - baseline
            score = _cvar_score(regret, alpha=robust_alpha, tau=cvar_tau)
        else:
            score = float(np.mean(covered))
        if overreferral_penalty > 0.0 and (not risk_high) and "safety_referral" in selected:
            score -= overreferral_penalty
        return score

    for combo in itertools.combinations(pool, min(need, len(pool))):
        score = _score(combo)
        if score > best_score + 1e-12:
            best_score = score
            best_combo = combo
    if best_combo is None:
        return req + pool[:need]
    return req + list(best_combo)


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


def _fit_pooled_overall_bt() -> dict[str, float]:
    """Canonical pooled BT-overall used by BT-Greedy / fair BoN.

    PA-SCT-DRO can anchor the K-set with this scorer's top-1 response.
    This preserves the verifier-aligned response chosen by BT-Greedy under
    the canonical single-verifier protocol while using the remaining K-1
    slots for panel/disagreement coverage.
    """
    from eval.bt_winrate_proxy import (
        _build_response_text_index, _collect_dim_games, _fit_bt,
    )
    return _fit_bt(
        _collect_dim_games(_build_response_text_index()).get("overall", []))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_set",
                    default="data/judge_eval_v10/v10_eval_contexts.jsonl")
    ap.add_argument("--responses",
                    default="data/judge_eval_v10/v10_responses.jsonl")
    ap.add_argument("--safety_overrides",
                    default="data/judge_eval_v10/v12_safety_overrides.jsonl")
    ap.add_argument("--panel_bt", default="results/panel_bt.json")
    ap.add_argument("--out_topk",
                    default="data/fair_bon_v12/psystate_pasct_topk.jsonl")
    ap.add_argument("--system_id", default="psystate_pasct")
    ap.add_argument("--K", type=int, default=3)
    ap.add_argument("--no_state_lift", action="store_true",
                    help="Ablation: drop state-conditioned dim lifts; "
                         "score each expert by overall BT logit only.")
    ap.add_argument("--no_clinical_constraints", action="store_true",
                    help="Ablation: drop clinical coverage constraints "
                         "(severe-risk safety floor, distress empathy, "
                         "low-clarity question floors).")
    ap.add_argument("--collapse_to_mean", action="store_true",
                    help="Ablation: replace the panel-aware submodular "
                         "objective with the BT-Greedy-by-mean objective "
                         "F(S) = sum_{a in topK_by_mean} 1[a in S].  This "
                         "is mathematically identical to BT-Greedy and "
                         "should leave PA-SCT no algorithmic edge.")
    ap.add_argument("--multidim_scenarios", action="store_true",
                    help="Use 3*7=21 (expert, dim) scenarios for the "
                         "panel-aware coverage objective; each dim is "
                         "weighted by the state-conditioned dim lift.  "
                         "Increases the algorithmic gap of submodular "
                         "coverage over plain top-K-by-mean by spreading "
                         "the K=3 budget across more axes of disagreement.")
    ap.add_argument("--exact_coverage", action="store_true",
                    help="PA-SCT++: exactly enumerate feasible K-sets in "
                         "the seven-strategy action space instead of using "
                         "greedy approximation.")
    ap.add_argument("--coverage_objective",
                    choices=("mean", "mean_min", "cvar", "regret"), default="mean",
                    help="Exact-coverage objective. mean is the exact "
                         "panel-stochastic verifier utility; mean_min adds "
                         "a worst-expert robustness term; cvar uses CVaR "
                         "at --cvar_tau with weight --robust_alpha; regret "
                         "optimises improvement over BT-Greedy top-1.")
    ap.add_argument("--robust_alpha", type=float, default=0.15,
                    help="Weight of the DRO term (worst-expert or CVaR) "
                         "in the objective.")
    ap.add_argument("--cvar_tau", type=float, default=0.33,
                    help="CVaR tail fraction (proportion of worst experts "
                         "to average over). Only used when "
                         "--coverage_objective=cvar.")
    ap.add_argument("--expand_virtual", action="store_true",
                    help="PA-SCT-CVaR: expand the observed expert set with "
                         "convex-interpolated virtual experts before "
                         "planning, making the planner robust to unseen "
                         "experts on the convex hull of the panel.")
    ap.add_argument("--n_virtual_interp", type=int, default=3,
                    help="Number of interpolation points per expert pair "
                         "when --expand_virtual is active.")
    ap.add_argument("--anchor_pooled_top1", action="store_true",
                    help="PA-SCT-DRO: force the canonical pooled-BT top-1 "
                         "candidate into the K-set, then optimise panel "
                         "coverage with the remaining slots. This directly "
                         "targets the BT-Greedy failure mode.")
    ap.add_argument("--conditional_anchor", action="store_true",
                    help="Only keep the pooled-BT top-1 anchor when the "
                         "panel conflict score is below "
                         "--anchor_conflict_eps. This preserves BT-Greedy "
                         "on easy contexts but lets PA-SCT-DRO diverge on "
                         "high-disagreement contexts.")
    ap.add_argument("--anchor_conflict_eps", type=float, default=0.20,
                    help="Conflict threshold for --conditional_anchor. "
                         "Conflict is max_e(max_a u_e(a)-u_e(a_pool)).")
    ap.add_argument("--disagreement_gate", action="store_true",
                    help="Use BT-Greedy's pooled top-K on low-disagreement "
                         "contexts and PA-SCT-DRO only when panel conflict "
                         "exceeds --disagreement_tau.")
    ap.add_argument("--disagreement_tau", type=float, default=0.10,
                    help="Panel-conflict threshold for --disagreement_gate.")
    ap.add_argument("--overreferral_penalty", type=float, default=0.0,
                    help="Subtract this score when a low-risk context selects "
                         "safety_referral. This reduces safety-referral "
                         "strategy bias without weakening hard safety "
                         "overrides.")
    ap.add_argument("--exclude_safety_low_risk", action="store_true",
                    help="Hard-exclude ``safety_referral`` from the K-set "
                         "candidate pool whenever the context has "
                         "``risk_level == none`` and is not in the safety-"
                         "overrides set. This is the principled fix for "
                         "the over-triggering of safety_referral observed in "
                         "the panel-aware planner: safety_referral is a "
                         "clinically high-cost action and should only be in "
                         "the K-set when there is a clinical risk indicator.")
    args = ap.parse_args()

    eval_rows = _load_jsonl(Path(args.eval_set))
    overrides = {
        str(r.get("sample_id")): r
        for r in _load_jsonl(Path(args.safety_overrides))
        if r.get("shield_fired") or r.get("shield_fired_hard")
        or r.get("decision") == "hard"
    }
    resp_idx = _response_index([Path(args.responses)])
    panel_bt = _load_panel_bt(Path(args.panel_bt))
    pooled_bt = _fit_pooled_overall_bt() if args.anchor_pooled_top1 else None
    print(f"[pasct] panel BT items per expert: " +
          ", ".join(f"{e}={len(panel_bt[e]['overall'])}" for e in EXPERT_IDS))

    Path(args.out_topk).parent.mkdir(parents=True, exist_ok=True)
    n_total = 0
    n_safety = 0
    n_missing = 0
    n_panel_evidence = 0  # contexts with >=2 BT-rated candidates
    n_anchor_dropped = 0
    n_gate_btgreedy = 0

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

            dim_lift = _state_dim_lift(ctx, state_lift=not args.no_state_lift)
            text_table, expert_scores = _candidate_table(
                sid, resp_idx, panel_bt, dim_lift=dim_lift,
                multidim_scenarios=bool(args.multidim_scenarios))
            if not expert_scores:
                n_missing += 1
                continue

            # Principled safety_referral pool filter.  safety_referral is a
            # clinically high-cost action; allowing it into the K-set on
            # contexts with no risk indicators causes the panel-aware planner
            # to over-trigger when the cached safety response happens to have
            # high panel BT logits.  When --exclude_safety_low_risk is set
            # and the context is low-risk (risk_level=none, not in the safety
            # override set), we drop safety_referral from the candidate pool.
            if args.exclude_safety_low_risk:
                low_risk = (_risk_level(ctx) == "none"
                            and not _is_risk(ctx)
                            and sid not in overrides)
                if low_risk and "safety_referral" in expert_scores:
                    expert_scores.pop("safety_referral", None)
                    text_table.pop("safety_referral", None)

            # Quick diagnostics: how many candidates are panel-rated
            n_rated = sum(int(np.linalg.norm(s) > 1e-6)
                          for s in expert_scores.values())
            if n_rated >= 2:
                n_panel_evidence += 1

            panel_conflict: dict[str, float] | None = None
            if args.collapse_to_mean:
                # Ablation: just rank by mean expert score and take top-K.
                ranked = sorted(expert_scores.keys(),
                                key=lambda s: -float(expert_scores[s].mean()))
                topk = ranked[: args.K]
                required: list[str] = []
            else:
                required = ([] if args.no_clinical_constraints
                            else _required_strategies(ctx, expert_scores))
                anchor = None
                conflict = {"max": 0.0, "mean": 0.0, "var": 0.0}
                pooled_ranked: list[str] = []
                if pooled_bt is not None:
                    def _pooled_score(strat: str) -> float:
                        text = text_table.get(strat, "")
                        return float(pooled_bt.get(text, 0.0))
                    pooled_ranked = sorted(
                        expert_scores.keys(), key=lambda s: -_pooled_score(s))
                    anchor = pooled_ranked[0] if pooled_ranked else None
                    conflict = _panel_conflict(expert_scores, anchor)
                    panel_conflict = conflict

                    if (args.disagreement_gate and
                            conflict["max"] < args.disagreement_tau):
                        topk = pooled_ranked[: args.K]
                        required = []
                        n_gate_btgreedy += 1
                        ftop.write(json.dumps({
                            "sample_id": sid,
                            "planner": args.system_id,
                            "candidate_strategies": topk,
                            "required_strategies": required,
                            "panel_state_lift": dim_lift.tolist(),
                            "expert_scores": {
                                k: v.tolist() for k, v in expert_scores.items()
                            },
                            "panel_conflict": conflict,
                            "decision": "btgreedy_gate",
                            "K": args.K,
                        }, ensure_ascii=False) + "\n")
                        continue

                    keep_anchor = bool(args.anchor_pooled_top1)
                    if args.conditional_anchor and conflict["max"] >= args.anchor_conflict_eps:
                        keep_anchor = False
                        n_anchor_dropped += 1

                    if keep_anchor and anchor is not None and anchor not in required:
                        required = [anchor] + required
                if args.exact_coverage:
                    regret_baseline = (
                        expert_scores.get(anchor) if anchor is not None else None)
                    topk = _exact_panel_aware(
                        expert_scores, args.K, required=required,
                        objective=args.coverage_objective,
                        robust_alpha=args.robust_alpha,
                        cvar_tau=args.cvar_tau,
                        expand_virtual=args.expand_virtual,
                        n_virtual_interp=args.n_virtual_interp,
                        regret_baseline=regret_baseline,
                        overreferral_penalty=args.overreferral_penalty,
                        risk_high=_risk_level(ctx) in {"severe", "imminent"})
                else:
                    topk = _greedy_panel_aware(
                        expert_scores, args.K, required=required)
            if not topk:
                n_missing += 1
                continue

            ftop.write(json.dumps({
                "sample_id": sid,
                "planner": args.system_id,
                "candidate_strategies": topk,
                "required_strategies": required,
                "panel_state_lift": dim_lift.tolist(),
                "expert_scores": {
                    k: v.tolist() for k, v in expert_scores.items()
                },
                "panel_conflict": panel_conflict,
                "K": args.K,
            }, ensure_ascii=False) + "\n")

    print(f"[pasct] total={n_total} safety_hard={n_safety} "
          f"missing={n_missing} panel_evidence>=2: {n_panel_evidence}")
    print(f"[pasct] gate_btgreedy={n_gate_btgreedy} "
          f"anchor_dropped={n_anchor_dropped}")
    print(f"[pasct] wrote {args.out_topk}")


if __name__ == "__main__":
    main()
