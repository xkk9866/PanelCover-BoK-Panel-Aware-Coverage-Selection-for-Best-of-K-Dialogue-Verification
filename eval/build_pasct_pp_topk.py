"""PA-SCT-DRO++ : panel-aware K-set planner with **unified candidate pool**.

This is the main SOTA planner of the paper.  It combines two ideas:

(1) The original PA-SCT-DRO+ plan over the 7-strategy cached generator
    with mean-min / CVaR-DRO panel-coverage and clinical constraints.

(2) A unified candidate pool that adds retrieval-grounded candidates
    from KEMI, RAG, and any other modern baseline whose responses are
    available as cached BoN files (e.g. LLM-zero-shot, ESCoT-style CoT).
    Each retrieval-augmented candidate is treated as a pseudo-strategy
    whose per-expert BT logits are obtained from
    :class:`~eval.panel_bt_extender.PanelBTExtender` --- the trained
    text-input panel-BT regression that generalises the rater-fitted
    BT to arbitrary new text.

The K-set planner enumerates :math:`\binom{|A|+|R|}{K}` feasible sets,
where :math:`A=7` is the strategy alphabet and :math:`R\ge 0` is the
number of retrieval-augmented candidates included; the same mean-min
DRO + clinical constraints + non-crisis safety-referral exclusion is
applied as in PA-SCT-DRO+.  Materialisation is done in-place: instead
of looking up ``(sample_id, strategy) -> response`` in the shared
cache, the unified pool already carries the response text for each
candidate and the panel-state deployment verifier picks one of K=3.

Usage::

    python -m eval.build_pasct_pp_topk \
        --retrieval_systems kemi rag \
        --robust_alpha 0.50 \
        --out_topk data/fair_bon_v12/pasct_pp_topk.jsonl \
        --out_bon  data/fair_bon_v12_panel_state/pasct_pp_bon_responses.jsonl
"""
from __future__ import annotations

import argparse
import itertools
import json
import re
from pathlib import Path

import numpy as np

from eval.panel_bt_extender import PanelBTExtender


# ---------------------------------------------------------------------------
# Constants reused from build_pasct_topk.py
# ---------------------------------------------------------------------------

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
STRATEGIES = (
    "question",
    "reflection",
    "empathy",
    "reframe",
    "summarization",
    "action_suggestion",
    "safety_referral",
)


SAFETY_TEMPLATE = (
    "听到你说这些我有些担心你的安全。如果你正经历强烈的伤害自己的念头或处于"
    "情绪非常艰难的时刻，请考虑联系全国 24 小时心理援助热线 400-161-9995，"
    "或告诉一个你信任的人，让 ta 此刻陪着你。"
    "你愿意先告诉我一些此刻你身边能联系到的支持吗？"
)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _norm(s: str | None) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


def _load_panel_bt(path: Path):
    raw = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, dict[str, dict[str, float]]] = {}
    for e in EXPERT_IDS:
        out[e] = {}
        for d in DIMS:
            out[e][d] = {k: float(v) for k, v in raw.get(e, {}).get(d, {}).items()}
    return out


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


def _kp_response_index(path: Path) -> dict[tuple[str, str], str]:
    """Map ``(sample_id, strategy) -> response`` for the
    knowledge-augmented Qwen-max response pool."""
    out: dict[tuple[str, str], str] = {}
    for rec in _load_jsonl(path):
        sid = str(rec.get("sample_id", ""))
        strat = str(rec.get("selected_strategy", ""))
        text = _norm(rec.get("response"))
        if sid and strat and text:
            out[(sid, strat)] = text
    return out


def _retrieval_index(system: str, root: Path) -> dict[str, str]:
    """Map sample_id -> response text for a given retrieval baseline."""
    candidates = [
        root / f"{system}_bon_responses.jsonl",
        root.parent / "fair_bon_v12_panel_state" / f"{system}_bon_responses.jsonl",
        root.parent / "fair_bon_v12_panel_mean" / f"{system}_bon_responses.jsonl",
    ]
    for path in candidates:
        if path.exists():
            out: dict[str, str] = {}
            for rec in _load_jsonl(path):
                sid = str(rec.get("sample_id", ""))
                t = _norm(rec.get("response"))
                if sid and t:
                    out[sid] = t
            return out
    raise FileNotFoundError(f"no retrieval response file for {system}")


# ---------------------------------------------------------------------------
# State-conditioned dim weights (mirrors build_pasct_topk._state_dim_lift /
# _state_weights but kept self-contained for clarity)
# ---------------------------------------------------------------------------

def _state(ctx: dict, key: str, default: float = 0.5) -> float:
    return float((ctx.get("posterior_state") or {}).get(key, default))


def _risk_level(ctx: dict) -> str:
    return str(ctx.get("risk_level") or "none")


def _is_risk(ctx: dict) -> bool:
    return bool(ctx.get("risk_any") or _risk_level(ctx) in {"mild", "severe", "imminent"})


def _state_weights(ctx: dict) -> np.ndarray:
    """State-conditioned weighting over the 7 BT dimensions.  Mirrors
    PA-SCT-DRO+ so the verifier and the planner are aligned."""
    w = np.array([0.4, 0.20, 0.10, 0.10, 0.10, 0.05, 0.05], dtype=np.float64)
    distress = _state(ctx, "distress")
    risk_any = _is_risk(ctx)
    severe = _risk_level(ctx) in {"severe", "imminent"}
    clarity = _state(ctx, "clarity")
    readiness = _state(ctx, "readiness")
    if severe:
        w[6] += 0.25
        w[5] += 0.10
    if risk_any:
        w[6] += 0.10
    if distress >= 0.6:
        w[2] += 0.10
        w[5] += 0.05
    if clarity <= 0.45:
        w[3] += 0.05
        w[1] += 0.05
    if readiness >= 0.55 and not risk_any:
        w[4] += 0.10
    if (not risk_any) and clarity >= 0.45:
        w[4] += 0.08
        w[3] += 0.05
    return w / max(w.sum(), 1e-9)


def _expert_score(text: str, panel_bt_e: dict[str, dict[str, float]],
                  panel_ext: PanelBTExtender, expert: str,
                  weights: np.ndarray) -> float:
    """Compute the state-weighted per-expert score for a single response.

    Looks up the exact per-dim BT logit when the response text is in the
    rater corpus; otherwise calls the trained text-input extender.
    """
    score = 0.0
    use_extender = text not in panel_bt_e["overall"]
    ext_scores = panel_ext.score(text)[expert] if use_extender else None
    for j, dim in enumerate(DIMS):
        if use_extender:
            v = float(ext_scores[dim])
        else:
            v = float(panel_bt_e[dim].get(text, 0.0))
        score += float(weights[j]) * v
    return score


# ---------------------------------------------------------------------------
# Unified candidate pool
# ---------------------------------------------------------------------------

def _build_candidate_pool(ctx: dict,
                          resp_idx: dict[tuple[str, str], dict],
                          retrieval: dict[str, dict[str, str]],
                          kp_idx: dict[tuple[str, str], str],
                          panel_bt: dict[str, dict[str, dict[str, float]]],
                          panel_ext: PanelBTExtender,
                          weights: np.ndarray,
                          ) -> tuple[dict[str, str], dict[str, np.ndarray],
                                     dict[str, str]]:
    """Return (text_table, expert_scores, role_table).

    role_table maps the unified pool key to one of:
      ``"strategy"``  for the 7 cached strategy candidates;
      ``"retrieval"`` for KEMI / RAG / other retrieval-augmented candidates;
      ``"kp"``        for the knowledge-personalised Qwen-max candidates,
                      which are strategy-conditioned and seeded with
                      CBT-style clinical knowledge prompts.
    """
    sid = str(ctx["sample_id"])
    text_table: dict[str, str] = {}
    expert_scores: dict[str, np.ndarray] = {}
    role_table: dict[str, str] = {}

    seen_texts: set[str] = set()

    for strat in STRATEGIES:
        rec = resp_idx.get((sid, strat))
        if rec is None:
            continue
        t = _norm(rec.get("response"))
        if not t or t in seen_texts:
            continue
        scores = np.array([
            _expert_score(t, panel_bt[e], panel_ext, e, weights)
            for e in EXPERT_IDS
        ], dtype=np.float64)
        text_table[strat] = t
        expert_scores[strat] = scores
        role_table[strat] = "strategy"
        seen_texts.add(t)

    # Knowledge-personalised Qwen-max responses (CBT-informed prompts,
    # one per strategy).  Skip texts that exactly duplicate a cached
    # strategy response.
    for strat in STRATEGIES:
        t = kp_idx.get((sid, strat))
        if not t:
            continue
        t = _norm(t)
        if not t or t in seen_texts:
            continue
        key = f"kp_{strat}"
        scores = np.array([
            _expert_score(t, panel_bt[e], panel_ext, e, weights)
            for e in EXPERT_IDS
        ], dtype=np.float64)
        text_table[key] = t
        expert_scores[key] = scores
        role_table[key] = "kp"
        seen_texts.add(t)

    for system, idx in retrieval.items():
        t = idx.get(sid)
        if not t:
            continue
        t = _norm(t)
        if not t or t in seen_texts:
            continue
        key = f"retrieval_{system}"
        scores = np.array([
            _expert_score(t, panel_bt[e], panel_ext, e, weights)
            for e in EXPERT_IDS
        ], dtype=np.float64)
        text_table[key] = t
        expert_scores[key] = scores
        role_table[key] = "retrieval"
        seen_texts.add(t)

    return text_table, expert_scores, role_table


# ---------------------------------------------------------------------------
# Clinical constraints + DRO objective
# ---------------------------------------------------------------------------

def _required(ctx: dict, expert_scores: dict[str, np.ndarray]) -> list[str]:
    req: list[str] = []
    severe = _risk_level(ctx) in {"severe", "imminent"}
    distress = _state(ctx, "distress")
    alliance = _state(ctx, "alliance")
    clarity = _state(ctx, "clarity")
    if severe and "safety_referral" in expert_scores:
        req.append("safety_referral")
    if distress >= 0.65 or alliance <= 0.35:
        for s in ("empathy", "reflection"):
            if s in expert_scores:
                req.append(s)
                break
    if clarity <= 0.30:
        for s in ("question", "summarization"):
            if s in expert_scores:
                req.append(s)
                break
    return req


def _objective(scores: np.ndarray, alpha: float) -> float:
    """Mean-min DRO objective.  ``scores`` is a (|S|, |E|) matrix; we
    take the per-expert max (coverage) and combine its mean and worst."""
    cov = scores.max(axis=0)
    return float((1.0 - alpha) * cov.mean() + alpha * cov.min())


def _enumerate(text_table, expert_scores, role_table, *, K, alpha, required,
               low_risk: bool, exclude_safety_low_risk: bool) -> list[str]:
    candidates = list(expert_scores.keys())
    if exclude_safety_low_risk and low_risk and "safety_referral" in candidates:
        candidates.remove("safety_referral")
    required = [r for r in required if r in candidates]
    if len(candidates) <= K:
        return candidates
    free = [c for c in candidates if c not in required]
    n_free = max(0, K - len(required))
    if n_free <= 0:
        return required[:K]
    best_set: list[str] = []
    best_obj = -np.inf
    for combo in itertools.combinations(free, n_free):
        s = required + list(combo)
        sc = np.stack([expert_scores[k] for k in s], axis=0)
        v = _objective(sc, alpha)
        if v > best_obj:
            best_obj = v
            best_set = s
    return best_set


# ---------------------------------------------------------------------------
# Deployment verifier (panel-state)
# ---------------------------------------------------------------------------

def _panel_state_pick(K_set: list[str], expert_scores: dict[str, np.ndarray]
                      ) -> str:
    """Final-response picker: highest panel-mean score among the K=3."""
    best = K_set[0]
    best_v = -np.inf
    for k in K_set:
        v = float(expert_scores[k].mean())
        if v > best_v:
            best_v = v
            best = k
    return best


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_set",
                    default="data/judge_eval_v10/v10_eval_contexts.jsonl")
    ap.add_argument("--responses",
                    default="data/judge_eval_v10/v10_responses.jsonl")
    ap.add_argument("--safety_overrides",
                    default="data/judge_eval_v10/v12_safety_overrides.jsonl")
    ap.add_argument("--panel_bt", default="results/panel_bt.json")
    ap.add_argument("--retrieval_systems", nargs="+",
                    default=["kemi", "rag"],
                    help="Retrieval baselines to add to the unified pool.")
    ap.add_argument("--retrieval_root", default="data/fair_bon_v12")
    ap.add_argument("--kp_responses",
                    default="data/judge_eval_v10/v10_responses_kp.jsonl",
                    help="Knowledge-personalised Qwen-max responses "
                         "(strategy-conditioned, CBT-informed prompts).")
    ap.add_argument("--K", type=int, default=3)
    ap.add_argument("--robust_alpha", type=float, default=0.50,
                    help="Weight of the worst-expert tilt in mean-min DRO.")
    ap.add_argument("--exclude_safety_low_risk", action="store_true",
                    default=True)
    ap.add_argument("--out_topk",
                    default="data/fair_bon_v12/pasct_pp_topk.jsonl")
    ap.add_argument("--out_bon",
                    default="data/fair_bon_v12_panel_state/"
                            "pasct_pp_bon_responses.jsonl")
    ap.add_argument("--system_id", default="pasct_pp")
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
    retrieval = {
        s: _retrieval_index(s, Path(args.retrieval_root))
        for s in args.retrieval_systems
    }
    panel_ext = PanelBTExtender()
    kp_idx = _kp_response_index(Path(args.kp_responses))
    print(f"[pasct-pp] retrieval systems: {list(retrieval)} "
          f"(sizes: {[len(v) for v in retrieval.values()]})")
    print(f"[pasct-pp] kp pool: {len(kp_idx)} "
          f"({len({sid for sid, _ in kp_idx})} contexts)")

    Path(args.out_topk).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_bon).parent.mkdir(parents=True, exist_ok=True)
    n_total = 0
    n_safety = 0
    n_retrieval_picked = 0
    n_kp_picked = 0
    role_picked: dict[str, int] = {"strategy": 0, "retrieval": 0,
                                    "kp": 0, "safety_template": 0}

    with open(args.out_topk, "w", encoding="utf-8") as ft, \
            open(args.out_bon, "w", encoding="utf-8") as fb:
        for ctx in eval_rows:
            sid = str(ctx["sample_id"])
            n_total += 1
            if sid in overrides:
                bon = {
                    "sample_id": sid,
                    "system_id": args.system_id,
                    "selected_strategy": "safety_referral",
                    "response": SAFETY_TEMPLATE,
                    "verifier_choice": "safety_hard",
                    "K": 1,
                }
                topk = {
                    "sample_id": sid,
                    "planner": args.system_id,
                    "candidate_strategies": ["safety_referral"],
                    "required_strategies": ["safety_referral"],
                    "decision": "safety_hard",
                    "K": 1,
                }
                fb.write(json.dumps(bon, ensure_ascii=False) + "\n")
                ft.write(json.dumps(topk, ensure_ascii=False) + "\n")
                n_safety += 1
                role_picked["safety_template"] += 1
                continue

            weights = _state_weights(ctx)
            text_table, expert_scores, role_table = _build_candidate_pool(
                ctx, resp_idx, retrieval, kp_idx,
                panel_bt, panel_ext, weights)
            if not expert_scores:
                continue

            low_risk = (_risk_level(ctx) == "none" and not _is_risk(ctx)
                        and sid not in overrides)
            required = _required(ctx, expert_scores)
            K_set = _enumerate(
                text_table, expert_scores, role_table,
                K=args.K, alpha=args.robust_alpha, required=required,
                low_risk=low_risk,
                exclude_safety_low_risk=args.exclude_safety_low_risk,
            )
            if not K_set:
                continue
            picked = _panel_state_pick(K_set, expert_scores)
            role = role_table.get(picked, "strategy")
            role_picked[role] = role_picked.get(role, 0) + 1
            if role == "retrieval":
                n_retrieval_picked += 1
            elif role == "kp":
                n_kp_picked += 1

            bon = {
                "sample_id": sid,
                "system_id": args.system_id,
                "selected_strategy": picked,
                "response": text_table[picked],
                "verifier_choice": "panel_state_pp",
                "K": args.K,
                "candidates": K_set,
                "candidate_role": {k: role_table[k] for k in K_set},
                "candidate_scores": {
                    k: expert_scores[k].tolist() for k in K_set
                },
            }
            topk = {
                "sample_id": sid,
                "planner": args.system_id,
                "candidate_strategies": K_set,
                "candidate_role": {k: role_table[k] for k in K_set},
                "required_strategies": required,
                "panel_state_lift": weights.tolist(),
                "expert_scores": {k: v.tolist()
                                   for k, v in expert_scores.items()},
                "K": args.K,
            }
            fb.write(json.dumps(bon, ensure_ascii=False) + "\n")
            ft.write(json.dumps(topk, ensure_ascii=False) + "\n")

    print(f"[pasct-pp] total={n_total} safety_hard={n_safety} "
          f"retrieval_picked={n_retrieval_picked} kp_picked={n_kp_picked}")
    print(f"[pasct-pp] role distribution: {role_picked}")
    print(f"[pasct-pp] wrote {args.out_topk}")
    print(f"[pasct-pp] wrote {args.out_bon}")


if __name__ == "__main__":
    main()
