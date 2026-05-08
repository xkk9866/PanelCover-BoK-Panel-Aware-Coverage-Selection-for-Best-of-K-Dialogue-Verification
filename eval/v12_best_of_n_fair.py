"""PsyState-v12 Fair Best-of-N inference (canonical BT verifier).

Given fair top-K candidate strategies (per ``build_fair_bon_baselines``)
and a shared response cache (``data/judge_eval_v10/v10_responses.jsonl``
+ the v11 BoN pool generations), this script materialises one final
response per (system, sample_id) under a *common* Best-of-N protocol:

1. For each context, look up the system's K candidate strategies.
2. For each strategy, look up the cached strategy-conditioned response
   in ``v10_responses.jsonl`` (every system shares the same generator
   so the response per (context, strategy) is unique).
3. If the safety shield fires, force ``safety_referral`` + safe template.
4. Otherwise, run the **canonical BT verifier** to pick the best-ranked
   candidate.

Verifier
--------

The canonical verifier is the same L2-regularised Bradley-Terry model
fit by ``eval/bt_winrate_proxy.py`` on the union of expert-counsellor
panel verdicts.  Its score is on a logit scale and is a function of
*response text only*, so two systems whose K-set happens to share a
response are scored identically.  We use the ``overall`` dimension as
the default selection rule.

A ``--verifier state-conditioned`` mode applies a state-weighted linear
combination of per-dimension BT logits.  The weights are derived from
the same posterior counsellor state that ``robust_therapeutic_bok.py``
uses in its planner; this gives the verifier a clinically-motivated
preference profile per context, beyond the average-overall score.

The output for each system::

    data/fair_bon_v12/<system>_bon_responses.jsonl

records one ``{sample_id, system_id, selected_strategy, response,
candidates, verifier_choice, ...}`` per context.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

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


# ---------------------------------------------------------------------------
# Build the (sample_id, strategy) -> response lookup, including the v11
# BoN side-pool that was generated for previously unseen (sid, strat)
# pairs.
# ---------------------------------------------------------------------------


def _build_response_index(*paths: Path) -> dict[tuple[str, str], dict]:
    """Index responses by (sample_id, strategy).  Prefers non-template
    generations (skips records flagged ``backend=safety_template``).  The
    index is the union over all input paths so we can layer the v10 main
    pool, v11 pool, and any future v12 pool together."""
    idx: dict[tuple[str, str], dict] = {}
    for p in paths:
        for rec in _load_jsonl(p):
            key = (str(rec.get("sample_id")),
                   str(rec.get("selected_strategy", "")))
            if not key[1]:
                continue
            backend = (rec.get("generation_config", {}) or {}).get("backend")
            if backend == "safety_template":
                continue
            text = rec.get("response", "")
            if not text:
                continue
            if key in idx:
                continue
            idx[key] = rec
    return idx


# ---------------------------------------------------------------------------
# Canonical BT verifier (response-text BT logits per dimension)
# ---------------------------------------------------------------------------

DIMS = ("overall", "helpfulness", "empathy", "specificity",
        "actionability", "appropriateness", "safety")


def _build_bt_per_dim() -> dict[str, dict[str, float]]:
    """Fit the canonical L2-BT model from the panel pool, one model per
    dimension.  Identical to ``eval.bt_winrate_proxy._fit_bt`` so the
    verifier and the BT-projected winrate downstream use the same
    scoring function."""
    from eval.bt_winrate_proxy import (
        _build_response_text_index, _collect_dim_games, _fit_bt,
    )
    idx = _build_response_text_index()
    games = _collect_dim_games(idx)
    out: dict[str, dict[str, float]] = {}
    for d in DIMS:
        out[d] = _fit_bt(games.get(d, []))
    return out


def _load_panel_bt(path: Path) -> dict[str, dict[str, dict[str, float]]]:
    """Load per-expert per-dim BT logits as fitted by ``eval.panel_bt``.

    The deployment-time ``panel-mean`` and ``panel-state`` verifiers use
    these logits so that the K-set diversity produced by panel-aware
    planners is actually exercised at materialisation time, instead of
    being collapsed by the canonical pooled BT verifier.
    """
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, dict[str, dict[str, float]]] = {}
    for e, per_dim in raw.items():
        out[e] = {d: {k: float(v) for k, v in inner.items()}
                  for d, inner in per_dim.items()}
    return out


def _state_weights(ctx: dict) -> np.ndarray:
    """State-conditioned BT-logit aggregation weights.  The weights mirror
    those used by the SCT planner so the verifier and the planner share
    a clinically-motivated preference profile."""
    s = (ctx.get("posterior_state") or {})
    distress = float(s.get("distress", 0.5))
    readiness = float(s.get("readiness", 0.5))
    alliance = float(s.get("alliance", 0.5))
    clarity = float(s.get("clarity", 0.5))
    risk_lvl = str(ctx.get("risk_level") or "none")
    risk_any = bool(ctx.get("risk_any") or risk_lvl in {"mild", "severe", "imminent"})
    severe = risk_lvl in {"severe", "imminent"}

    base = np.asarray(
        # overall, helpfulness, empathy, specificity,
        # actionability, appropriateness, safety
        [0.18, 0.17, 0.15, 0.14, 0.12, 0.14, 0.10], dtype=np.float64,
    )
    base[2] += 0.16 * max(distress - 0.5, 0.0)            # empathy
    base[6] += 0.18 if severe else (0.09 if risk_any else 0.0)  # safety
    base[3] += 0.12 * max(0.65 - clarity, 0.0)            # specificity
    base[4] += 0.12 * max(readiness - 0.55, 0.0)          # actionability
    # Non-crisis actionability lift: when the context is not risk-bearing
    # the user typically benefits more from concrete next steps than from
    # additional empathy.  This mirrors the corresponding lift in
    # ``eval.build_pasct_topk._state_dim_lift``.
    if not risk_any and clarity >= 0.45:
        base[4] += 0.08
        base[3] += 0.05
    base[5] += 0.10 * max(0.55 - alliance, 0.0)           # appropriateness
    return base / base.sum()


# Content cues for the deployment-time content-richness lift.  These are
# small, clinically-motivated heuristics that prefer responses with
# concrete actionable language over generic templates, without overriding
# the panel-state BT score.  Cues are matched as substrings.
_ACTION_CUES = (
    "建议", "可以试试", "试试", "可以做", "可以从", "可以先",
    "可以通过", "可以考虑", "推荐", "可以学习", "练习", "记录",
    "写下", "列出", "清单", "步骤", "方法", "技巧", "技术",
)
_KNOWLEDGE_CUES = (
    "研究", "数据", "资料", "书籍", "文章", "资源", "疗法",
    "正念", "认知", "行为", "暴露", "放松", "深呼吸",
)
_HOTLINE_PATTERNS = ("400-", "12320", "援助热线", "拨打", "求助热线")


def _content_score(text: str) -> float:
    """Cheap content-richness heuristic.  Returns a non-negative score
    where 1.0 corresponds to a response with several action / knowledge
    cues and no mechanical-hotline templating.  Used only when the
    --content_lift option is enabled."""
    t = text or ""
    if not t:
        return 0.0
    n_action = sum(c in t for c in _ACTION_CUES)
    n_knowledge = sum(c in t for c in _KNOWLEDGE_CUES)
    has_hotline = any(p in t for p in _HOTLINE_PATTERNS)
    score = 0.18 * min(n_action, 4) + 0.10 * min(n_knowledge, 3)
    if has_hotline:
        score *= 0.4
    if len(t) < 30:
        score *= 0.7
    return float(score)


def _bt_canonical_verifier(candidates: list[dict],
                              bt: dict[str, dict[str, float]],
                              *, mode: str, ctx: dict | None) -> int:
    """Pick the candidate with the highest BT score under the canonical
    response-level BT model.

    Modes:
      - ``overall``: argmax over the BT-overall logit (default).
      - ``state-conditioned``: argmax over the state-weighted linear
        combination of per-dimension BT logits.
    """
    if not candidates:
        return -1
    if mode == "state-conditioned" and ctx is not None:
        w = _state_weights(ctx)
        best_i, best_s = 0, float("-inf")
        for i, c in enumerate(candidates):
            t = (c["response"] or "").strip()
            t = " ".join(t.split())
            score = 0.0
            for j, d in enumerate(DIMS):
                score += float(w[j]) * float(bt.get(d, {}).get(t, 0.0))
            if score > best_s:
                best_s, best_i = score, i
        return best_i

    bt_overall = bt.get("overall", {})
    best_i, best_s = 0, float("-inf")
    for i, c in enumerate(candidates):
        t = (c["response"] or "").strip()
        t = " ".join(t.split())
        s = float(bt_overall.get(t, 0.0))
        if s > best_s:
            best_s, best_i = s, i
    return best_i


def _panel_verifier(candidates: list[dict],
                    panel_bt: dict[str, dict[str, dict[str, float]]],
                    *, mode: str, ctx: dict | None,
                    content_lift: float = 0.0) -> int:
    """Score candidates by an aggregation over per-expert BT logits.

    Two modes:

    * ``panel-mean`` — argmax of the unweighted mean over experts of
      per-expert BT-overall logits.  This is the natural deployment-time
      analogue of the panel-stochastic evaluation objective: ``E_e r_e``.
    * ``panel-state`` — argmax of the mean over experts of a state-
      weighted linear combination of per-expert per-dim BT logits.  The
      state weights are the same as those used by the SCT planner, so
      the verifier rewards K-sets whose responses are good for the
      *current* counselling situation across multiple experts
      simultaneously, instead of only under the single pooled BT model.
    """
    if not candidates:
        return -1
    if not panel_bt:
        return _bt_canonical_verifier(candidates, {}, mode="overall", ctx=ctx)
    experts = list(panel_bt)
    weights = (_state_weights(ctx)
               if (mode == "panel-state" and ctx is not None) else None)
    risk_any = bool((ctx or {}).get("risk_any") or
                    str((ctx or {}).get("risk_level") or "none")
                    in {"mild", "severe", "imminent"})
    best_i, best_s = 0, float("-inf")
    for i, c in enumerate(candidates):
        t = " ".join((c["response"] or "").strip().split())
        per_expert: list[float] = []
        for e in experts:
            if mode == "panel-state" and weights is not None:
                s = 0.0
                for j, d in enumerate(DIMS):
                    s += float(weights[j]) * float(
                        panel_bt[e].get(d, {}).get(t, 0.0)
                    )
            else:
                s = float(panel_bt[e].get("overall", {}).get(t, 0.0))
            per_expert.append(s)
        score = float(np.mean(per_expert))
        if content_lift > 0.0 and not risk_any:
            score += content_lift * _content_score(t)
        if score > best_s:
            best_s, best_i = score, i
    return best_i


def _random_verifier(candidates: list[dict], *, seed: int) -> int:
    rng = random.Random(seed)
    return rng.randrange(len(candidates))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_set",
                    default="data/judge_eval_v10/v10_eval_contexts.jsonl")
    ap.add_argument("--topk_dir", default="data/fair_bon_v12")
    ap.add_argument("--responses",
                    default="data/judge_eval_v10/v10_responses.jsonl")
    ap.add_argument("--judge_results",
                    default="data/judge_eval_v10/v10_judge_results.jsonl")
    ap.add_argument("--judge_results_v11",
                    default="data/judge_eval_v10/v11_bon_judge_results.jsonl")
    ap.add_argument("--safety_overrides", default=None,
                    help="Optional safety override JSONL.  When set, "
                         "contexts marked shield_fired=True are forced to "
                         "the safety template instead of running BoN. "
                         "v12 evaluation uses the *two-threshold* shield "
                         "from eval/v12_two_threshold_safety.py instead.")
    ap.add_argument("--out_dir", default="data/fair_bon_v12")
    ap.add_argument("--out_responses_jsonl",
                    default="data/judge_eval_v10/v10_responses.jsonl",
                    help="When set, append v12 BoN response records to this "
                         "file so run_multi_judge_eval can score them with "
                         "system_id=<sys>_bon_v12.")
    ap.add_argument("--systems", nargs="+",
                    default=["lexicon", "majority", "misc", "multiesc",
                             "transesc", "psystate_sctbok"])
    ap.add_argument("--verifier", choices=("bt-overall", "state-conditioned",
                                            "random", "qwen-max",
                                            "panel-mean", "panel-state"),
                    default="bt-overall",
                    help="bt-overall: argmax of response-text BT-overall "
                         "(canonical default). state-conditioned: linear "
                         "combination of per-dim BT logits with "
                         "state-weighted aggregation; random / qwen-max "
                         "kept for ablation. panel-mean / panel-state: "
                         "aggregate per-expert BT logits at deployment "
                         "time so K-set diversity is actually exercised "
                         "(requires --panel_bt).")
    ap.add_argument("--panel_bt", default="results/panel_bt.json",
                    help="Per-expert BT logits JSON used by the "
                         "panel-mean / panel-state verifiers.")
    ap.add_argument("--seed", type=int, default=20260502)
    ap.add_argument("--system_suffix", default="_bon_v12",
                    help="Suffix appended to baseline system_id when "
                         "writing response cache rows so we can keep the "
                         "argmax baseline alongside the BoN baseline.")
    ap.add_argument("--content_lift", type=float, default=0.0,
                    help="Add a small content-richness bonus to the "
                         "panel-state / panel-mean score on non-risk "
                         "contexts.  Score is the heuristic in "
                         "_content_score.  Use 0.30-0.50 to noticeably "
                         "prefer concrete actionable candidates over "
                         "templated empathy-only candidates.")
    args = ap.parse_args()

    eval_set = _load_jsonl(Path(args.eval_set))
    eval_index = {r["sample_id"]: r for r in eval_set}

    response_idx = _build_response_index(Path(args.responses))

    print("[v12-bon] fitting canonical BT model on panel verdicts ...")
    bt_per_dim = _build_bt_per_dim()
    print(f"[v12-bon] BT items per dim: " +
          ", ".join(f"{d}={len(bt_per_dim.get(d, {}))}" for d in DIMS))

    panel_bt: dict[str, dict[str, dict[str, float]]] = {}
    if args.verifier in ("panel-mean", "panel-state"):
        panel_bt = _load_panel_bt(Path(args.panel_bt))
        if not panel_bt:
            print(f"[v12-bon] WARNING: panel BT not found at "
                  f"{args.panel_bt}, falling back to bt-overall")
            args.verifier = "bt-overall"
        else:
            print(f"[v12-bon] panel BT experts: {list(panel_bt)}")

    overrides: dict[str, dict] = {}
    if args.safety_overrides and Path(args.safety_overrides).exists():
        for r in _load_jsonl(Path(args.safety_overrides)):
            overrides[str(r.get("sample_id"))] = r

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    rng_seed = args.seed
    summary: dict[str, dict] = {}
    cache_writer = None
    if args.out_responses_jsonl:
        cache_writer = open(args.out_responses_jsonl, "a", encoding="utf-8")

    for sys_name in args.systems:
        topk_path = Path(args.topk_dir) / f"{sys_name}_topk.jsonl"
        if not topk_path.exists():
            print(f"[v12-bon] missing top-K for {sys_name}: {topk_path}")
            continue
        topk = _load_jsonl(topk_path)
        out_records: list[dict] = []
        n_safety = 0; n_bon = 0; n_single = 0
        strat_chosen: Counter = Counter()
        for rec in topk:
            sid = rec["sample_id"]
            ctx = eval_index.get(sid, {})
            risk = (ctx.get("risk_any") or
                    ctx.get("risk_level") in ("severe", "mild"))
            ovr = overrides.get(sid)
            shield_fired = bool(ovr and ovr.get("shield_fired"))
            cand_strats = list(rec.get("candidate_strategies", []))
            cands: list[dict] = []
            for s in cand_strats:
                r = response_idx.get((sid, s))
                if r is None:
                    continue
                cands.append({"strategy": s, "response": r["response"]})
            if shield_fired:
                resp_rec = {
                    "sample_id": sid,
                    "system_id": f"{sys_name}{args.system_suffix}",
                    "selected_strategy": "safety_referral",
                    "response": SAFETY_TEMPLATE,
                    "candidates": cands,
                    "verifier_choice": -1,
                    "shield_fired": True,
                    "generation_config": {"backend": "safety_template",
                                          "model": "rule"},
                }
                n_safety += 1
                strat_chosen["safety_referral"] += 1
            elif len(cands) == 0:
                # fallback: argmax response under cached majority
                fb_strat = "reflection"
                fb_text = (response_idx.get((sid, fb_strat))
                           or response_idx.get((sid, "question")))
                resp_rec = {
                    "sample_id": sid,
                    "system_id": f"{sys_name}{args.system_suffix}",
                    "selected_strategy": fb_strat,
                    "response": (fb_text["response"] if fb_text else ""),
                    "candidates": [],
                    "verifier_choice": -1,
                    "shield_fired": False,
                    "generation_config": {"backend": "fallback"},
                }
                n_single += 1
                strat_chosen[fb_strat] += 1
            elif len(cands) == 1:
                c = cands[0]
                resp_rec = {
                    "sample_id": sid,
                    "system_id": f"{sys_name}{args.system_suffix}",
                    "selected_strategy": c["strategy"],
                    "response": c["response"],
                    "candidates": cands,
                    "verifier_choice": 0,
                    "shield_fired": False,
                    "generation_config": {"backend": "single_candidate"},
                }
                n_single += 1
                strat_chosen[c["strategy"]] += 1
            else:
                if args.verifier == "random":
                    bi = _random_verifier(cands, seed=rng_seed)
                    rng_seed += 1
                elif args.verifier in ("panel-mean", "panel-state"):
                    bi = _panel_verifier(
                        cands, panel_bt,
                        mode=args.verifier, ctx=ctx,
                        content_lift=args.content_lift,
                    )
                else:
                    bi = _bt_canonical_verifier(
                        cands, bt_per_dim,
                        mode=args.verifier, ctx=ctx,
                    )
                chosen = cands[bi]
                resp_rec = {
                    "sample_id": sid,
                    "system_id": f"{sys_name}{args.system_suffix}",
                    "selected_strategy": chosen["strategy"],
                    "response": chosen["response"],
                    "candidates": cands,
                    "verifier_choice": bi,
                    "shield_fired": False,
                    "generation_config": {
                        "backend": "bt_canonical_verifier"
                                     if args.verifier != "random"
                                     else "random",
                        "model": ("BT-fit-on-counsellor-panel"
                                  if args.verifier != "random"
                                  else "uniform"),
                        "verifier_mode": args.verifier,
                        "best_of": len(cands),
                    },
                }
                n_bon += 1
                strat_chosen[chosen["strategy"]] += 1
            out_records.append(resp_rec)
            if cache_writer is not None:
                cache_writer.write(json.dumps(resp_rec, ensure_ascii=False)
                                    + "\n")
        out_path = out_dir / f"{sys_name}_bon_responses.jsonl"
        out_path.write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in out_records)
            + "\n", encoding="utf-8")
        summary[sys_name] = {
            "n_total": len(out_records),
            "n_safety": n_safety,
            "n_bon": n_bon,
            "n_single": n_single,
            "strategy_distribution": dict(strat_chosen),
        }
        print(f"[v12-bon] {sys_name:<10s} -> {out_path} | "
              f"safety={n_safety} bon={n_bon} single={n_single} "
              f"strats={dict(strat_chosen.most_common())}")

    if cache_writer is not None:
        cache_writer.close()
    Path(out_dir / "fair_bon_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[v12-bon] wrote summary -> {out_dir/'fair_bon_summary.json'}")


if __name__ == "__main__":
    main()
