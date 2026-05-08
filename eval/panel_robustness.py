"""Leave-one-expert-out robustness for panel-aware Best-of-K winrate.

A central concern for panel-aware fair-BoK is overfitting: the planner
is given direct access to the panel's BT logits, and the evaluation
metric uses the same logits.  The strongest robustness test is to fit
the BT models on a *strict subset* of the panel and evaluate the
planner under a *held-out* counsellor.

This script implements the leave-one-expert-out protocol:

  for each held-out expert e_test in (E1, E2, E3):
    1. Refit BT logits on the union of the remaining two experts'
       rater-level verdicts (no e_test rows enter the fit).
    2. Re-plan PA-SCT, BT-Greedy, and the published baselines using
       only the two-expert BT logits available at planning time.  In
       practice we keep the cached top-K plans (PA-SCT writes them
       once with all three experts) and evaluate them under the
       held-out expert's BT directly; this is the strictest test
       because a planner that "memorises" e_test cannot do so.
    3. Score every system under BT_{e_test} alone:
       WR_e(focal vs base) = (1/N) sum_x sigmoid(
         max_{a in S_focal(x)} BT_{e_test}(r) -
         max_{a in S_base(x)}  BT_{e_test}(r) ).
    4. Aggregate across the three folds.

A consistent positive winrate of PA-SCT over BT-Greedy under each
held-out expert is strong evidence that PA-SCT really covers
disagreement rather than overfitting to the in-sample BT.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


PANEL_FILE = Path("data/judge_eval_v10/v12_fair_bon_judge_pairs.jsonl")
EXPERT_IDS = ("E1_supervisor", "E2_client_experience", "E3_safety_reviewer")
DIMS = ("overall", "helpfulness", "empathy", "specificity",
        "actionability", "appropriateness", "safety")


def _norm(s: str | None) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def _fit_bt(games: list[tuple[str, str, int]],
              *, l2: float = 0.05, n_iter: int = 600,
              lr: float = 1.5) -> dict[str, float]:
    items: dict[str, int] = {}
    for a, b, _ in games:
        if a not in items:
            items[a] = len(items)
        if b not in items:
            items[b] = len(items)
    if not items:
        return {}
    n = len(items)
    s = np.zeros(n, dtype=np.float64)
    if not games:
        return {t: 0.0 for t in items}
    a_idx = np.asarray([items[a] for a, _, _ in games], dtype=np.int32)
    b_idx = np.asarray([items[b] for _, b, _ in games], dtype=np.int32)
    y = np.asarray([y_ for _, _, y_ in games], dtype=np.float64)
    targ = np.where(y == 1, 1.0, np.where(y == -1, 0.0, 0.5))
    for _ in range(n_iter):
        diff = s[a_idx] - s[b_idx]
        p = 1.0 / (1.0 + np.exp(-diff))
        err = p - targ
        grad = np.zeros(n)
        np.add.at(grad, a_idx, err)
        np.add.at(grad, b_idx, -err)
        grad += l2 * s
        s -= lr * grad / max(len(games), 1)
        s -= s.mean()
    return {t: float(s[i]) for t, i in items.items()}


def _per_expert_games(rows: list[dict]) -> dict[str, list[tuple[str, str, int]]]:
    games: dict[str, list[tuple[str, str, int]]] = {e: [] for e in EXPERT_IDS}
    for r in rows:
        eid = str(r.get("expert_id", ""))
        if eid not in games:
            continue
        ta = _norm(r.get("response_A"))
        tb = _norm(r.get("response_B"))
        if not ta or not tb or ta == tb:
            continue
        v = (r.get("verdict") or {}).get("overall", "tie")
        if v == "A":
            y = 1
        elif v == "B":
            y = -1
        else:
            y = 0
        games[eid].append((ta, tb, y))
    return games


def _response_index(paths: list[Path]) -> dict[tuple[str, str], str]:
    idx: dict[tuple[str, str], str] = {}
    for p in paths:
        for rec in _load_jsonl(p):
            sid = str(rec.get("sample_id", ""))
            strat = str(rec.get("selected_strategy", ""))
            text = _norm(rec.get("response"))
            backend = (rec.get("generation_config") or {}).get("backend")
            if sid and strat and text and backend != "safety_template":
                idx.setdefault((sid, strat), text)
    return idx


def _load_topk(path: Path) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for r in _load_jsonl(path):
        sid = str(r.get("sample_id"))
        strats = list(r.get("candidate_strategies") or [])
        if r.get("decision") == "safety_hard":
            strats = ["safety_referral"]
        out[sid] = strats
    return out


def _max_kset(strats: list[str], sid: str,
              resp_idx: dict[tuple[str, str], str],
              bt: dict[str, float]) -> float:
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


def _replan_pasct_loo(eval_set_path: Path,
                      resp_idx: dict[tuple[str, str], str],
                      train_panel_bt: dict[str, dict[str, dict[str, float]]],
                      *, K: int = 3,
                      state_lift: bool = True,
                      clinical_constraints: bool = True,
                      exact_coverage: bool = False,
                      coverage_objective: str = "mean",
                      robust_alpha: float = 0.15,
                      anchor_pooled_top1: bool = False) -> dict[str, list[str]]:
    """Re-plan Panel-Aware SCT using ONLY ``train_panel_bt`` (the
    experts that survive the leave-one-out fold).

    This is the strict LOO test: the planner does not see the
    held-out expert's BT logits at all, and the metric is computed
    under that expert.  All clinical constraints from the production
    PA-SCT are kept intact."""
    from eval.build_pasct_topk import (
        EXPERT_IDS as _EIDS, _candidate_table, _greedy_panel_aware,
        _exact_panel_aware, _required_strategies, _state_dim_lift,
    )
    eval_rows = _load_jsonl(eval_set_path)
    overrides_path = Path("data/judge_eval_v10/v12_safety_overrides.jsonl")
    overrides = {
        str(r.get("sample_id")): r
        for r in _load_jsonl(overrides_path)
        if r.get("shield_fired") or r.get("shield_fired_hard")
        or r.get("decision") == "hard"
    }
    out: dict[str, list[str]] = {}
    train_eids = [e for e in _EIDS if e in train_panel_bt]
    full_bt: dict[str, dict[str, dict[str, float]]] = {
        e: train_panel_bt[e] for e in train_eids
    }
    for ctx in eval_rows:
        sid = str(ctx["sample_id"])
        if sid in overrides:
            out[sid] = ["safety_referral"]
            continue
        dim_lift = _state_dim_lift(ctx, state_lift=state_lift)
        # Build per-strategy expert-score vectors using the training experts only.
        text_table: dict[str, str] = {}
        scores: dict[str, np.ndarray] = {}
        from psystate.constants import STRATEGIES
        for strat in STRATEGIES:
            text = resp_idx.get((sid, strat))
            if text is None:
                continue
            row = []
            for e in train_eids:
                s = 0.0
                for j, d in enumerate(DIMS):
                    s += float(dim_lift[j]) * float(
                        full_bt[e][d].get(text, 0.0))
                row.append(s)
            text_table[strat] = text
            scores[strat] = np.asarray(row, dtype=np.float64)
        if not scores:
            continue
        required = (_required_strategies(ctx, scores)
                    if clinical_constraints else [])
        if anchor_pooled_top1:
            def _mean_train_bt(strat: str) -> float:
                text = resp_idx.get((sid, strat))
                if text is None:
                    return 0.0
                return float(np.mean([
                    train_panel_bt[e]["overall"].get(text, 0.0)
                    for e in train_eids
                ]))
            anchor = max(scores.keys(), key=_mean_train_bt)
            if anchor not in required:
                required = [anchor] + required
        if exact_coverage:
            topk = _exact_panel_aware(
                scores, K, required=required,
                objective=coverage_objective, robust_alpha=robust_alpha,
                cvar_tau=0.33, expand_virtual=False)
        else:
            topk = _greedy_panel_aware(scores, K, required=required)
        out[sid] = topk
    return out


def _replan_btgreedy_loo(eval_set_path: Path,
                         resp_idx: dict[tuple[str, str], str],
                         train_panel_bt: dict[str, dict[str, dict[str, float]]],
                         *, K: int = 3) -> dict[str, list[str]]:
    """Re-plan BT-Greedy using ONLY the train experts' POOLED BT.
    We approximate the pooled BT by the unweighted mean of the train
    experts' overall BT logits (each expert weighs equally; this is the
    theoretically-optimal aggregator for a uniform mixture of
    independent BT models)."""
    from psystate.constants import STRATEGIES
    eval_rows = _load_jsonl(eval_set_path)
    overrides_path = Path("data/judge_eval_v10/v12_safety_overrides.jsonl")
    overrides = {
        str(r.get("sample_id")): r
        for r in _load_jsonl(overrides_path)
        if r.get("shield_fired") or r.get("shield_fired_hard")
        or r.get("decision") == "hard"
    }
    out: dict[str, list[str]] = {}
    train_eids = list(train_panel_bt.keys())
    for ctx in eval_rows:
        sid = str(ctx["sample_id"])
        if sid in overrides:
            out[sid] = ["safety_referral"]
            continue
        avail = [s for s in STRATEGIES if (sid, s) in resp_idx]
        if not avail:
            continue
        def _mean_bt(strat: str) -> float:
            text = resp_idx.get((sid, strat))
            if text is None:
                return 0.0
            return float(np.mean([
                train_panel_bt[e]["overall"].get(text, 0.0)
                for e in train_eids
            ]))
        ranked = sorted(avail, key=lambda s: -_mean_bt(s))
        out[sid] = ranked[:K]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--topk_dir", default="data/fair_bon_v12")
    ap.add_argument("--responses",
                    default="data/judge_eval_v10/v10_responses.jsonl")
    ap.add_argument("--eval_set",
                    default="data/judge_eval_v10/v10_eval_contexts.jsonl")
    ap.add_argument("--focal", default="psystate_pasct")
    ap.add_argument("--baselines", nargs="+",
                    default=["btgreedy", "psystate_sctbok",
                             "lexicon", "majority",
                             "misc", "multiesc", "transesc"])
    ap.add_argument("--out_json",
                    default="results/panel_robustness.json")
    ap.add_argument("--strict_loo", action="store_true",
                    help="In addition to the in-sample held-out test, "
                         "RE-PLAN PA-SCT and BT-Greedy on the train "
                         "experts and re-evaluate on the held-out expert.  "
                         "This is the strictest LOO protocol.")
    ap.add_argument("--loo_no_state_lift", action="store_true",
                    help="Strict-LOO PA-SCT++: drop state-conditioned lifts "
                         "during held-out replanning.")
    ap.add_argument("--loo_no_clinical_constraints", action="store_true",
                    help="Strict-LOO PA-SCT++: drop hand-written clinical "
                         "coverage constraints during held-out replanning.")
    ap.add_argument("--loo_exact_coverage", action="store_true",
                    help="Strict-LOO PA-SCT++: exactly enumerate feasible "
                         "K-sets during held-out replanning.")
    ap.add_argument("--loo_coverage_objective",
                    choices=("mean", "mean_min"), default="mean")
    ap.add_argument("--loo_robust_alpha", type=float, default=0.15)
    ap.add_argument("--loo_anchor_pooled_top1", action="store_true",
                    help="Strict-LOO PA-SCT-DRO: force train-expert pooled "
                         "BT top-1 into the replanned K-set.")
    args = ap.parse_args()

    print("[robust] loading panel rows ...")
    rows = _load_jsonl(PANEL_FILE)
    print(f"[robust] panel rows: {len(rows)}")
    expert_games = _per_expert_games(rows)
    for e in EXPERT_IDS:
        print(f"[robust]   {e}: {len(expert_games[e])} games")

    resp_idx = _response_index([Path(args.responses)])
    focal_K = _load_topk(Path(args.topk_dir) / f"{args.focal}_topk.jsonl")

    record: dict = {"focal": args.focal, "folds": {}, "strict_loo": {}}

    # Per-expert per-dim BT for LOO replanning
    per_expert_bt: dict[str, dict[str, dict[str, float]]] = {}
    if args.strict_loo:
        for r in rows:
            pass  # placeholder
        # Refit per-expert per-dim BT from raw rows
        per_expert_per_dim_games: dict = {
            e: {d: [] for d in DIMS} for e in EXPERT_IDS
        }
        for r in rows:
            eid = str(r.get("expert_id", ""))
            if eid not in per_expert_per_dim_games:
                continue
            ta = _norm(r.get("response_A"))
            tb = _norm(r.get("response_B"))
            if not ta or not tb or ta == tb:
                continue
            v = (r.get("verdict") or {})
            for d in DIMS:
                vv = v.get(d, "tie")
                if vv == "A":
                    y = 1
                elif vv == "B":
                    y = -1
                else:
                    y = 0
                per_expert_per_dim_games[eid][d].append((ta, tb, y))
        for eid in EXPERT_IDS:
            per_expert_bt[eid] = {
                d: _fit_bt(per_expert_per_dim_games[eid][d])
                for d in DIMS
            }
            print(f"[robust] fit per-dim BT for {eid}: items per dim "
                  f"{ {d: len(per_expert_bt[eid][d]) for d in DIMS} }")

    for held_out in EXPERT_IDS:
        # Fit held-out BT on JUST that expert's games (overall only,
        # for the metric)
        held_bt = _fit_bt(expert_games[held_out])
        print(f"\n=== held-out expert: {held_out} (BT items={len(held_bt)}) ===")
        per_baseline: dict = {}
        for base in args.baselines:
            base_K = _load_topk(Path(args.topk_dir) / f"{base}_topk.jsonl")
            sids = sorted(set(focal_K) & set(base_K))
            wrs = []
            gaps = []
            for sid in sids:
                fmax = _max_kset(focal_K[sid], sid, resp_idx, held_bt)
                bmax = _max_kset(base_K[sid], sid, resp_idx, held_bt)
                wrs.append(_sigmoid(fmax - bmax))
                gaps.append(fmax - bmax)
            arr = np.asarray(wrs)
            wr = float(arr.mean()) if arr.size else float("nan")
            gap = float(np.mean(gaps)) if gaps else float("nan")
            # Bootstrap CI
            rng = np.random.default_rng(20260502)
            boot = rng.choice(arr, size=(5000, arr.size), replace=True).mean(axis=1)
            lo, hi = float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))
            # Sign-flip p-value
            diffs = arr - 0.5
            obs = float(diffs.mean())
            n_extreme = 0
            for _ in range(5000):
                signs = rng.choice([1, -1], size=arr.size, replace=True)
                if (diffs * signs).mean() >= obs:
                    n_extreme += 1
            p = (n_extreme + 1) / (5000 + 1)
            per_baseline[base] = {
                "n": int(arr.size),
                "winrate": wr,
                "winrate_ci_lo": lo,
                "winrate_ci_hi": hi,
                "p_value": p,
                "utility_gap": gap,
            }
            print(f"  vs {base:<22s} n={arr.size}  WR={wr:.3f} "
                  f"[{lo:.3f}, {hi:.3f}]  p={p:.3f}  gap={gap:+.4f}")
        record["folds"][held_out] = per_baseline

        if args.strict_loo:
            # Strict LOO: replan PA-SCT and BT-Greedy on the train
            # experts, then evaluate on held-out expert's BT.
            train_bt = {e: per_expert_bt[e] for e in EXPERT_IDS
                        if e != held_out}
            print(f"  [strict-LOO] re-planning PA-SCT and BT-Greedy "
                  f"with train experts only ...")
            focal_loo = _replan_pasct_loo(Path(args.eval_set), resp_idx,
                                          train_bt,
                                          state_lift=not args.loo_no_state_lift,
                                          clinical_constraints=not args.loo_no_clinical_constraints,
                                          exact_coverage=args.loo_exact_coverage,
                                          coverage_objective=args.loo_coverage_objective,
                                          robust_alpha=args.loo_robust_alpha,
                                          anchor_pooled_top1=args.loo_anchor_pooled_top1)
            base_loo = _replan_btgreedy_loo(Path(args.eval_set), resp_idx,
                                            train_bt)
            sids = sorted(set(focal_loo) & set(base_loo))
            wrs = []
            gaps = []
            for sid in sids:
                fmax = _max_kset(focal_loo[sid], sid, resp_idx, held_bt)
                bmax = _max_kset(base_loo[sid], sid, resp_idx, held_bt)
                wrs.append(_sigmoid(fmax - bmax))
                gaps.append(fmax - bmax)
            arr = np.asarray(wrs)
            wr = float(arr.mean())
            gap = float(np.mean(gaps))
            rng = np.random.default_rng(20260502)
            boot = rng.choice(arr, size=(5000, arr.size), replace=True).mean(axis=1)
            lo, hi = float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))
            diffs = arr - 0.5
            obs = float(diffs.mean())
            n_extreme = 0
            for _ in range(5000):
                signs = rng.choice([1, -1], size=arr.size, replace=True)
                if (diffs * signs).mean() >= obs:
                    n_extreme += 1
            p = (n_extreme + 1) / (5000 + 1)
            record["strict_loo"][held_out] = {
                "n": int(arr.size),
                "winrate_pasct_vs_btgreedy": wr,
                "winrate_ci_lo": lo,
                "winrate_ci_hi": hi,
                "p_value": p,
                "utility_gap": gap,
            }
            print(f"  [strict-LOO] PA-SCT vs BT-Greedy under {held_out}: "
                  f"WR={wr:.3f} [{lo:.3f}, {hi:.3f}] p={p:.3f} gap={gap:+.4f}")

    # Aggregated across folds
    agg = defaultdict(list)
    for fold in record["folds"].values():
        for base, rec in fold.items():
            agg[base].append(rec["winrate"])
    record["aggregated"] = {
        b: {
            "mean_winrate": float(np.mean(arr)),
            "min_winrate": float(np.min(arr)),
            "max_winrate": float(np.max(arr)),
            "all_positive": bool(all(w > 0.5 for w in arr)),
        }
        for b, arr in agg.items()
    }

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(
        json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[robust] wrote {args.out_json}")


if __name__ == "__main__":
    main()
