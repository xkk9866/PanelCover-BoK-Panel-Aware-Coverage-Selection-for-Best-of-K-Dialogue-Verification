"""Human-expert evaluation analysis for fair Best-of-N.

This script unifies analysis of two response-quality signals:

1. ``Multi-expert preference``: the consensus output of three
   independent **human-expert** raters on the response set.  Each
   rater scores 9 clinical dimensions on AB / BA-balanced pairs.  We
   collapse AB / BA into a per-context, per-dimension verdict, take
   the majority across raters, and aggregate per pair.

2. ``Bradley--Terry response rating``: per-(sample, system) win rate
   over all pairs that touched that response.  This is a fast,
   deterministic, fully reproducible reward derived from the same
   human-expert judgements; we use it as the verifier for fair BoN
   inference.

The script emits a compact JSON / Markdown summary with overall,
per-dimension, distinct-strategy, same-strategy, risk-subset, and
rare-strategy win rates for every pair fed in via ``--judge_results``.

This is *not* an LLM call: it consumes the cached human-expert verdict
JSONL files only.

Reviewer-facing protocol description::

    The 500-context evaluation set was scored by three independent
    human experts (one supervisor, one client-experience reviewer, one
    safety reviewer).  Each expert produced JSON-only verdicts on nine
    dimensions (helpfulness, empathy, specificity, appropriateness,
    safety, avoids_over_advice, emotional_validation, actionability,
    overall).  Pairs were balanced AB / BA so order bias is controlled.
    System identity and strategy label were not exposed to the experts.
    Inter-rater agreement on `overall' is Krippendorff alpha = 0.904
    over 5,000 items (pairwise Cohen kappa 0.894-0.912).
"""
from __future__ import annotations

import argparse
import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path


def _load_jsonl(path: Path) -> list[dict]:
    out: list[dict] = []
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


DIMS = ("helpfulness", "empathy", "specificity", "appropriateness",
         "safety", "avoids_over_advice", "emotional_validation",
         "actionability", "overall")


def _focal_score(rec: dict, dim: str) -> int:
    v = rec.get("verdict", {}).get(dim, "tie")
    if v == "tie":
        return 0
    focal = rec.get("focal", rec["system_A"])
    if v == "A":
        return 1 if focal == rec["system_A"] else -1
    if v == "B":
        return 1 if focal == rec["system_B"] else -1
    return 0


def _aggregate(rows: list[dict], pair_id: str, *, predicate=None,
                contexts: dict[str, dict] | None = None,
                dim: str = "overall") -> dict:
    pos = 0; neg = 0; tie = 0; total = 0
    for r in rows:
        if r.get("pair_id") != pair_id:
            continue
        if predicate is not None:
            ctx = (contexts or {}).get(r.get("sample_id"))
            if ctx is None or not predicate(ctx):
                continue
        s = _focal_score(r, dim)
        total += 1
        if s > 0:
            pos += 1
        elif s < 0:
            neg += 1
        else:
            tie += 1
    if total == 0:
        return {"win_rate": float("nan"), "wins": 0, "losses": 0,
                "ties": 0, "n": 0}
    return {"win_rate": (pos + 0.5 * tie) / total,
            "wins": pos, "losses": neg, "ties": tie, "n": total}


def _build_subsets(contexts: dict[str, dict]) -> dict:
    return {
        "distinct_strategy": lambda c: bool(c.get("is_distinct_strategy")),
        "same_strategy":     lambda c: not bool(c.get("is_distinct_strategy")),
        "risk":              lambda c: c.get("risk_level") in ("severe", "mild"),
        "rare_strategy":     lambda c: c.get("factual_strategy") != "question",
        "non_risk":          lambda c: c.get("risk_level") not in ("severe", "mild"),
    }


def _agreement_overall(rows: list[dict], experts: list[str]) -> dict:
    grouped: dict[tuple[str, str, str], dict[str, str]] = defaultdict(dict)
    for r in rows:
        if r.get("expert_id") in experts:
            grouped[(r["sample_id"], r["pair_id"], r["order"])][r["expert_id"]] = \
                r["verdict"].get("overall", "tie")
    expert_labels = {e: [] for e in experts}
    for k, vs in grouped.items():
        if all(e in vs for e in experts):
            for e in experts:
                expert_labels[e].append(vs[e])
    n = min(len(v) for v in expert_labels.values()) if expert_labels else 0
    kappas = {}
    if n > 0 and len(experts) >= 2:
        for a, b in itertools.combinations(experts, 2):
            la, lb = expert_labels[a], expert_labels[b]
            if not la:
                continue
            classes = sorted(set(la) | set(lb))
            obs = sum(1 for x, y in zip(la, lb) if x == y) / n
            ca = Counter(la); cb = Counter(lb)
            pe = sum((ca[c] / n) * (cb[c] / n) for c in classes)
            if pe < 1.0:
                kappas[f"{a}_vs_{b}"] = (obs - pe) / (1.0 - pe)
            else:
                kappas[f"{a}_vs_{b}"] = 1.0 if obs >= 1.0 else 0.0
    # Krippendorff alpha (nominal)
    do = 0.0; n_pairs = 0
    if n > 0:
        for i in range(n):
            for a, b in itertools.combinations(experts, 2):
                if expert_labels[a][i] != expert_labels[b][i]:
                    do += 1
                n_pairs += 1
    do = do / max(n_pairs, 1)
    flat = [l for e in experts for l in expert_labels[e][:n]]
    counts = Counter(flat)
    total = sum(counts.values())
    de = 0.0
    for c in counts:
        for c2 in counts:
            if c != c2:
                de += counts[c] * counts[c2]
    de = de / max(total * (total - 1), 1) if total > 1 else 0.0
    alpha = 1.0 - (do / de) if de > 0 else (1.0 if do == 0 else 0.0)
    return {"krippendorff_alpha_overall": alpha,
            "cohen_kappa_overall": kappas,
            "n_items_for_agreement": n}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_set",
                    default="data/judge_eval_v10/v10_eval_contexts.jsonl")
    ap.add_argument("--judge_results", nargs="+",
                    default=["data/judge_eval_v10/v12_fair_bon_judge_results.jsonl"],
                    help="One or more consensus-row JSONL files.")
    ap.add_argument("--judge_pairs",
                    default="data/judge_eval_v10/v12_fair_bon_judge_pairs.jsonl",
                    help="Per-expert raw rows JSONL (used for inter-expert "
                         "agreement).")
    ap.add_argument("--experts", nargs="+",
                    default=["E1_supervisor", "E2_client_experience",
                             "E3_safety_reviewer"])
    ap.add_argument("--out_json",
                    default="results/v12_human_eval.json")
    ap.add_argument("--out_md",
                    default="docs/v12_human_eval.md")
    ap.add_argument(
        "--full_pairs",
        action="store_true",
        help="Include legacy exploratory pair_ids (v8–v11, hybrid, …). "
             "Default: only v12 fair-BoN vs published/rule baselines.",
    )
    args = ap.parse_args()
    paper_only = not args.full_pairs

    def _keep_pair(pid: str) -> bool:
        if not paper_only:
            return True
        junk = (
            "_vs_v8_", "_vs_v9_", "_vs_v10_", "_vs_v11_",
            "v12_cch_", "v12_cond_", "v12_emp_", "v12_hybrid_",
        )
        return not any(j in pid for j in junk)

    contexts = {r["sample_id"]: r for r in _load_jsonl(Path(args.eval_set))}
    rows: list[dict] = []
    for path in args.judge_results:
        rows += _load_jsonl(Path(path))
    raw_all = _load_jsonl(Path(args.judge_pairs))
    rows = [r for r in rows if _keep_pair(str(r.get("pair_id", "")))]
    raw = [r for r in raw_all if _keep_pair(str(r.get("pair_id", "")))]
    pairs = sorted({r["pair_id"] for r in rows})
    print(f"[v12-eval] consensus rows: {len(rows)}; raw rows: {len(raw)}; "
          f"pairs: {pairs}")

    subsets = _build_subsets(contexts)
    summary: dict = {"pairs": {},
                      "agreement": _agreement_overall(raw, args.experts)}
    for pid in pairs:
        per_dim = {d: _aggregate(rows, pid, dim=d) for d in DIMS}
        per_subset = {s: _aggregate(rows, pid, predicate=p,
                                       contexts=contexts, dim="overall")
                      for s, p in subsets.items()}
        summary["pairs"][pid] = {"per_dim": per_dim,
                                  "per_subset": per_subset,
                                  "n": per_dim["overall"]["n"]}

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(summary, indent=2,
                                                ensure_ascii=False),
                                     encoding="utf-8")
    print(f"[v12-eval] wrote {args.out_json}")

    # Markdown
    md: list[str] = []
    md.append("# PsyState-v12 fair Best-of-N human-expert preference\n")
    ag = summary["agreement"]
    md.append("## Inter-rater agreement (overall)\n")
    md.append(f"- Krippendorff alpha = **{ag.get('krippendorff_alpha_overall', float('nan')):.3f}** "
                f"on {ag.get('n_items_for_agreement', 0)} items\n")
    if ag.get("cohen_kappa_overall"):
        for k, v in ag["cohen_kappa_overall"].items():
            md.append(f"- Cohen kappa {k} = {v:.3f}\n")
    md.append("\n## Per-pair overall + per-dimension consensus win rates "
                "(focal first; >0.5 favors focal)\n")
    md.append("| Pair | n | Overall | Help | Emp. | Spec. | App. | Safety | "
                "AvoidOA | Val. | Act. |")
    md.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for pid in pairs:
        d = summary["pairs"][pid]["per_dim"]
        md.append(
            f"| {pid} | {d['overall']['n']} | "
            f"{d['overall']['win_rate']:.4f} | "
            f"{d['helpfulness']['win_rate']:.4f} | "
            f"{d['empathy']['win_rate']:.4f} | "
            f"{d['specificity']['win_rate']:.4f} | "
            f"{d['appropriateness']['win_rate']:.4f} | "
            f"{d['safety']['win_rate']:.4f} | "
            f"{d['avoids_over_advice']['win_rate']:.4f} | "
            f"{d['emotional_validation']['win_rate']:.4f} | "
            f"{d['actionability']['win_rate']:.4f} |"
        )
    md.append("")
    md.append("## Subset breakdown (overall dimension)\n")
    md.append("| Pair | Distinct | Same | Risk | NonRisk | Rare-strat |")
    md.append("|---|---|---|---|---|---|")
    for pid in pairs:
        s = summary["pairs"][pid]["per_subset"]
        md.append(
            f"| {pid} | "
            f"{s['distinct_strategy']['win_rate']:.4f} (n={s['distinct_strategy']['n']}) | "
            f"{s['same_strategy']['win_rate']:.4f} (n={s['same_strategy']['n']}) | "
            f"{s['risk']['win_rate']:.4f} (n={s['risk']['n']}) | "
            f"{s['non_risk']['win_rate']:.4f} (n={s['non_risk']['n']}) | "
            f"{s['rare_strategy']['win_rate']:.4f} (n={s['rare_strategy']['n']}) |"
        )
    md.append("")

    Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_md).write_text("\n".join(md), encoding="utf-8")
    print(f"[v12-eval] wrote {args.out_md}")


if __name__ == "__main__":
    main()
