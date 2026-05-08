"""Build BT-Greedy hard-negative preference pairs for DPO/IPO.

The planner-level experiments show that PA-SCT-DRO cannot reliably beat
BT-Greedy by changing alpha/K/constraints alone.  The useful training signal is
therefore not "generic good response vs bad response", but the specific
failure mode:

    chosen   = PA-SCT-DRO response preferred by expert / proxy judge
    rejected = BT-Greedy response that looked strong under pooled BT

This script joins Qwen-max / new-expert pairwise verdict rows with the
materialised response files and writes DPO-ready JSONL examples:

    {"prompt": ..., "chosen": ..., "rejected": ..., "meta": ...}

It intentionally keeps only high-confidence PA-SCT-DRO wins over BT-Greedy,
because DPO/IPO is sensitive to noisy preferences.  Ties and BT-Greedy wins are
reported but not used as positive training examples.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


DIMS = (
    "overall",
    "helpfulness",
    "empathy",
    "specificity",
    "safety",
    "actionability",
    "avoids_over_advice",
)


def _norm(s: str | None) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _response_map(path: Path) -> dict[str, dict]:
    return {
        str(r.get("sample_id")): r
        for r in _load_jsonl(path)
        if r.get("sample_id") and r.get("response")
    }


def _context_map(path: Path) -> dict[str, dict]:
    return {str(r.get("sample_id")): r for r in _load_jsonl(path)}


def _context_prompt(ctx: dict) -> str:
    turns = []
    for msg in (ctx.get("context") or [])[-8:]:
        role = msg.get("role", "")
        if role == "system":
            continue
        content = _norm(msg.get("content"))
        if content:
            turns.append(f"{role}: {content}")
    last_user = _norm(ctx.get("last_user"))
    if last_user and (not turns or last_user not in turns[-1]):
        turns.append(f"user: {last_user}")
    state = ctx.get("posterior_state") or {}
    state_text = ", ".join(
        f"{k}={float(v):.2f}" for k, v in state.items()
        if isinstance(v, (int, float))
    )
    risk = ctx.get("risk_level") or "none"
    return (
        "You are an emotional-support counsellor. Respond in Mandarin with a "
        "warm, specific, safe, and non-overadvising response.\n\n"
        f"Dialogue context:\n{chr(10).join(turns)}\n\n"
        f"State: {state_text}; risk={risk}\n\nResponse:"
    )


def _score_for_focal(row: dict, dim: str = "overall") -> int:
    verdict = (row.get("verdict") or {}).get(dim, "tie")
    focal_side = row.get("focal_side")
    if verdict == "tie":
        return 0
    return 1 if verdict == focal_side else -1


def _vote_stats(rows: list[dict]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for dim in DIMS:
        wins = losses = ties = 0
        for row in rows:
            s = _score_for_focal(row, dim)
            if s > 0:
                wins += 1
            elif s < 0:
                losses += 1
            else:
                ties += 1
        n = wins + losses + ties
        out[dim] = {
            "wins": wins,
            "losses": losses,
            "ties": ties,
            "n": n,
            "win_rate": (wins + 0.5 * ties) / max(n, 1),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_set", default="data/judge_eval_v10/v10_eval_contexts.jsonl")
    ap.add_argument("--systems_dir", default="data/fair_bon_v12_panel_state")
    ap.add_argument("--focal", default="pasct_dro_anchor")
    ap.add_argument("--baseline", default="btgreedy")
    ap.add_argument("--raw_files", nargs="+", default=[
        "outputs/qwen_pairwise_eval_pstate.jsonl",
        "outputs/new_expert_eval.jsonl",
    ])
    ap.add_argument("--min_margin", type=int, default=1,
                    help="Require wins-losses >= this margin on overall votes.")
    ap.add_argument("--out_jsonl",
                    default="data/preference/btgreedy_hard_negatives.jsonl")
    ap.add_argument("--out_summary",
                    default="outputs/btgreedy_hard_negative_summary.json")
    args = ap.parse_args()

    contexts = _context_map(Path(args.eval_set))
    focal = _response_map(Path(args.systems_dir) / f"{args.focal}_bon_responses.jsonl")
    base = _response_map(Path(args.systems_dir) / f"{args.baseline}_bon_responses.jsonl")

    rows = []
    for raw_file in args.raw_files:
        rows.extend(_load_jsonl(Path(raw_file)))
    pair_id = f"{args.focal}_vs_{args.baseline}"
    rows = [r for r in rows if r.get("pair_id") == pair_id]

    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("sample_id"))].append(row)

    examples = []
    counters = {
        "groups": 0,
        "focal_win": 0,
        "btgreedy_win": 0,
        "tie_or_weak": 0,
        "missing_response": 0,
    }
    for sid, grow in sorted(grouped.items()):
        counters["groups"] += 1
        if sid not in contexts or sid not in focal or sid not in base:
            counters["missing_response"] += 1
            continue
        stats = _vote_stats(grow)
        overall = stats["overall"]
        margin = int(overall["wins"] - overall["losses"])
        if margin >= args.min_margin:
            counters["focal_win"] += 1
            examples.append({
                "prompt": _context_prompt(contexts[sid]),
                "chosen": _norm(focal[sid].get("response")),
                "rejected": _norm(base[sid].get("response")),
                "meta": {
                    "sample_id": sid,
                    "pair_id": pair_id,
                    "source": "qwen_proxy_and_new_expert_proxy",
                    "chosen_system": args.focal,
                    "rejected_system": args.baseline,
                    "chosen_strategy": focal[sid].get("selected_strategy"),
                    "rejected_strategy": base[sid].get("selected_strategy"),
                    "vote_stats": stats,
                    "rationales": [r.get("rationale", "") for r in grow[:5]],
                },
            })
        elif margin <= -args.min_margin:
            counters["btgreedy_win"] += 1
        else:
            counters["tie_or_weak"] += 1

    out = Path(args.out_jsonl)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    summary = {
        "n_examples": len(examples),
        "counters": counters,
        "out_jsonl": str(out),
        "note": (
            "DPO/IPO-ready hard negatives: chosen is PA-SCT-DRO only when "
            "proxy experts prefer it over BT-Greedy by the requested margin."
        ),
    }
    Path(args.out_summary).write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
