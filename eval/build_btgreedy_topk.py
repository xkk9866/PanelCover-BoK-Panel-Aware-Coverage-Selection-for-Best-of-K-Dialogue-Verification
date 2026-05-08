"""BT-Greedy planner: top-K strategies by BT-overall score.

Under the canonical BT-overall verifier, the absolute upper bound on
selected-response BT score is the K-set whose top-1 BT-overall is
maximal.  Greedily picking the K strategies with the highest BT-overall
on their cached responses achieves exactly that.  This planner is
therefore the *verifier-aligned* upper bound on the canonical fair-BoN
metric: any state-conditioned planner that disagrees with this greedy
order is sacrificing in-protocol score for off-protocol coverage.

We report this baseline so reviewers can see how much of SCT-BoK's
gain is "pick high-BT responses" vs how much is genuinely
state-conditioned coverage.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from psystate.constants import STRATEGIES


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


def _response_index(path: Path) -> dict[tuple[str, str], str]:
    idx: dict[tuple[str, str], str] = {}
    for r in _load_jsonl(path):
        sid = str(r.get("sample_id", ""))
        strat = str(r.get("selected_strategy", ""))
        text = _norm(r.get("response"))
        backend = (r.get("generation_config") or {}).get("backend")
        if sid and strat and text and backend != "safety_template":
            idx.setdefault((sid, strat), text)
    return idx


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_set", default="data/judge_eval_v10/v10_eval_contexts.jsonl")
    ap.add_argument("--responses", default="data/judge_eval_v10/v10_responses.jsonl")
    ap.add_argument("--safety_overrides", default="data/judge_eval_v10/v12_safety_overrides.jsonl")
    ap.add_argument("--out_topk", default="data/fair_bon_v12/btgreedy_topk.jsonl")
    ap.add_argument("--K", type=int, default=3)
    args = ap.parse_args()

    from eval.bt_winrate_proxy import (
        _build_response_text_index, _collect_dim_games, _fit_bt,
    )
    print("[btgreedy] fitting canonical BT-overall ...")
    bt = _fit_bt(_collect_dim_games(_build_response_text_index()).get("overall", []))

    eval_rows = _load_jsonl(Path(args.eval_set))
    overrides = {
        str(r.get("sample_id")): r
        for r in _load_jsonl(Path(args.safety_overrides))
        if r.get("shield_fired") or r.get("shield_fired_hard") or r.get("decision") == "hard"
    }
    resp_idx = _response_index(Path(args.responses))

    Path(args.out_topk).parent.mkdir(parents=True, exist_ok=True)
    n_total = 0; n_safety = 0; n_missing = 0
    with open(args.out_topk, "w", encoding="utf-8") as fout:
        for ctx in eval_rows:
            sid = str(ctx["sample_id"])
            n_total += 1
            if sid in overrides:
                fout.write(json.dumps({
                    "sample_id": sid,
                    "planner": "btgreedy",
                    "candidate_strategies": ["safety_referral"],
                    "decision": "safety_hard",
                    "K": args.K,
                }, ensure_ascii=False) + "\n")
                n_safety += 1
                continue
            avail = [s for s in STRATEGIES if (sid, s) in resp_idx]
            if not avail:
                n_missing += 1
                continue
            scored = sorted(avail, key=lambda s: -bt.get(resp_idx[(sid, s)], 0.0))
            cand = scored[: args.K]
            fout.write(json.dumps({
                "sample_id": sid,
                "planner": "btgreedy",
                "candidate_strategies": cand,
                "K": args.K,
            }, ensure_ascii=False) + "\n")
    print(f"[btgreedy] total={n_total} safety_hard={n_safety} missing={n_missing}")
    print(f"[btgreedy] wrote {args.out_topk}")


if __name__ == "__main__":
    main()
