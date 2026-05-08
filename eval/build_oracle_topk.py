"""Oracle-strategy planner: emit the dataset gold strategy as a K=1 set.

For each context, the gold next-turn strategy comes from the PsyDial-
style annotation (``factual_strategy``); this is the strategy a human
counsellor actually used in the reference dialogue.  Oracle-strategy
emits a singleton K=1 plan so the fair-BoN verifier is forced to use
exactly the gold strategy's response.  When the gold response is not
in the cache (or the safety shield fires), the planner falls back to
the strategy whose cached response has the highest BT-overall score.

This is reported as an upper-bound reference; it is *not* a deployable
system because real test contexts have no gold strategy.  K=1 is also
why oracle gets *no* Best-of-K boost; any per-context lift over a K=1
non-oracle baseline is purely from picking the gold intent.
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
    ap.add_argument("--out_topk", default="data/fair_bon_v12/oracle_topk.jsonl")
    args = ap.parse_args()

    from eval.bt_winrate_proxy import (
        _build_response_text_index, _collect_dim_games, _fit_bt,
    )

    print("[oracle] fitting canonical BT-overall ...")
    bt = _fit_bt(_collect_dim_games(_build_response_text_index()).get("overall", []))

    eval_rows = _load_jsonl(Path(args.eval_set))
    overrides = {
        str(r.get("sample_id")): r
        for r in _load_jsonl(Path(args.safety_overrides))
        if r.get("shield_fired") or r.get("shield_fired_hard") or r.get("decision") == "hard"
    }
    resp_idx = _response_index(Path(args.responses))

    Path(args.out_topk).parent.mkdir(parents=True, exist_ok=True)
    n_total = 0; n_gold = 0; n_safety = 0; n_missing = 0; n_fallback = 0
    with open(args.out_topk, "w", encoding="utf-8") as fout:
        for ctx in eval_rows:
            sid = str(ctx["sample_id"])
            n_total += 1
            if sid in overrides:
                fout.write(json.dumps({
                    "sample_id": sid,
                    "planner": "oracle_strategy",
                    "candidate_strategies": ["safety_referral"],
                    "decision": "safety_hard",
                    "K": 1,
                }, ensure_ascii=False) + "\n")
                n_safety += 1
                continue
            avail = [s for s in STRATEGIES if (sid, s) in resp_idx]
            if not avail:
                n_missing += 1
                continue
            gold = str(ctx.get("factual_strategy") or "").strip()
            if gold and gold in avail:
                cand = [gold]
                n_gold += 1
            else:
                cand = [max(avail, key=lambda s: bt.get(resp_idx[(sid, s)], 0.0))]
                n_fallback += 1
            fout.write(json.dumps({
                "sample_id": sid,
                "planner": "oracle_strategy",
                "candidate_strategies": cand,
                "gold_strategy": gold,
                "K": 1,
            }, ensure_ascii=False) + "\n")
    print(
        f"[oracle] total={n_total} gold_hit={n_gold} fallback={n_fallback} "
        f"safety_hard={n_safety} missing={n_missing}"
    )
    print(f"[oracle] wrote {args.out_topk}")


if __name__ == "__main__":
    main()
