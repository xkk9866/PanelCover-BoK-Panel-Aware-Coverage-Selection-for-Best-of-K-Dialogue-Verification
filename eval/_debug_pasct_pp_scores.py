"""Quick diagnostic: per-context candidate scores in PA-SCT-DRO++."""
import json
from pathlib import Path

import numpy as np

from eval.build_pasct_pp_topk import (
    _build_candidate_pool, _load_jsonl, _load_panel_bt,
    _response_index, _retrieval_index, _state_weights,
)
from eval.panel_bt_extender import PanelBTExtender


ROOT = Path(".").resolve()
EVAL = ROOT / "data/judge_eval_v10/v10_eval_contexts.jsonl"
RESP = ROOT / "data/judge_eval_v10/v10_responses.jsonl"
PBT = ROOT / "results/panel_bt.json"


def main():
    eval_rows = _load_jsonl(EVAL)
    resp_idx = _response_index([RESP])
    panel_bt = _load_panel_bt(PBT)
    retrieval = {
        "kemi": _retrieval_index("kemi", ROOT / "data/fair_bon_v12"),
        "rag": _retrieval_index("rag", ROOT / "data/fair_bon_v12"),
    }
    panel_ext = PanelBTExtender()

    ranks = {"strategy": [], "retrieval_kemi": [], "retrieval_rag": []}
    pick_counts = {"strategy": 0, "retrieval_kemi": 0, "retrieval_rag": 0}
    margin_kemi = []
    margin_rag = []
    examples = 0

    for ctx in eval_rows[:100]:
        weights = _state_weights(ctx)
        text_table, expert_scores, role_table = _build_candidate_pool(
            ctx, resp_idx, retrieval, panel_bt, panel_ext, weights)
        means = {k: float(v.mean()) for k, v in expert_scores.items()}
        if not means:
            continue
        sorted_keys = sorted(means.keys(), key=lambda k: -means[k])
        top = sorted_keys[0]
        if top == "retrieval_kemi":
            pick_counts["retrieval_kemi"] += 1
        elif top == "retrieval_rag":
            pick_counts["retrieval_rag"] += 1
        else:
            pick_counts["strategy"] += 1

        for k in expert_scores:
            r = sorted_keys.index(k)
            cat = (k if k in ("retrieval_kemi", "retrieval_rag") else "strategy")
            ranks.setdefault(cat, []).append(r)

        if "retrieval_kemi" in means:
            margin_kemi.append(means["retrieval_kemi"] - means[top])
        if "retrieval_rag" in means:
            margin_rag.append(means["retrieval_rag"] - means[top])

        if examples < 5 and "retrieval_kemi" in means:
            print(f"\nctx {ctx['sample_id']}:")
            for k in sorted_keys:
                print(f"  {k:<22s} mean={means[k]:+.3f}  text={text_table[k][:60]}...")
            examples += 1

    print(f"\npick_counts (top-1 of unified pool): {pick_counts}")
    print(f"\nrank distribution (lower is better):")
    for cat, rs in ranks.items():
        if rs:
            print(f"  {cat:<22s} mean_rank={np.mean(rs):.2f}  n={len(rs)}")
    print(f"\nmargin retrieval_kemi - top1: mean={np.mean(margin_kemi):+.3f} max={np.max(margin_kemi):+.3f}")
    print(f"margin retrieval_rag  - top1: mean={np.mean(margin_rag):+.3f} max={np.max(margin_rag):+.3f}")


if __name__ == "__main__":
    main()
