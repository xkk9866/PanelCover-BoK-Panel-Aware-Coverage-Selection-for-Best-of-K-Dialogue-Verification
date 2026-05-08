"""Panel-aware win rate, extended with prompt-only baseline generators.

This script reuses the panel-stochastic BT-projected win rate from
``eval.panel_aware_winrate`` but adds support for *prompt-only* baseline
generators that do not produce a K-set (LLM-direct, ESCoT, Self-Refine).
For these baselines the K-set is degenerate ($|S|=1$) and the response
text is read directly from
``data/fair_bon_v12/{system}_bon_responses.jsonl``.  Their text is
scored at the per-expert per-dimension BT level by the fitted
``PanelBTExtender`` (TF-IDF + Ridge on the panel BT corpus); for texts
that already exist in the panel BT corpus we return the exact fitted
logit and only genuinely new text goes through the regression.

The focal system is still read from a top-K file
(``{focal}_topk.jsonl``) and its K-set is scored from the original
panel BT logits via the response-cache lookup.  This is the same path
as ``eval.panel_aware_winrate`` for compatibility; the only added
machinery is the per-system, per-context single-response score path
for the prompt baselines.

Output goes to
``results/panel_aware_winrate_pasct_plus_with_prompts.json``; we keep
the original ``panel_aware_winrate_pasct_plus.json`` untouched so the
older results remain reproducible.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np

from eval.panel_aware_winrate import (
    _kset_for_system,
    _max_over_kset,
    _norm,
    _sigmoid,
    bootstrap_ci_and_pvalue,
    panel_winrate_pair,
    _load_panel_bt,
    _response_index,
    EXPERT_IDS,
    DIMS,
)
from eval.panel_bt_extender import PanelBTExtender


def _prompt_baseline_index(path: Path) -> dict[str, str]:
    """Return ``{sample_id: response_text}`` for a prompt baseline."""
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        sid = str(rec.get("sample_id", ""))
        text = _norm(rec.get("response"))
        if sid and text:
            out[sid] = text
    return out


def _max_over_prompt(text: str,
                     extender_score: dict[str, dict[str, float]],
                     dim: str) -> dict[str, float]:
    return {e: float(extender_score[e][dim]) for e in EXPERT_IDS}


def panel_winrate_focal_vs_prompt(
        focal_K: dict[str, list[str]],
        prompt_idx: dict[str, str],
        resp_idx: dict[tuple[str, str], str],
        panel_bt: dict[str, dict[str, dict[str, float]]],
        extender: PanelBTExtender,
        *, dim: str = "overall") -> dict:
    sids = sorted(set(focal_K) & set(prompt_idx))
    per_ctx: list[float] = []
    per_ctx_gap: list[float] = []
    for sid in sids:
        score_b = extender.score(prompt_idx[sid])
        wrs = []
        gaps = []
        for e in EXPERT_IDS:
            fmax = _max_over_kset(focal_K[sid], resp_idx, sid,
                                  panel_bt[e][dim])
            bmax = float(score_b[e][dim])
            wrs.append(_sigmoid(fmax - bmax))
            gaps.append(fmax - bmax)
        per_ctx.append(float(np.mean(wrs)))
        per_ctx_gap.append(float(np.mean(gaps)))
    arr = np.asarray(per_ctx, dtype=np.float64)
    arr_gap = np.asarray(per_ctx_gap, dtype=np.float64)
    return {
        "n": int(arr.size),
        "winrate": float(arr.mean()) if arr.size else float("nan"),
        "utility_gap": float(arr_gap.mean()) if arr.size else float("nan"),
        "per_context": arr.tolist(),
        "per_context_gap": arr_gap.tolist(),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--topk_dir", default="data/fair_bon_v12")
    ap.add_argument("--bon_dir", default="data/fair_bon_v12")
    ap.add_argument("--responses",
                    default="data/judge_eval_v10/v10_responses.jsonl")
    ap.add_argument("--panel_bt", default="results/panel_bt.json")
    ap.add_argument("--focal", default="pasct_plus")
    ap.add_argument("--topk_baselines", nargs="+",
                    default=["misc", "multiesc", "transesc", "kemi", "rag"])
    ap.add_argument("--prompt_baselines", nargs="+",
                    default=["llm_direct", "escot", "self_refine"])
    ap.add_argument("--out_json",
                    default="results/panel_aware_winrate_pasct_plus_with_prompts.json")
    args = ap.parse_args()

    panel_bt = _load_panel_bt(Path(args.panel_bt))
    resp_idx = _response_index([Path(args.responses)])
    extender = PanelBTExtender()

    focal_K = _kset_for_system(Path(args.topk_dir) / f"{args.focal}_topk.jsonl")
    print(f"[panel-wr-prompt] focal={args.focal} contexts={len(focal_K)}")

    record: dict = {"focal": args.focal, "baselines": {}}

    for base in args.topk_baselines:
        bp = Path(args.topk_dir) / f"{base}_topk.jsonl"
        if not bp.exists():
            print(f"[panel-wr-prompt] missing topk baseline {base}: {bp}")
            continue
        base_K = _kset_for_system(bp)
        out = panel_winrate_pair(focal_K, base_K, resp_idx, panel_bt,
                                 dim="overall")
        lo, hi, p = bootstrap_ci_and_pvalue(out["per_context"])
        gap_lo, gap_hi, _ = bootstrap_ci_and_pvalue(out["per_context_gap"])
        record["baselines"][base] = {
            "n": out["n"],
            "winrate": out["winrate"],
            "winrate_ci_lo": lo,
            "winrate_ci_hi": hi,
            "winrate_p_value": p,
            "utility_gap": out["utility_gap"],
            "utility_gap_ci_lo": gap_lo,
            "utility_gap_ci_hi": gap_hi,
            "type": "topk",
        }
        print(f"[panel-wr-prompt] {args.focal} vs {base:<14s} "
              f"n={out['n']:>4d}  WR={out['winrate']:.3f}  "
              f"[{lo:.3f}, {hi:.3f}]  p={p:.3f}")

    for base in args.prompt_baselines:
        bp = Path(args.bon_dir) / f"{base}_bon_responses.jsonl"
        if not bp.exists():
            print(f"[panel-wr-prompt] missing prompt baseline {base}: {bp}")
            continue
        prompt_idx = _prompt_baseline_index(bp)
        out = panel_winrate_focal_vs_prompt(
            focal_K, prompt_idx, resp_idx, panel_bt, extender,
            dim="overall")
        lo, hi, p = bootstrap_ci_and_pvalue(out["per_context"])
        gap_lo, gap_hi, _ = bootstrap_ci_and_pvalue(out["per_context_gap"])
        record["baselines"][base] = {
            "n": out["n"],
            "winrate": out["winrate"],
            "winrate_ci_lo": lo,
            "winrate_ci_hi": hi,
            "winrate_p_value": p,
            "utility_gap": out["utility_gap"],
            "utility_gap_ci_lo": gap_lo,
            "utility_gap_ci_hi": gap_hi,
            "type": "prompt",
        }
        print(f"[panel-wr-prompt] {args.focal} vs {base:<14s} "
              f"n={out['n']:>4d}  WR={out['winrate']:.3f}  "
              f"[{lo:.3f}, {hi:.3f}]  p={p:.3f}")

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(
        json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[panel-wr-prompt] wrote {args.out_json}")


if __name__ == "__main__":
    main()
