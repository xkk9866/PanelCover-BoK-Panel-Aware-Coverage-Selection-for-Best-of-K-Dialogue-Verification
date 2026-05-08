r"""Strategy-distribution analysis for fair Best-of-N planners.

For each system in ``data/fair_bon_v12/{system}_bon_responses.jsonl``:

  - histogram over the 7 strategies in ``selected_strategy``
  - Shannon entropy over the same histogram (in bits)
  - top-1 accuracy of ``selected_strategy`` against the gold strategy from
    ``v10_eval_contexts.jsonl`` (the field ``factual_strategy`` from the
    PsyDial weak-label pipeline; same gold used by every prior planner
    paper that reports strategy accuracy on PsyDial)
  - macro-F1 of strategy prediction
  - KL divergence to the train-set prior (uniform train_freq from
    ``metadata.majority_strategy_train_freq``); reported as a sanity check
    on whether a planner just collapses to the corpus-frequency mode

Outputs ``results/strategy_distribution.json``.

Usage::

    python -m eval.strategy_distribution_analysis
"""
from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
EVAL_CTX = REPO / "data/judge_eval_v10/v10_eval_contexts.jsonl"
RESP_DIR = REPO / "data/fair_bon_v12"
OUT = REPO / "results/strategy_distribution.json"

SYSTEMS: list[tuple[str, str]] = [
    ("pasct_plus",     "pasct_plus_bon_responses.jsonl"),
    ("pasct_dro_anchor", "pasct_dro_anchor_bon_responses.jsonl"),
    ("psystate_pasct", "psystate_pasct_bon_responses.jsonl"),
    ("psystate_sctbok", "psystate_sctbok_bon_responses.jsonl"),
    ("oracle",         "oracle_bon_responses.jsonl"),
    ("misc",           "misc_bon_responses.jsonl"),
    ("multiesc",       "multiesc_bon_responses.jsonl"),
    ("transesc",       "transesc_bon_responses.jsonl"),
    ("kemi",           "kemi_bon_responses.jsonl"),
    ("rag",            "rag_bon_responses.jsonl"),
    ("majority",       "majority_bon_responses.jsonl"),
    ("lexicon",        "lexicon_bon_responses.jsonl"),
]

STRATEGIES = [
    "question",
    "reflection",
    "empathy",
    "reframe",
    "summarization",
    "action_suggestion",
    "safety_referral",
]


def _entropy_bits(p: dict[str, float]) -> float:
    h = 0.0
    for v in p.values():
        if v > 0:
            h -= v * math.log2(v)
    return h


def _macro_f1(preds: list[str], golds: list[str]) -> float:
    classes = sorted(set(golds) | set(preds))
    f1s: list[float] = []
    for c in classes:
        tp = sum(1 for p, g in zip(preds, golds) if p == c and g == c)
        fp = sum(1 for p, g in zip(preds, golds) if p == c and g != c)
        fn = sum(1 for p, g in zip(preds, golds) if g == c and p != c)
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        f1s.append(f1)
    return sum(f1s) / max(1, len(f1s))


def _load_gold_strategies() -> dict[str, str]:
    out: dict[str, str] = {}
    with EVAL_CTX.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            out[row["sample_id"]] = row.get("factual_strategy", "")
    return out


def _load_predictions(fname: str) -> dict[str, str]:
    fp = RESP_DIR / fname
    out: dict[str, str] = {}
    with fp.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            sid = row["sample_id"]
            strat = row.get("selected_strategy") or ""
            out[sid] = strat
    return out


def _hist(strats: list[str]) -> dict[str, float]:
    c = Counter(strats)
    n = max(1, sum(c.values()))
    return {s: c.get(s, 0) / n for s in STRATEGIES}


def main() -> None:
    golds = _load_gold_strategies()
    out: dict[str, dict] = {"systems": {}, "n_eval": len(golds)}

    train_prior = {s: 1.0 / len(STRATEGIES) for s in STRATEGIES}

    for name, fname in SYSTEMS:
        try:
            preds = _load_predictions(fname)
        except FileNotFoundError:
            print(f"[strat-dist] missing {fname}, skipping {name}")
            continue
        sids = sorted(set(preds) & set(golds))
        plist = [preds[s] for s in sids]
        glist = [golds[s] for s in sids]
        hist = _hist(plist)
        ent = _entropy_bits(hist)
        acc = sum(1 for p, g in zip(plist, glist) if p == g) / max(1, len(plist))
        f1 = _macro_f1(plist, glist)
        kl_uniform = sum(
            v * math.log2(max(1e-12, v) / train_prior[k])
            for k, v in hist.items() if v > 0
        )
        rec = {
            "n": len(plist),
            "histogram": hist,
            "entropy_bits": ent,
            "kl_to_uniform_bits": kl_uniform,
            "top1_accuracy_vs_factual": acc,
            "macro_f1_vs_factual": f1,
        }
        out["systems"][name] = rec
        print(
            f"[strat-dist] {name:<14s} | n={rec['n']:3d} | H={ent:.3f} | "
            f"acc={acc:.3f} | F1={f1:.3f} | KL_unif={kl_uniform:.3f}"
        )
        for s, v in hist.items():
            print(f"    {s:<20s} {v:.3f}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[strat-dist] wrote {OUT}")


if __name__ == "__main__":
    main()
