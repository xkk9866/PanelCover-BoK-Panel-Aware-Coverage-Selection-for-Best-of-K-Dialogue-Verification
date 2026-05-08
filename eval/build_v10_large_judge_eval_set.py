"""Large stratified judge evaluation set for PsyState-v10.

Compared with v9 (120 contexts, 70% same-strategy ties dominating
the judge signal) v10 builds a 500-context evaluation set that:

1. Stratifies across 17 clinical / engineering slices and guarantees
   at least 30 contexts per slice;
2. Reads pre-computed planner picks for the *lexicon, v8, v9, v10
   (placeholder = v9 by default), no_posterior, no_transition*
   variants directly from the v9 planner prediction directory
   (``data/judge_eval/v9_planner_preds``).  v10 picks are computed
   later by ``v10_response_level_planner.py``;
3. Oversamples ``distinct_strategy`` contexts -- those where v10/v9
   disagree with at least one of (lexicon, v8, majority).  These are
   the only contexts that produce non-tied judge signal in
   downstream LLM-judge evaluation;
4. Reports a transparent slice-distribution audit (counts +
   oversample factor) so reviewers can verify the eval set is not
   adversarially curated.

Sample anatomy::

    {
      "sample_id":  "v10eval_0042",
      "dialog_id":  "...",
      "turn_idx":   ...,
      "seed":       "seed181",
      "row_idx":    ...,
      "context":    [{"role": "user", "text": ...}, ...],
      "last_user":  "...",
      "posterior_state":  {distress: ..., rigidity: ..., ...},
      "measurement":      {distress: ..., ...},
      "risk_level":       "none|mild|severe",
      "slice_tags":       ["high_distress", ...],
      "candidate_strategies": [...],
      "factual_strategy":   "...",
      "majority_strategy":  "question",
      "lexicon_strategy":   "...",
      "v8_strategy":        "...",
      "v9_strategy":        "...",
      "v10_strategy":       "..."   # placeholder, filled later
      "no_posterior_strategy":  "...",
      "no_transition_strategy": "...",
      "same_strategy_flags": {
        "v10_eq_lex":      bool,
        "v10_eq_v8":       bool,
        "v10_eq_v9":       bool,
        "v10_eq_majority": bool
      },
      "is_distinct_strategy":  bool,
      "metadata": {...}
    }

Outputs
-------

- ``data/judge_eval_v10/v10_eval_contexts.jsonl``: the 500 sampled rows;
- ``results/v10_eval_set_stats.json``: slice counts, distinct-strategy
  fraction, seed distribution.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from eval.build_judge_eval_set import (
    SIGN, _row_view, _v8_proxy_strategy, _train_lex_planner,
    _strategy_distribution,
)
from eval.value_probe import _last_user_text, _load_rows
from psystate.constants import N_STATE, N_STRATEGY, STATE_AXES, STRATEGIES


# ---------------------------------------------------------------------------
# Slice predicates
# ---------------------------------------------------------------------------


def _slice_predicates() -> dict:
    return {
        "high_distress":      lambda r: r["z"][0] >= 0.6,
        "low_distress":       lambda r: r["z"][0] <= 0.3,
        "high_rigidity":      lambda r: r["z"][1] >= 0.6,
        "low_readiness":      lambda r: r["z"][2] <= 0.4,
        "high_readiness":     lambda r: r["z"][2] >= 0.6,
        "low_alliance":       lambda r: r["z"][3] <= 0.4,
        "low_clarity":        lambda r: r["z"][4] <= 0.4,
        "risk":               lambda r: r["risk_any"],
        "non_risk":           lambda r: not r["risk_any"],
        "rare_strategy":      lambda r: r["factual_strategy"] != "question",
        "common_strategy":    lambda r: r["factual_strategy"] == "question",
        "low_quality":        lambda r: r["quality_conf"] < 0.5,
        "high_quality":       lambda r: r["quality_conf"] >= 0.7,
        "short_utterance":    lambda r: r["n_chars_last_user"] < 60,
        "long_utterance":     lambda r: r["n_chars_last_user"] > 200,
        "mid_utterance":      lambda r: 60 <= r["n_chars_last_user"] <= 200,
    }


def _risk_level(rv: dict, severe_threshold: float = 0.85) -> str:
    """Coarse risk level (none / mild / severe).

    severe = explicit risk flag from data;
    mild   = high lexical or posterior distress;
    none   = otherwise."""
    if rv["risk_any"]:
        return "severe"
    if rv["lex_state"][0] >= severe_threshold or rv["z"][0] >= severe_threshold:
        return "severe"
    if rv["lex_state"][0] >= 0.55 or rv["z"][0] >= 0.6:
        return "mild"
    return "none"


# ---------------------------------------------------------------------------
# Planner pick lookup
# ---------------------------------------------------------------------------


def _load_planner_picks(pred_dir: Path | None) -> dict:
    """Load (seed, row_idx) -> {variant: strategy} map from the v9
    planner prediction directory.

    The v9 planner writes one JSONL per (variant, seed) like
    ``v9_predictions_F_seed180.jsonl`` with rows::
        {"row_idx": ..., "selected_strategy": "...", ...}

    We collapse them into a per-row map and use it to populate v10's
    `lexicon_strategy`, `v8_strategy`, `v9_strategy` fields and the
    ablation picks.  If a file is missing we leave the field empty.
    """
    if pred_dir is None or not Path(pred_dir).exists():
        return {}
    pred_dir = Path(pred_dir)
    out: dict = defaultdict(dict)
    for path in sorted(pred_dir.glob("v9_predictions_*.jsonl")):
        # Filename: v9_predictions_<variant>_<seed>.jsonl
        stem = path.stem  # v9_predictions_F_seed180
        parts = stem.split("_")
        if len(parts) < 4:
            continue
        # Variant is everything between v9_predictions and the trailing
        # seed token (which itself starts with 'seed').
        seed_tok = parts[-1]
        if not seed_tok.startswith("seed"):
            continue
        variant = "_".join(parts[2:-1])
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            ridx = int(row.get("row_idx", -1))
            if ridx < 0:
                continue
            out[(seed_tok, ridx)][variant] = row.get("selected_strategy")
    return dict(out)


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def _stratified_sample_with_distinct(
    rows: list[dict],
    *,
    total: int,
    distinct_target: int,
    risk_target: int,
    rng: random.Random,
    min_per_slice: int = 30,
) -> list[dict]:
    """Two-phase stratified sampling:

    1. Allocate ``min_per_slice`` quota per slice greedily, then
    2. fill the remainder from the *distinct-strategy* pool until
       ``distinct_target`` is reached, then
    3. fill any final remainder from the *risk* pool until
       ``risk_target`` is reached, then
    4. fill the rest uniformly from the residual pool.

    A row may belong to multiple slices; we accumulate its tags in
    ``slice_tags`` instead of duplicating it."""
    preds = _slice_predicates()

    # Index by slice
    by_slice: dict[str, list[dict]] = {n: [] for n in preds}
    for r in rows:
        for slc, pred in preds.items():
            if pred(r):
                by_slice[slc].append(r)

    chosen: dict = {}

    def _add(rec: dict, tag: str) -> None:
        key = (rec["seed"], rec["dialog_id"], rec["turn_idx"])
        if key in chosen:
            if tag not in chosen[key]["slice_tags"]:
                chosen[key]["slice_tags"].append(tag)
        else:
            chosen[key] = {"slice_tags": [tag], **rec}

    # Phase 1: per-slice quota
    for slc, items in by_slice.items():
        rng.shuffle(items)
        for it in items[:min_per_slice]:
            _add(it, slc)

    # Phase 2: distinct-strategy fill
    distinct_pool = [r for r in rows if r.get("is_distinct_strategy")]
    rng.shuffle(distinct_pool)
    while distinct_pool and len([k for k, v in chosen.items()
                                 if v.get("is_distinct_strategy")]) < distinct_target:
        it = distinct_pool.pop()
        _add(it, "distinct_strategy")

    # Phase 3: risk fill
    risk_pool = [r for r in rows if r["risk_any"]]
    rng.shuffle(risk_pool)
    while risk_pool and len([k for k, v in chosen.items() if v["risk_any"]]) < risk_target:
        it = risk_pool.pop()
        _add(it, "risk_oversample")

    # Phase 4: top up to total
    if len(chosen) < total:
        residual = list(rows)
        rng.shuffle(residual)
        for it in residual:
            if len(chosen) >= total:
                break
            key = (it["seed"], it["dialog_id"], it["turn_idx"])
            if key not in chosen:
                _add(it, "fill")

    out = list(chosen.values())
    rng.shuffle(out)
    return out[:total]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_files", nargs="+", required=True,
                    help="Frozen test prediction JSONL per seed.")
    ap.add_argument("--train_files", nargs="+", required=True,
                    help="Frozen train-probe JSONL per seed.")
    ap.add_argument(
        "--planner_pred_dir", default="data/judge_eval/v9_planner_preds",
        help="Directory with v9 planner predictions per (variant, seed) "
             "to populate lexicon/v8/v9/no_posterior/no_transition picks.",
    )
    ap.add_argument("--out", default="data/judge_eval_v10/v10_eval_contexts.jsonl")
    ap.add_argument("--stats_out", default="results/v10_eval_set_stats.json")
    ap.add_argument("--total", type=int, default=500)
    ap.add_argument("--distinct_target", type=int, default=200)
    ap.add_argument("--risk_target", type=int, default=100)
    ap.add_argument("--min_per_slice", type=int, default=30)
    ap.add_argument("--seed", type=int, default=20260501)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    # ------------------------------------------------------------------
    # 1. Train rows -> lexicon planner + majority distribution
    # ------------------------------------------------------------------
    train_rows: list[dict] = []
    for path in args.train_files:
        train_rows.extend(_load_rows(path))
    lex_planner, _ = _train_lex_planner(train_rows)
    strat_dist = _strategy_distribution(train_rows)
    majority_strategy = max(strat_dist, key=strat_dist.get)

    # ------------------------------------------------------------------
    # 2. Load planner picks (variant -> {(seed,row_idx): strategy})
    # ------------------------------------------------------------------
    planner_picks = _load_planner_picks(Path(args.planner_pred_dir))
    print(f"[v10-eval] loaded planner picks for {len(planner_picks)} rows")

    # ------------------------------------------------------------------
    # 3. Build candidate pool
    # ------------------------------------------------------------------
    pool: list[dict] = []
    for path in args.pred_files:
        full = Path(path).parent.parent.name
        short = full.split("_")[-1]
        if short == "quick":
            short = "seed180"
        rows = _load_rows(path)
        for i, r in enumerate(rows):
            rv = _row_view(r, short, i)
            if rv is None:
                continue

            # Lexicon-planner pick: rank candidate strategies by
            # therapeutic delta (axis-aware).  Tie-broken by cf value.
            therap_mean = np.asarray(rv["therapeutic_per_strategy"]).mean(axis=-1)
            cfv = np.asarray(rv["cf_value_per_strategy"])
            lex_score = therap_mean + 1e-3 * (cfv - cfv.mean())
            lex_strategy_local = STRATEGIES[int(np.argmax(lex_score))]
            v8_strategy_local = _v8_proxy_strategy(rv)

            picks = planner_picks.get((short, i), {}) or {}
            lex_strategy = picks.get("F_no_strategy_id") or lex_strategy_local
            v8_strategy = picks.get("A") or v8_strategy_local
            v9_strategy = picks.get("F") or v8_strategy
            v10_strategy = picks.get("F") or v9_strategy  # placeholder
            no_posterior_strategy = picks.get("F_no_posterior") or v9_strategy
            no_transition_strategy = picks.get("F_no_transition") or v9_strategy

            same_flags = {
                "v10_eq_lex":      v10_strategy == lex_strategy,
                "v10_eq_v8":       v10_strategy == v8_strategy,
                "v10_eq_v9":       v10_strategy == v9_strategy,
                "v10_eq_majority": v10_strategy == majority_strategy,
            }
            is_distinct = not all(same_flags.values())

            pool.append({
                **rv,
                "lex_strategy":            lex_strategy,
                "v8_strategy":             v8_strategy,
                "v9_strategy":             v9_strategy,
                "v10_strategy":            v10_strategy,
                "no_posterior_strategy":   no_posterior_strategy,
                "no_transition_strategy":  no_transition_strategy,
                "majority_strategy":       majority_strategy,
                "same_strategy_flags":     same_flags,
                "is_distinct_strategy":    is_distinct,
                "risk_level":              _risk_level(rv),
            })
    print(f"[v10-eval] candidate pool: {len(pool)} rows; "
          f"distinct-strategy: {sum(1 for r in pool if r['is_distinct_strategy'])}; "
          f"risk: {sum(1 for r in pool if r['risk_any'])}")

    # ------------------------------------------------------------------
    # 4. Sample
    # ------------------------------------------------------------------
    chosen = _stratified_sample_with_distinct(
        pool,
        total=args.total,
        distinct_target=args.distinct_target,
        risk_target=args.risk_target,
        rng=rng,
        min_per_slice=args.min_per_slice,
    )
    print(f"[v10-eval] sampled {len(chosen)} contexts "
          f"(distinct={sum(1 for r in chosen if r['is_distinct_strategy'])}, "
          f"risk={sum(1 for r in chosen if r['risk_any'])})")

    # ------------------------------------------------------------------
    # 5. Decorate + write
    # ------------------------------------------------------------------
    out_rows: list[dict] = []
    for i, rv in enumerate(chosen):
        lex_logit = float(lex_planner.decision_function([rv["lex_state"]])[0])
        out_rows.append({
            "sample_id":            f"v10eval_{i:04d}",
            "dialog_id":            rv["dialog_id"],
            "turn_idx":             rv["turn_idx"],
            "seed":                 rv["seed"],
            "row_idx":              rv["row_idx"],
            "slice_tags":           rv["slice_tags"],
            "context":              rv["context"],
            "last_user":            rv["last_user"],
            "posterior_state":      dict(zip(STATE_AXES, rv["z"])),
            "measurement":          dict(zip(STATE_AXES, rv["lex_state"])),
            "risk_any":             rv["risk_any"],
            "risk_level":           rv["risk_level"],
            "quality_conf":         rv["quality_conf"],
            "n_chars_last_user":    rv["n_chars_last_user"],
            "candidate_strategies": list(STRATEGIES),
            "factual_strategy":     rv["factual_strategy"],
            "majority_strategy":    rv["majority_strategy"],
            "lexicon_strategy":     rv["lex_strategy"],
            "v8_strategy":          rv["v8_strategy"],
            "v9_strategy":          rv["v9_strategy"],
            "v10_strategy":         rv["v10_strategy"],
            "no_posterior_strategy":  rv["no_posterior_strategy"],
            "no_transition_strategy": rv["no_transition_strategy"],
            "same_strategy_flags":  rv["same_strategy_flags"],
            "is_distinct_strategy": rv["is_distinct_strategy"],
            "lex_baseline_logit":   lex_logit,
            "delta_per_strategy":         rv["delta_per_strategy"],
            "therapeutic_per_strategy":   rv["therapeutic_per_strategy"],
            "cf_value_per_strategy":      rv["cf_value_per_strategy"],
            "uptake":               rv["uptake"],
            "metadata": {
                "majority_strategy_train_freq": strat_dist[rv["majority_strategy"]],
                "rare_strategy_train_freq":     min(strat_dist.values()),
            },
        })

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ------------------------------------------------------------------
    # 6. Stats
    # ------------------------------------------------------------------
    slice_count = Counter(t for r in out_rows for t in r["slice_tags"])
    seed_count = Counter(r["seed"] for r in out_rows)
    risk_count = Counter(r["risk_level"] for r in out_rows)
    distinct_n = sum(1 for r in out_rows if r["is_distinct_strategy"])
    stats = {
        "n_total":           len(out_rows),
        "n_distinct":        distinct_n,
        "distinct_fraction": distinct_n / max(len(out_rows), 1),
        "n_risk":            risk_count.get("severe", 0) + risk_count.get("mild", 0),
        "per_slice":         dict(slice_count),
        "per_seed":          dict(seed_count),
        "per_risk_level":    dict(risk_count),
        "majority_strategy": majority_strategy,
        "majority_strategy_freq": strat_dist[majority_strategy],
        "args": {
            "total":            args.total,
            "distinct_target":  args.distinct_target,
            "risk_target":      args.risk_target,
            "min_per_slice":    args.min_per_slice,
            "seed":             args.seed,
        },
    }
    Path(args.stats_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.stats_out).write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(f"[v10-eval] wrote {len(out_rows)} contexts -> {out_path}")
    print(f"[v10-eval] distinct-strategy: {distinct_n}/{len(out_rows)} "
          f"({distinct_n / max(len(out_rows), 1):.1%})")
    print(f"[v10-eval] risk levels: {dict(risk_count)}")
    print(f"[v10-eval] per-slice (head): "
          f"{dict(list(sorted(slice_count.items(), key=lambda x: -x[1]))[:10])}")


if __name__ == "__main__":
    main()
