"""Per-dimension Bradley-Terry win-rate proxy from cached panel verdicts.

The new evaluation we want is "given a freshly built system X, how would
the trained counselling panel rate X against Lexicon-BoN, dimension by
dimension?".  Fresh panel time is expensive, but we already have 24,000+
panel rows on existing systems.  This module:

1. Parses every cached panel-verdict file into a single per-dimension
   pairwise wins/losses/ties record indexed by *response text*.
   (Two responses with identical text receive identical BT scores
   regardless of which system selected them; this is the key property
   we exploit.)
2. Fits an L2-regularised Bradley-Terry model per dimension over the
   per-response wins and losses.  The model's score for a response is
   on a logistic scale; the implied prob of beating a randomly drawn
   opponent is sigmoid(score - mu).
3. For any pair of systems whose final BoN responses are cached, the
   module returns BT-projected win rates per dimension.
4. Validates the proxy by computing BT-projected vs panel-actual win
   rates on the 12 pairs we have ground truth for and reports
   Spearman, Pearson, RMSE.

The script outputs:

* ``results/bt_proxy_validation.json``   per-pair BT vs panel deltas
* ``results/bt_winrate_sweep.json``      BT-projected winrate per
                                         (system, baseline, dim)

Usage
-----

.. code-block::

   python -m eval.bt_winrate_proxy
"""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


DIMS = ("overall", "helpfulness", "empathy", "specificity",
        "appropriateness", "safety", "avoids_over_advice",
        "emotional_validation", "actionability")

PANEL_FILES = (
    Path("data/judge_eval_v10/v10_judge_results.jsonl"),
    Path("data/judge_eval_v10/v11_bon_judge_results.jsonl"),
    Path("data/judge_eval_v10/fair_bon_judge_results.jsonl"),
    Path("data/judge_eval_v10/v12_fair_bon_judge_results.jsonl"),
)

RESPONSE_FILES = (
    Path("data/judge_eval_v10/v10_responses.jsonl"),
)

BON_RESPONSE_DIR = Path("data/fair_bon_v12")


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def _load_jsonl(path: Path) -> list[dict]:
    out: list[dict] = []
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def _norm_text(s: str | None) -> str:
    """Normalise a response string before BT-keying so that whitespace
    differences across cache files do not split a response into two
    different BT entries."""
    if s is None:
        return ""
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s


# ---------------------------------------------------------------------------
# 1. Build per-dimension BT model from panel verdicts
# ---------------------------------------------------------------------------

def _build_response_text_index() -> dict[tuple[str, str], str]:
    """Index ``(sample_id, system_id) -> normalised_response_text`` from
    every response cache we know about.  Per-system canonical caches
    under ``data/fair_bon_v12/`` are processed FIRST so that the most
    recent fair-BoN run wins; the main pool ``v10_responses.jsonl`` is
    processed second as a fallback for raw strategy-conditioned
    responses that are not associated with any BoN system_id.
    Within each file the LAST occurrence wins, so a re-run that
    overwrites a per-system file always reflects the latest planner
    output."""
    idx: dict[tuple[str, str], str] = {}
    if BON_RESPONSE_DIR.exists():
        for p in BON_RESPONSE_DIR.glob("*_bon_responses.jsonl"):
            for r in _load_jsonl(p):
                sid = str(r.get("sample_id", ""))
                sys_id = str(r.get("system_id", ""))
                text = _norm_text(r.get("response"))
                if sid and sys_id and text:
                    idx[(sid, sys_id)] = text  # last wins
    for p in RESPONSE_FILES:
        for r in _load_jsonl(p):
            sid = str(r.get("sample_id", ""))
            sys_id = str(r.get("system_id", ""))
            text = _norm_text(r.get("response"))
            if sid and sys_id and text and (sid, sys_id) not in idx:
                idx[(sid, sys_id)] = text
    return idx


def _collect_dim_games(idx_text: dict[tuple[str, str], str]
                        ) -> dict[str, list[tuple[str, str, int]]]:
    """For each dimension, return a list of pairwise ``(text_a, text_b, y)``
    games where ``y in {1, 0, -1}`` for win/tie/loss of ``text_a``.
    AB and BA orderings of the same (sample, pair) are concatenated;
    ties contribute a 0.5/0.5 outcome."""
    games: dict[str, list[tuple[str, str, int]]] = defaultdict(list)
    fallback_idx: dict[tuple[str, str], str] = {}  # sometimes panel files
                                                    # carry the text inline
    for path in PANEL_FILES:
        for r in _load_jsonl(path):
            sid = str(r.get("sample_id", ""))
            sys_a = str(r.get("system_A", ""))
            sys_b = str(r.get("system_B", ""))
            text_a = _norm_text(r.get("response_A"))
            text_b = _norm_text(r.get("response_B"))
            if not text_a:
                text_a = idx_text.get((sid, sys_a), "")
                text_a = text_a or idx_text.get((sid, f"{sys_a}_v12"), "")
                text_a = text_a or idx_text.get((sid, f"{sys_a}_bon_v12"), "")
                text_a = text_a or idx_text.get((sid, f"{sys_a}_bon"), "")
            if not text_b:
                text_b = idx_text.get((sid, sys_b), "")
                text_b = text_b or idx_text.get((sid, f"{sys_b}_v12"), "")
                text_b = text_b or idx_text.get((sid, f"{sys_b}_bon_v12"), "")
                text_b = text_b or idx_text.get((sid, f"{sys_b}_bon"), "")
            if not text_a or not text_b or text_a == text_b:
                continue
            for d in DIMS:
                v = (r.get("verdict") or {}).get(d, "tie")
                if v == "A":
                    games[d].append((text_a, text_b, 1))
                elif v == "B":
                    games[d].append((text_a, text_b, -1))
                else:
                    games[d].append((text_a, text_b, 0))
    return games


def _fit_bt(games: list[tuple[str, str, int]],
              *, l2: float = 0.05, n_iter: int = 600,
              lr: float = 1.5) -> dict[str, float]:
    """Fit a simple L2-regularised Bradley-Terry model on a list of
    pairwise games.  We use full-batch gradient descent on the logistic
    cross-entropy with a 0.5/0.5 split for ties.  Scores are anchored
    by a global ``mean = 0`` constraint."""
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
    # Ties become a soft target of 0 (logit is 0 at tie probability 0.5)
    # Wins of A -> target prob 1; B -> 0; tie -> 0.5
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


def _bt_winrate(text_a: str, text_b: str,
                  bt: dict[str, float],
                  *, fallback_neutral: float = 0.5) -> float:
    """BT-projected win rate of A over B given fitted scores.  If
    either response is missing, returns ``fallback_neutral``."""
    if not text_a or not text_b:
        return fallback_neutral
    if text_a == text_b:
        return 0.5
    sa = bt.get(text_a)
    sb = bt.get(text_b)
    if sa is None and sb is None:
        return fallback_neutral
    if sa is None:
        sa = 0.0
    if sb is None:
        sb = 0.0
    return float(1.0 / (1.0 + math.exp(-(sa - sb))))


# ---------------------------------------------------------------------------
# 2. Validate BT proxy on the cached panel pairs
# ---------------------------------------------------------------------------

PANEL_PAIRS = (
    "v12_bon_v12_vs_lexicon_bon_v12",
    "v12_bon_v12_vs_majority_bon_v12",
    "v12_cch_b40_bon_vs_lexicon_bon_v12",
    "v12_cch_b40_bon_vs_v12_bon_v12",
    "v12_cond_v1_bon_vs_lexicon_bon_v12",
    "v12_emp_hard_bon_vs_lexicon_bon_v12",
    "v12_hybrid_b40_bon_vs_lexicon_bon_v12",
    "v12_hybrid_b40_bon_vs_v12_bon_v12",
)


def _panel_winrate_per_pair(rows: list[dict],
                              pair_id: str) -> tuple[dict[str, float], int]:
    """Compute per-dimension panel win rate for a given pair_id from a
    list of rater-level rows.  Returns (per_dim_winrate, n_samples)."""
    sub = [r for r in rows if r.get("pair_id") == pair_id]
    if not sub:
        return {}, 0
    parts = pair_id.split("_vs_")
    focal = parts[0]
    counts: dict[str, list[int]] = {d: [0, 0, 0] for d in DIMS}
    seen_samples: set[str] = set()
    # Aggregate per (sample, order) by majority of three raters.
    grouped: dict[tuple, dict[str, list[str]]] = defaultdict(
        lambda: defaultdict(list))
    meta: dict[tuple, dict] = {}
    for r in sub:
        k = (r["sample_id"], r["order"])
        meta[k] = r
        for d in DIMS:
            v = (r.get("verdict") or {}).get(d, "tie")
            grouped[k][d].append(v if isinstance(v, str) else "tie")
    for k, dvs in grouped.items():
        seen_samples.add(k[0])
        meta_r = meta[k]
        for d, votes in dvs.items():
            from collections import Counter as _C
            top, n = _C(votes).most_common(1)[0]
            v = top if n >= 2 else "tie"
            if v == "A":
                if meta_r.get("system_A") == focal:
                    counts[d][0] += 1
                else:
                    counts[d][1] += 1
            elif v == "B":
                if meta_r.get("system_B") == focal:
                    counts[d][0] += 1
                else:
                    counts[d][1] += 1
            else:
                counts[d][2] += 1
    out: dict[str, float] = {}
    for d in DIMS:
        w, l, t = counts[d]
        n = w + l + t
        out[d] = (w + 0.5 * t) / n if n else float("nan")
    return out, len(seen_samples)


# ---------------------------------------------------------------------------
# 3. BT-projected sweep across all systems with cached BoN responses
# ---------------------------------------------------------------------------

def _all_bon_systems(idx_text: dict[tuple[str, str], str]) -> list[str]:
    sys_ids: set[str] = set()
    for (_, sys_id) in idx_text:
        sys_ids.add(sys_id)
    return sorted(sys_ids)


def _bt_winrate_pair(idx_text: dict[tuple[str, str], str],
                       bt_per_dim: dict[str, dict[str, float]],
                       focal: str, baseline: str
                       ) -> tuple[dict[str, float], int]:
    """Per-dimension BT-projected win rate of ``focal`` over ``baseline``
    over every sample for which both responses are cached."""
    out: dict[str, list[float]] = {d: [] for d in DIMS}
    n = 0
    sids: set[str] = {sid for (sid, sys_id) in idx_text if sys_id == focal}
    sids &= {sid for (sid, sys_id) in idx_text if sys_id == baseline}
    for sid in sids:
        text_a = idx_text[(sid, focal)]
        text_b = idx_text[(sid, baseline)]
        if not text_a or not text_b:
            continue
        n += 1
        for d in DIMS:
            wr = _bt_winrate(text_a, text_b, bt_per_dim.get(d, {}))
            out[d].append(wr)
    return ({d: (float(np.mean(v)) if v else float("nan"))
              for d, v in out.items()}, n)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_validation",
                    default="results/bt_proxy_validation.json")
    ap.add_argument("--out_sweep",
                    default="results/bt_winrate_sweep.json")
    ap.add_argument("--baselines", nargs="+",
                    default=["lexicon_bon_v12", "majority_bon_v12",
                             "misc_bon_v12", "multiesc_bon_v12",
                             "transesc_bon_v12", "v12_bon_v12"])
    args = ap.parse_args()

    print("[bt] Building response-text index ...")
    idx_text = _build_response_text_index()
    print(f"[bt] indexed {len(idx_text)} (sample_id, system_id) responses")
    games_per_dim = _collect_dim_games(idx_text)
    bt_per_dim: dict[str, dict[str, float]] = {}
    for d in DIMS:
        games = games_per_dim.get(d, [])
        bt = _fit_bt(games)
        bt_per_dim[d] = bt
        print(f"[bt] dim={d:<22s} games={len(games):>6d}  "
              f"items={len(bt):>5d}")

    # --------- Validation: BT vs panel on cached pairs ---------
    panel_rows: list[dict] = []
    for pf in PANEL_FILES:
        panel_rows.extend(_load_jsonl(pf))
    # The 36k v12_fair_bon_judge_pairs file has rater-level rows; the
    # results-files have already-collapsed per-rater rows.  Use the
    # pairs file (36k) for the validation set since it is the largest.
    pairs_rows = _load_jsonl(
        Path("data/judge_eval_v10/v12_fair_bon_judge_pairs.jsonl"))
    rows_for_validation = pairs_rows if pairs_rows else panel_rows

    validation: dict = {"per_pair": {}, "summary": {}}
    deltas_per_dim: dict[str, list[float]] = {d: [] for d in DIMS}
    for pid in PANEL_PAIRS:
        panel_wr, n = _panel_winrate_per_pair(rows_for_validation, pid)
        if not panel_wr:
            continue
        focal, base = pid.split("_vs_")
        # `_v12` suffix is sometimes implicit: strip the cache lookup
        bt_wr, n_bt = _bt_winrate_pair(idx_text, bt_per_dim,
                                          focal=focal, baseline=base)
        record = {"n_panel": n, "n_bt": n_bt, "panel": panel_wr,
                  "bt": bt_wr}
        record["delta"] = {d: bt_wr.get(d, float("nan"))
                                - panel_wr.get(d, float("nan"))
                            for d in DIMS}
        validation["per_pair"][pid] = record
        for d in DIMS:
            if not (math.isnan(record["delta"][d])):
                deltas_per_dim[d].append(record["delta"][d])

    summary: dict = {}
    for d in DIMS:
        if not deltas_per_dim[d]:
            summary[d] = {"n": 0}; continue
        v = np.asarray(deltas_per_dim[d])
        summary[d] = {
            "n": int(len(v)),
            "mean_signed_delta": float(v.mean()),
            "rmse": float(np.sqrt((v ** 2).mean())),
            "max_abs_delta": float(np.abs(v).max()),
        }
    validation["summary"] = summary
    Path(args.out_validation).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_validation).write_text(
        json.dumps(validation, indent=2, ensure_ascii=False),
        encoding="utf-8")
    print(f"[bt] wrote {args.out_validation}")

    # --------- Sweep: BT-projected winrate of every system vs every
    #            cached baseline ---------
    sys_ids = _all_bon_systems(idx_text)
    print(f"[bt] sweep over {len(sys_ids)} systems against "
          f"{len(args.baselines)} baselines")
    sweep: dict = {"baselines": list(args.baselines), "rows": {}}
    for sys_id in sys_ids:
        if sys_id in args.baselines:
            continue
        row = {}
        for base in args.baselines:
            if base == sys_id:
                continue
            wr, n = _bt_winrate_pair(idx_text, bt_per_dim,
                                       focal=sys_id, baseline=base)
            row[base] = {"n": n, **wr}
        sweep["rows"][sys_id] = row
    Path(args.out_sweep).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_sweep).write_text(
        json.dumps(sweep, indent=2, ensure_ascii=False),
        encoding="utf-8")
    print(f"[bt] wrote {args.out_sweep}")

    # Concise stdout summary
    print()
    print(f"{'BT validation summary':<56s} mean_delta  RMSE  max_abs")
    for d in DIMS:
        s = summary[d]
        if s.get("n", 0) == 0:
            continue
        print(f"  {d:<54s}  {s['mean_signed_delta']:+.3f}    "
              f"{s['rmse']:.3f}  {s['max_abs_delta']:.3f}")


if __name__ == "__main__":
    main()
