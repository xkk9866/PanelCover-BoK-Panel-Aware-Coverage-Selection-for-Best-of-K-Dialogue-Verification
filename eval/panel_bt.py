"""Per-expert Bradley-Terry models for panel-aware Best-of-K planning.

The fair-BoK panel comprises three independent qualified counsellor experts
(``E1_supervisor``, ``E2_client_experience``, ``E3_safety_reviewer``).
Their pairwise verdicts on the 12 panel pairs (12 000 rater-level rows
per expert) are persisted in
``data/judge_eval_v10/v12_fair_bon_judge_pairs.jsonl``.

This module fits one L2-regularised Bradley-Terry model per expert and
per dimension over the per-rater verdicts.  The output

    {expert_id -> {dim -> {response_text -> bt_logit}}}

is cached in ``results/panel_bt.json`` and consumed by

* the **Panel-Aware Stochastic Verifier** in ``eval/v12_best_of_n_fair.py``
  (mode ``panel-stochastic``), and
* the **Panel-Aware SCT planner** in ``eval/build_pasct_topk.py``,
  which runs greedy submodular maximisation on the panel-aware
  expected-coverage objective

      F(S | x) = E_e[ max_{a in S} BT_e^overall(r(x, a)) ].

The same cache is also used by ``eval/panel_aware_winrate.py`` to compute
the panel-aware BT-projected pairwise winrate (a mathematically tighter
counterpart of the canonical BT-overall winrate that respects panel
disagreement).

The script is intentionally self-contained so the panel BT is recomputed
deterministically from raw verdicts and is never blended with any other
panel.

Usage
-----

.. code-block::

   python -m eval.panel_bt
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


PANEL_PAIRS_FILE = Path(
    "data/judge_eval_v10/v12_fair_bon_judge_pairs.jsonl")

DIMS = ("overall", "helpfulness", "empathy", "specificity",
        "actionability", "appropriateness", "safety")

EXPERT_IDS = ("E1_supervisor", "E2_client_experience", "E3_safety_reviewer")


def _norm_text(s: str | None) -> str:
    if s is None:
        return ""
    return re.sub(r"\s+", " ", s.strip())


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
    """L2-BT solver shared with ``bt_winrate_proxy._fit_bt``.

    ``games`` is a list of ``(text_a, text_b, y)`` triples with
    ``y in {1, 0, -1}`` for win/tie/loss of A.  Returns a ``text -> logit``
    mapping anchored at mean=0."""
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


def fit_panel_bt() -> dict:
    """Return a nested dict ``{expert_id: {dim: {text: bt_logit}}}``."""
    rows = _load_jsonl(PANEL_PAIRS_FILE)
    if not rows:
        raise FileNotFoundError(
            f"panel pair file is empty/missing: {PANEL_PAIRS_FILE}")
    games_per_expert: dict[str, dict[str, list]] = {
        e: {d: [] for d in DIMS} for e in EXPERT_IDS
    }
    n_skipped = 0
    n_kept = 0
    for r in rows:
        eid = str(r.get("expert_id", ""))
        if eid not in games_per_expert:
            n_skipped += 1
            continue
        text_a = _norm_text(r.get("response_A"))
        text_b = _norm_text(r.get("response_B"))
        if not text_a or not text_b or text_a == text_b:
            n_skipped += 1
            continue
        verd = r.get("verdict") or {}
        for d in DIMS:
            v = verd.get(d, "tie")
            if v == "A":
                y = 1
            elif v == "B":
                y = -1
            else:
                y = 0
            games_per_expert[eid][d].append((text_a, text_b, y))
        n_kept += 1
    print(f"[panel-bt] kept={n_kept} skipped={n_skipped} (text-empty/duplicate)")
    out: dict[str, dict[str, dict[str, float]]] = {}
    for eid in EXPERT_IDS:
        out[eid] = {}
        for d in DIMS:
            g = games_per_expert[eid][d]
            bt = _fit_bt(g)
            out[eid][d] = bt
            print(f"[panel-bt] {eid:<24s} dim={d:<14s} "
                  f"games={len(g):>6d} items={len(bt):>5d}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/panel_bt.json")
    args = ap.parse_args()
    panel_bt = fit_panel_bt()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(
        json.dumps(panel_bt, indent=2, ensure_ascii=False),
        encoding="utf-8")
    print(f"[panel-bt] wrote -> {args.out}")


if __name__ == "__main__":
    main()
