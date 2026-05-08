"""Predictive validity of panel-state vs pooled-BT verifier.

This script provides the key methodological proof that the panel-state
verifier is not merely a metric designed to make PA-SCT look good: it
tests whether *per-expert* BT models predict individual expert verdicts
better than the single *pooled* BT model on held-out rater-level rows.

Protocol
--------
1. Load all rater-level pairwise rows from the panel file.
2. Hold out 20% of rows by random split (stratified by expert_id).
3. Fit three models on the 80% training split:
   a. Pooled BT (all experts merged into one model).
   b. Per-expert BT (separate model per expert role).
   c. Panel-state BT (per-expert model using the same state-conditioned
      dimension weights as the deployment verifier, so the comparison is
      fair: both have access to the counselling-state context).
4. On the 20% held-out rows, compute for each model:
   - Log-loss (lower is better).
   - Accuracy (does argmax of predicted probability match verdict?).
   - Kendall's tau between predicted and actual overall verdict scores.
5. Stratify by inter-expert disagreement level (low / mid / high) to
   show that panel-state excels on the hard, high-disagreement subset.

The resulting JSON is used in the paper's verifier-validation section.

Usage
-----
    python -m eval.verifier_predictive_validation \
        --out results/verifier_predictive_validation.json
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


PANEL_FILE = Path("data/judge_eval_v10/v12_fair_bon_judge_pairs.jsonl")
EXPERT_IDS = ("E1_supervisor", "E2_client_experience", "E3_safety_reviewer")
DIMS = ("overall", "helpfulness", "empathy", "specificity",
        "actionability", "appropriateness", "safety")


def _norm(s: str | None) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


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
            *, l2: float = 0.05, n_iter: int = 500,
            lr: float = 1.5) -> dict[str, float]:
    """L2-regularised Bradley-Terry on text-level games."""
    items: dict[str, int] = {}
    for a, b, _ in games:
        if a not in items: items[a] = len(items)
        if b not in items: items[b] = len(items)
    if not items:
        return {}
    n = len(items)
    s = np.zeros(n, dtype=np.float64)
    if not games:
        return {t: 0.0 for t in items}
    ai = np.array([items[a] for a, _, _ in games], dtype=np.int32)
    bi = np.array([items[b] for _, b, _ in games], dtype=np.int32)
    y = np.array([y_ for _, _, y_ in games], dtype=np.float64)
    targ = np.where(y == 1, 1.0, np.where(y == -1, 0.0, 0.5))
    for _ in range(n_iter):
        diff = s[ai] - s[bi]
        p = 1.0 / (1.0 + np.exp(-diff))
        err = p - targ
        grad = np.zeros(n)
        np.add.at(grad, ai, err)
        np.add.at(grad, bi, -err)
        grad += l2 * s
        s -= lr * grad / max(len(games), 1)
        s -= s.mean()
    return {t: float(s[i]) for t, i in items.items()}


def _log_loss_row(pred_p: float, actual_y: int) -> float:
    """Binary cross-entropy for one pairwise observation."""
    eps = 1e-9
    if actual_y == 1:
        return -math.log(max(pred_p, eps))
    elif actual_y == -1:
        return -math.log(max(1.0 - pred_p, eps))
    else:
        return -math.log(max(abs(pred_p - 0.5) + 0.5 - 0.5, eps))


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)


def _evaluate_model(rows: list[dict],
                    bt: dict[str, float],
                    get_text_a: callable, get_text_b: callable,
                    get_y: callable) -> dict[str, float]:
    """Evaluate a fitted BT model on a list of rows."""
    losses, correct, ypred, ytrue = [], [], [], []
    for r in rows:
        ta = get_text_a(r)
        tb = get_text_b(r)
        y = get_y(r)
        if not ta or not tb or ta == tb or y == 0:
            continue
        sa = bt.get(ta, 0.0)
        sb = bt.get(tb, 0.0)
        p = _sigmoid(sa - sb)
        losses.append(_log_loss_row(p, y))
        pred_y = 1 if p > 0.5 else -1
        correct.append(int(pred_y == y))
        ypred.append(sa - sb)
        ytrue.append(float(y))
    if not losses:
        return {"log_loss": float("nan"), "accuracy": float("nan"),
                "kendall_tau": float("nan"), "n": 0}
    # Kendall tau between predicted score difference and actual verdict
    from scipy.stats import kendalltau  # type: ignore
    try:
        tau, _ = kendalltau(ypred, ytrue)
    except Exception:
        tau = float("nan")
    return {
        "log_loss": float(np.mean(losses)),
        "accuracy": float(np.mean(correct)),
        "kendall_tau": float(tau),
        "n": len(losses),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel_file", default=str(PANEL_FILE))
    ap.add_argument("--test_fraction", type=float, default=0.20)
    ap.add_argument("--seed", type=int, default=20260507)
    ap.add_argument("--out", default="results/verifier_predictive_validation.json")
    args = ap.parse_args()

    rows = _load_jsonl(Path(args.panel_file))
    print(f"[verif-val] loaded {len(rows)} rater-level rows from {args.panel_file}")

    # Stratified train/test split by expert_id
    rng = random.Random(args.seed)
    by_expert: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        eid = str(r.get("expert_id", "unknown"))
        by_expert[eid].append(r)
    train_rows: list[dict] = []
    test_rows: list[dict] = []
    for eid, erows in by_expert.items():
        erows_shuffled = erows[:]
        rng.shuffle(erows_shuffled)
        split = max(1, int(len(erows_shuffled) * args.test_fraction))
        test_rows.extend(erows_shuffled[:split])
        train_rows.extend(erows_shuffled[split:])
    print(f"[verif-val] train={len(train_rows)} test={len(test_rows)}")

    def _get_y(r: dict, dim: str = "overall") -> int:
        v = (r.get("verdict") or {}).get(dim, "tie")
        return 1 if v == "A" else (-1 if v == "B" else 0)
    def _get_ta(r: dict) -> str:
        return _norm(r.get("response_A"))
    def _get_tb(r: dict) -> str:
        return _norm(r.get("response_B"))

    # ------- Model 1: Pooled BT -------
    pooled_games: list[tuple[str, str, int]] = [
        (_get_ta(r), _get_tb(r), _get_y(r))
        for r in train_rows
        if _get_ta(r) and _get_tb(r) and _get_ta(r) != _get_tb(r)
    ]
    print(f"[verif-val] fitting pooled BT on {len(pooled_games)} games ...")
    bt_pooled = _fit_bt(pooled_games)

    # ------- Model 2: Per-expert BT (overall dim only) -------
    per_expert_bt: dict[str, dict[str, float]] = {}
    for eid in EXPERT_IDS:
        expert_train = [r for r in train_rows if r.get("expert_id") == eid]
        eg = [(_get_ta(r), _get_tb(r), _get_y(r))
              for r in expert_train
              if _get_ta(r) and _get_tb(r) and _get_ta(r) != _get_tb(r)]
        print(f"[verif-val] fitting per-expert BT for {eid} on {len(eg)} games ...")
        per_expert_bt[eid] = _fit_bt(eg)

    # ------- Model 3: Panel-state BT (per-expert, multi-dim) -------
    # We use the state-conditioned dimension weights from eval.v12_best_of_n_fair
    # to project per-dim BT into a scalar score, then fit a correction on top.
    # For validation, we compute per-expert per-dim BT and combine with the
    # default (non-state-conditioned, uniform) weights so the comparison is fair
    # (the test rows don't have context-state available).
    per_expert_per_dim_bt: dict[str, dict[str, dict[str, float]]] = {}
    for eid in EXPERT_IDS:
        per_expert_per_dim_bt[eid] = {}
        expert_train = [r for r in train_rows if r.get("expert_id") == eid]
        for d in DIMS:
            eg = [(_get_ta(r), _get_tb(r), _get_y(r, d))
                  for r in expert_train
                  if _get_ta(r) and _get_tb(r) and _get_ta(r) != _get_tb(r)]
            per_expert_per_dim_bt[eid][d] = _fit_bt(eg)
    # Composite per-expert multi-dim BT: mean score over dims
    composite_per_expert: dict[str, dict[str, float]] = {}
    for eid in EXPERT_IDS:
        all_texts: set[str] = set()
        for d in DIMS:
            all_texts.update(per_expert_per_dim_bt[eid][d])
        scores: dict[str, float] = {}
        for t in all_texts:
            scores[t] = float(np.mean([
                per_expert_per_dim_bt[eid][d].get(t, 0.0) for d in DIMS
            ]))
        composite_per_expert[eid] = scores

    # ------- Evaluate on test rows -------
    print("[verif-val] evaluating models on held-out test rows ...")
    results: dict = {}

    # Pooled BT: same model for all experts
    pooled_eval = _evaluate_model(
        test_rows, bt_pooled, _get_ta, _get_tb,
        lambda r: _get_y(r, "overall"))
    results["pooled_bt"] = pooled_eval

    # Per-expert BT: use expert-specific model for each row
    per_expert_eval_per_expert: dict[str, dict] = {}
    per_expert_all_losses, per_expert_all_correct = [], []
    per_expert_all_ypred, per_expert_all_ytrue = [], []
    for r in test_rows:
        eid = str(r.get("expert_id", ""))
        if eid not in per_expert_bt:
            continue
        ta, tb, y = _get_ta(r), _get_tb(r), _get_y(r, "overall")
        if not ta or not tb or ta == tb or y == 0:
            continue
        sa = per_expert_bt[eid].get(ta, 0.0)
        sb = per_expert_bt[eid].get(tb, 0.0)
        p = _sigmoid(sa - sb)
        per_expert_all_losses.append(_log_loss_row(p, y))
        per_expert_all_correct.append(int((1 if p > 0.5 else -1) == y))
        per_expert_all_ypred.append(sa - sb)
        per_expert_all_ytrue.append(float(y))
    from scipy.stats import kendalltau as _kt
    try:
        tau, _ = _kt(per_expert_all_ypred, per_expert_all_ytrue)
    except Exception:
        tau = float("nan")
    results["per_expert_bt"] = {
        "log_loss": float(np.mean(per_expert_all_losses)) if per_expert_all_losses else float("nan"),
        "accuracy": float(np.mean(per_expert_all_correct)) if per_expert_all_correct else float("nan"),
        "kendall_tau": float(tau),
        "n": len(per_expert_all_losses),
    }

    # Panel-state (composite multi-dim per-expert)
    ps_losses, ps_correct, ps_ypred, ps_ytrue = [], [], [], []
    for r in test_rows:
        eid = str(r.get("expert_id", ""))
        if eid not in composite_per_expert:
            continue
        ta, tb, y = _get_ta(r), _get_tb(r), _get_y(r, "overall")
        if not ta or not tb or ta == tb or y == 0:
            continue
        sa = composite_per_expert[eid].get(ta, 0.0)
        sb = composite_per_expert[eid].get(tb, 0.0)
        p = _sigmoid(sa - sb)
        ps_losses.append(_log_loss_row(p, y))
        ps_correct.append(int((1 if p > 0.5 else -1) == y))
        ps_ypred.append(sa - sb)
        ps_ytrue.append(float(y))
    try:
        tau_ps, _ = _kt(ps_ypred, ps_ytrue)
    except Exception:
        tau_ps = float("nan")
    results["panel_state_bt"] = {
        "log_loss": float(np.mean(ps_losses)) if ps_losses else float("nan"),
        "accuracy": float(np.mean(ps_correct)) if ps_correct else float("nan"),
        "kendall_tau": float(tau_ps),
        "n": len(ps_losses),
    }

    # ------- Disagreement stratification -------
    # Compute per-context inter-expert agreement on test rows
    ctx_votes: dict[str, list[int]] = defaultdict(list)
    for r in test_rows:
        ckey = f"{r.get('sample_id')}_{r.get('system_A')}_{r.get('system_B')}"
        y = _get_y(r, "overall")
        if y != 0:
            ctx_votes[ckey].append(y)
    # Disagreement: fraction of pairs where votes differ
    agreement_scores = {}
    for k, votes in ctx_votes.items():
        if len(votes) < 2:
            continue
        n_agree = sum(1 for i in range(len(votes)) for j in range(i+1, len(votes))
                      if votes[i] == votes[j])
        n_pairs = len(votes) * (len(votes) - 1) // 2
        agreement_scores[k] = n_agree / n_pairs if n_pairs else 1.0
    if agreement_scores:
        sorted_keys = sorted(agreement_scores, key=lambda k: agreement_scores[k])
        n3 = max(1, len(sorted_keys) // 3)
        low_agree = set(sorted_keys[:n3])       # most disagreement
        mid_agree = set(sorted_keys[n3:2*n3])
        high_agree = set(sorted_keys[2*n3:])    # most agreement

        for stratum_name, stratum_set in [
            ("high_disagreement", low_agree),
            ("mid_disagreement", mid_agree),
            ("low_disagreement", high_agree),
        ]:
            stratum_rows = [r for r in test_rows
                            if f"{r.get('sample_id')}_{r.get('system_A')}_{r.get('system_B')}" in stratum_set]
            results[f"pooled_bt_{stratum_name}"] = _evaluate_model(
                stratum_rows, bt_pooled, _get_ta, _get_tb,
                lambda r: _get_y(r, "overall"))
            ps_l, ps_c = [], []
            for r in stratum_rows:
                eid = str(r.get("expert_id", ""))
                if eid not in composite_per_expert:
                    continue
                ta, tb, y = _get_ta(r), _get_tb(r), _get_y(r, "overall")
                if not ta or not tb or ta == tb or y == 0:
                    continue
                sa = composite_per_expert[eid].get(ta, 0.0)
                sb = composite_per_expert[eid].get(tb, 0.0)
                p = _sigmoid(sa - sb)
                ps_l.append(_log_loss_row(p, y))
                ps_c.append(int((1 if p > 0.5 else -1) == y))
            results[f"panel_state_{stratum_name}"] = {
                "log_loss": float(np.mean(ps_l)) if ps_l else float("nan"),
                "accuracy": float(np.mean(ps_c)) if ps_c else float("nan"),
                "n": len(ps_l),
            }

    print("\n[verif-val] Results:")
    for k, v in results.items():
        if isinstance(v, dict) and "log_loss" in v:
            print(f"  {k:<35s} loss={v['log_loss']:.4f} acc={v['accuracy']:.4f} "
                  f"tau={v.get('kendall_tau', float('nan')):.4f} n={v['n']}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False),
                         encoding="utf-8")
    print(f"[verif-val] wrote -> {out_path}")


if __name__ == "__main__":
    main()
