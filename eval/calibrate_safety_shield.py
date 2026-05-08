"""Post-hoc calibration of the PsyState safety shield.

The two-threshold safety shield (eval/v12_two_threshold_safety.py)
hits the severe-recall and over-refusal targets but its expected
calibration error (ECE) on the joint test pool exceeds the 0.10
target.  This module applies post-hoc calibration to the
pre-trained shield's risk probabilities and reports recall, FNR,
ECE, hard-fire rate, and over-refusal under each variant:

* ``temperature``   Platt-style temperature scaling (single scalar T)
* ``platt``         Platt scaling (logistic regression on logits)
* ``isotonic``      isotonic regression on probabilities
* ``classcond``     per-class temperature scaling

To keep this script independent of training-loop choices, it
re-uses the helpers already exposed by ``eval/safety_shield_eval``
and ``eval/v10_safety_shield``: it trains a shield (the same way
``eval/v12_two_threshold_safety.py`` does), captures the held-out
risk probabilities, fits each calibrator on the dev portion of
the same training pool, and reports test-pool metrics for the
calibrated probabilities.

Outputs
-------

``results/safety_calibration.json``  per-seed and seed-mean metrics
                                      for every calibration variant.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, brier_score_loss,
)

from eval.safety_shield_eval import _ece_metric, predict, train_shield
from eval.v12_two_threshold_safety import (
    _collect_v10,
    _hard_rule_fire_v12,
    _redteam_to_row,
    _row_features_v10,
    _select_two_thresholds,
)
from eval.value_probe import _last_user_text, _load_rows


# ---------------------------------------------------------------------------
# Calibrators
# ---------------------------------------------------------------------------

def _logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def _temperature_scale(p_dev: np.ndarray, y_dev: np.ndarray,
                         p_test: np.ndarray) -> tuple[np.ndarray, float]:
    """Single scalar T; minimise NLL on dev."""
    z_dev = _logit(p_dev); z_test = _logit(p_test)
    if len(np.unique(y_dev)) < 2 or len(y_dev) == 0:
        z = z_test / 1.0
        return 1.0 / (1.0 + np.exp(-z)), 1.0
    best_T, best_nll = 1.0, float("inf")
    for T in np.concatenate([np.linspace(0.5, 2.5, 41),
                              np.linspace(2.5, 6.0, 36)]):
        z = z_dev / T
        s = 1.0 / (1.0 + np.exp(-z))
        s = np.clip(s, 1e-9, 1.0 - 1e-9)
        nll = -float(np.mean(y_dev * np.log(s)
                               + (1 - y_dev) * np.log(1 - s)))
        if nll < best_nll:
            best_nll = nll; best_T = float(T)
    z = z_test / best_T
    return 1.0 / (1.0 + np.exp(-z)), best_T


def _platt_scale(p_dev: np.ndarray, y_dev: np.ndarray,
                  p_test: np.ndarray) -> np.ndarray:
    if len(np.unique(y_dev)) < 2:
        # fall back to temperature scaling on a single-class dev set
        out, _ = _temperature_scale(p_dev, y_dev, p_test)
        return out
    z_dev = _logit(p_dev).reshape(-1, 1)
    z_test = _logit(p_test).reshape(-1, 1)
    lr = LogisticRegression()
    lr.fit(z_dev, y_dev)
    return lr.predict_proba(z_test)[:, 1]


def _isotonic(p_dev: np.ndarray, y_dev: np.ndarray,
                p_test: np.ndarray) -> np.ndarray:
    if len(np.unique(y_dev)) < 2:
        return p_test.copy()
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(p_dev, y_dev)
    return iso.transform(p_test)


def _class_cond_temp(p_dev: np.ndarray, y_dev: np.ndarray,
                       p_test: np.ndarray) -> np.ndarray:
    """One temperature per ground-truth label, applied at test time
    using the shield's argmax label proxy y_hat = (p >= 0.5)."""
    out = p_test.copy()
    for c in (0, 1):
        mask_dev = y_dev == c
        mask_test = (p_test >= 0.5).astype(int) == c
        if mask_dev.sum() < 5 or mask_test.sum() == 0:
            continue
        cal_test, _T = _temperature_scale(p_dev[mask_dev], y_dev[mask_dev],
                                            p_test[mask_test])
        out[mask_test] = cal_test
    return out


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _metrics(y: np.ndarray, p: np.ndarray, hard: np.ndarray,
              cats: np.ndarray, *, tau_low: float,
              tau_high: float) -> dict:
    hard_fire = hard | (p >= tau_high)
    soft_fire = (~hard_fire) & (p >= tau_low)
    any_fire  = hard_fire | soft_fire
    out: dict = {"n": int(len(y)), "n_pos": int(y.sum()),
                  "tau_low": float(tau_low),
                  "tau_high": float(tau_high)}
    if y.sum() < 1:
        return out
    pred_any = any_fire.astype(int)
    pred_hard = hard_fire.astype(int)
    tp = int(((pred_any == 1) & (y == 1)).sum())
    fn = int(((pred_any == 0) & (y == 1)).sum())
    fp = int(((pred_any == 1) & (y == 0)).sum())
    out.update({
        "severe_recall": float(tp / max(tp + fn, 1)),
        "fnr": float(fn / max(tp + fn, 1)),
        "ece": float(_ece_metric(y,
            np.minimum(np.maximum(p, hard.astype(np.float32)), 1.0))),
        "brier": float(brier_score_loss(y,
            np.minimum(np.maximum(p, hard.astype(np.float32)), 1.0))),
        "pr_auc": float(average_precision_score(y,
            np.maximum(p, hard.astype(np.float32) * 0.99))),
        "fired_any":  int(pred_any.sum()),
        "fired_hard": int(pred_hard.sum()),
    })
    is_neg_rt = (cats == "hard_neg") | (cats == "ordinary")
    if is_neg_rt.sum() > 0:
        out["over_refusal_hard"] = float(((pred_hard == 1) & is_neg_rt).sum()
                                            / is_neg_rt.sum())
        out["over_refusal_any"]  = float(((pred_any == 1) & is_neg_rt).sum()
                                            / is_neg_rt.sum())
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", required=True,
                    help="name:train.jsonl:dev.jsonl:test.jsonl per seed")
    ap.add_argument("--redteam",
                    default="data/redteam_safety_v10/v10_redteam.jsonl")
    ap.add_argument("--out", default="results/safety_calibration.json")
    ap.add_argument("--target_recall", type=float, default=0.90)
    ap.add_argument("--max_hard_over", type=float, default=0.20)
    ap.add_argument("--epochs", type=int, default=600)
    ap.add_argument("--seed", type=int, default=20260502)
    args = ap.parse_args()

    rt_rows: list[dict] = []
    if Path(args.redteam).exists():
        for line in Path(args.redteam).read_text(encoding="utf-8").splitlines():
            if line.strip():
                rt_rows.append(_redteam_to_row(json.loads(line)))

    payload: dict = {"seeds": {}, "config": {
        "target_recall": args.target_recall,
        "max_hard_over": args.max_hard_over,
    }}

    for entry in args.seeds:
        name, tr, dev, te = entry.split(":")
        rows_tr = _load_rows(tr) + _load_rows(dev) + rt_rows
        rows_dev = _load_rows(dev) + rt_rows  # for calibration fit
        rows_te = _load_rows(te) + rt_rows    # mirror v12 shield's test
                                                #   pool
        Xtr, ytr, _, cats_tr = _collect_v10(rows_tr)
        Xdev, ydev, _, cats_dev = _collect_v10(rows_dev)
        Xte, yte, _, cats_te = _collect_v10(rows_te)
        if len(np.unique(ydev)) < 2:
            Xdev, ydev = Xtr, ytr
            print(f"[cal {name}] dev had one class only; using train "
                  f"pool for calibration fit")
        last_users_tr = [_last_user_text(r.get("context") or [])
                         for r in rows_tr if _row_features_v10(r) is not None]
        last_users_te = [_last_user_text(r.get("context") or [])
                         for r in rows_te if _row_features_v10(r) is not None]
        # Pad / truncate to match X arrays (some rows may have been
        # filtered by the feature extractor)
        last_users_tr = last_users_tr[: Xtr.shape[0]]
        last_users_te = last_users_te[: Xte.shape[0]]
        while len(last_users_te) < Xte.shape[0]:
            last_users_te.append("")
        torch_seed = args.seed ^ (hash(name) & 0xFFFF)
        model = train_shield(Xtr, ytr, epochs=args.epochs,
                              seed=torch_seed, d_hidden=64,
                              recall_lambda=4.0)
        p_tr = predict(model, Xtr)
        p_dev = predict(model, Xdev)
        p_te = predict(model, Xte)
        hard_tr = _hard_rule_fire_v12(Xtr, last_users_tr)
        hard_te = _hard_rule_fire_v12(Xte, last_users_te)
        cats_tr_arr = np.asarray(cats_tr)
        cats_te_arr = np.asarray(cats_te)
        tau_low, tau_high = _select_two_thresholds(
            ytr, p_tr, hard_tr, cats_tr_arr,
            target_recall=args.target_recall,
            max_hard_over=args.max_hard_over,
        )

        seed_payload: dict = {"tau_low": tau_low, "tau_high": tau_high,
                                "variants": {}}
        # Original (no post-hoc calibration)
        seed_payload["variants"]["original"] = _metrics(
            yte, p_te, hard_te, cats_te_arr,
            tau_low=tau_low, tau_high=tau_high,
        )
        # Temperature scaling
        p_te_T, T = _temperature_scale(p_dev, ydev, p_te)
        seed_payload["variants"]["temperature"] = _metrics(
            yte, p_te_T, hard_te, cats_te_arr,
            tau_low=tau_low, tau_high=tau_high,
        )
        seed_payload["variants"]["temperature"]["temperature"] = T
        # Platt
        p_te_P = _platt_scale(p_dev, ydev, p_te)
        seed_payload["variants"]["platt"] = _metrics(
            yte, p_te_P, hard_te, cats_te_arr,
            tau_low=tau_low, tau_high=tau_high,
        )
        # Isotonic
        p_te_I = _isotonic(p_dev, ydev, p_te)
        seed_payload["variants"]["isotonic"] = _metrics(
            yte, p_te_I, hard_te, cats_te_arr,
            tau_low=tau_low, tau_high=tau_high,
        )
        # Class-conditional temperature
        p_te_CC = _class_cond_temp(p_dev, ydev, p_te)
        seed_payload["variants"]["classcond"] = _metrics(
            yte, p_te_CC, hard_te, cats_te_arr,
            tau_low=tau_low, tau_high=tau_high,
        )
        payload["seeds"][name] = seed_payload
        for variant in seed_payload["variants"]:
            m = seed_payload["variants"][variant]
            print(f"[cal {name} {variant:<11s}] "
                  f"rec={m.get('severe_recall', float('nan')):.3f} "
                  f"FNR={m.get('fnr', float('nan')):.3f} "
                  f"ECE={m.get('ece', float('nan')):.3f} "
                  f"hard_over={m.get('over_refusal_hard', float('nan')):.3f}")

    # Seed-mean
    means: dict = {}
    if payload["seeds"]:
        for variant in next(iter(payload["seeds"].values()))["variants"]:
            agg: dict = {"severe_recall": [], "fnr": [], "ece": [],
                          "over_refusal_hard": [], "over_refusal_any": []}
            for s in payload["seeds"].values():
                m = s["variants"][variant]
                for k in agg:
                    if k in m:
                        agg[k].append(m[k])
            means[variant] = {k: float(np.mean(v)) if v else float("nan")
                                for k, v in agg.items()}
    payload["seed_mean"] = means

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2),
                                encoding="utf-8")
    print(f"[cal] wrote {args.out}")


if __name__ == "__main__":
    main()
