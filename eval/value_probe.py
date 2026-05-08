"""Held-out value-probe evaluation for PsyState representations.

This script trains simple calibrated probes on one prediction file and scores
them on another.  Unlike ``latent_diagnosis.py`` (which is a capacity diagnostic
on a fixed prediction set), this gives a held-out estimate of whether the
measurement-aware posterior state supports a deployable value planner.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from psystate.constants import STATE_AXES
from psystate.weak_label import weak_state


def _load_rows(path: str) -> list[dict]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _last_user_text(context: list[dict]) -> str:
    for msg in reversed(context or []):
        if msg.get("role") == "user":
            return msg.get("content", "")
    return ""


def _feature_row(row: dict, mode: str) -> list[float] | None:
    z_pred = row.get("z_pred")
    if not isinstance(z_pred, dict):
        return None

    lex = row.get("state_target")
    if not isinstance(lex, dict):
        lex = weak_state(_last_user_text(row.get("context") or []))

    z = [float(z_pred[a]) for a in STATE_AXES]
    lex_vec = [float(lex[a]) for a in STATE_AXES]

    if mode == "z":
        return z
    if mode == "lexicon":
        return lex_vec
    if mode == "z_plus_lexicon":
        return z + lex_vec
    if mode == "trajectory":
        return _trajectory_features(row)
    if mode == "z_subspace_plus_trajectory":
        residual = row.get("z_axis_residual")
        traj = _trajectory_features(row)
        if residual is None or traj is None:
            return None
        flat: list[float] = []
        for i, a in enumerate(STATE_AXES):
            flat.append(float(z_pred[a]))
            flat.extend(float(x) for x in residual[i])
        return flat + traj
    if mode == "transition_value":
        z_next = row.get("z_next_pred")
        if isinstance(z_next, dict):
            z_next_vec = [float(z_next[a]) for a in STATE_AXES]
        elif isinstance(z_next, list) and len(z_next) >= len(STATE_AXES):
            z_next_vec = [float(z_next[i]) for i in range(len(STATE_AXES))]
        else:
            return None
        delta = {a: z_next_vec[i] - float(z_pred[a]) for i, a in enumerate(STATE_AXES)}
        value = (
            -delta["distress"]
            -delta["rigidity"]
            +delta["readiness"]
            +delta["alliance"]
            +delta["clarity"]
        ) / 5.0
        return [float(value)]
    if mode == "learned_transition_value":
        v = row.get("transition_value_logit")
        if v is None:
            return None
        return [float(v)]
    if mode == "transition_plus_trajectory":
        tv = _feature_row(row, "transition_value")
        traj = _trajectory_features(row)
        if tv is None or traj is None:
            return None
        return tv + traj
    if mode == "transition_plus_z":
        tv = _feature_row(row, "transition_value")
        if tv is None:
            return None
        return tv + z
    if mode == "planner_value":
        tv = _feature_row(row, "transition_value")
        learned = _feature_row(row, "learned_transition_value") or []
        traj = _trajectory_features(row)
        residual = row.get("z_axis_residual")
        if tv is None or traj is None or residual is None:
            return None
        flat: list[float] = []
        for i, a in enumerate(STATE_AXES):
            flat.append(float(z_pred[a]))
            flat.extend(float(x) for x in residual[i])
        return tv + learned + flat + traj
    if mode == "z_subspace":
        residual = row.get("z_axis_residual")
        if residual is None:
            return None
        flat: list[float] = []
        for i, a in enumerate(STATE_AXES):
            flat.append(float(z_pred[a]))
            flat.extend(float(x) for x in residual[i])
        return flat
    raise ValueError(f"unknown feature mode: {mode}")


def _trajectory_features(row: dict) -> list[float] | None:
    """Clinical measurement over the full observed user trajectory.

    Raw lexicon uses only the last user turn.  For planning, the state should
    include history: current level, recent average, session-level average,
    short-term trend, volatility, and dialogue progress.
    """

    users = [
        msg.get("content", "")
        for msg in (row.get("context") or [])
        if msg.get("role") == "user"
    ]
    if not users:
        return None
    seq = np.asarray(
        [[float(weak_state(text)[a]) for a in STATE_AXES] for text in users],
        dtype=np.float32,
    )
    last = seq[-1]
    recent = seq[-3:].mean(axis=0)
    mean = seq.mean(axis=0)
    trend = seq[-1] - seq[-2] if len(seq) >= 2 else np.zeros(len(STATE_AXES), dtype=np.float32)
    volatility = seq[-5:].std(axis=0) if len(seq) >= 2 else np.zeros(len(STATE_AXES), dtype=np.float32)
    progress = np.asarray([min(len(users) / 20.0, 1.0)], dtype=np.float32)
    return np.concatenate([last, recent, mean, trend, volatility, progress]).astype(float).tolist()


def _matrix(rows: list[dict], mode: str) -> tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for row in rows:
        outcome = row.get("outcome_short_target") or {}
        uptake = outcome.get("uptake") if isinstance(outcome, dict) else None
        if uptake is None:
            continue
        feat = _feature_row(row, mode)
        if feat is None:
            continue
        X.append(feat)
        y.append(int(uptake))
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int64)


def _score(train_rows: list[dict], test_rows: list[dict], mode: str) -> dict:
    X_tr, y_tr = _matrix(train_rows, mode)
    X_te, y_te = _matrix(test_rows, mode)
    out = {
        "mode": mode,
        "n_train": int(len(y_tr)),
        "n_test": int(len(y_te)),
        "train_pos_rate": float(y_tr.mean()) if len(y_tr) else None,
        "test_pos_rate": float(y_te.mean()) if len(y_te) else None,
    }
    if len(set(y_tr.tolist())) < 2 or len(set(y_te.tolist())) < 2:
        out["auroc"] = None
        out["note"] = "degenerate labels"
        return out
    clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(X_tr, y_tr)
    out["auroc"] = float(roc_auc_score(y_te, clf.predict_proba(X_te)[:, 1]))
    out["coef_l2"] = float(np.linalg.norm(clf.coef_))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_preds", required=True)
    ap.add_argument("--test_preds", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--modes",
        nargs="+",
        default=[
            "lexicon",
            "trajectory",
            "z",
            "z_subspace",
            "z_plus_lexicon",
            "z_subspace_plus_trajectory",
            "transition_value",
            "learned_transition_value",
            "transition_plus_trajectory",
            "transition_plus_z",
            "planner_value",
        ],
    )
    args = ap.parse_args()

    train_rows = _load_rows(args.train_preds)
    test_rows = _load_rows(args.test_preds)
    results = [_score(train_rows, test_rows, mode) for mode in args.modes]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")

    for r in results:
        auc = r.get("auroc")
        auc_s = "n/a" if auc is None else f"{auc:.3f}"
        print(f"{r['mode']:15s} AUROC={auc_s} n_train={r['n_train']} n_test={r['n_test']}")
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
