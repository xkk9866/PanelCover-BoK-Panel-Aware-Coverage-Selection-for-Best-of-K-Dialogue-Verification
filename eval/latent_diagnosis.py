"""Diagnose whether the learned latent state ``z`` adds information beyond the
weak-supervision lexicon — the central interpretability question.

We answer:

    Q1 (lexicon mimicry):  How close is z to weak_state(client_text)?
                           Per-axis Pearson + MAE.  If ρ ≈ 1, z *is* the
                           lexicon and adds nothing.

    Q2 (outcome utility):  Does ``z`` predict outcome (uptake) better than
                           the lexicon does?  We fit two simple logistic
                           regressions on the test set:
                             outcome ~ logit(lexicon_axes)
                             outcome ~ logit(z_axes)
                           If z >> lexicon AUROC, the model is doing real
                           latent inference; if z ≤ lexicon AUROC, z is a
                           noisy lexicon clone with no extra signal.

    Q3 (axis identifiability):  Can we *swap* axis names in z and still get
                           the same outcome AUROC?  If yes, axis labels are
                           meaningless; if no, axes carry directional info.

    Q4 (counterfactual sanity):  When the model sees a high-distress
                           client, does its ``z_cf_pred[safety_referral, distress]``
                           predict a *larger* drop than other strategies?
                           Aggregated over the test set, this is the
                           behavioural interpretability test.

The script consumes ``preds_<split>.jsonl`` files from
:func:`eval.generate.main`, so no GPU is needed.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from psystate.constants import STATE_AXES, STRATEGIES
from psystate.weak_label import weak_state


def _last_user_text(context: list[dict]) -> str:
    for m in reversed(context):
        if m.get("role") == "user":
            return m.get("content", "")
    return ""


def load_preds(p: str) -> list[dict]:
    rows = []
    for line in Path(p).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def q1_lexicon_mimicry(rows: list[dict]) -> dict:
    """Per-axis correlation between predicted z and weak_state(last user)."""
    out: dict[str, dict] = {}
    z_mat, lex_mat = [], []
    for r in rows:
        z_pred = r.get("z_pred")
        if not z_pred:
            continue
        ut = _last_user_text(r.get("context", []))
        lex = weak_state(ut)
        z_mat.append([z_pred[a] for a in STATE_AXES])
        lex_mat.append([lex[a] for a in STATE_AXES])
    if not z_mat:
        return {"n": 0}
    z_mat = np.array(z_mat); lex_mat = np.array(lex_mat)
    for j, a in enumerate(STATE_AXES):
        r, p = pearsonr(z_mat[:, j], lex_mat[:, j])
        rs, ps = spearmanr(z_mat[:, j], lex_mat[:, j])
        out[a] = {
            "pearson_r": float(r), "pearson_p": float(p),
            "spearman_r": float(rs),
            "mae": float(np.mean(np.abs(z_mat[:, j] - lex_mat[:, j]))),
        }
    out["n"] = len(z_mat)
    return out


def q2_outcome_utility(rows: list[dict]) -> dict:
    """Compare AUROC(uptake | predictor) for several predictors:

    * ``lexicon``     — logistic regression on the 5 lexicon axes
    * ``z``           — logistic regression on the model's predicted ``z``
    * ``z_mlp``       — small MLP on ``z`` (probes for non-linear signal)
    * ``z_concat_lex``— logistic regression on ``[z; lexicon]`` (does z add
                        signal *over and above* the lexicon?)
    * ``model_uptake``— directly use the model's own ``uptake_pred`` field
                        (this uses *all* of the model's representation,
                        i.e. the head's full forward pass; in v5 this is
                        ``z_only`` so it equals what an MLP-on-z would get).

    Δ AUROC vs lexicon is the headline reviewer-asked metric: if z fails
    to match or beat the lexicon, calling z "the latent psychological state"
    is a misnomer.
    """
    from sklearn.neural_network import MLPClassifier

    z_mat, z_subspace_mat, lex_mat, model_p, residual_p, y = [], [], [], [], [], []
    for r in rows:
        z_pred = r.get("z_pred")
        if z_pred is None:
            continue
        meta_o = r.get("outcome_short_target") or {}
        uptake_t = meta_o.get("uptake") if isinstance(meta_o, dict) else None
        if uptake_t is None:
            continue
        ut = _last_user_text(r.get("context", []))
        lex = weak_state(ut)
        z_mat.append([z_pred[a] for a in STATE_AXES])
        z_res = r.get("z_axis_residual")
        if z_res is not None:
            flat = []
            for i, a in enumerate(STATE_AXES):
                flat.append(float(z_pred[a]))
                flat.extend(float(x) for x in z_res[i])
            z_subspace_mat.append(flat)
        lex_mat.append([lex[a] for a in STATE_AXES])
        model_p.append(float(r.get("uptake_pred", 0.5)))
        residual_p.append(r.get("residual_uptake_pred"))
        y.append(int(uptake_t))
    if len(set(y)) < 2:
        return {"n": len(y), "note": "uptake target degenerate (single class)"}
    z_mat = np.array(z_mat); lex_mat = np.array(lex_mat); y = np.array(y)
    z_subspace_mat = np.array(z_subspace_mat) if len(z_subspace_mat) == len(y) else None
    model_p = np.array(model_p)
    residual_p_arr = np.array([0.5 if v is None else float(v) for v in residual_p])
    out = {"n": int(len(y)), "pos_rate": float(y.mean())}

    def _logreg_auroc(X):
        try:
            clf = LogisticRegression(max_iter=2000).fit(X, y)
            return float(roc_auc_score(y, clf.predict_proba(X)[:, 1]))
        except Exception:
            return None

    out["auroc_lexicon_logreg"] = _logreg_auroc(lex_mat)
    out["auroc_z_logreg"]       = _logreg_auroc(z_mat)
    out["auroc_z_subspace_logreg"] = _logreg_auroc(z_subspace_mat) if z_subspace_mat is not None else None
    out["auroc_z_concat_lex"]   = _logreg_auroc(np.concatenate([z_mat, lex_mat], axis=-1))
    try:
        clf = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=500,
                            random_state=0).fit(z_mat, y)
        out["auroc_z_mlp"] = float(roc_auc_score(y, clf.predict_proba(z_mat)[:, 1]))
    except Exception:
        out["auroc_z_mlp"] = None
    try:
        out["auroc_model_uptake_head"] = float(roc_auc_score(y, model_p))
    except Exception:
        out["auroc_model_uptake_head"] = None
    try:
        out["auroc_residual_leakage_head"] = float(roc_auc_score(y, residual_p_arr)) if any(v is not None for v in residual_p) else None
    except Exception:
        out["auroc_residual_leakage_head"] = None

    if out["auroc_z_logreg"] is not None and out["auroc_lexicon_logreg"] is not None:
        out["delta_auroc_z_vs_lex"] = out["auroc_z_logreg"] - out["auroc_lexicon_logreg"]
    if out.get("auroc_z_subspace_logreg") is not None and out["auroc_lexicon_logreg"] is not None:
        out["delta_auroc_zsubspace_vs_lex"] = out["auroc_z_subspace_logreg"] - out["auroc_lexicon_logreg"]
    if out.get("auroc_z_mlp") is not None and out["auroc_lexicon_logreg"] is not None:
        out["delta_auroc_zmlp_vs_lex"] = out["auroc_z_mlp"] - out["auroc_lexicon_logreg"]
    return out


def q3_axis_identifiability(rows: list[dict]) -> dict:
    """Axis-identifiability diagnostics.

    D-PERM-v2 trains an outcome probe on the original z ordering, then tests
    the *same fitted probe* on permuted columns. This is stronger than
    refitting after permutation: if axis ordering matters to a fixed decision
    rule, AUROC should drop under test-time permutation.

    D-AXIS checks diagonal dominance against the weak axis proxies: z_i should
    predict lexicon proxy i better than other z_j do. This is still proxy-based
    rather than human-labelled, but it directly tests whether the named axes
    are anchored to their intended semantics.
    """
    z_mat, lex_mat, y = [], [], []
    for r in rows:
        z_pred = r.get("z_pred")
        if z_pred is None:
            continue
        lex = r.get("state_target")
        if not isinstance(lex, dict):
            ut = r.get("client_text") or r.get("x") or r.get("user") or ""
            lex = weak_state(ut)
        meta_o = r.get("outcome_short_target") or {}
        upt = meta_o.get("uptake") if isinstance(meta_o, dict) else None
        if upt is None: continue
        z_mat.append([z_pred[a] for a in STATE_AXES])
        lex_mat.append([lex[a] for a in STATE_AXES])
        y.append(int(upt))
    if len(set(y)) < 2:
        return {"n": len(y), "note": "uptake degenerate"}
    z = np.array(z_mat); lex = np.array(lex_mat); y = np.array(y)

    def fit_probe(X, y):
        try:
            return LogisticRegression(max_iter=1000).fit(X, y)
        except Exception:
            return None

    def score_probe(clf, X, y):
        if clf is None:
            return None
        try:
            return float(roc_auc_score(y, clf.predict_proba(X)[:, 1]))
        except Exception:
            return None

    clf = fit_probe(z, y)
    base = score_probe(clf, z, y)
    rng = np.random.default_rng(0)
    perm_aurocs = []
    for _ in range(200):
        idx = rng.permutation(z.shape[1])
        a = score_probe(clf, z[:, idx], y)
        if a is not None: perm_aurocs.append(a)

    axis_auc = np.full((len(STATE_AXES), len(STATE_AXES)), np.nan)
    axis_notes: list[str] = []
    for target_idx, axis in enumerate(STATE_AXES):
        proxy = lex[:, target_idx]
        # Prefer a clinically meaningful 0.5 threshold. If it is degenerate on
        # this slice, fall back to a median split so the diagnostic remains
        # computable and records the fallback.
        label = (proxy >= 0.5).astype(int)
        if len(set(label.tolist())) < 2:
            med = float(np.median(proxy))
            label = (proxy > med).astype(int)
            axis_notes.append(f"{axis}: median_split@{med:.3f}")
        if len(set(label.tolist())) < 2:
            axis_notes.append(f"{axis}: degenerate_proxy")
            continue
        for pred_idx in range(len(STATE_AXES)):
            try:
                axis_auc[target_idx, pred_idx] = float(roc_auc_score(label, z[:, pred_idx]))
            except Exception:
                pass

    diag_auc = np.diag(axis_auc)
    off_best = []
    diag_margin = []
    for i in range(len(STATE_AXES)):
        row = axis_auc[i].copy()
        row[i] = np.nan
        best = float(np.nanmax(row)) if np.isfinite(row).any() else float("nan")
        off_best.append(best)
        diag_margin.append(float(diag_auc[i] - best) if np.isfinite(diag_auc[i]) and np.isfinite(best) else float("nan"))

    return {
        "n": int(len(y)),
        "d_perm_v2_train_original_test_original_auroc": base,
        "d_perm_v2_test_permuted_auroc_mean": float(np.mean(perm_aurocs)) if perm_aurocs else None,
        "d_perm_v2_test_permuted_auroc_std":  float(np.std(perm_aurocs))  if perm_aurocs else None,
        "d_perm_v2_mean_drop": float(base - np.mean(perm_aurocs)) if base is not None and perm_aurocs else None,
        "d_perm_v2_p_permuted_ge_original": float(np.mean([a >= base for a in perm_aurocs])) if base is not None and perm_aurocs else None,
        "d_axis_auc_matrix_rows_targets_cols_z_axes": axis_auc.tolist(),
        "d_axis_diag_auc": {axis: float(diag_auc[i]) if np.isfinite(diag_auc[i]) else None for i, axis in enumerate(STATE_AXES)},
        "d_axis_best_offdiag_auc": {axis: off_best[i] if np.isfinite(off_best[i]) else None for i, axis in enumerate(STATE_AXES)},
        "d_axis_diag_margin": {axis: diag_margin[i] if np.isfinite(diag_margin[i]) else None for i, axis in enumerate(STATE_AXES)},
        "d_axis_mean_diag_margin": float(np.nanmean(diag_margin)) if np.isfinite(diag_margin).any() else None,
        "d_axis_notes": axis_notes,
        "d_swap_note": (
            "D-SWAP requires a model forward pass after swapping named z axes "
            "before the transition/generator. It is not computable from "
            "preds-only jsonl files without re-running the model."
        ),
    }


def q4_counterfactual_clinical_test(rows: list[dict]) -> dict:
    """Behavioural interpretability test using `z_cf_pred[s, axis]` (the
    counterfactual next-state under each strategy).

    For each test turn we compute Δz_a^{(s)} = z_cf[s, a] - z_pred[a].
    For three pre-registered clinical predictions we ask: "is the strategy
    predicted by theory the one whose Δz is most extreme in the right
    direction?"

      P1: safety_referral should reduce distress more than empathy.
      P2: reframe should reduce rigidity more than question.
      P3: action_suggestion should raise readiness more than reflection.
    """
    rules = [
        # ----- pre-registered, *trained* in L_cf (§4.3, §7.5) -----
        ("safety_lowers_distress_more_than_empathy",
         "safety_referral", "empathy", "distress", -1),
        ("reframe_lowers_rigidity_more_than_question",
         "reframe", "question", "rigidity", -1),
        ("action_raises_readiness_more_than_reflection",
         "action_suggestion", "reflection", "readiness", +1),
        # ----- held-out, NOT trained in L_cf (§8 R4-R6) -----
        ("HELDOUT_R4_empathy_raises_alliance_more_than_question",
         "empathy", "question", "alliance", +1),
        ("HELDOUT_R5_summarization_raises_clarity_more_than_empathy",
         "summarization", "empathy", "clarity", +1),
        ("HELDOUT_R6_safety_raises_alliance_more_than_action",
         "safety_referral", "action_suggestion", "alliance", +1),
    ]
    S_IDX = {s: i for i, s in enumerate(STRATEGIES)}
    A_IDX = {a: i for i, a in enumerate(STATE_AXES)}
    out: dict = {}
    for name, sA, sB, axis, sign in rules:
        wins = ties = losses = 0
        diffs = []
        for r in rows:
            cf = r.get("z_cf_pred")
            z = r.get("z_pred")
            if cf is None or z is None: continue
            iA, iB, j = S_IDX[sA], S_IDX[sB], A_IDX[axis]
            try:
                dA = cf[iA][j] - z[axis]
                dB = cf[iB][j] - z[axis]
            except Exception:
                continue
            # We want Δ_A "more in the right direction" than Δ_B.
            # If sign = -1, we want dA < dB; else dA > dB.
            if sign < 0:
                if dA < dB - 1e-4: wins += 1
                elif dB < dA - 1e-4: losses += 1
                else: ties += 1
                diffs.append(dB - dA)  # positive = clinical theory wins
            else:
                if dA > dB + 1e-4: wins += 1
                elif dB > dA + 1e-4: losses += 1
                else: ties += 1
                diffs.append(dA - dB)
        n = wins + losses + ties
        out[name] = {
            "n": n,
            "win_rate": wins / n if n else 0.0,
            "tie_rate": ties / n if n else 0.0,
            "loss_rate": losses / n if n else 0.0,
            "mean_margin_in_theory_direction": float(np.mean(diffs)) if diffs else 0.0,
            "binom_p_>0.5":  # one-sided sign test
                _binom_p(wins, n - ties) if n - ties else None,
        }
    return out


def _binom_p(k: int, n: int) -> float:
    if n <= 0: return float("nan")
    from math import comb
    return float(sum(comb(n, i) for i in range(k, n + 1)) / (2 ** n))


# ---------------------------------------------------------------------------


def analyse(preds_path: str) -> dict:
    rows = load_preds(preds_path)
    return {
        "preds_path": preds_path,
        "n_rows": len(rows),
        "q1_lexicon_mimicry": q1_lexicon_mimicry(rows),
        "q2_outcome_utility": q2_outcome_utility(rows),
        "q3_axis_identifiability": q3_axis_identifiability(rows),
        "q4_counterfactual_clinical": q4_counterfactual_clinical_test(rows),
    }


def render(rep: dict) -> str:
    L = []
    L.append(f"\n=== {rep['preds_path']}  (n={rep['n_rows']}) ===")
    L.append("Q1 (lexicon mimicry — does z just copy the lexicon?)")
    q1 = rep["q1_lexicon_mimicry"]
    if q1.get("n", 0) > 0:
        for a in STATE_AXES:
            d = q1[a]
            L.append(f"  {a:10s} pearson={d['pearson_r']:+.3f}  spearman={d['spearman_r']:+.3f}  MAE={d['mae']:.3f}")
    L.append("Q2 (outcome utility — does z beat lexicon at predicting uptake?)")
    q2 = rep["q2_outcome_utility"]
    if q2.get("auroc_z_logreg") is not None:
        L.append(f"  AUROC(lexicon, logreg)      = {q2['auroc_lexicon_logreg']:.3f}")
        L.append(f"  AUROC(z, logreg)            = {q2['auroc_z_logreg']:.3f}    Δ={q2.get('delta_auroc_z_vs_lex', float('nan')):+.3f}")
        if q2.get("auroc_z_subspace_logreg") is not None:
            L.append(f"  AUROC(axis-subspace Z)      = {q2['auroc_z_subspace_logreg']:.3f}    Δ={q2.get('delta_auroc_zsubspace_vs_lex', float('nan')):+.3f}")
        L.append(f"  AUROC(z, mlp)               = {q2.get('auroc_z_mlp', float('nan')):.3f}    Δ={q2.get('delta_auroc_zmlp_vs_lex', float('nan')):+.3f}")
        L.append(f"  AUROC(z + lex, logreg)      = {q2.get('auroc_z_concat_lex', float('nan')):.3f}")
        L.append(f"  AUROC(model uptake head)    = {q2.get('auroc_model_uptake_head', float('nan')):.3f}")
        if q2.get("auroc_residual_leakage_head") is not None:
            L.append(f"  AUROC(residual leakage head)= {q2['auroc_residual_leakage_head']:.3f}")
    else:
        L.append(f"  {q2}")
    L.append("Q3 (axis identifiability — fixed-probe permutation + diagonal dominance)")
    q3 = rep["q3_axis_identifiability"]
    if q3.get("d_perm_v2_train_original_test_original_auroc") is not None:
        L.append(f"  D-PERM-v2 AUROC(original)       = {q3['d_perm_v2_train_original_test_original_auroc']:.3f}")
        L.append(f"  D-PERM-v2 AUROC(test perm mean) = {q3['d_perm_v2_test_permuted_auroc_mean']:.3f} ± {q3['d_perm_v2_test_permuted_auroc_std']:.3f}")
        L.append(f"  D-PERM-v2 mean drop             = {q3['d_perm_v2_mean_drop']:+.3f}")
        L.append(f"  D-PERM-v2 p(perm ≥ original)    = {q3['d_perm_v2_p_permuted_ge_original']:.3f}")
        mean_margin = q3.get("d_axis_mean_diag_margin")
        if mean_margin is not None:
            L.append(f"  D-AXIS mean diagonal margin     = {mean_margin:+.3f}")
        else:
            L.append("  D-AXIS mean diagonal margin     = n/a")
        margins = q3.get("d_axis_diag_margin") or {}
        for a in STATE_AXES:
            val = margins.get(a)
            if val is not None:
                L.append(f"    {a:10s} diag_margin={val:+.3f}")
        if q3.get("d_axis_notes"):
            L.append(f"  D-AXIS notes: {', '.join(q3['d_axis_notes'])}")
        L.append(f"  D-SWAP: {q3.get('d_swap_note')}")
    L.append("Q4 (counterfactual clinical predictions — is z_cf clinically right?)")
    q4 = rep["q4_counterfactual_clinical"]
    for k, v in q4.items():
        L.append(f"  {k:50s} win={v['win_rate']:.2f} tie={v['tie_rate']:.2f}"
                 f" loss={v['loss_rate']:.2f}  margin={v['mean_margin_in_theory_direction']:+.4f}"
                 f"  p(binom)={v['binom_p_>0.5']}")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", nargs="+", required=True)
    ap.add_argument("--out", default="results/latent_diagnosis.json")
    args = ap.parse_args()
    reports = []
    for p in args.preds:
        if not os.path.isfile(p):
            print(f"[skip] {p} missing")
            continue
        rep = analyse(p)
        print(render(rep))
        reports.append(rep)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(reports, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    main()
