"""Held-out validation of the panel BT text-input extender.

Re-fits the extender on a random 80% of the panel BT responses and
reports per-expert / per-dim Pearson r and MAE on the held-out 20%.
This is the strict generalisation test used to justify deploying the
extender on new (KEMI / RAG / LLM-direct) responses in PA-SCT-DRO++.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.linear_model import Ridge
from sklearn.feature_extraction.text import TfidfVectorizer

from eval.panel_bt_extender import (
    DIMS,
    EXPERTS,
    PANEL_BT,
    _structural_matrix,
)


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    panel_bt = json.loads(PANEL_BT.read_text(encoding="utf-8"))
    responses = sorted(set(panel_bt[EXPERTS[0]][DIMS[0]].keys()))
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(responses))
    n_test = max(1, int(round(len(responses) * 0.2)))
    test_idx = set(perm[:n_test].tolist())
    train_texts = [t for i, t in enumerate(responses) if i not in test_idx]
    test_texts = [t for i, t in enumerate(responses) if i in test_idx]
    print(f"[validate] train={len(train_texts)}  test={len(test_texts)}")

    tfidf = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(1, 3),
        min_df=2,
        max_df=0.98,
        sublinear_tf=True,
        max_features=20000,
    ).fit(train_texts)

    X_tr = hstack([tfidf.transform(train_texts),
                   csr_matrix(_structural_matrix(train_texts))]).tocsr()
    X_te = hstack([tfidf.transform(test_texts),
                   csr_matrix(_structural_matrix(test_texts))]).tocsr()

    rows = []
    for expert in EXPERTS:
        for dim in DIMS:
            scored = panel_bt[expert][dim]
            y_tr = np.array([scored[t] for t in train_texts], dtype=np.float32)
            y_te = np.array([scored[t] for t in test_texts], dtype=np.float32)
            ridge = Ridge(alpha=2.0, random_state=0).fit(X_tr, y_tr)
            yhat = ridge.predict(X_te)
            r = float(np.corrcoef(y_te, yhat)[0, 1]) if y_te.std() > 0 else 0.0
            mae = float(np.mean(np.abs(y_te - yhat)))
            rmse = float(np.sqrt(np.mean((y_te - yhat) ** 2)))
            rows.append((expert, dim, r, mae, rmse))
            print(f"[validate] {expert:<22s} {dim:<14s} r={r:.3f}  mae={mae:.3f}  rmse={rmse:.3f}")

    out_path = ROOT / "results" / "panel_bt_extender_validation.json"
    out_path.write_text(
        json.dumps(
            {"n_train": len(train_texts), "n_test": len(test_texts),
             "rows": [
                 {"expert": e, "dim": d, "r": r, "mae": m, "rmse": rm}
                 for e, d, r, m, rm in rows
             ]},
            indent=2, ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    mean_r = float(np.mean([r for _, _, r, _, _ in rows]))
    mean_mae = float(np.mean([m for _, _, _, m, _ in rows]))
    print(f"[validate] mean held-out r={mean_r:.3f}  mae={mean_mae:.3f}")
    print(f"[validate] wrote {out_path}")


if __name__ == "__main__":
    main()
