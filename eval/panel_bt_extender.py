"""Text-input panel BT extender.

The fitted panel BT model (``results/panel_bt.json``) stores per-expert,
per-dimension Bradley-Terry logits keyed by response *text*; this lets
us read off a logit for any response that was rated by the panel, but
not for a new response (e.g.\ a KEMI / RAG / LLM-direct response that
the panel never rated).

This module fits a simple TF-IDF + Ridge regression model that maps an
arbitrary response string to a per-expert, per-dimension BT logit.  We
use Chinese character n-grams (1-3) so that the same featuriser works
for Mandarin emotional-support text, plus a small set of structural
features (length, distinct-2, action / knowledge cue density) that
mental-health reviewers are known to attend to.

Trained models are cached at
``models/panel_bt_extender/{expert}__{dim}.joblib`` and reloaded by the
PA-SCT-DRO++ unified-pool selector.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
import joblib


ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models" / "panel_bt_extender"
PANEL_BT = ROOT / "results" / "panel_bt.json"

EXPERTS = ("E1_supervisor", "E2_client_experience", "E3_safety_reviewer")
DIMS = (
    "overall",
    "helpfulness",
    "empathy",
    "specificity",
    "actionability",
    "appropriateness",
    "safety",
)


_ACTION_CUES = (
    "建议", "可以试试", "试试", "可以做", "可以从", "可以先",
    "可以通过", "可以考虑", "推荐", "练习", "记录",
    "写下", "列出", "清单", "步骤", "方法", "技巧", "技术",
)
_RETRIEVAL_CUES = (
    "研究", "数据", "资料", "书籍", "文章", "资源", "疗法",
    "正念", "认知", "行为", "暴露", "放松", "深呼吸",
    "心理咨询", "专业人员", "热线", "诊所", "咨询师",
)
_SAFETY_CUES = (
    "热线", "急诊", "危机干预", "立即就医", "拨打", "12320",
    "010-82951332", "心理援助", "马上", "尽快",
)


def _structural_features(text: str) -> np.ndarray:
    if not isinstance(text, str):
        text = ""
    n_chars = len(text)
    n_question = text.count("?") + text.count("？")
    bigrams = {text[i:i + 2] for i in range(max(0, n_chars - 1))}
    distinct_2 = len(bigrams) / max(1, n_chars - 1)
    action = sum(text.count(c) for c in _ACTION_CUES)
    retrieval = sum(text.count(c) for c in _RETRIEVAL_CUES)
    safety = sum(text.count(c) for c in _SAFETY_CUES)
    return np.array(
        [
            n_chars,
            np.log1p(n_chars),
            n_question,
            distinct_2,
            action,
            retrieval,
            safety,
            action / max(1, n_chars / 50.0),
            retrieval / max(1, n_chars / 50.0),
        ],
        dtype=np.float32,
    )


def _structural_matrix(texts):
    return np.vstack([_structural_features(t) for t in texts])


def _make_pipeline() -> Pipeline:
    return Pipeline(
        [
            (
                "feats",
                FunctionTransformer(_structural_matrix, validate=False),
            ),
        ]
    )


def _build_features(texts):
    char_tfidf = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(1, 3),
        min_df=2,
        max_df=0.98,
        sublinear_tf=True,
        max_features=20000,
    )
    char_tfidf.fit(texts)
    return char_tfidf


def fit_all() -> None:
    """Fit one Ridge regressor per (expert, dimension) and persist to
    ``models/panel_bt_extender/{expert}__{dim}.joblib``.

    Each model is a small sklearn pipeline so it can be re-loaded and
    applied to arbitrary response strings.
    """
    panel_bt = json.loads(PANEL_BT.read_text(encoding="utf-8"))

    all_texts = set()
    for expert in EXPERTS:
        for dim in DIMS:
            all_texts.update(panel_bt[expert][dim].keys())
    all_texts = list(all_texts)
    print(f"[panel-bt-ext] training on {len(all_texts)} unique responses")

    tfidf = _build_features(all_texts)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(tfidf, MODELS_DIR / "tfidf.joblib")
    print(f"[panel-bt-ext] saved tfidf vectoriser ({len(tfidf.vocabulary_)} features)")

    summary: Dict[str, Dict[str, float]] = {}
    for expert in EXPERTS:
        summary[expert] = {}
        for dim in DIMS:
            scored = panel_bt[expert][dim]
            X_text = list(scored.keys())
            y = np.array(list(scored.values()), dtype=np.float32)
            X_tfidf = tfidf.transform(X_text)
            X_struct = _structural_matrix(X_text)
            from scipy.sparse import hstack, csr_matrix
            X = hstack([X_tfidf, csr_matrix(X_struct)]).tocsr()

            ridge = Ridge(alpha=2.0, random_state=0)
            ridge.fit(X, y)
            yhat = ridge.predict(X)
            corr = float(np.corrcoef(y, yhat)[0, 1])
            mae = float(np.mean(np.abs(y - yhat)))
            summary[expert][dim] = {"r": corr, "mae": mae, "n": len(y)}

            joblib.dump(ridge, MODELS_DIR / f"{expert}__{dim}.joblib")
            print(
                f"[panel-bt-ext] {expert:<22s} {dim:<14s} "
                f"n={len(y):4d}  r={corr:.3f}  mae={mae:.3f}"
            )

    summary_path = MODELS_DIR / "training_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[panel-bt-ext] training summary -> {summary_path}")


class PanelBTExtender:
    """Loads the persisted Ridge ensemble and scores arbitrary text.

    For a query response, ``score(text)`` returns a nested dict
    ``{expert: {dim: logit}}`` that drops in wherever the original
    panel-BT lookup was used.

    For texts that already exist in the original panel BT corpus we
    return the *exact* fitted logit (not the regression prediction);
    only genuinely new responses go through the regression.
    """

    def __init__(self) -> None:
        self.tfidf = joblib.load(MODELS_DIR / "tfidf.joblib")
        self.models: Dict[str, Dict[str, Ridge]] = {}
        for expert in EXPERTS:
            self.models[expert] = {}
            for dim in DIMS:
                p = MODELS_DIR / f"{expert}__{dim}.joblib"
                self.models[expert][dim] = joblib.load(p)
        self.panel_bt = json.loads(PANEL_BT.read_text(encoding="utf-8"))

    def _featurise(self, text: str):
        from scipy.sparse import hstack, csr_matrix

        X_tfidf = self.tfidf.transform([text])
        X_struct = _structural_matrix([text])
        return hstack([X_tfidf, csr_matrix(X_struct)]).tocsr()

    def score(self, text: str) -> Dict[str, Dict[str, float]]:
        out: Dict[str, Dict[str, float]] = {}
        feats = None
        for expert in EXPERTS:
            out[expert] = {}
            for dim in DIMS:
                exact = self.panel_bt.get(expert, {}).get(dim, {}).get(text)
                if exact is not None:
                    out[expert][dim] = float(exact)
                    continue
                if feats is None:
                    feats = self._featurise(text)
                out[expert][dim] = float(
                    self.models[expert][dim].predict(feats)[0]
                )
        return out


if __name__ == "__main__":
    fit_all()
