"""Weak-supervision labelers for client state and counselor strategy.

These are lightweight lexicon + pattern matchers. They produce:

* ``weak_state(text) -> dict[str, float]`` in ``[0, 1]`` for each of the 5 axes.
* ``weak_strategy(text) -> (label: str, one_hot: list[float])`` over the 7 labels.
* ``risk_flag(text) -> {self_harm, harm_others, severe_distress, any}``.

They are intentionally non-neural: they ground the latent space during training
so that dim *k* actually means axis *k*. A learned model will refine them.
"""

from __future__ import annotations

import json
import math
import re
from functools import lru_cache
from pathlib import Path

import jieba

from .constants import N_STATE, N_STRATEGY, STATE_AXES, STRATEGIES

LEX_DIR = Path(__file__).resolve().parent.parent / "data" / "lexicons"


@lru_cache(maxsize=1)
def _load_state_lex() -> dict:
    return json.loads((LEX_DIR / "cn_state_lexicon.json").read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def _load_strategy_lex() -> dict:
    return json.loads((LEX_DIR / "cn_strategy_lexicon.json").read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def _load_risk_lex() -> dict:
    return json.loads((LEX_DIR / "risk_terms.json").read_text(encoding="utf-8"))


def _hits(text: str, terms: list[str]) -> int:
    return sum(1 for t in terms if t in text)


def weak_state(text: str) -> dict[str, float]:
    """Score a client utterance along the 5 axes, in [0, 1].

    Each axis uses matched-term counts vs. counter-term counts, squashed through
    a logistic. Missing signal → 0.5 (neutral prior).
    """

    if not text:
        return {a: 0.5 for a in STATE_AXES}

    lex = _load_state_lex()
    out: dict[str, float] = {}
    # Normalize by text length (in chars) to reduce bias from long turns.
    n = max(len(text), 20)
    scale = math.sqrt(n / 60.0)  # ~1 for a 60-char turn

    for axis in STATE_AXES:
        pos = _hits(text, lex[axis]["pos"])
        neg = _hits(text, lex[axis]["neg"])
        # readiness-neg means lower readiness; alliance-neg means lower alliance; etc.
        logit = (pos - neg) / scale
        out[axis] = 1.0 / (1.0 + math.exp(-logit))
    return out


def weak_strategy(text: str) -> tuple[str, list[float]]:
    """Return most-likely counselor strategy and soft distribution over 7 labels."""

    if not text:
        probs = [1.0 / N_STRATEGY] * N_STRATEGY
        return "reflection", probs

    lex = _load_strategy_lex()
    scores = []
    for strat in STRATEGIES:
        cfg = lex[strat]
        s = 0.0
        for pat in cfg["patterns"]:
            if pat in text:
                s += cfg["weight"]
        scores.append(s)

    # Prior: reflection is the default if nothing matches.
    if max(scores) == 0:
        probs = [0.0] * N_STRATEGY
        probs[STRATEGIES.index("reflection")] = 1.0
        return "reflection", probs

    # Soft-normalize with temperature.
    T = 0.8
    exps = [math.exp(s / T) for s in scores]
    Z = sum(exps)
    probs = [e / Z for e in exps]
    label = STRATEGIES[probs.index(max(probs))]
    return label, probs


def risk_flag(text: str) -> dict[str, bool]:
    """Detect risk markers in a client utterance + a graded severity level.

    Severity ordinal (mirrors :data:`psystate.heads.SEVERITY_LABELS`):

    * ``0`` — none: no risk markers.
    * ``1`` — mild: severe distress / harm-others mentions only ("崩溃",
              "想打人", etc.) and no self-harm intent.
    * ``2`` — severe: any self-harm or suicidal-ideation marker without
              actionable plan or means ("不想活", "活不下去", "轻生", "自残").
    * ``3`` — imminent: explicit plan / means / declared self-harm
              ("自杀", "想死", "结束生命", "跳楼", "割腕", "安眠药",
              "伤害自己").

    The severity field is consumed by :class:`psystate.heads.RiskSeverityHead`
    during training and by the safety-router gate at inference.
    """

    lex = _load_risk_lex()
    flags = {k: any(t in text for t in v) for k, v in lex.items()}

    # Decompose self-harm into "imminent" (plan/means) vs "severe" (ideation).
    imminent_terms = (
        "自杀", "想死", "结束生命", "跳楼", "割腕", "安眠药", "伤害自己",
    )
    is_imminent = any(t in text for t in imminent_terms)

    if is_imminent:
        sev = 3
    elif flags.get("self_harm", False):
        sev = 2
    elif flags.get("severe_distress", False) or flags.get("harm_others", False):
        sev = 1
    else:
        sev = 0

    flags["any"] = any(flags.values())
    flags["severity"] = int(sev)
    return flags


def derive_short_outcome(curr_state: dict, next_state: dict) -> dict[str, float]:
    """Short-term outcome `o_short`: differences next - curr (bounded to [-1, 1])."""

    out = {}
    for axis in STATE_AXES:
        d = next_state[axis] - curr_state[axis]
        out[f"d_{axis}"] = max(-1.0, min(1.0, d))
    # simple aggregate 'uptake' label: client less distressed + more aligned + clearer.
    good = (-out["d_distress"] + out["d_alliance"] + out["d_clarity"]
            + out["d_readiness"] - out["d_rigidity"]) / 5.0
    out["uptake"] = 1.0 if good > 0.02 else 0.0
    out["uptake_soft"] = 0.5 + 0.5 * math.tanh(good * 3.0)
    return out
