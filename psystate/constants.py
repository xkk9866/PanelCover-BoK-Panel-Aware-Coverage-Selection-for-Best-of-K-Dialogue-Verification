"""Global constants for PsyState.

The 5 interpretable state axes (order matters!) and the 7 counselor strategies
plus a latent residual codebook size. Everything else should import from here
so dims stay consistent.
"""

from __future__ import annotations

STATE_AXES: tuple[str, ...] = (
    "distress",   # d_t
    "rigidity",   # r_t
    "readiness",  # e_t
    "alliance",   # a_t
    "clarity",    # c_t
)
N_STATE: int = len(STATE_AXES)

STRATEGIES: tuple[str, ...] = (
    "question",
    "reflection",
    "empathy",
    "reframe",
    "summarization",
    "action_suggestion",
    "safety_referral",
)
N_STRATEGY: int = len(STRATEGIES)

N_LATENT_ACTION: int = 16  # residual codebook size
PREFIX_LEN: int = 8        # decoder soft-prefix length
