"""Bayesian clinical priors over the strategy-conditioned bias matrix ``b_s``.

Reviewer-fix (interpretability): in v3 we used hard sign hinges
(``ReLU``-style penalties) on a hand-picked subset of cells in
``b ∈ R^{7 × 5}``. This made the interpretability claim a **tautology**: we
constrained 14 cells, then reported that those same 14 cells respect the
priors.  v4 replaces the binary "constrained / unconstrained" partition with a
single, principled regulariser: every cell has a Gaussian prior

    b[s, a] ~ N( mu_clinical[s, a],  sigma_clinical[s, a]^2 )

with three intensity tiers grounded in the MI / CBT / WAT literature:

* ``"strong"``  (sigma = 0.04) — well-replicated clinical effects, e.g.
                                 empathy → alliance ↑, safety_referral →
                                 distress ↓, reframe → rigidity ↓.
* ``"weak"``    (sigma = 0.10) — directional but mixed evidence, e.g.
                                 reflection → alliance ↑, action_suggestion →
                                 rigidity ↓.
* ``"neutral"`` (sigma = 0.40) — no strong clinical commitment; we still
                                 prefer near-zero biases for stability.

The held-out cells reported in :mod:`eval.holdout_interp` are scored from the
"weak" tier — the model is only weakly nudged, so a high held-out match rate
is genuine evidence of clinical generalisation rather than memorisation.

We additionally expose a **diagonal prior** over ``A_s`` (self-coupling) that
mirrors the v3 hinges: the diagonal is centred at 1 with sigma = 0.05, and
*off-diagonal* is centred at 0 with sigma = 0.05. This keeps the dynamics
near-identity unless the data + priors actively push it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import torch

from .constants import N_STATE, N_STRATEGY, STATE_AXES, STRATEGIES

# ---------------------------------------------------------------------------
# Clinical prior table — rows = strategy, cols = state axis.
# Sign convention: positive value means the strategy is expected to *raise*
# that axis on average; negative means it should *lower* that axis.
# ---------------------------------------------------------------------------

# Effect magnitudes (the prior mean ``mu``). These are conservative ~3-5 %
# nudges on the bias scale that match v3's empirical magnitudes.
STRONG = 0.05
WEAK   = 0.03

# Default sigmas per tier.
# v4 used (0.04, 0.10, 0.40) — the strong tier was so tight that the data
# could never move ``b`` more than 0.2 σ away from ``μ``. Reviewer feedback:
# this produced a *circular* interpretability story (prior ≈ posterior).
# v5 default loosens the tiers so the data has a real chance to push back:
SIGMA_STRONG  = 0.10
SIGMA_WEAK    = 0.20
SIGMA_NEUTRAL = 0.40


def _make_prior_tables(
    sigma_strong: float = SIGMA_STRONG,
    sigma_weak: float = SIGMA_WEAK,
    sigma_neutral: float = SIGMA_NEUTRAL,
) -> tuple[np.ndarray, np.ndarray, list[list[str]]]:
    """Return ``(mu, sigma, tier_table)`` shaped ``[S, K]``."""

    mu  = np.zeros((N_STRATEGY, N_STATE), dtype=np.float64)
    sig = np.full((N_STRATEGY, N_STATE), sigma_neutral, dtype=np.float64)
    tier = [["neutral"] * N_STATE for _ in range(N_STRATEGY)]

    S = {s: i for i, s in enumerate(STRATEGIES)}
    A = {a: i for i, a in enumerate(STATE_AXES)}

    def set_strong(s: str, a: str, sign: float) -> None:
        i, j = S[s], A[a]
        mu[i, j]  = sign * STRONG
        sig[i, j] = sigma_strong
        tier[i][j] = "strong"

    def set_weak(s: str, a: str, sign: float) -> None:
        i, j = S[s], A[a]
        mu[i, j]  = sign * WEAK
        sig[i, j] = sigma_weak
        tier[i][j] = "weak"

    # ---- STRONG priors (well-replicated effects) ---------------------------
    set_strong("empathy",            "alliance",   +1.0)   # warmth → alliance ↑
    set_strong("empathy",            "rigidity",   -1.0)   # warmth shouldn't amplify rigidity
    set_strong("reframe",            "rigidity",   -1.0)   # cognitive restructuring ↓
    set_strong("summarization",      "clarity",    +1.0)
    set_strong("question",           "clarity",    +1.0)
    set_strong("action_suggestion",  "readiness",  +1.0)
    set_strong("safety_referral",    "distress",   -1.0)
    # Adjacent strong priors (consistent with v3's L_consist):
    set_strong("reflection",         "rigidity",   -1.0)
    set_strong("question",           "rigidity",   -1.0)
    set_strong("summarization",      "rigidity",   -1.0)

    # ---- WEAK priors (directional but mixed evidence; held-out tier) -------
    set_weak("empathy",            "readiness",  +1.0)
    set_weak("empathy",            "distress",   -1.0)
    set_weak("empathy",            "clarity",    +1.0)   # support clarifies a bit
    set_weak("reflection",         "alliance",   +1.0)
    set_weak("reflection",         "distress",   -1.0)
    set_weak("reflection",         "clarity",    +1.0)
    set_weak("reflection",         "readiness",  +1.0)
    set_weak("reframe",            "clarity",    +1.0)
    set_weak("reframe",            "distress",   -1.0)
    set_weak("reframe",            "readiness",  +1.0)
    set_weak("summarization",      "alliance",   +1.0)
    set_weak("summarization",      "distress",   -1.0)
    set_weak("question",           "readiness",  +1.0)
    set_weak("action_suggestion",  "clarity",    +1.0)
    set_weak("action_suggestion",  "rigidity",   -1.0)
    set_weak("safety_referral",    "alliance",   +1.0)
    set_weak("safety_referral",    "clarity",    +1.0)
    set_weak("safety_referral",    "readiness",  +1.0)

    return mu, sig, tier


_MU_NP, _SIG_NP, _TIER = _make_prior_tables()


@dataclass
class ClinicalPriors:
    """Tensor-valued container for the prior tables (lazy device move)."""

    mu: torch.Tensor   # [S, K]  prior means for ``b``
    sigma: torch.Tensor  # [S, K] prior std-devs for ``b``
    tier: list[list[str]]
    diag_mu: float = 1.0
    diag_sigma: float = 0.05
    offdiag_mu: float = 0.0
    offdiag_sigma: float = 0.05

    def to(self, device, dtype: torch.dtype = torch.float32) -> "ClinicalPriors":
        return ClinicalPriors(
            mu=self.mu.to(device=device, dtype=dtype),
            sigma=self.sigma.to(device=device, dtype=dtype),
            tier=self.tier,
            diag_mu=self.diag_mu, diag_sigma=self.diag_sigma,
            offdiag_mu=self.offdiag_mu, offdiag_sigma=self.offdiag_sigma,
        )


def make_clinical_priors(
    sigma_strong: float = SIGMA_STRONG,
    sigma_weak: float = SIGMA_WEAK,
    sigma_neutral: float = SIGMA_NEUTRAL,
) -> ClinicalPriors:
    """Return a fresh ``ClinicalPriors`` whose tier sigmas can be overridden.

    Reviewer fix: v4 used the module-level defaults of (0.04, 0.10, 0.40)
    which caused the posterior to coincide with the prior (no data signal in
    ``b``). v5's default is (0.10, 0.20, 0.40); we expose the override so
    the v5 sweep (`v5-full-prior` / `v5-weak-prior` / `v5-no-prior`) can be
    driven from a single config entry.
    """

    if (sigma_strong, sigma_weak, sigma_neutral) == (SIGMA_STRONG, SIGMA_WEAK, SIGMA_NEUTRAL):
        mu_np, sig_np, tier_table = _MU_NP, _SIG_NP, _TIER
    else:
        mu_np, sig_np, tier_table = _make_prior_tables(sigma_strong, sigma_weak, sigma_neutral)
    return ClinicalPriors(
        mu=torch.from_numpy(mu_np.copy()).float(),
        sigma=torch.from_numpy(sig_np.copy()).float(),
        tier=[row[:] for row in tier_table],
    )


def cells_by_tier(tier_name: str) -> list[tuple[str, str]]:
    """Return ``[(strategy, axis), ...]`` cells classified at ``tier_name``."""

    out: list[tuple[str, str]] = []
    for i, s in enumerate(STRATEGIES):
        for j, a in enumerate(STATE_AXES):
            if _TIER[i][j] == tier_name:
                out.append((s, a))
    return out


def gaussian_prior_loss(
    bias: torch.Tensor,         # [S, K]
    A: torch.Tensor,            # [S, K, K]
    priors: ClinicalPriors,
) -> torch.Tensor:
    """Sum of Gaussian NLLs (up to constants).

    ``L = 0.5 * sum_{s,a} ((b[s,a] - mu[s,a]) / sigma[s,a])^2``  +
        diagonal/off-diagonal regularisers on ``A``.
    """

    # --- bias term ---
    diff_b = (bias - priors.mu) / priors.sigma
    L_b = 0.5 * (diff_b * diff_b).mean()

    # --- A_s diagonal (self-coupling) toward 1 ---
    eye = torch.eye(N_STATE, device=A.device, dtype=A.dtype)
    diag_A = torch.diagonal(A, dim1=-2, dim2=-1)              # [S, K]
    diff_d = (diag_A - priors.diag_mu) / priors.diag_sigma
    L_d = 0.5 * (diff_d * diff_d).mean()

    # --- A_s off-diagonal toward 0 ---
    off_A = A * (1.0 - eye).unsqueeze(0)                      # [S, K, K]
    diff_o = off_A / priors.offdiag_sigma
    L_o = 0.5 * (diff_o * diff_o).mean()

    return L_b + L_d + L_o


def posterior_shift_report(
    bias: torch.Tensor,         # [S, K]
    priors: ClinicalPriors,
    z_threshold: float = 1.0,
) -> dict:
    """Posterior summary: where the data moved each cell relative to its prior.

    For every cell we report ``(b_hat, mu, sigma, z = (b_hat - mu)/sigma,
    sign_match)``.  Cells with ``|z| > z_threshold`` are tagged "shifted":
    these are places where the **data** wanted something different from the
    prior — exactly the empirical evidence that interpretability is not just
    a tautology.
    """

    b = bias.detach().float().cpu().numpy()
    mu = priors.mu.detach().float().cpu().numpy()
    sig = priors.sigma.detach().float().cpu().numpy()

    cells: list[dict] = []
    n_shift = 0
    n_strong = 0; n_strong_match = 0
    n_weak = 0; n_weak_match = 0
    for i, s in enumerate(STRATEGIES):
        for j, a in enumerate(STATE_AXES):
            t = _TIER[i][j]
            z = float((b[i, j] - mu[i, j]) / max(sig[i, j], 1e-8))
            sign_data = int(np.sign(b[i, j])) if abs(b[i, j]) > 1e-4 else 0
            sign_prior = int(np.sign(mu[i, j])) if abs(mu[i, j]) > 1e-4 else 0
            sign_match = (sign_data == sign_prior) if sign_prior != 0 else None
            shifted = abs(z) > z_threshold
            if shifted:
                n_shift += 1
            if t == "strong":
                n_strong += 1
                if sign_match: n_strong_match += 1
            elif t == "weak":
                n_weak += 1
                if sign_match: n_weak_match += 1
            cells.append({
                "strategy": s, "axis": a,
                "tier": t,
                "b": float(b[i, j]),
                "mu": float(mu[i, j]),
                "sigma": float(sig[i, j]),
                "z": z,
                "sign_data": sign_data,
                "sign_prior": sign_prior,
                "sign_match": sign_match,
                "shifted": shifted,
            })
    summary = {
        "n_strong": n_strong,
        "n_strong_match": n_strong_match,
        "strong_match_rate": (n_strong_match / n_strong) if n_strong else 0.0,
        "n_weak": n_weak,
        "n_weak_match": n_weak_match,
        "weak_match_rate": (n_weak_match / n_weak) if n_weak else 0.0,
        "n_shifted_cells": n_shift,
        "z_threshold": z_threshold,
        "cells": cells,
    }
    return summary


# Helper exposed for ``eval.holdout_interp``: lift the previous "constrained"
# vs "held-out" partition out of ``losses.py`` so it stays in sync.
WEAK_HELDOUT_CELLS: list[tuple[str, str]] = cells_by_tier("weak")
STRONG_CELLS: list[tuple[str, str]] = cells_by_tier("strong")
NEUTRAL_CELLS: list[tuple[str, str]] = cells_by_tier("neutral")
