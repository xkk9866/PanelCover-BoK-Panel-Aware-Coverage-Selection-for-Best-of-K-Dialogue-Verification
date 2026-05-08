"""Composite objective for PsyState training.

Implements the 7 loss terms described in ``TASK_OUTLINE.md``:

* ``L_gen``        — teacher-forcing CE on counselor reply tokens.
* ``L_state``      — MSE between predicted axis values and weak labels.
* ``L_strategy``   — CE on strategy label (7 classes).
* ``L_transition`` — MSE between the transition model's prediction of
                     ``z_{t+1}`` and the teacher's weak label for ``z_{t+1}``.
* ``L_outcome``    — BCE on uptake + MSE on the 4 quality dims.
* ``L_consist``    — sign/monotonicity penalties on the transition matrices
                     (structure prior).
* ``L_sep``        — strategy-separation: encodes *differential* clinical
                     priors between pairs of strategies (e.g.  reframe
                     reduces rigidity *more* than empathy does), plus an
                     axis-wise dispersion term that keeps ``A_s`` / ``b_s``
                     from collapsing to a single shared dynamics.
* ``L_safety``     — hinge: if risk is detected, strategy should prefer
                     ``safety_referral`` and predicted distress should not
                     decrease spuriously (no false reassurance).
* ``L_commit``     — VQ commitment (from the latent action head).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .bayes_priors import ClinicalPriors, gaussian_prior_loss
from .constants import N_STATE, N_STRATEGY, STATE_AXES, STRATEGIES
from .heads import N_SEVERITY, SEVERE_INDEX


def axis_decorrelation_loss(
    z: torch.Tensor,
    running_mean: torch.Tensor | None = None,
    running_var: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Penalise off-diagonal correlations of ``z``.

    v5 reviewer fix: ``axis_permutation_AUROC`` tests on v3/v4 showed the
    5 axes of ``z`` are statistically interchangeable. Without a
    decorrelation pressure, a ``z_only_outcome`` head can collapse all five
    dims onto a single shared latent score, re-introducing the same
    identifiability failure.  This term encourages ``Corr(z) ~= I``.

    Two regimes are supported so the loss works at any batch size:

    * ``B >= 2``: in-batch normalised correlation matrix; standard
      ``||Corr(z) - I||^2``.
    * ``B == 1`` (our QLoRA default): we use detached EMA statistics
      ``running_mean`` / ``running_var`` (one number per axis) supplied by
      the trainer, normalise the single sample, and penalise the squared
      off-diagonal cross-products of ``(z - μ) / σ``. Gradients still flow
      through ``z`` (and hence through ``StateHead``); the EMA stats are
      detached so we don't optimise them away.
    """

    if z.numel() == 0:
        return z.new_zeros(())

    K = z.size(-1)
    eye = torch.eye(K, device=z.device, dtype=z.dtype)

    if z.size(0) >= 2:
        zc = z - z.mean(dim=0, keepdim=True)
        std = zc.std(dim=0, keepdim=True).clamp_min(eps)
        zn = zc / std
        corr = (zn.t() @ zn) / max(zn.size(0) - 1, 1)
        return ((corr - eye) ** 2).mean()

    if running_mean is None or running_var is None:
        return z.new_zeros(())
    mu = running_mean.to(z.device, z.dtype).detach()
    var = running_var.to(z.device, z.dtype).detach().clamp_min(eps)
    sd = var.sqrt()
    zc = (z - mu) / sd                            # [B, K]
    cross = zc.unsqueeze(-1) * zc.unsqueeze(-2)   # [B, K, K]
    cross = cross * (1.0 - eye)
    return cross.pow(2).mean()


def gen_loss_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Standard next-token CE (labels already shifted by HF convention)."""

    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        return F.mse_loss(pred, target)
    d = (pred - target).pow(2)
    if d.dim() > mask.dim():
        mask = mask.unsqueeze(-1).expand_as(d)
    denom = mask.float().sum().clamp_min(1.0)
    return (d * mask.float()).sum() / denom


@dataclass
class LossConfig:
    lam_gen: float = 1.0
    lam_state: float = 0.3
    lam_strategy: float = 0.3
    lam_transition: float = 0.2
    lam_outcome: float = 0.3
    lam_consist: float = 0.05
    lam_sep: float = 0.0  # strategy-separation (differential priors)
    lam_safety: float = 0.2
    lam_commit: float = 0.05
    # v4 -----------------------------------------------------------------
    lam_bayes: float = 0.0          # Gaussian clinical prior on b_s, A_s
    lam_counterfactual: float = 0.0 # per-example counterfactual contrastive
    lam_router: float = 0.0         # independent risk-severity head
    lam_uncertainty: float = 0.0    # heteroscedastic NLL on state head
    cf_margin: float = 0.05         # margin for counterfactual hinges
    router_focal_gamma: float = 2.0
    router_recall_weight: float = 3.0
    router_pos_weight: float = 8.0  # binary severe-vs-rest weight in the BCE side-loss
    # v5: axis decorrelation on the predicted ``z``. Penalises pairwise axis
    # correlations toward identity so the 5 dims do not collapse into a
    # single shared latent score.
    lam_decor: float = 0.0
    # v6: mediation and planning losses.
    lam_adv: float = 0.0          # residual adversary via GRL
    lam_pref: float = 0.0         # pairwise clinical strategy preference
    lam_q: float = 0.0            # Q-value predicts observed uptake for taken strategy
    lam_outcome_rank: float = 0.0 # pairwise ranking loss for uptake AUROC
    lam_value_rank: float = 0.0   # pairwise ranking for transition-derived value
    lam_value_reg: float = 0.0    # weak factual value calibration
    # v5: tunable Bayesian sigmas (per-tier). When non-default, the trainer
    # will rebuild ``ClinicalPriors`` with these widths instead of the v5
    # defaults (0.10, 0.20, 0.40).
    sigma_strong: float = 0.10
    sigma_weak: float = 0.20
    sigma_neutral: float = 0.40
    # Class-balancing for uptake BCE; 0 disables (plain BCE).  A value of e.g.
    # 2.0 amplifies the positive-class gradient by 2× — useful when uptake is
    # very imbalanced.
    uptake_pos_weight: float = 0.0
    # Which heads are active (for ablations).
    use_state: bool = True
    use_strategy: bool = True
    use_transition: bool = True
    use_outcome: bool = True
    use_consist: bool = True
    use_sep: bool = True
    use_safety: bool = True
    # v4 toggles ---------------------------------------------------------
    use_bayes: bool = False
    use_counterfactual: bool = False
    use_router: bool = False
    use_uncertainty: bool = False
    use_adversary: bool = False
    use_q_planner: bool = False
    use_transition_value: bool = False


def q_preference_loss(q_values: torch.Tensor, margin: float = 0.0) -> torch.Tensor:
    """Pairwise clinical preference loss over Q-values.

    Uses the same broad clinical pair bank as the counterfactual transition
    hinges, but trains a value function: Q(s_pos) should exceed Q(s_neg).
    This is a minimal v6 preference bank; it can later be replaced by a
    data-backed JSONL bank without changing the model interface.
    """

    if q_values is None or q_values.dim() != 2:
        return torch.tensor(0.0)
    S_IDX = {s: i for i, s in enumerate(STRATEGIES)}
    pen = q_values.new_zeros(())
    for s_pos, s_neg, _axis, _sign, w in COUNTERFACTUAL_PAIRS:
        pi, ni = S_IDX[s_pos], S_IDX[s_neg]
        diff = q_values[:, pi] - q_values[:, ni] - margin
        pen = pen + float(w) * F.softplus(-diff).mean()
    return pen / max(len(COUNTERFACTUAL_PAIRS), 1)


def binary_pairwise_rank_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid: torch.Tensor | None,
    margin: float = 0.0,
) -> torch.Tensor:
    """Rank positive-uptake examples above negative examples in the batch."""

    if valid is not None:
        logits = logits[valid]
        targets = targets[valid]
    if logits.numel() < 2:
        return logits.new_zeros(())
    pos = logits[targets > 0.5]
    neg = logits[targets <= 0.5]
    if pos.numel() == 0 or neg.numel() == 0:
        return logits.new_zeros(())
    diff = pos[:, None] - neg[None, :] - margin
    return F.softplus(-diff).mean()


def structural_consistency_loss(transition) -> torch.Tensor:
    """Sign/monotonicity priors on the transition (A_s, b_s).

    Clinical priors from 思想.md we enforce *softly* (ReLU hinge):

    * *safety_referral* must not increase distress
      (b[safety, distress] ≤ 0 and A[safety, distress, distress] ≤ 1).
    * *empathy* must not decrease alliance
      (b[empathy, alliance] ≥ 0 and A[empathy, alliance, alliance] ≥ 1 −δ).
    * *reframe* must not increase rigidity
      (b[reframe, rigidity] ≤ 0 and A[reframe, rigidity, rigidity] ≤ 1).
    * *action_suggestion* should push readiness up
      (b[action, readiness] ≥ 0).
    * *summarization* should raise clarity
      (b[summary, clarity] ≥ 0 and A[summary, clarity, clarity] ≥ 1 −δ).
    * *question* should raise clarity on average
      (b[question, clarity] ≥ 0).

    Additionally we keep each A_s close to I for numerical stability.
    """

    A = transition.A     # [S, K, K]
    b = transition.bias  # [S, K]

    S_IDX = {s: i for i, s in enumerate(STRATEGIES)}
    A_IDX = {a: i for i, a in enumerate(STATE_AXES)}

    pen = A.new_zeros(())
    delta = 0.05  # slack for the diagonal priors

    # --- bias-level priors -------------------------------------------------
    pen = pen + F.relu( b[S_IDX["safety_referral"],    A_IDX["distress"]]).pow(2)
    pen = pen + F.relu(-b[S_IDX["empathy"],            A_IDX["alliance"]]).pow(2)
    pen = pen + F.relu( b[S_IDX["reframe"],            A_IDX["rigidity"]]).pow(2)
    pen = pen + F.relu(-b[S_IDX["action_suggestion"],  A_IDX["readiness"]]).pow(2)
    pen = pen + F.relu(-b[S_IDX["summarization"],      A_IDX["clarity"]]).pow(2)
    pen = pen + F.relu(-b[S_IDX["question"],           A_IDX["clarity"]]).pow(2)

    # v3 additions: tighten coverage on the *rigidity* axis.  In v2 we saw
    # ``b[empathy, rigidity] = +44e-3`` and ``b[summarisation, rigidity] =
    # +37e-3`` — both clinically wrong (supportive talk moves should not
    # systematically amplify rigid thinking).  These absolute priors close
    # that gap without blocking the much stronger ``b[reframe, rigidity]``
    # reduction that L_sep differentially demands.
    pen = pen + F.relu( b[S_IDX["empathy"],       A_IDX["rigidity"]]).pow(2)
    pen = pen + F.relu( b[S_IDX["summarization"], A_IDX["rigidity"]]).pow(2)
    # and generally: reflection / question should not raise rigidity either.
    pen = pen + F.relu( b[S_IDX["reflection"],    A_IDX["rigidity"]]).pow(2)
    pen = pen + F.relu( b[S_IDX["question"],      A_IDX["rigidity"]]).pow(2)

    # --- diagonal priors on A_s (self-coupling) ---------------------------
    # safety_referral must not amplify distress (diag ≤ 1)
    pen = pen + F.relu(A[S_IDX["safety_referral"], A_IDX["distress"], A_IDX["distress"]] - 1.0).pow(2)
    # reframe must not amplify rigidity
    pen = pen + F.relu(A[S_IDX["reframe"], A_IDX["rigidity"], A_IDX["rigidity"]] - 1.0).pow(2)
    # empathy should keep alliance (diag ≥ 1 −δ)
    pen = pen + F.relu(1.0 - delta
                       - A[S_IDX["empathy"], A_IDX["alliance"], A_IDX["alliance"]]).pow(2)
    # summarization should keep clarity
    pen = pen + F.relu(1.0 - delta
                       - A[S_IDX["summarization"], A_IDX["clarity"], A_IDX["clarity"]]).pow(2)

    # --- off-diagonal prior: supportive actions shouldn't flip sign -------
    # Empathy off-diag from distress to alliance ought to be non-negative:
    # calming distress should not *reduce* alliance.
    pen = pen + F.relu(-A[S_IDX["empathy"], A_IDX["alliance"], A_IDX["distress"]]).pow(2)

    # --- stability: keep each A_s close to I ------------------------------
    I = torch.eye(N_STATE, device=A.device).unsqueeze(0).expand_as(A)
    pen = pen + 1e-2 * (A - I).pow(2).mean()
    return pen


def strategy_separation_loss(transition) -> torch.Tensor:
    """Differential priors *between* strategies + axis-wise dispersion.

    A plain ``(A_s - I)^2`` regularizer (present in ``L_consist``) pulls every
    strategy toward the same identity dynamics; without a counter-force the
    learned ``A_s`` and ``b_s`` collapse to a single shared operator and the
    per-strategy intervention probe becomes uninformative.  This term injects
    two complementary pushes:

    (1) **Pairwise differential hinges** — ordered inequalities motivated by
        the clinical literature in 思想.md:

        - ``b[reframe, rigidity]    <  b[empathy, rigidity]   - m``
          (a reframe should reduce rigidity *more* than plain empathy).
        - ``b[summarization, clarity] > b[empathy, clarity]   + m``
          (summary should raise clarity more than empathy).
        - ``b[question, clarity]      > b[reflection, clarity] + m``
          (probing questions clarify more than mirroring).
        - ``b[action_suggestion, readiness] > b[reflection, readiness] + m``
          (action suggestions raise readiness more than reflection).
        - ``b[safety_referral, distress] < b[empathy, distress] - m``
          (safety referral should dominate distress reduction in crises).

        Each hinge is a squared ReLU with margin ``m = 0.005`` on the raw
        bias scale.

    (2) **Axis-wise dispersion** — for each axis ``a``, reward the spread of
        ``b_{·,a}`` across the 7 strategies by subtracting its mean-free
        squared norm (in a *soft* way: a small negative variance, clipped).
        This encodes "at least one of the seven strategies should dominate on
        each axis" and is what actually breaks the collapsed minimum.
    """

    b = transition.bias  # [S, K]
    S_IDX = {s: i for i, s in enumerate(STRATEGIES)}
    A_IDX = {a: i for i, a in enumerate(STATE_AXES)}
    m = 5e-3

    pen = b.new_zeros(())
    # Differential bias hinges ------------------------------------------------
    # reframe reduces rigidity MORE than empathy
    pen = pen + F.relu(
        b[S_IDX["reframe"],           A_IDX["rigidity"]]
        - b[S_IDX["empathy"],         A_IDX["rigidity"]] + m
    ).pow(2)
    # summarisation raises clarity MORE than empathy
    pen = pen + F.relu(
        b[S_IDX["empathy"],           A_IDX["clarity"]]
        - b[S_IDX["summarization"],   A_IDX["clarity"]] + m
    ).pow(2)
    # probing questions clarify MORE than mirroring
    pen = pen + F.relu(
        b[S_IDX["reflection"],        A_IDX["clarity"]]
        - b[S_IDX["question"],        A_IDX["clarity"]] + m
    ).pow(2)
    # action_suggestion raises readiness MORE than reflection
    pen = pen + F.relu(
        b[S_IDX["reflection"],        A_IDX["readiness"]]
        - b[S_IDX["action_suggestion"], A_IDX["readiness"]] + m
    ).pow(2)
    # safety_referral reduces distress MORE than plain empathy
    pen = pen + F.relu(
        b[S_IDX["empathy"],           A_IDX["distress"]]
        - b[S_IDX["safety_referral"], A_IDX["distress"]] + m
    ).pow(2)

    # v3 additions: richer rigidity coverage -----------------------------------
    # reframe reduces rigidity MORE than reflection  (reflection is mild)
    pen = pen + F.relu(
        b[S_IDX["reframe"],           A_IDX["rigidity"]]
        - b[S_IDX["reflection"],      A_IDX["rigidity"]] + m
    ).pow(2)
    # reframe reduces rigidity MORE than summarisation
    pen = pen + F.relu(
        b[S_IDX["reframe"],           A_IDX["rigidity"]]
        - b[S_IDX["summarization"],   A_IDX["rigidity"]] + m
    ).pow(2)
    # question raises clarity MORE than empathy (targeted probing vs support)
    pen = pen + F.relu(
        b[S_IDX["empathy"],           A_IDX["clarity"]]
        - b[S_IDX["question"],        A_IDX["clarity"]] + m
    ).pow(2)

    # Axis-wise dispersion reward (soft, capped) -----------------------------
    # var over the 7 strategies, per axis.  We *subtract* a capped variance
    # so the optimizer is rewarded for a bit of spread but doesn't blow up.
    var_axis = b.var(dim=0, unbiased=False)          # [K]
    # cap the reward at variance=1e-3 per axis, so we don't push to infinity.
    disp_reward = torch.clamp(var_axis, max=1e-3).sum()
    pen = pen - 1.0 * disp_reward
    return pen


def safety_loss(
    risk_mask: torch.Tensor,                # [B] bool
    strategy_logits: torch.Tensor,          # [B, 7]
    z_pred: torch.Tensor,                   # [B, 5]
    z_target: torch.Tensor,                 # [B, 5]
) -> torch.Tensor:
    """When risk is present, strategy should lean toward `safety_referral`,
    and predicted distress should remain *above* target distress (no false
    minimization).
    """

    if risk_mask.sum() == 0:
        return strategy_logits.new_zeros(())

    sr = STRATEGIES.index("safety_referral")
    probs = F.softmax(strategy_logits, dim=-1)
    # Margin hinge: want P(safety_referral) ≥ 0.4 on risk samples.
    m = 0.4
    lose_strategy = F.relu(m - probs[:, sr])
    # Distress minimization check: penalty if predicted d < target d - 0.1 under risk.
    d_pred = z_pred[:, STATE_AXES.index("distress")]
    d_true = z_target[:, STATE_AXES.index("distress")]
    lose_distress = F.relu(d_true - 0.1 - d_pred)
    total = (lose_strategy + lose_distress) * risk_mask.float()
    return total.sum() / risk_mask.float().sum().clamp_min(1.0)


def heteroscedastic_state_nll(
    z_pred: torch.Tensor,            # [B, K]
    z_target: torch.Tensor,          # [B, K]
    log_var: torch.Tensor | None,    # [B, K] or None
    valid: torch.Tensor | None,      # [B] bool or None
) -> torch.Tensor:
    """Per-axis Gaussian NLL with learnt variance ``sigma^2 = exp(log_var)``.

    Encourages the head to admit *uncertainty* on noisy weak labels rather
    than overfit them.  Falls back to plain MSE when ``log_var`` is ``None``.
    """

    if log_var is None:
        return masked_mse(z_pred, z_target, valid)
    inv_var = torch.exp(-log_var)
    sq = (z_pred - z_target).pow(2)
    nll = 0.5 * (sq * inv_var + log_var)
    if valid is not None:
        if valid.dim() < nll.dim():
            valid_e = valid.unsqueeze(-1).expand_as(nll)
        else:
            valid_e = valid
        denom = valid_e.float().sum().clamp_min(1.0)
        return (nll * valid_e.float()).sum() / denom
    return nll.mean()


# ---------------------------------------------------------------------------
# v4: per-example counterfactual contrastive loss
# ---------------------------------------------------------------------------

# Each tuple: (strategy_pos, strategy_neg, axis, sign, weight)
#   sign = +1 means we want  Δz[a] under pos to be HIGHER than under neg.
#   sign = -1 means we want  Δz[a] under pos to be LOWER  than under neg.
# These pair-wise priors *condition on z_t and the observed context* — i.e.
# they're applied per-example rather than only on the parameter ``b``.
COUNTERFACTUAL_PAIRS: tuple[tuple[str, str, str, int, float], ...] = (
    # cognitive restructuring
    ("reframe",            "empathy",            "rigidity",  -1, 1.0),
    ("reframe",            "reflection",         "rigidity",  -1, 1.0),
    ("reframe",            "summarization",      "rigidity",  -1, 0.7),
    # alliance & warmth
    ("empathy",            "question",           "alliance",  +1, 1.0),
    ("reflection",         "question",           "alliance",  +1, 0.7),
    # focus & clarity
    ("summarization",      "empathy",            "clarity",   +1, 1.0),
    ("question",           "empathy",            "clarity",   +1, 0.7),
    ("question",           "reflection",         "clarity",   +1, 1.0),
    # behavioural change motivation
    ("action_suggestion",  "reflection",         "readiness", +1, 1.0),
    ("action_suggestion",  "empathy",            "readiness", +1, 0.7),
    # crisis routing dominates on distress
    ("safety_referral",    "empathy",            "distress",  -1, 1.5),
    ("safety_referral",    "question",           "distress",  -1, 1.5),
    # warmth shouldn't increase distress
    ("empathy",            "question",           "distress",  -1, 0.4),
)


def counterfactual_separation_loss(
    z_t: torch.Tensor,                 # [B, K]
    z_cf: torch.Tensor,                # [B, S, K]  z_{t+1}^{(s)}
    margin: float = 0.05,
) -> torch.Tensor:
    """Per-example contrastive priors over counterfactual ``Δz`` deltas.

    ``z_cf[b, s, a]`` is what the model predicts the next state's axis ``a``
    would be if the counsellor took strategy ``s`` at example ``b``.

    For each pair ``(s_pos, s_neg, a, sign)`` we want

        sign * ((Δz_pos[a] - Δz_neg[a]) - margin) ≥ 0

    where Δz_s = z_cf[s] - z_t.  Hinge-square: ``relu(...)^2``.
    """

    if z_cf.dim() != 3:
        return z_cf.new_zeros(())
    delta = z_cf - z_t.unsqueeze(1)               # [B, S, K]
    S_IDX = {s: i for i, s in enumerate(STRATEGIES)}
    A_IDX = {a: i for i, a in enumerate(STATE_AXES)}

    pen = z_cf.new_zeros(())
    for s_pos, s_neg, a, sign, w in COUNTERFACTUAL_PAIRS:
        pi, ni, ai = S_IDX[s_pos], S_IDX[s_neg], A_IDX[a]
        diff = delta[:, pi, ai] - delta[:, ni, ai]   # [B]
        # We want  sign * diff >= margin   ->   margin - sign*diff <= 0.
        violation = F.relu(margin - sign * diff)
        pen = pen + w * violation.pow(2).mean()
    return pen


# ---------------------------------------------------------------------------
# v4: independent safety-router loss (ordinal-aware focal CE + recall weight)
# ---------------------------------------------------------------------------


def _focal_ce(logits: torch.Tensor, targets: torch.Tensor,
              gamma: float = 2.0, weights: torch.Tensor | None = None) -> torch.Tensor:
    """Multi-class focal cross-entropy. Stable under bf16 forward.

    ``logits``: [B, C].  ``targets``: [B] long.  ``weights``: optional [C].
    """

    log_probs = F.log_softmax(logits.float(), dim=-1)
    probs = log_probs.exp()
    if weights is None:
        weights = log_probs.new_ones(logits.size(-1))
    nll = -log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    p_t = probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1).clamp(min=1e-6)
    w_t = weights[targets]
    loss = w_t * (1.0 - p_t).pow(gamma) * nll
    return loss.mean()


def safety_router_loss(
    risk_logits: torch.Tensor,         # [B, 4]
    severity_target: torch.Tensor,     # [B] long in [0,3]
    *,
    focal_gamma: float = 2.0,
    recall_weight: float = 3.0,
    pos_weight: float = 8.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Multi-component safety loss.

    1. **Focal multi-class CE** with linearly-increasing class weights
       ``[1, 1.5, recall_weight, recall_weight^2]`` so severe / imminent
       errors dominate the gradient.

    2. **Binary "severe-or-worse" BCE** on ``logsumexp(logits[:, ≥severe])``
       vs target ``severity ≥ severe``.  This is the gate the inference
       router actually uses; we calibrate it directly with ``pos_weight``.

    3. **Recall-margin penalty**: if the predicted severity for a
       severe / imminent target falls below severe (i.e. the gate would
       miss the alarm), add a hinge that pushes the severe-vs-rest
       margin up by 0.5 logits.
    """

    device = risk_logits.device
    weights = risk_logits.new_tensor([1.0, 1.5, recall_weight, recall_weight * 1.5])
    L_focal = _focal_ce(risk_logits, severity_target, gamma=focal_gamma, weights=weights)

    # Binary severe-vs-rest gate.
    severe_mask = (severity_target >= SEVERE_INDEX).float()
    severe_logit = torch.logsumexp(risk_logits[:, SEVERE_INDEX:], dim=-1) \
                   - torch.logsumexp(risk_logits[:, :SEVERE_INDEX], dim=-1)
    pw = risk_logits.new_tensor(float(pos_weight))
    L_gate = F.binary_cross_entropy_with_logits(
        severe_logit, severe_mask, pos_weight=pw,
    )

    # Recall-margin hinge: severe-vs-rest logit should exceed +0.5 on positives.
    margin = 0.5
    if severe_mask.sum() > 0:
        L_margin = F.relu(margin - severe_logit[severe_mask.bool()]).pow(2).mean()
    else:
        L_margin = risk_logits.new_zeros(())

    L_total = L_focal + L_gate + 0.5 * L_margin
    return L_total, {
        "focal": L_focal.detach(),
        "gate":  L_gate.detach(),
        "margin": L_margin.detach(),
    }


def compute_losses(
    *,
    cfg: LossConfig,
    gen_logits: torch.Tensor | None,
    gen_labels: torch.Tensor | None,
    z_pred: torch.Tensor,
    strategy_logits: torch.Tensor,
    outcome_pred: dict,
    transition_module,
    z_trans_pred: torch.Tensor | None,
    targets: dict,
    commit_loss: torch.Tensor,
    z_log_var: torch.Tensor | None = None,
    z_cf_all: torch.Tensor | None = None,
    risk_logits: torch.Tensor | None = None,
    residual_adv_logit: torch.Tensor | None = None,
    q_values: torch.Tensor | None = None,
    transition_value_logit: torch.Tensor | None = None,
    clinical_priors: ClinicalPriors | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Sum up all enabled terms. ``targets`` keys:

    * ``state_target``:      [B, 5] float in [0,1]
    * ``state_next_target``: [B, 5] or None
    * ``state_valid``:       [B] bool
    * ``strategy_target``:   [B] long in [0, 7)
    * ``strategy_valid``:    [B] bool
    * ``uptake_target``:     [B] float in {0,1}
    * ``uptake_valid``:      [B] bool
    * ``quality_target``:    [B, 4] float in [0,1]
    * ``quality_valid``:     [B] bool
    * ``risk_mask``:         [B] bool
    """

    logs: dict[str, torch.Tensor] = {}
    zero = z_pred.new_zeros(())

    if gen_logits is not None and gen_labels is not None:
        L_gen = gen_loss_from_logits(gen_logits, gen_labels)
    else:
        L_gen = zero
    logs["L_gen"] = L_gen.detach()

    # State (weak-supervised). v4 swaps in heteroscedastic NLL when the head
    # emits a per-axis log-variance, which both calibrates uncertainty and
    # down-weights noisy lexicon labels (reviewer C3).
    if cfg.use_state:
        if cfg.use_uncertainty and z_log_var is not None:
            L_state = heteroscedastic_state_nll(
                z_pred, targets["state_target"], z_log_var, targets.get("state_valid"),
            )
        else:
            L_state = masked_mse(z_pred, targets["state_target"], targets.get("state_valid"))
    else:
        L_state = zero
    logs["L_state"] = L_state.detach()

    # Strategy (CE).
    if cfg.use_strategy:
        valid = targets.get("strategy_valid")
        tgt = targets["strategy_target"]
        if valid is not None and valid.sum() > 0:
            L_strategy = F.cross_entropy(strategy_logits[valid], tgt[valid])
        else:
            L_strategy = zero
    else:
        L_strategy = zero
    logs["L_strategy"] = L_strategy.detach()

    # Transition.
    if cfg.use_transition and z_trans_pred is not None:
        valid = targets.get("state_next_valid")
        next_tgt = targets.get("state_next_target")
        if next_tgt is not None and valid is not None and valid.sum() > 0:
            L_trans = masked_mse(z_trans_pred, next_tgt, valid)
        else:
            L_trans = zero
    else:
        L_trans = zero
    logs["L_trans"] = L_trans.detach()

    # Outcome.
    if cfg.use_outcome:
        up_logit = outcome_pred["uptake_logit"]
        up_tgt = targets.get("uptake_target")
        up_valid = targets.get("uptake_valid")
        if up_tgt is not None and up_valid is not None and up_valid.sum() > 0:
            if cfg.uptake_pos_weight and cfg.uptake_pos_weight > 0:
                pw = up_logit.new_tensor(float(cfg.uptake_pos_weight))
                L_up = F.binary_cross_entropy_with_logits(
                    up_logit[up_valid], up_tgt[up_valid], pos_weight=pw
                )
            else:
                L_up = F.binary_cross_entropy_with_logits(
                    up_logit[up_valid], up_tgt[up_valid]
                )
        else:
            L_up = zero
        L_rank = binary_pairwise_rank_loss(up_logit, up_tgt, up_valid) if up_tgt is not None else zero
        q_pred = outcome_pred["quality"]
        q_tgt = targets.get("quality_target")
        q_valid = targets.get("quality_valid")
        if q_tgt is not None and q_valid is not None and q_valid.sum() > 0:
            L_q = F.mse_loss(q_pred[q_valid], q_tgt[q_valid])
        else:
            L_q = zero
        L_outcome = L_up + L_q
    else:
        L_outcome = zero
        L_rank = zero
    logs["L_outcome"] = L_outcome.detach()
    logs["L_outcome_rank"] = L_rank.detach()

    # Structural consistency on A, b.
    if cfg.use_consist:
        L_consist = structural_consistency_loss(transition_module)
    else:
        L_consist = zero
    logs["L_consist"] = L_consist.detach()

    # Strategy separation (differential priors + dispersion reward).
    if cfg.use_sep and cfg.lam_sep > 0:
        L_sep = strategy_separation_loss(transition_module)
    else:
        L_sep = zero
    logs["L_sep"] = L_sep.detach()

    # Safety.
    if cfg.use_safety:
        L_safe = safety_loss(
            risk_mask=targets.get("risk_mask", torch.zeros(z_pred.size(0), dtype=torch.bool, device=z_pred.device)),
            strategy_logits=strategy_logits,
            z_pred=z_pred,
            z_target=targets["state_target"],
        )
    else:
        L_safe = zero
    logs["L_safety"] = L_safe.detach()

    logs["L_commit"] = commit_loss.detach() if isinstance(commit_loss, torch.Tensor) else torch.tensor(float(commit_loss))

    # ------------------------------------------------------------------
    # v4 additions
    # ------------------------------------------------------------------

    # Bayesian clinical prior on b_s and A_s. The caller is expected to
    # pre-place the priors on the correct device; we move only when needed.
    if cfg.use_bayes and cfg.lam_bayes > 0 and clinical_priors is not None:
        if clinical_priors.mu.device != transition_module.bias.device:
            clinical_priors = clinical_priors.to(
                transition_module.bias.device, transition_module.bias.dtype,
            )
        L_bayes = gaussian_prior_loss(
            transition_module.bias, transition_module.A, clinical_priors,
        )
    else:
        L_bayes = zero
    logs["L_bayes"] = L_bayes.detach()

    # Per-example counterfactual contrastive loss.
    if cfg.use_counterfactual and cfg.lam_counterfactual > 0 and z_cf_all is not None:
        prev = targets.get("state_target")
        if prev is None:
            L_cf = zero
        else:
            L_cf = counterfactual_separation_loss(prev, z_cf_all, margin=cfg.cf_margin)
    else:
        L_cf = zero
    logs["L_cf"] = L_cf.detach()

    # Independent safety-router loss.
    if cfg.use_router and cfg.lam_router > 0 and risk_logits is not None:
        sev_tgt = targets.get("risk_severity")
        if sev_tgt is None:
            # Back-compat: derive a binary 0/2 severity from risk_mask.
            mask = targets.get("risk_mask",
                               torch.zeros(z_pred.size(0), dtype=torch.bool, device=z_pred.device))
            sev_tgt = mask.long() * SEVERE_INDEX
        L_router, router_logs = safety_router_loss(
            risk_logits, sev_tgt,
            focal_gamma=cfg.router_focal_gamma,
            recall_weight=cfg.router_recall_weight,
            pos_weight=cfg.router_pos_weight,
        )
        for k, v in router_logs.items():
            logs[f"L_router_{k}"] = v
    else:
        L_router = zero
    logs["L_router"] = L_router.detach()

    # v5 axis decorrelation on the predicted ``z`` (prevents axis collapse).
    if getattr(cfg, "lam_decor", 0.0) > 0:
        L_decor = axis_decorrelation_loss(
            z_pred,
            running_mean=targets.get("z_running_mean"),
            running_var=targets.get("z_running_var"),
        )
    else:
        L_decor = zero
    logs["L_decor"] = L_decor.detach()

    # v6 adversarial mediation: residual variables should not retain outcome
    # signal.  The gradient reversal is inside the adversary head, so this BCE
    # simultaneously trains the adversary and pushes the shared residual path to
    # hide uptake information.
    if getattr(cfg, "use_adversary", False) and cfg.lam_adv > 0 and residual_adv_logit is not None:
        up_tgt = targets.get("uptake_target")
        up_valid = targets.get("uptake_valid")
        if up_tgt is not None and up_valid is not None and up_valid.sum() > 0:
            L_adv = F.binary_cross_entropy_with_logits(
                residual_adv_logit[up_valid], up_tgt[up_valid]
            )
        else:
            L_adv = zero
    else:
        L_adv = zero
    logs["L_adv"] = L_adv.detach()

    # v6 Q-planner: (i) pairwise clinical preferences over candidate strategy
    # values and (ii) a weak observed-policy target: the Q-value of the observed
    # strategy should predict next-turn uptake.
    if getattr(cfg, "use_q_planner", False) and q_values is not None:
        L_pref = q_preference_loss(q_values)
        up_tgt = targets.get("uptake_target")
        up_valid = targets.get("uptake_valid")
        strat_tgt = targets.get("strategy_target")
        strat_valid = targets.get("strategy_valid")
        valid = None
        if up_valid is not None and strat_valid is not None:
            valid = up_valid & strat_valid
        if (
            cfg.lam_q > 0 and up_tgt is not None and strat_tgt is not None
            and valid is not None and valid.sum() > 0
        ):
            strat_idx = strat_tgt.clamp(0, q_values.size(1) - 1)
            q_taken = q_values[torch.arange(q_values.size(0), device=q_values.device), strat_idx]
            L_qplan = F.binary_cross_entropy_with_logits(q_taken[valid], up_tgt[valid])
        else:
            L_qplan = zero
    else:
        L_pref = zero
        L_qplan = zero
    logs["L_pref"] = L_pref.detach()
    logs["L_qplan"] = L_qplan.detach()

    if getattr(cfg, "use_transition_value", False) and transition_value_logit is not None:
        up_tgt = targets.get("uptake_target")
        up_soft = targets.get("uptake_soft_target", up_tgt)
        up_valid = targets.get("uptake_valid")
        if up_tgt is not None and up_valid is not None and up_valid.sum() > 0:
            L_value_rank = binary_pairwise_rank_loss(transition_value_logit, up_tgt, up_valid)
            L_value_reg = F.binary_cross_entropy_with_logits(
                transition_value_logit[up_valid], up_soft[up_valid].to(transition_value_logit.dtype),
            )
        else:
            L_value_rank = zero
            L_value_reg = zero
    else:
        L_value_rank = zero
        L_value_reg = zero
    logs["L_value_rank"] = L_value_rank.detach()
    logs["L_value_reg"] = L_value_reg.detach()

    L_total = (
        cfg.lam_gen * L_gen
        + cfg.lam_state * L_state
        + cfg.lam_strategy * L_strategy
        + cfg.lam_transition * L_trans
        + cfg.lam_outcome * L_outcome
        + getattr(cfg, "lam_outcome_rank", 0.0) * L_rank
        + cfg.lam_consist * L_consist
        + cfg.lam_sep * L_sep
        + cfg.lam_safety * L_safe
        + cfg.lam_commit * (commit_loss if isinstance(commit_loss, torch.Tensor) else zero)
        + cfg.lam_bayes * L_bayes
        + cfg.lam_counterfactual * L_cf
        + cfg.lam_router * L_router
        + getattr(cfg, "lam_decor", 0.0) * L_decor
        + getattr(cfg, "lam_adv", 0.0) * L_adv
        + getattr(cfg, "lam_pref", 0.0) * L_pref
        + getattr(cfg, "lam_q", 0.0) * L_qplan
        + getattr(cfg, "lam_value_rank", 0.0) * L_value_rank
        + getattr(cfg, "lam_value_reg", 0.0) * L_value_reg
    )
    logs["L_total"] = L_total.detach()
    return L_total, logs
