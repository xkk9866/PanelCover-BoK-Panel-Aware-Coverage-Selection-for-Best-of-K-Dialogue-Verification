"""Task heads that sit on top of the LLM backbone.

All heads operate on a pooled context representation ``h_ctx`` of shape
``[B, D]`` taken from the last hidden state at the position immediately before
the counselor reply (i.e. the last token of the context).

They are intentionally small: most of the modeling capacity already lives in
the backbone + LoRA. These heads make the latent axes interpretable, plug
action dynamics into the generation, and provide the outcome supervision
pathway demanded by the paper's thesis.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .constants import N_LATENT_ACTION, N_STATE, N_STRATEGY, PREFIX_LEN


def _mlp(in_dim: int, hidden: int, out_dim: int, dropout: float = 0.1) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden, out_dim),
    )


class StateHead(nn.Module):
    """``q_phi(z | h_ctx)`` for 5 interpretable axes in [0,1].

    v4 also emits a per-axis log-variance ``log_var`` that is consumed by the
    heteroscedastic Gaussian NLL state loss (``losses.heteroscedastic_state_nll``)
    and by the safety router's uncertainty-aware abstention rule.

    The variance head is gated by ``emit_uncertainty`` so v3 / ablations that
    don't need it pay zero compute and keep the same parameter count.
    """

    def __init__(self, hidden_size: int, proj: int = 256, emit_uncertainty: bool = False):
        super().__init__()
        self.net = _mlp(hidden_size, proj, N_STATE)
        self.emit_uncertainty = emit_uncertainty
        if emit_uncertainty:
            self.log_var_net = _mlp(hidden_size, proj, N_STATE)

    def forward(self, h: torch.Tensor):  # [B,D] -> [B,5] (+ [B,5] log-var)
        z = torch.sigmoid(self.net(h))
        if self.emit_uncertainty:
            # Clamp log_var to a sane range so the NLL term cannot diverge if the
            # head is briefly miscalibrated early in training.
            log_var = self.log_var_net(h).clamp(-6.0, 1.5)
            return z, log_var
        return z


class AxisSubspaceStateHead(nn.Module):
    """v6 axis-subspace state head.

    Each clinical axis keeps a scalar anchor ``s_k`` for inspection and weak
    supervision, plus a small residual subspace ``r_k`` for outcome-relevant
    information that cannot fit into a single scalar.  The public ``z`` remains
    [B, 5] for transition / legacy diagnostics; ``z_repr`` is [B, 5*(1+d)] and
    feeds the v6 outcome and Q heads.
    """

    def __init__(
        self,
        hidden_size: int,
        proj: int = 256,
        residual_dim: int = 8,
        emit_uncertainty: bool = False,
    ):
        super().__init__()
        self.residual_dim = int(residual_dim)
        self.scalar_net = _mlp(hidden_size, proj, N_STATE)
        self.residual_net = _mlp(hidden_size, proj, N_STATE * self.residual_dim)
        self.emit_uncertainty = emit_uncertainty
        if emit_uncertainty:
            self.log_var_net = _mlp(hidden_size, proj, N_STATE)

    @property
    def repr_dim(self) -> int:
        return N_STATE * (1 + self.residual_dim)

    def forward(self, h: torch.Tensor):
        z_scalar = torch.sigmoid(self.scalar_net(h))
        residual = torch.tanh(self.residual_net(h)).view(h.size(0), N_STATE, self.residual_dim)
        z_repr = torch.cat([z_scalar.unsqueeze(-1), residual], dim=-1).flatten(1)
        if self.emit_uncertainty:
            log_var = self.log_var_net(h).clamp(-6.0, 1.5)
            return z_scalar, z_repr, residual, log_var
        return z_scalar, z_repr, residual


class StrategyHead(nn.Module):
    """Explicit strategy classifier (7 labels)."""

    def __init__(self, hidden_size: int, proj: int = 256):
        super().__init__()
        self.net = _mlp(hidden_size, proj, N_STRATEGY)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)  # logits


# ---------------------------------------------------------------------------
# v4: Independent risk severity head & generation-time safety router
# ---------------------------------------------------------------------------

# Severity ordinal: index 0 = none, 1 = mild, 2 = severe, 3 = imminent.
# Anything ``>= SEVERE_THRESHOLD`` (default 2) routes generation through the
# safety template and overrides the strategy-conditioned response policy.
SEVERITY_LABELS = ("none", "mild", "severe", "imminent")
N_SEVERITY = len(SEVERITY_LABELS)
SEVERE_INDEX = 2


class RiskSeverityHead(nn.Module):
    """Independent ordinal-aware 4-class risk severity classifier.

    Reviewer-fix: in v3 we treated ``safety_referral`` as one of the 7
    strategy classes. In a long-tailed corpus the strategy softmax is
    dominated by frequent classes (question / reflection / empathy) and
    severe-risk turns *never* win top-1, even when the reply should be a
    crisis referral. v4 promotes risk to its own dedicated channel: a
    4-class severity classifier whose head is **decoupled** from
    strategy logits and trained with focal + recall-weighted loss on
    oversampled risk turns. The router rule lives in
    :func:`route_severe`.
    """

    def __init__(self, hidden_size: int, proj: int = 256):
        super().__init__()
        self.net = _mlp(hidden_size, proj, N_SEVERITY)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)  # logits [B, 4]


def route_severe(severity_logits: torch.Tensor, threshold: int = SEVERE_INDEX) -> torch.Tensor:
    """Predicate: ``True`` if argmax-severity ≥ threshold (severe / imminent).

    A sigmoid on the cumulative tail probability would be smoother but at
    inference time we only need a hard gate that flips the generation policy
    to the crisis-support template.
    """

    pred = severity_logits.argmax(dim=-1)
    return pred >= threshold


class LatentActionHead(nn.Module):
    """Residual latent action with a small learnable codebook.

    Forward returns (latent_code_probs, latent_embed, commit_loss).
    We use a Gumbel-softmax relaxation during training so the downstream
    transition / prefix modules receive a differentiable code.
    """

    def __init__(self, hidden_size: int, k: int = N_LATENT_ACTION, code_dim: int = 32):
        super().__init__()
        self.proj = _mlp(hidden_size, 256, k)
        self.codebook = nn.Embedding(k, code_dim)
        self.k = k
        self.code_dim = code_dim

    def forward(self, h: torch.Tensor, tau: float = 1.0, hard: bool = False):
        logits = self.proj(h)                          # [B, K]
        probs = F.softmax(logits, dim=-1)
        # Gumbel-softmax sample
        y = F.gumbel_softmax(logits, tau=tau, hard=hard)  # [B, K]
        emb = y @ self.codebook.weight                  # [B, code_dim]
        # Commitment-style regularizer: encourage confident codes.
        ent = -(probs * probs.clamp_min(1e-8).log()).sum(-1).mean()
        commit = 0.1 * ent
        return {"probs": probs, "logits": logits, "embed": emb, "commit": commit, "onehot_st": y}


@dataclass
class TransitionOutput:
    z_pred: torch.Tensor       # [B, 5]
    obs_feat: torch.Tensor     # [B, Fobs]


class ControlledTransition(nn.Module):
    """``z_{t+1} = sum_s u_s * ( A_s @ z_t + B_s @ g(x) + b_s ) + residual``.

    * ``u_s`` are soft probabilities over 7 strategies (broadcast by the latent
      residual embedding, added as a bias).
    * ``g(x)`` is a projection of the context features.

    Sign constraints (implemented as a penalty in ``losses.py``):

    * ``empathy`` row on alliance >= 0
    * ``safety_referral`` row on distress <= 0 (reduces distress)
    * ``reframe`` row on rigidity <= 0 (reduces rigidity) *when alliance high*
    """

    def __init__(self, obs_feat: int = 64, latent_dim: int = 32):
        super().__init__()
        # A_s: [S, K, K] initialized close to identity so transitions are stable.
        self.A = nn.Parameter(torch.eye(N_STATE).unsqueeze(0).repeat(N_STRATEGY, 1, 1))
        # B_s: [S, K, Fobs]
        self.B = nn.Parameter(torch.zeros(N_STRATEGY, N_STATE, obs_feat))
        # b_s: [S, K]
        self.bias = nn.Parameter(torch.zeros(N_STRATEGY, N_STATE))
        # latent residual bias lookup (from latent action embed)
        self.lat_bias = nn.Linear(latent_dim, N_STATE, bias=False)
        nn.init.normal_(self.lat_bias.weight, std=1e-3)

    def forward(
        self,
        z_prev: torch.Tensor,          # [B, 5]
        strategy_probs: torch.Tensor,  # [B, 7]
        obs_feat: torch.Tensor,        # [B, Fobs]
        latent_embed: torch.Tensor,    # [B, latent_dim]
    ) -> torch.Tensor:
        # Subscripts: b=batch, s=strategy(7), i/j=state(5), f=obs_feat.
        # sum_s u_s * (A_s @ z) -> [B, I]
        Az = torch.einsum("bs,sij,bj->bi", strategy_probs, self.A, z_prev)
        # sum_s u_s * (B_s @ obs) -> [B, I]
        Bx = torch.einsum("bs,sif,bf->bi", strategy_probs, self.B, obs_feat)
        # sum_s u_s * b_s -> [B, I]
        bs = torch.einsum("bs,si->bi", strategy_probs, self.bias)
        lat = self.lat_bias(latent_embed)
        z_next = torch.sigmoid(Az + Bx + bs + lat)
        return z_next

    def counterfactual(
        self,
        z_prev: torch.Tensor,          # [B, 5]
        obs_feat: torch.Tensor,        # [B, Fobs]
        latent_embed: torch.Tensor,    # [B, latent_dim]
    ) -> torch.Tensor:
        """Compute ``z_{t+1}^{(s)}`` *for every strategy s simultaneously*.

        Unlike :meth:`forward` (which mixes the 7 strategies through soft
        ``u``), this method returns a tensor of shape ``[B, S, K]`` carrying
        the next-state under each *one-hot* counterfactual intervention.
        Used by ``losses.counterfactual_separation_loss`` to drive
        differential clinical priors at the **example** level (conditioning
        on ``z_t`` and ``g(x_t)``) rather than only at the parameter level.
        """

        # einsum subscripts: b=batch, s=strategy, i/j=state, f=obs_feat.
        Az = torch.einsum("sij,bj->bsi", self.A, z_prev)         # [B, S, I]
        Bx = torch.einsum("sif,bf->bsi", self.B, obs_feat)       # [B, S, I]
        b  = self.bias.unsqueeze(0)                              # [1, S, I]
        lat = self.lat_bias(latent_embed).unsqueeze(1)           # [B, 1, I]
        return torch.sigmoid(Az + Bx + b + lat)                  # [B, S, I]


class OutcomeHead(nn.Module):
    """Short- and long-term outcome predictor.

    Default (v1–v4): input is ``[z_t; strategy_probs; latent_embed]`` so the
    head can use the high-bandwidth ``lat_embed`` (32-dim) directly. Empirically
    this caused a **bottleneck failure**: gradient from ``L_outcome`` flowed
    through ``lat_embed`` rather than through the 5-dim ``z``, so ``z`` ended up
    a noisy clone of the weak-supervision lexicon and was *worse* than the
    lexicon at predicting uptake (Δ AUROC ≈ −0.25 on test).

    v5 fix (``z_only=True``): the outcome head consumes only ``z``. This
    removes the parallel pathway, *forces* ``z`` to carry every bit of
    outcome-relevant information, and converts the 5-dim latent into an actual
    information bottleneck. Subsequent diagnostics
    (:mod:`eval.latent_diagnosis`) then test whether ``z`` finally beats the
    lexicon and whether axis permutation finally hurts AUROC.
    """

    def __init__(
        self,
        latent_dim: int = 32,
        proj: int = 128,
        z_only: bool = False,
        axis_dropout_p: float = 0.0,
        z_dim: int = N_STATE,
        head_type: str = "mlp",
    ):
        super().__init__()
        self.z_only = z_only
        self.z_dim = int(z_dim)
        self.head_type = str(head_type)
        # v5 axis dropout: at training time we randomly drop axes of ``z``
        # before feeding it into the head. This prevents the head from
        # collapsing onto a single axis (e.g. only ``distress``) — every
        # axis has to carry usable signal because any of them might be
        # masked. Off at eval time (``self.training`` flag).
        self.axis_dropout_p = float(axis_dropout_p)
        if z_only:
            in_dim = self.z_dim
        else:
            in_dim = self.z_dim + N_STRATEGY + latent_dim
        self.backbone = _mlp(in_dim, proj, proj)
        self.uptake = nn.Linear(proj, 1)
        self.linear_uptake = nn.Linear(in_dim, 1)
        self.quality = nn.Linear(proj, 4)

    def _maybe_axis_dropout(self, z: torch.Tensor) -> torch.Tensor:
        if not self.training or self.axis_dropout_p <= 0.0:
            return z
        keep = 1.0 - self.axis_dropout_p
        # Sample once per batch element so axis usage is independent across
        # examples; rescale by 1/keep to preserve expectation.
        mask = torch.bernoulli(torch.full_like(z, keep))
        return z * mask / max(keep, 1e-6)

    def forward(self, z: torch.Tensor, u_probs: torch.Tensor, lat_embed: torch.Tensor) -> dict:
        z_in = self._maybe_axis_dropout(z)
        if self.z_only:
            x = z_in
        else:
            x = torch.cat([z_in, u_probs, lat_embed], dim=-1)
        h = self.backbone(x)
        mlp_logit = self.uptake(h).squeeze(-1)
        linear_logit = self.linear_uptake(x).squeeze(-1)
        if self.head_type == "linear":
            uptake_logit = linear_logit
        elif self.head_type == "ensemble":
            uptake_logit = 0.5 * (mlp_logit + linear_logit)
        else:
            uptake_logit = mlp_logit
        return {
            "uptake_logit": uptake_logit,
            "uptake_logit_mlp": mlp_logit,
            "uptake_logit_linear": linear_logit,
            "quality": torch.sigmoid(self.quality(h)),
        }


class _GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambd: float):
        ctx.lambd = float(lambd)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.lambd * grad_output, None


def grad_reverse(x: torch.Tensor, lambd: float = 1.0) -> torch.Tensor:
    return _GradientReverse.apply(x, lambd)


class ResidualAdversaryHead(nn.Module):
    """Predict outcome from bypass variables through a gradient-reversal layer.

    The adversary itself learns to predict uptake, while GRL pushes the shared
    representation to remove uptake information from ``[u_probs; lat_embed;
    h_ctx]``.  Evaluation can still read the adversary AUROC as residual
    leakage.
    """

    def __init__(self, hidden_size: int, latent_dim: int = 32, proj: int = 128):
        super().__init__()
        in_dim = hidden_size + N_STRATEGY + latent_dim
        self.net = _mlp(in_dim, proj, proj)
        self.uptake = nn.Linear(proj, 1)

    def forward(
        self,
        h_ctx: torch.Tensor,
        u_probs: torch.Tensor,
        lat_embed: torch.Tensor,
        grl_lambda: float = 1.0,
    ) -> torch.Tensor:
        x = torch.cat([h_ctx, u_probs, lat_embed], dim=-1)
        x = grad_reverse(x, grl_lambda)
        return self.uptake(self.net(x)).squeeze(-1)


class CounterfactualQHead(nn.Module):
    """Value head over counterfactual strategy choices.

    For every strategy s, reads ``[state_repr; onehot(s); z_next_cf[s]]`` and
    emits a scalar Q-value.  The planner can rank strategies by Q and the loss
    can train pairwise clinical preferences or uptake-based targets.
    """

    def __init__(self, state_dim: int = N_STATE, proj: int = 128):
        super().__init__()
        self.state_dim = int(state_dim)
        in_dim = self.state_dim + N_STRATEGY + N_STATE
        self.net = _mlp(in_dim, proj, proj)
        self.q = nn.Linear(proj, 1)

    def forward(self, state_repr: torch.Tensor, z_cf: torch.Tensor) -> torch.Tensor:
        B, S, _ = z_cf.shape
        state = state_repr.unsqueeze(1).expand(B, S, self.state_dim)
        eye = torch.eye(S, device=z_cf.device, dtype=z_cf.dtype).unsqueeze(0).expand(B, S, S)
        x = torch.cat([state, eye, z_cf], dim=-1)
        return self.q(self.net(x)).squeeze(-1)


class PrefixProjector(nn.Module):
    """Projects ``[z; u_probs; lat_embed]`` to ``PREFIX_LEN`` soft-prompt tokens.

    Used for optional generation conditioning (``cfg.use_prefix=True``):
    the prefix tokens are inserted between the context and the assistant reply
    in ``inputs_embeds``.
    """

    def __init__(self, hidden_size: int, latent_dim: int = 32, prefix_len: int = PREFIX_LEN):
        super().__init__()
        in_dim = N_STATE + N_STRATEGY + latent_dim
        self.prefix_len = prefix_len
        self.hidden_size = hidden_size
        self.proj = _mlp(in_dim, 512, prefix_len * hidden_size)

    def forward(self, z: torch.Tensor, u_probs: torch.Tensor, lat_embed: torch.Tensor) -> torch.Tensor:
        x = torch.cat([z, u_probs, lat_embed], dim=-1)
        out = self.proj(x)
        return out.view(x.shape[0], self.prefix_len, self.hidden_size)


class ObsProjector(nn.Module):
    """Reduces the backbone hidden size to a compact ``g(x)`` for transition."""

    def __init__(self, hidden_size: int, obs_feat: int = 64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(hidden_size, obs_feat), nn.GELU())

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)
