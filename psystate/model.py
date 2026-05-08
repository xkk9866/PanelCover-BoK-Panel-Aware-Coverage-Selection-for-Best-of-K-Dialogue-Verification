"""PsyStateModel: backbone + all auxiliary heads, single forward path.

Given a batch of counselor prediction examples, we:

1. run the LLM backbone on ``[context || reply]`` tokens;
2. pick the hidden state at the last context token as ``h_ctx``;
3. run the state / strategy / latent-action / outcome heads on ``h_ctx``;
4. compute the transition prediction using the previous client's weak state.

The LM loss uses the same forward-pass logits (no separate generation pass),
so the total cost is **one** backbone call per step.

For optional prefix conditioning we support ``--use_prefix`` which shifts
inputs_embeds by the projected prefix, at the cost of a second short forward
over only the reply segment. This is off by default for speed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn

from .constants import N_LATENT_ACTION, N_STATE, N_STRATEGY
from .heads import (
    AxisSubspaceStateHead,
    ControlledTransition,
    CounterfactualQHead,
    LatentActionHead,
    ObsProjector,
    OutcomeHead,
    PrefixProjector,
    ResidualAdversaryHead,
    RiskSeverityHead,
    StateHead,
    StrategyHead,
)


class TherapeuticValueHead(nn.Module):
    """Constrained therapeutic value from predicted state movement."""

    def __init__(self):
        super().__init__()
        init = torch.tensor([0.25, 0.20, 0.20, 0.20, 0.15], dtype=torch.float32)
        self.weight_logits = nn.Parameter(init.log())
        self.scale = nn.Parameter(torch.tensor(4.0))
        self.bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, z_t: torch.Tensor, z_next: torch.Tensor) -> torch.Tensor:
        delta = z_next - z_t
        signed = torch.stack(
            [-delta[:, 0], -delta[:, 1], delta[:, 2], delta[:, 3], delta[:, 4]],
            dim=-1,
        )
        weights = torch.softmax(self.weight_logits, dim=-1).to(signed.device, signed.dtype)
        value = (signed * weights).sum(dim=-1)
        return self.scale.clamp(0.1, 20.0) * value + self.bias

    def counterfactual(self, z_t: torch.Tensor, z_cf: torch.Tensor) -> torch.Tensor:
        B, S, K = z_cf.shape
        z_flat = z_t.unsqueeze(1).expand(B, S, K).reshape(B * S, K)
        cf_flat = z_cf.reshape(B * S, K)
        return self.forward(z_flat, cf_flat).view(B, S)


@dataclass
class PsyStateConfig:
    hidden_size: int = 4096
    obs_feat: int = 64
    latent_dim: int = 32
    n_strategy: int = N_STRATEGY
    n_state: int = N_STATE
    n_latent: int = N_LATENT_ACTION
    use_prefix: bool = False  # optional decoder prefix conditioning (slower)
    # Ablation toggles propagate to forward & loss.
    use_state: bool = True
    use_strategy: bool = True
    use_transition: bool = True
    use_outcome: bool = True
    # v4 toggles ---------------------------------------------------------
    use_safety_router: bool = False
    use_uncertainty: bool = False
    use_counterfactual: bool = False
    # v5 toggle: route outcome (and optional prefix) through z alone, removing
    # the high-bandwidth ``lat_embed`` parallel pathway so that ``z`` becomes
    # the actual information bottleneck for outcome prediction.
    z_only_outcome: bool = False
    # v5 axis-dropout probability for the OutcomeHead input. Forces every axis
    # of ``z`` to carry usable information (no axis collapse).
    axis_dropout_p: float = 0.0
    # v6: preserve scalar axis anchors but add per-axis residual subspaces for
    # outcome-relevant variation.  ``z`` remains [B,5]; ``z_repr`` is
    # [B, 5*(1+state_residual_dim)] and can feed outcome / Q heads.
    use_axis_subspace: bool = False
    state_residual_dim: int = 0
    # v6: adversarial mediation bottleneck and counterfactual Q planner.
    use_adversarial_bottleneck: bool = False
    use_q_planner: bool = False
    # v6.1: measurement-aware state.  The weak clinical state is no longer only
    # a target; it is treated as an observed measurement and the neural state
    # head learns a residual correction in logit space.  This avoids the failed
    # assumption that a 5-D state should emerge from hidden activations alone.
    use_observed_state_anchor: bool = False
    anchor_residual_scale: float = 1.0
    outcome_head_type: str = "mlp"
    use_reliability_gated_posterior: bool = False
    measurement_quality_dim: int = 4
    use_transition_value: bool = False
    transition_from_posterior: bool = False


class PsyStateModel(nn.Module):
    def __init__(self, backbone: nn.Module, cfg: PsyStateConfig):
        super().__init__()
        self.backbone = backbone
        self.cfg = cfg
        H = cfg.hidden_size

        if cfg.use_state and cfg.use_axis_subspace and cfg.state_residual_dim > 0:
            self.state_head = AxisSubspaceStateHead(
                H,
                residual_dim=cfg.state_residual_dim,
                emit_uncertainty=cfg.use_uncertainty,
            )
            state_repr_dim = self.state_head.repr_dim
        elif cfg.use_state:
            self.state_head = StateHead(H, emit_uncertainty=cfg.use_uncertainty)
            state_repr_dim = cfg.n_state
        else:
            self.state_head = None
            state_repr_dim = cfg.n_state
        self.strategy_head = StrategyHead(H) if cfg.use_strategy else None
        self.latent_action = LatentActionHead(H, k=cfg.n_latent, code_dim=cfg.latent_dim)
        self.obs_proj = ObsProjector(H, obs_feat=cfg.obs_feat) if cfg.use_transition else None
        self.transition = ControlledTransition(obs_feat=cfg.obs_feat, latent_dim=cfg.latent_dim) if cfg.use_transition else None
        self.outcome_head = (
            OutcomeHead(
                latent_dim=cfg.latent_dim,
                z_only=cfg.z_only_outcome,
                axis_dropout_p=cfg.axis_dropout_p,
                z_dim=state_repr_dim if cfg.z_only_outcome else cfg.n_state,
                head_type=cfg.outcome_head_type,
            )
            if cfg.use_outcome else None
        )
        self.prefix_proj = PrefixProjector(H, latent_dim=cfg.latent_dim) if cfg.use_prefix else None
        self.risk_head = RiskSeverityHead(H) if cfg.use_safety_router else None
        self.residual_adv = (
            ResidualAdversaryHead(H, latent_dim=cfg.latent_dim)
            if cfg.use_adversarial_bottleneck else None
        )
        self.q_head = CounterfactualQHead(state_dim=state_repr_dim) if cfg.use_q_planner else None
        if cfg.use_reliability_gated_posterior:
            self.posterior_delta = nn.Sequential(
                nn.Linear(H + cfg.n_state, 128),
                nn.GELU(),
                nn.Linear(128, cfg.n_state),
                nn.Tanh(),
            )
            self.posterior_gate = nn.Sequential(
                nn.Linear(H + cfg.n_state + cfg.measurement_quality_dim, 128),
                nn.GELU(),
                nn.Linear(128, cfg.n_state),
                nn.Sigmoid(),
            )
        else:
            self.posterior_delta = None
            self.posterior_gate = None
        self.transition_value_head = TherapeuticValueHead() if cfg.use_transition_value else None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _last_ctx_hidden(hidden: torch.Tensor, ctx_lens: torch.Tensor) -> torch.Tensor:
        """Select ``hidden[b, ctx_lens[b]-1, :]`` per batch item."""

        B, T, D = hidden.shape
        idx = (ctx_lens - 1).clamp(min=0).view(B, 1, 1).expand(B, 1, D)
        return hidden.gather(dim=1, index=idx).squeeze(1)

    @staticmethod
    def _logit(x: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
        x = x.clamp(eps, 1.0 - eps)
        return torch.log(x) - torch.log1p(-x)

    def _apply_state_anchor(
        self,
        z_neural: torch.Tensor,
        z_repr: torch.Tensor,
        z_axis_residual: torch.Tensor | None,
        state_anchor: torch.Tensor | None,
        h_ctx: torch.Tensor | None = None,
        measurement_quality: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        if (
            not self.cfg.use_observed_state_anchor
            or state_anchor is None
        ):
            return z_neural, z_repr, z_axis_residual, None, None, None

        anchor = state_anchor.to(z_neural.device, z_neural.dtype).clamp(1e-4, 1.0 - 1e-4)
        if self.cfg.use_reliability_gated_posterior and h_ctx is not None:
            q = measurement_quality
            if q is None:
                q = anchor.new_zeros(anchor.size(0), self.cfg.measurement_quality_dim)
            q = q.to(anchor.device, anchor.dtype)
            delta = self.posterior_delta(torch.cat([h_ctx.to(anchor.dtype), anchor], dim=-1))
            gate = self.posterior_gate(torch.cat([h_ctx.to(anchor.dtype), anchor, q], dim=-1))
            residual = gate * delta * float(self.cfg.anchor_residual_scale)
        else:
            gate = None
            # v6.1 fixed posterior update.
            residual = (z_neural - 0.5) * float(self.cfg.anchor_residual_scale)
        z_post = torch.sigmoid(self._logit(anchor) + residual)

        if z_axis_residual is not None:
            z_repr_post = torch.cat(
                [z_post.unsqueeze(-1), z_axis_residual],
                dim=-1,
            ).flatten(1)
        else:
            z_repr_post = z_post
        return z_post, z_repr_post, z_axis_residual, anchor, residual, gate

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        ctx_lens: torch.Tensor,
        prev_state: Optional[torch.Tensor] = None,
        state_anchor: Optional[torch.Tensor] = None,
        measurement_quality: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> dict:
        out = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        hidden = out.hidden_states[-1]                       # [B, T, D]
        gen_logits = out.logits                              # [B, T, V]

        h_ctx = self._last_ctx_hidden(hidden, ctx_lens)      # [B, D]

        # Cast context feature to FP32 for the small auxiliary heads.
        # The backbone hidden states may be BF16 (mixed precision training or
        # a pure-BF16 eval pass). The heads are kept in FP32 for numerical
        # stability and to avoid dtype-mismatch errors when autocast is off
        # (e.g., during ``evaluate`` which runs under ``torch.no_grad``).
        h_ctx = h_ctx.to(torch.float32)

        # Heads.
        z_log_var = None
        z_repr = None
        z_axis_residual = None
        if self.state_head is not None:
            out_z = self.state_head(h_ctx)
            if self.cfg.use_axis_subspace and self.cfg.state_residual_dim > 0:
                if len(out_z) == 4:
                    z, z_repr, z_axis_residual, z_log_var = out_z
                else:
                    z, z_repr, z_axis_residual = out_z
            elif isinstance(out_z, tuple):
                z, z_log_var = out_z
            else:
                z = out_z
            if z_repr is None:
                z_repr = z
        else:
            z = torch.zeros(
                h_ctx.size(0), self.cfg.n_state, device=h_ctx.device, dtype=h_ctx.dtype
            )
            z_repr = z
        z_neural = z
        z, z_repr, z_axis_residual, z_anchor, z_anchor_residual, z_anchor_gate = self._apply_state_anchor(
            z_neural, z_repr, z_axis_residual, state_anchor, h_ctx=h_ctx,
            measurement_quality=measurement_quality,
        )
        if self.strategy_head is not None:
            u_logits = self.strategy_head(h_ctx)
            u_probs = torch.softmax(u_logits, dim=-1)
        else:
            u_logits = torch.zeros(h_ctx.size(0), self.cfg.n_strategy, device=h_ctx.device, dtype=h_ctx.dtype)
            u_probs = torch.softmax(u_logits, dim=-1)

        lat = self.latent_action(h_ctx, tau=1.0, hard=False)
        lat_embed = lat["embed"]
        commit_loss = lat["commit"]

        # Outcome.
        if self.outcome_head is not None:
            outcome_input = z_repr if self.cfg.z_only_outcome else z
            outcome_pred = self.outcome_head(outcome_input, u_probs, lat_embed)
        else:
            outcome_pred = {
                "uptake_logit": h_ctx.new_zeros(h_ctx.size(0)),
                "quality": h_ctx.new_zeros(h_ctx.size(0), 4),
            }

        # Transition.
        z_trans_pred = None
        z_cf_all = None
        obs_feat = None
        if self.transition is not None and prev_state is not None:
            obs_feat = self.obs_proj(h_ctx)
            transition_state = z if self.cfg.transition_from_posterior else prev_state
            z_trans_pred = self.transition(transition_state, u_probs, obs_feat, lat_embed)
            if self.cfg.use_counterfactual:
                z_cf_all = self.transition.counterfactual(transition_state, obs_feat, lat_embed)

        transition_value_logit = None
        transition_cf_values = None
        if self.transition_value_head is not None and z_trans_pred is not None:
            transition_value_logit = self.transition_value_head(z, z_trans_pred)
            if z_cf_all is not None:
                transition_cf_values = self.transition_value_head.counterfactual(z, z_cf_all)

        residual_adv_logit = None
        if self.residual_adv is not None:
            residual_adv_logit = self.residual_adv(h_ctx, u_probs, lat_embed)

        q_values = None
        if self.q_head is not None and z_cf_all is not None:
            q_values = self.q_head(z_repr, z_cf_all)

        # v4 risk severity head (independent of strategy softmax).
        risk_logits = None
        if self.risk_head is not None:
            risk_logits = self.risk_head(h_ctx)

        return {
            "gen_logits": gen_logits,
            "gen_labels": labels,
            "h_ctx": h_ctx,
            "z": z,
            "z_neural": z_neural,
            "z_repr": z_repr,
            "z_axis_residual": z_axis_residual,
            "z_anchor": z_anchor,
            "z_anchor_residual": z_anchor_residual,
            "z_anchor_gate": z_anchor_gate,
            "z_log_var": z_log_var,
            "u_logits": u_logits,
            "u_probs": u_probs,
            "latent_embed": lat_embed,
            "latent_probs": lat["probs"],
            "outcome_pred": outcome_pred,
            "z_trans_pred": z_trans_pred,
            "z_cf_all": z_cf_all,
            "transition_value_logit": transition_value_logit,
            "transition_cf_values": transition_cf_values,
            "q_values": q_values,
            "obs_feat": obs_feat,
            "risk_logits": risk_logits,
            "residual_adv_logit": residual_adv_logit,
            "commit_loss": commit_loss,
        }
