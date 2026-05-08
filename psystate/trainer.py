"""End-to-end trainer for PsyState.

Uses standard ``transformers`` + ``peft`` (LoRA) — this keeps us agnostic to
``ms-swift``'s internal Trainer class while still being able to compare against
``swift sft`` for the vanilla baseline.
"""

from __future__ import annotations

import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .bayes_priors import make_clinical_priors
from .constants import N_LATENT_ACTION, N_STATE, N_STRATEGY
from .data_collator import PsyChatDataset, PsyCollator
from .losses import LossConfig, compute_losses
from .model import PsyStateConfig, PsyStateModel


@dataclass
class TrainArgs:
    # Paths
    model_name_or_path: str = "Qwen/Qwen2.5-7B-Instruct"
    train_file: str = "data/processed/train.chat.jsonl"
    dev_file: str = "data/processed/dev.chat.jsonl"
    output_dir: str = "runs/psystate_full"
    # Optimization
    learning_rate: float = 1e-4
    head_learning_rate: float = 3e-4
    weight_decay: float = 0.0
    num_train_steps: int = 2000
    per_device_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    warmup_ratio: float = 0.03
    max_length: int = 2048
    max_ctx_length: int = 1800
    bf16: bool = True            # compute dtype for mixed precision
    # Quantization: 4-bit NF4 (QLoRA) keeps the 7B backbone at ~5 GB so we
    # can train on a contended 48 GB card alongside other workloads.
    load_in_4bit: bool = True
    bnb_quant_type: str = "nf4"  # {nf4, fp4}
    bnb_double_quant: bool = True
    # "sdpa" is faster; "eager" is slower but can avoid driver bugs with 4bit + grad ckpt.
    attn_implementation: str = "sdpa"
    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target: str = "q_proj,k_proj,v_proj,o_proj"
    # Data
    train_subsample: int = 20000   # number of chat rows to use
    dev_subsample: int = 400
    # RAG (optional, used by baseline_rag / baseline_kemi / baseline_cpsycoun)
    use_rag: bool = False
    rag_corpus: str = "data/processed/train.chat.jsonl"
    rag_k: int = 3
    rag_mode: str = "global"          # "global" | "emotion_typed" (KEMI 2023)
    rag_memory_chars: int = 600       # soft budget for retrieved context
    # Schedule / logging
    log_every: int = 10
    eval_every: int = 300
    save_every: int = 500
    seed: int = 42
    # Loss weights (mirrors psystate.losses.LossConfig)
    loss_cfg: LossConfig = field(default_factory=LossConfig)
    # Ablations (also propagated to PsyStateConfig)
    use_state: bool = True
    use_strategy: bool = True
    use_transition: bool = True
    use_outcome: bool = True
    use_consist: bool = True
    use_sep: bool = True
    use_safety: bool = True
    # v4 ----------------------------------------------------------------
    use_bayes: bool = False
    use_counterfactual: bool = False
    use_router: bool = False
    use_uncertainty: bool = False
    # Curriculum: oversample risk-flagged turns during training so the
    # router sees many positives even at small batch size.  ``risk_oversample``
    # is the multiplicative factor (1 = off, 4 = quadruple risk frequency).
    risk_oversample: float = 1.0
    # v5: route outcome head through z alone (no lat_embed bypass).
    z_only_outcome: bool = False
    # v5 axis-dropout on outcome input ([0, 1)).
    axis_dropout_p: float = 0.0
    # v6: axis-subspace state and planning toggles.
    use_axis_subspace: bool = False
    state_residual_dim: int = 0
    use_adversary: bool = False
    use_q_planner: bool = False
    use_observed_state_anchor: bool = False
    anchor_residual_scale: float = 1.0
    outcome_head_type: str = "mlp"
    use_reliability_gated_posterior: bool = False
    use_transition_value: bool = False
    transition_from_posterior: bool = False


# ---------------------------------------------------------------------------
# Backbone + LoRA setup
# ---------------------------------------------------------------------------


def build_backbone(args: TrainArgs):
    """Instantiate the LLM backbone and wrap with LoRA (QLoRA by default).

    With ``load_in_4bit=True`` we use the QLoRA recipe: the base weights are
    stored in NF4 (~4.5 GB for a 7B model), LoRA adapters stay in fp32/bf16,
    gradient checkpointing is on. This fits comfortably alongside other jobs
    on a 48 GB card.
    """

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    tok = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    compute_dtype = torch.bfloat16 if args.bf16 else torch.float16

    load_kwargs: dict[str, Any] = dict(
        trust_remote_code=True,
        attn_implementation=getattr(args, "attn_implementation", "sdpa"),
    )

    if args.load_in_4bit:
        from transformers import BitsAndBytesConfig

        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type=args.bnb_quant_type,
            bnb_4bit_use_double_quant=args.bnb_double_quant,
        )
        load_kwargs.update(quantization_config=bnb_cfg, device_map={"": 0})
    else:
        load_kwargs.update(torch_dtype=compute_dtype)

    base = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **load_kwargs)
    try:
        base.config.use_cache = False
    except Exception:
        pass

    if args.load_in_4bit:
        base = prepare_model_for_kbit_training(
            base, use_gradient_checkpointing=True,
        )

    # LoRA
    target = [x.strip() for x in args.lora_target.split(",") if x.strip()]
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=target,
        task_type="CAUSAL_LM",
    )
    base = get_peft_model(base, lora_cfg)
    if args.load_in_4bit:
        # kbit prepare_model enables grad ckpt with default reentrancy, which
        # can trigger illegal-memory CUDA errors in backward on some Windows
        # drivers. Re-enable with use_reentrant=False.
        try:
            if hasattr(base, "gradient_checkpointing_disable"):
                base.gradient_checkpointing_disable()
        except Exception:
            pass
        try:
            base.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            if hasattr(base, "enable_input_require_grads"):
                base.enable_input_require_grads()
        except Exception as e:  # pragma: no cover
            print(f"[warn] 4bit gradient checkpointing (use_reentrant=False): {e}")
    else:
        try:
            base.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            if hasattr(base, "enable_input_require_grads"):
                base.enable_input_require_grads()
        except Exception as e:  # pragma: no cover
            print(f"[warn] gradient checkpointing not enabled: {e}")
    base.print_trainable_parameters()

    hidden_size = getattr(base.config, "hidden_size", None)
    if hidden_size is None:
        hidden_size = getattr(base.config, "n_embd", None) or getattr(base.config, "d_model", None)
    assert hidden_size is not None, "cannot infer hidden size from backbone config"
    return base, tok, int(hidden_size)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_scheduler(optimizer, num_steps: int, warmup_ratio: float):
    from transformers import get_cosine_schedule_with_warmup
    warmup = int(num_steps * warmup_ratio)
    return get_cosine_schedule_with_warmup(optimizer, warmup, num_steps)


def _move(batch: dict, device) -> dict:
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
    return out


@torch.no_grad()
def evaluate(model: PsyStateModel, loader, device) -> dict[str, float]:
    model.eval()
    sums: dict[str, float] = {}
    n = 0
    for batch in loader:
        batch = _move(batch, device)
        out = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
            ctx_lens=batch["ctx_lens"],
            prev_state=batch["state_target"],
            state_anchor=batch["state_target"],
            measurement_quality=batch.get("measurement_quality"),
        )
        L, logs = compute_losses(
            cfg=model.loss_cfg,
            gen_logits=out["gen_logits"],
            gen_labels=out["gen_labels"],
            z_pred=out["z"],
            strategy_logits=out["u_logits"],
            outcome_pred=out["outcome_pred"],
            transition_module=model.transition if model.transition is not None else _Stub(),
            z_trans_pred=out["z_trans_pred"],
            targets=batch,
            commit_loss=out["commit_loss"],
            z_log_var=out.get("z_log_var"),
            z_cf_all=out.get("z_cf_all"),
            risk_logits=out.get("risk_logits"),
            residual_adv_logit=out.get("residual_adv_logit"),
            q_values=out.get("q_values"),
            transition_value_logit=out.get("transition_value_logit"),
            clinical_priors=getattr(model, "clinical_priors", None),
        )
        for k, v in logs.items():
            sums[k] = sums.get(k, 0.0) + float(v)
        n += 1
    model.train()
    return {k: v / max(n, 1) for k, v in sums.items()}


class _Stub(nn.Module):
    """Placeholder transition module when transition is disabled: no parameters."""

    def __init__(self):
        super().__init__()
        self.A = nn.Parameter(torch.eye(N_STATE).unsqueeze(0).repeat(N_STRATEGY, 1, 1), requires_grad=False)
        self.bias = nn.Parameter(torch.zeros(N_STRATEGY, N_STATE), requires_grad=False)


def train(args: TrainArgs) -> None:
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    Path(args.output_dir, "train_args.json").write_text(
        json.dumps({k: (asdict(v) if hasattr(v, "__dataclass_fields__") else v) for k, v in asdict(args).items()}, indent=2),
        encoding="utf-8",
    )

    backbone, tok, hidden_size = build_backbone(args)

    model_cfg = PsyStateConfig(
        hidden_size=hidden_size,
        use_state=args.use_state,
        use_strategy=args.use_strategy,
        use_transition=args.use_transition,
        use_outcome=args.use_outcome,
        use_safety_router=getattr(args, "use_router", False),
        use_uncertainty=getattr(args, "use_uncertainty", False),
        use_counterfactual=getattr(args, "use_counterfactual", False),
        z_only_outcome=getattr(args, "z_only_outcome", False),
        axis_dropout_p=getattr(args, "axis_dropout_p", 0.0),
        use_axis_subspace=getattr(args, "use_axis_subspace", False),
        state_residual_dim=getattr(args, "state_residual_dim", 0),
        use_adversarial_bottleneck=getattr(args, "use_adversary", False),
        use_q_planner=getattr(args, "use_q_planner", False),
        use_observed_state_anchor=getattr(args, "use_observed_state_anchor", False),
        anchor_residual_scale=getattr(args, "anchor_residual_scale", 1.0),
        outcome_head_type=getattr(args, "outcome_head_type", "mlp"),
        use_reliability_gated_posterior=getattr(args, "use_reliability_gated_posterior", False),
        use_transition_value=getattr(args, "use_transition_value", False),
        transition_from_posterior=getattr(args, "transition_from_posterior", False),
    )
    model = PsyStateModel(backbone, model_cfg)
    if getattr(args, "use_bayes", False):
        clinical_priors = make_clinical_priors(
            sigma_strong=getattr(args.loss_cfg, "sigma_strong", 0.10),
            sigma_weak=getattr(args.loss_cfg, "sigma_weak", 0.20),
            sigma_neutral=getattr(args.loss_cfg, "sigma_neutral", 0.40),
        )
    else:
        clinical_priors = None
    # Attach loss cfg for eval convenience.
    loss_cfg = LossConfig(
        lam_gen=args.loss_cfg.lam_gen,
        lam_state=args.loss_cfg.lam_state,
        lam_strategy=args.loss_cfg.lam_strategy,
        lam_transition=args.loss_cfg.lam_transition,
        lam_outcome=args.loss_cfg.lam_outcome,
        lam_consist=args.loss_cfg.lam_consist,
        lam_sep=getattr(args.loss_cfg, "lam_sep", 0.0),
        lam_safety=args.loss_cfg.lam_safety,
        lam_commit=args.loss_cfg.lam_commit,
        lam_bayes=getattr(args.loss_cfg, "lam_bayes", 0.0),
        lam_counterfactual=getattr(args.loss_cfg, "lam_counterfactual", 0.0),
        lam_router=getattr(args.loss_cfg, "lam_router", 0.0),
        lam_uncertainty=getattr(args.loss_cfg, "lam_uncertainty", 0.0),
        cf_margin=getattr(args.loss_cfg, "cf_margin", 0.05),
        router_focal_gamma=getattr(args.loss_cfg, "router_focal_gamma", 2.0),
        router_recall_weight=getattr(args.loss_cfg, "router_recall_weight", 3.0),
        router_pos_weight=getattr(args.loss_cfg, "router_pos_weight", 8.0),
        uptake_pos_weight=getattr(args.loss_cfg, "uptake_pos_weight", 0.0),
        lam_decor=getattr(args.loss_cfg, "lam_decor", 0.0),
        lam_adv=getattr(args.loss_cfg, "lam_adv", 0.0),
        lam_pref=getattr(args.loss_cfg, "lam_pref", 0.0),
        lam_q=getattr(args.loss_cfg, "lam_q", 0.0),
        lam_outcome_rank=getattr(args.loss_cfg, "lam_outcome_rank", 0.0),
        lam_value_rank=getattr(args.loss_cfg, "lam_value_rank", 0.0),
        lam_value_reg=getattr(args.loss_cfg, "lam_value_reg", 0.0),
        sigma_strong=getattr(args.loss_cfg, "sigma_strong", 0.10),
        sigma_weak=getattr(args.loss_cfg, "sigma_weak", 0.20),
        sigma_neutral=getattr(args.loss_cfg, "sigma_neutral", 0.40),
        use_state=args.use_state,
        use_strategy=args.use_strategy,
        use_transition=args.use_transition,
        use_outcome=args.use_outcome,
        use_consist=args.use_consist,
        use_sep=getattr(args, "use_sep", True),
        use_safety=args.use_safety,
        use_bayes=getattr(args, "use_bayes", False),
        use_counterfactual=getattr(args, "use_counterfactual", False),
        use_router=getattr(args, "use_router", False),
        use_uncertainty=getattr(args, "use_uncertainty", False),
        use_adversary=getattr(args, "use_adversary", False),
        use_q_planner=getattr(args, "use_q_planner", False),
        use_transition_value=getattr(args, "use_transition_value", False),
    )
    model.loss_cfg = loss_cfg
    model.clinical_priors = clinical_priors

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # With QLoRA the backbone is already on GPU via ``device_map``; calling
    # ``.to(device)`` on a 4-bit model raises. Move only the auxiliary heads.
    if args.load_in_4bit:
        for name, mod in model.named_children():
            if name == "backbone":
                continue
            mod.to(device)
    else:
        model.to(device)

    # Place clinical priors on-device once (avoids per-step .to() copies).
    if clinical_priors is not None:
        clinical_priors = clinical_priors.to(device, torch.float32)
        model.clinical_priors = clinical_priors

    # Optional RAG memory.
    rag = None
    if args.use_rag:
        from baselines.rag_retriever import RAGMemory
        print(f"[rag] building BM25 memory from {args.rag_corpus} (mode={args.rag_mode}) ...")
        rag = RAGMemory(
            args.rag_corpus,
            k=args.rag_k,
            max_memory_chars=args.rag_memory_chars,
            mode=args.rag_mode,
        )
        print(f"[rag] indexed {len(rag.client_texts)} memory items")

    # Datasets.
    t0 = time.time()
    print(f"[data] loading train from {args.train_file} ...", flush=True)
    train_ds = PsyChatDataset(
        args.train_file, tok,
        max_length=args.max_length, max_ctx_length=args.max_ctx_length,
        subsample=args.train_subsample, rag_memory=rag,
    )
    print(f"[data] train loaded ({len(train_ds)}) in {time.time()-t0:.1f}s", flush=True)
    t0 = time.time()
    dev_ds = PsyChatDataset(
        args.dev_file, tok,
        max_length=args.max_length, max_ctx_length=args.max_ctx_length,
        subsample=args.dev_subsample, rag_memory=rag,
    )
    print(f"[data] dev loaded ({len(dev_ds)}) in {time.time()-t0:.1f}s", flush=True)
    collator = PsyCollator(tok)
    # ------------------------------------------------------------------
    # v4 risk-curriculum: oversample risk-flagged turns so the safety
    # router actually sees positives at small batch sizes.  Falls back to
    # plain shuffle when ``risk_oversample <= 1``.
    # ------------------------------------------------------------------
    train_sampler = None
    if getattr(args, "risk_oversample", 1.0) > 1.0:
        from torch.utils.data import WeightedRandomSampler

        risk_weights = []
        n_risk = 0
        for r in train_ds.rows:
            risk = (r.get("meta", {}) or {}).get("risk", {}) or {}
            sev = int(risk.get("severity", 2 if risk.get("any") else 0))
            w = float(args.risk_oversample) if sev >= 2 else 1.0
            if sev >= 2:
                n_risk += 1
            risk_weights.append(w)
        train_sampler = WeightedRandomSampler(
            weights=risk_weights, num_samples=len(risk_weights), replacement=True,
        )
        print(f"[risk-curriculum] oversampling {n_risk}/{len(risk_weights)} severe-risk turns "
              f"by {args.risk_oversample}x")

    train_loader = DataLoader(
        train_ds, batch_size=args.per_device_batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=0, collate_fn=collator, pin_memory=True, drop_last=True,
    )
    dev_loader = DataLoader(
        dev_ds, batch_size=args.per_device_batch_size * 2, shuffle=False,
        num_workers=0, collate_fn=collator, pin_memory=True,
    )

    # Separate parameter groups: backbone LoRA + head params.
    head_params, base_params = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if n.startswith("backbone."):
            base_params.append(p)
        else:
            head_params.append(p)

    # Under QLoRA we use paged 8-bit AdamW on the LoRA params to cut optimizer
    # memory ~4x. The small auxiliary heads use standard AdamW.
    if args.load_in_4bit:
        try:
            import bitsandbytes as bnb

            base_opt_cls: Any = bnb.optim.PagedAdamW8bit
        except Exception as e:  # pragma: no cover
            print(f"[warn] bnb 8-bit optimizer unavailable ({e}); falling back to AdamW")
            base_opt_cls = torch.optim.AdamW
    else:
        base_opt_cls = torch.optim.AdamW

    optimizer = base_opt_cls(
        [
            {"params": base_params, "lr": args.learning_rate,
             "weight_decay": args.weight_decay},
            {"params": head_params, "lr": args.head_learning_rate,
             "weight_decay": 0.0},
        ],
        betas=(0.9, 0.999),
    )
    sched = make_scheduler(optimizer, args.num_train_steps, args.warmup_ratio)
    scaler = None  # bf16 is native, no scaler needed

    step = 0
    step_in_accum = 0
    microstep = 0
    optimizer.zero_grad(set_to_none=True)
    t_last = time.time()
    running: dict[str, float] = {}

    print("[train] entering loop ...", flush=True)
    model.train()
    data_iter = iter(train_loader)

    # v5: EMA stats of ``z`` so the axis-decorrelation loss is well-defined
    # even with ``per_device_batch_size = 1``.  Updated detached (no grad)
    # after every microstep using the current model's ``z`` predictions.
    z_running_mean = torch.full((N_STATE,), 0.5, device=device)
    z_running_var  = torch.full((N_STATE,), 0.05, device=device)
    z_ema_alpha    = 0.99
    while step < args.num_train_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)
        batch = _move(batch, device)
        microstep += 1
        # Early heartbeat: print timing for the first few microsteps so we can
        # diagnose hangs before the first ``log_every`` boundary.
        if microstep <= 5:
            print(f"[train] microstep {microstep} got batch "
                  f"input_ids={tuple(batch['input_ids'].shape)} ctx_lens={batch['ctx_lens'].tolist()}",
                  flush=True)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=args.bf16):
            out = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
                ctx_lens=batch["ctx_lens"],
                prev_state=batch["state_target"],
                state_anchor=batch["state_target"],
                measurement_quality=batch.get("measurement_quality"),
            )
            trans_mod = model.transition if model.transition is not None else _Stub().to(device)
            # Inject the EMA stats so axis_decorrelation_loss works at B=1.
            batch_with_stats = dict(batch)
            batch_with_stats["z_running_mean"] = z_running_mean
            batch_with_stats["z_running_var"]  = z_running_var
            L, logs = compute_losses(
                cfg=loss_cfg,
                gen_logits=out["gen_logits"],
                gen_labels=out["gen_labels"],
                z_pred=out["z"],
                strategy_logits=out["u_logits"],
                outcome_pred=out["outcome_pred"],
                transition_module=trans_mod,
                z_trans_pred=out["z_trans_pred"],
                targets=batch_with_stats,
                commit_loss=out["commit_loss"],
                z_log_var=out.get("z_log_var"),
                z_cf_all=out.get("z_cf_all"),
                risk_logits=out.get("risk_logits"),
                residual_adv_logit=out.get("residual_adv_logit"),
                q_values=out.get("q_values"),
                transition_value_logit=out.get("transition_value_logit"),
                clinical_priors=clinical_priors,
            )

        # Update EMA stats from the current ``z``. Detached; no grad.
        with torch.no_grad():
            z_det = out["z"].detach().float()
            mb = z_det.mean(dim=0)
            vb = z_det.var(dim=0, unbiased=False) if z_det.size(0) >= 2 else (z_det - z_running_mean).pow(2).mean(dim=0)
            z_running_mean.mul_(z_ema_alpha).add_(mb, alpha=1.0 - z_ema_alpha)
            z_running_var.mul_(z_ema_alpha).add_(vb, alpha=1.0 - z_ema_alpha)
        (L / args.gradient_accumulation_steps).backward()
        if microstep <= 5:
            # detach() avoids the autograd warning on float(L) when grad still attached
            print(f"[train] microstep {microstep} fwd+bwd done  L={float(L.detach()):.4f}", flush=True)

        for k, v in logs.items():
            running[k] = running.get(k, 0.0) * 0.9 + float(v) * 0.1
        step_in_accum += 1

        if step_in_accum == args.gradient_accumulation_steps:
            torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), 1.0)
            optimizer.step()
            sched.step()
            optimizer.zero_grad(set_to_none=True)
            step_in_accum = 0
            step += 1

            if step % args.log_every == 0:
                dt = time.time() - t_last
                t_last = time.time()
                msg = " ".join(f"{k}={v:.4f}" for k, v in running.items())
                print(f"[step {step:5d}] lr={sched.get_last_lr()[0]:.2e} dt={dt:.1f}s | {msg}")

            if step % args.eval_every == 0:
                ev = evaluate(model, dev_loader, device)
                print(f"[eval {step}] " + " ".join(f"{k}={v:.4f}" for k, v in ev.items()))

            if step % args.save_every == 0 or step == args.num_train_steps:
                save_dir = Path(args.output_dir, f"ckpt-{step}")
                save_dir.mkdir(parents=True, exist_ok=True)
                # Save LoRA adapter
                model.backbone.save_pretrained(str(save_dir / "adapter"))
                # Save heads
                heads_state = {k: v.detach().cpu() for k, v in model.state_dict().items()
                               if not k.startswith("backbone.")}
                torch.save(heads_state, save_dir / "heads.pt")
                # Save tokenizer once
                tok.save_pretrained(str(save_dir / "adapter"))
                # Save cfg
                (save_dir / "psystate_config.json").write_text(json.dumps(asdict(model_cfg), indent=2))
                print(f"[save] -> {save_dir}")

    print("[done]")
