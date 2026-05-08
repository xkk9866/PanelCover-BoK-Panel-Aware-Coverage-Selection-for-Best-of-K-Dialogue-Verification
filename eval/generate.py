"""Batched greedy/sampled generation for a PsyState or baseline checkpoint.

Writes a JSONL of ``{dialog_id, turn_idx, context, reference, prediction,
z, u, outcome, meta}`` to ``<ckpt>/preds_<split>.jsonl``.

V4 inference-time safety routing (innovation, not tech-stacking):
the trained risk router is used as a *hard gate* before autoregressive
sampling.  When the gate fires (`P(severe) + P(imminent) > tau`) we emit a
crisis-support template instead of letting the LLM autoregress.  This
converts the router from a training-time loss term into an actual safety
behaviour and is the lever that fixes "0 / 30 referrals on the risk slice".
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from eval.load_ckpt import load_psystate
from psystate.data_collator import PsyChatDataset, PsyCollator
from psystate.constants import STRATEGIES, STATE_AXES


# ---------------------------------------------------------------------------
# Crisis-response templates (zh_CN).  Used by the inference-time risk gate
# when the router emits severe/imminent.  Two intensities:
#   - severe   : encourage professional help, emotion validation, no advice.
#   - imminent : explicit hotline + emergency guidance + safety planning.
# ---------------------------------------------------------------------------

CRISIS_TEMPLATE_SEVERE = (
    "我听到你正在经历非常难受的时刻，谢谢你愿意把这些告诉我。"
    "你的安全和感受是最重要的——你不需要一个人扛。"
    "如果你感到难以承受，请考虑联系专业心理援助：北京心理危机干预中心 "
    "010-82951332，或全国心理援助热线 400-161-9995（24 小时）。"
    "如果你愿意的话，我可以陪你一起想想：现在最让你难受的是什么？"
)

CRISIS_TEMPLATE_IMMINENT = (
    "我非常担心你的安全，你说的让我意识到这件事现在很紧急。"
    "请立即联系专业人员：120 急救 / 110 报警，或者 24 小时心理援助热线 "
    "400-161-9995；北京心理危机干预中心 010-82951332。"
    "如果身边有人，请告诉他们你的感受并请他们陪着你；"
    "如果手边有可能造成伤害的物品，请把它放到不容易拿到的地方。"
    "在你联系到帮助之前，我会一直在这里陪你——你不必独自面对这件事。"
)


def crisis_template(severity: str) -> str:
    if severity == "imminent":
        return CRISIS_TEMPLATE_IMMINENT
    return CRISIS_TEMPLATE_SEVERE


def gate_decision(
    risk_probs: list[float] | None,
    tau_severe: float = 0.30,
    tau_imminent: float = 0.20,
) -> tuple[bool, str | None, float]:
    """Return ``(fired, severity, score)`` where ``severity`` is in
    ``{"severe", "imminent"}`` if the gate fires."""
    if risk_probs is None:
        return False, None, 0.0
    p_imm = float(risk_probs[3])
    p_sev = float(risk_probs[2]) + p_imm
    if p_imm >= tau_imminent:
        return True, "imminent", p_imm
    if p_sev >= tau_severe:
        return True, "severe", p_sev
    return False, None, p_sev


@torch.no_grad()
def generate_batch(
    model,
    tok,
    batch: dict,
    max_new_tokens: int = 160,
    risk_probs: list[list[float]] | None = None,
    use_safety_gate: bool = False,
    tau_severe: float = 0.30,
    tau_imminent: float = 0.20,
) -> tuple[list[str], list[dict]]:
    """Batched generation with optional inference-time safety routing.

    Returns (texts, gate_meta) where ``gate_meta[i]`` is::
        {"fired": bool, "severity": "severe"|"imminent"|None, "score": float}
    so we can audit per-turn how often the gate triggers.
    """

    input_ids = batch["input_ids"]
    attn = batch["attention_mask"]
    ctx_lens = batch["ctx_lens"].tolist()

    outs: list[str] = []
    gates: list[dict] = []
    for i in range(input_ids.size(0)):
        rp = risk_probs[i] if (risk_probs is not None and i < len(risk_probs)) else None
        fired, sev, score = (False, None, 0.0)
        if use_safety_gate:
            fired, sev, score = gate_decision(rp, tau_severe, tau_imminent)
        gates.append({"fired": bool(fired), "severity": sev, "score": float(score)})
        if fired:
            outs.append(crisis_template(sev))
            continue
        ids = input_ids[i, : ctx_lens[i]].unsqueeze(0)
        am = attn[i, : ctx_lens[i]].unsqueeze(0)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
            gen = model.backbone.generate(
                input_ids=ids.to(model.backbone.device),
                attention_mask=am.to(model.backbone.device),
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id,
            )
        new_tokens = gen[0, ids.shape[1]:]
        text = tok.decode(new_tokens, skip_special_tokens=True).strip()
        outs.append(text)
    return outs, gates


@torch.no_grad()
def infer_heads(model, batch):
    out = model(
        input_ids=batch["input_ids"].to(model.backbone.device),
        attention_mask=batch["attention_mask"].to(model.backbone.device),
        labels=batch["labels"].to(model.backbone.device),
        ctx_lens=batch["ctx_lens"].to(model.backbone.device),
        prev_state=batch["state_target"].to(model.backbone.device),
        state_anchor=batch["state_target"].to(model.backbone.device),
        measurement_quality=batch.get("measurement_quality").to(model.backbone.device)
        if batch.get("measurement_quality") is not None else None,
    )
    z = out["z"].detach().float().cpu().tolist()
    z_neural = out.get("z_neural")
    z_neural = (
        z_neural.detach().float().cpu().tolist()
        if z_neural is not None else [None] * len(z)
    )
    z_anchor = out.get("z_anchor")
    z_anchor = (
        z_anchor.detach().float().cpu().tolist()
        if z_anchor is not None else [None] * len(z)
    )
    z_anchor_residual = out.get("z_anchor_residual")
    z_anchor_residual = (
        z_anchor_residual.detach().float().cpu().tolist()
        if z_anchor_residual is not None else [None] * len(z)
    )
    z_anchor_gate = out.get("z_anchor_gate")
    z_anchor_gate = (
        z_anchor_gate.detach().float().cpu().tolist()
        if z_anchor_gate is not None else [None] * len(z)
    )
    u = out["u_probs"].detach().float().cpu().tolist()
    uptake = out["outcome_pred"]["uptake_logit"].sigmoid().float().cpu().tolist()
    quality = out["outcome_pred"]["quality"].detach().float().cpu().tolist()
    z_next = out["z_trans_pred"]
    z_next = z_next.detach().float().cpu().tolist() if z_next is not None else [None] * len(z)

    # v4: risk router logits (4-class severity) + per-axis log-variance + counterfactual.
    risk_logits = out.get("risk_logits")
    risk_probs = (
        risk_logits.softmax(dim=-1).detach().float().cpu().tolist()
        if risk_logits is not None else [None] * len(z)
    )
    z_log_var = out.get("z_log_var")
    z_unc = (
        z_log_var.detach().float().cpu().tolist() if z_log_var is not None
        else [None] * len(z)
    )
    z_cf = out.get("z_cf_all")
    z_cf_list = (
        z_cf.detach().float().cpu().tolist() if z_cf is not None
        else [None] * len(z)
    )
    z_axis_res = out.get("z_axis_residual")
    z_axis_res = (
        z_axis_res.detach().float().cpu().tolist()
        if z_axis_res is not None else [None] * len(z)
    )
    q_values = out.get("q_values")
    q_values = (
        q_values.detach().float().cpu().tolist()
        if q_values is not None else [None] * len(z)
    )
    residual_adv = out.get("residual_adv_logit")
    residual_adv = (
        residual_adv.sigmoid().detach().float().cpu().tolist()
        if residual_adv is not None else [None] * len(z)
    )
    transition_value = out.get("transition_value_logit")
    transition_value = (
        transition_value.detach().float().cpu().tolist()
        if transition_value is not None else [None] * len(z)
    )
    transition_cf_values = out.get("transition_cf_values")
    transition_cf_values = (
        transition_cf_values.detach().float().cpu().tolist()
        if transition_cf_values is not None else [None] * len(z)
    )
    return (
        z, z_neural, z_anchor, z_anchor_residual,
        u, uptake, quality, z_next, risk_probs, z_unc, z_cf_list,
        z_axis_res, q_values, residual_adv, z_anchor_gate,
        transition_value, transition_cf_values,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--base_model", default="D:/models_cache/Qwen/Qwen2___5-7B-Instruct")
    ap.add_argument("--split", default="test")
    ap.add_argument("--data_file", default=None)
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--max_new_tokens", type=int, default=160)
    ap.add_argument("--out", default=None)
    ap.add_argument("--safety_gate", action="store_true",
                    help="If set, route severe/imminent risk turns through a crisis template "
                         "instead of letting the backbone autoregress.")
    ap.add_argument("--tau_severe", type=float, default=0.30,
                    help="P(severe)+P(imminent) >= tau_severe triggers severe template.")
    ap.add_argument("--tau_imminent", type=float, default=0.20,
                    help="P(imminent) >= tau_imminent triggers imminent template.")
    args = ap.parse_args()

    data_file = args.data_file or f"data/processed/{args.split}.chat.jsonl"
    out_path = Path(args.out) if args.out else Path(args.ckpt) / f"preds_{args.split}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model, tok, _ = load_psystate(args.ckpt, args.base_model)
    ds = PsyChatDataset(data_file, tok, max_length=2048, max_ctx_length=1800, subsample=args.limit)
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=PsyCollator(tok))

    n = 0
    n_gated = 0
    with out_path.open("w", encoding="utf-8") as f:
        for batch in loader:
            (
                z, z_neural, z_anchor, z_anchor_residual,
                u, uptake, quality, z_next, risk_probs, z_unc, z_cf,
                z_axis_res, q_values, residual_adv, z_anchor_gate,
                transition_value, transition_cf_values,
            ) = infer_heads(model, batch)
            preds, gates = generate_batch(
                model, tok, batch,
                max_new_tokens=args.max_new_tokens,
                risk_probs=risk_probs,
                use_safety_gate=args.safety_gate,
                tau_severe=args.tau_severe,
                tau_imminent=args.tau_imminent,
            )
            if gates and gates[0]["fired"]:
                n_gated += 1
            # Reconstruct inputs: context text is easiest to fetch from raw row.
            raw = ds.rows[n]
            meta = raw.get("meta", {})
            ref = raw["messages"][-1]["content"]
            ctx_msgs = raw["messages"][:-1]
            rec = {
                "dialog_id": meta.get("dialog_id"),
                "turn_idx": meta.get("turn_idx"),
                "source": meta.get("source"),
                "context": ctx_msgs,
                "reference": ref,
                "prediction": preds[0],
                "z_pred": {k: float(z[0][i]) for i, k in enumerate(STATE_AXES)},
                "z_neural_pred": (
                    {k: float(z_neural[0][i]) for i, k in enumerate(STATE_AXES)}
                    if z_neural and z_neural[0] is not None else None
                ),
                "z_anchor": (
                    {k: float(z_anchor[0][i]) for i, k in enumerate(STATE_AXES)}
                    if z_anchor and z_anchor[0] is not None else None
                ),
                "z_anchor_residual": (
                    {k: float(z_anchor_residual[0][i]) for i, k in enumerate(STATE_AXES)}
                    if z_anchor_residual and z_anchor_residual[0] is not None else None
                ),
                "z_anchor_gate": (
                    {k: float(z_anchor_gate[0][i]) for i, k in enumerate(STATE_AXES)}
                    if z_anchor_gate and z_anchor_gate[0] is not None else None
                ),
                "z_log_var": z_unc[0] if z_unc and z_unc[0] is not None else None,
                "z_axis_residual": z_axis_res[0] if z_axis_res and z_axis_res[0] is not None else None,
                "u_pred": {k: float(u[0][i]) for i, k in enumerate(STRATEGIES)},
                "uptake_pred": float(uptake[0]),
                "residual_uptake_pred": float(residual_adv[0]) if residual_adv and residual_adv[0] is not None else None,
                "quality_pred": quality[0],
                "z_next_pred": z_next[0],
                "z_cf_pred": z_cf[0] if z_cf and z_cf[0] is not None else None,
                "transition_value_logit": (
                    float(transition_value[0])
                    if transition_value and transition_value[0] is not None else None
                ),
                "transition_cf_values": (
                    {k: float(transition_cf_values[0][i]) for i, k in enumerate(STRATEGIES)}
                    if transition_cf_values and transition_cf_values[0] is not None else None
                ),
                "q_values": (
                    {k: float(q_values[0][i]) for i, k in enumerate(STRATEGIES)}
                    if q_values and q_values[0] is not None else None
                ),
                "risk_pred": (
                    {"none": float(risk_probs[0][0]),
                     "mild": float(risk_probs[0][1]),
                     "severe": float(risk_probs[0][2]),
                     "imminent": float(risk_probs[0][3])}
                    if risk_probs and risk_probs[0] is not None else None
                ),
                "state_target": meta.get("state"),
                "next_state_target": meta.get("next_state"),
                "strategy_target": meta.get("strategy_label"),
                "outcome_short_target": meta.get("outcome_short"),
                "session_feedback_target": meta.get("session_feedback"),
                "risk_target": meta.get("risk"),
                "safety_gate": gates[0] if gates else {"fired": False},
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
            if n % 20 == 0:
                print(f"  gen {n}/{len(ds)}  gated={n_gated}")
    print(f"[gen] wrote {n} preds  ({n_gated} routed via safety gate) -> {out_path}")


if __name__ == "__main__":
    main()
