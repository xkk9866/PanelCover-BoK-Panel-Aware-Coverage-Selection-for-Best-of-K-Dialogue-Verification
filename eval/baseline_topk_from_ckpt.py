"""Run a trained baseline planner over the v10 fair-BoN eval contexts and
write a fair-BoN top-K JSONL that ``v12_best_of_n_fair`` can consume.

For each sample in ``data/judge_eval_v10/v10_eval_contexts.jsonl`` we:
  1. Render the dialogue context with the tokenizer chat template.
  2. Forward through the LoRA-tuned PsyState backbone.
  3. Read off ``u_probs`` (per-strategy probability) from the strategy head.
  4. Pick the top-K strategies and write
     ``data/fair_bon_v12/<system>_topk.jsonl`` with one row per context::

         {sample_id, candidate_strategies, scores_per_strategy, planner}

This matches the contract used by ``v12_best_of_n_fair.py`` so the new
baseline plugs into the same Best-of-N + BT-verifier pipeline as every
other system.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from eval.load_ckpt import load_psystate
from psystate.constants import STRATEGIES


def _load_jsonl(path: Path) -> list[dict]:
    out: list[dict] = []
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def _render_ctx(tok, ctx_msgs: list[dict], max_ctx_length: int = 1800) -> torch.Tensor:
    try:
        rendered = tok.apply_chat_template(
            ctx_msgs, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        buf = []
        for m in ctx_msgs:
            buf.append(f"<|{m['role']}|>\n{m['content']}")
        buf.append("<|assistant|>\n")
        rendered = "".join(buf)
    ids = tok(
        rendered,
        add_special_tokens=False,
        truncation=True,
        max_length=max_ctx_length,
        return_tensors="pt",
    )["input_ids"]
    return ids


@torch.no_grad()
def _strategy_probs(model, tok, ctx_msgs: list[dict], state_vec: list[float],
                    risk_any: bool, max_ctx_length: int) -> list[float]:
    ids = _render_ctx(tok, ctx_msgs, max_ctx_length=max_ctx_length).to(model.backbone.device)
    attn = torch.ones_like(ids)
    ctx_len = torch.tensor([ids.shape[1]], dtype=torch.long, device=ids.device)
    state_t = torch.tensor([state_vec], dtype=torch.float32, device=ids.device)

    last_user = next(
        (m.get("content", "") for m in reversed(ctx_msgs) if m.get("role") == "user"),
        "",
    )
    confidence = float(torch.mean(torch.abs(state_t.cpu() - 0.5) * 2.0).clamp(0, 1))
    entropy = float(torch.mean(4.0 * state_t.cpu() * (1.0 - state_t.cpu())).clamp(0, 1))
    length = min(len(last_user) / 120.0, 1.0)
    meas = torch.tensor(
        [[confidence, entropy, length, float(risk_any)]],
        dtype=torch.float32, device=ids.device,
    )

    out = model(
        input_ids=ids,
        attention_mask=attn,
        labels=ids,           # labels unused for strategy head
        ctx_lens=ctx_len,
        prev_state=state_t,
        state_anchor=state_t,
        measurement_quality=meas,
    )
    probs = out["u_probs"][0].detach().float().cpu().tolist()
    return probs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Path to runs/<baseline>/ckpt-<step>")
    ap.add_argument("--base_model",
                    default="D:/models_cache/Qwen/Qwen2___5-7B-Instruct")
    ap.add_argument("--eval_set",
                    default="data/judge_eval_v10/v10_eval_contexts.jsonl")
    ap.add_argument("--out_topk", required=True)
    ap.add_argument("--system_name", required=True,
                    help="Used as ``planner`` field in the topk rows; the "
                         "downstream fair-BoN script reads candidate_strategies "
                         "and writes responses with system_id=<system_name>_bon_v12.")
    ap.add_argument("--K", type=int, default=3)
    ap.add_argument("--max_ctx_length", type=int, default=1800)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    print(f"[bon-bridge] loading {args.ckpt}")
    model, tok, _ = load_psystate(args.ckpt, args.base_model)
    print(f"[bon-bridge] loaded; reading {args.eval_set}")

    rows = _load_jsonl(Path(args.eval_set))
    if args.limit:
        rows = rows[: args.limit]
    print(f"[bon-bridge] {len(rows)} contexts")

    out_path = Path(args.out_topk)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    n_done = 0
    with out_path.open("w", encoding="utf-8") as fout:
        for r in rows:
            sid = str(r.get("sample_id"))
            ctx_msgs = list(r.get("context", []))
            risk_any = bool(r.get("risk_any") or
                             r.get("risk_level") in ("severe", "imminent"))
            state_vec = [
                float((r.get("posterior_state") or {}).get(k, 0.5))
                for k in ("distress", "rigidity", "readiness", "alliance", "clarity")
            ]
            probs = _strategy_probs(model, tok, ctx_msgs, state_vec,
                                     risk_any, args.max_ctx_length)
            order = sorted(range(len(STRATEGIES)),
                           key=lambda i: -probs[i])
            cand = [STRATEGIES[i] for i in order[: args.K]]
            scores = {STRATEGIES[i]: float(probs[i])
                      for i in range(len(STRATEGIES))}
            rec = {
                "sample_id": sid,
                "planner": args.system_name,
                "candidate_strategies": cand,
                "scores_per_strategy": scores,
                "K": args.K,
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_done += 1
            if n_done % 50 == 0:
                dt = time.time() - t0
                print(f"  [{n_done}/{len(rows)}]  {dt:.1f}s  "
                      f"{dt / n_done:.2f}s/ctx", flush=True)
    dt = time.time() - t0
    print(f"[bon-bridge] wrote {n_done} rows -> {out_path}  ({dt:.1f}s total)")


if __name__ == "__main__":
    main()
