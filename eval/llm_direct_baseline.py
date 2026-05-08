"""LLM direct generation baseline via DashScope / Qwen-max.

This generates responses for all eval contexts using a powerful LLM
(qwen-max or qwen-plus) with a simple zero-shot prompt, optionally with
the safety shield applied afterward. It serves as a strong LLM-era
baseline that PA-SCT-DRO must outperform.

Two variants:
  * llm_direct: Qwen-max zero-shot with clinical context prompt.
  * llm_safe: Same but with safety shield override applied post-hoc.

The outputs are materialised as BoN-compatible response files so they
plug directly into the panel-aware win-rate evaluation pipeline.

Usage
-----
    python -m eval.llm_direct_baseline \\
        --model qwen-max \\
        --out_responses data/fair_bon_v12/llm_direct_bon_responses.jsonl \\
        --max_workers 8
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


SYSTEM_PROMPT = (
    "你是一位专业的心理支持顾问，经验丰富，温暖、专业、不说教。"
    "请根据来访者的对话背景，给出一条简洁、有针对性的心理支持回复（100-200字）。"
    "遵循以下原则：\n"
    "1. 优先共情和理解，不急于给建议；\n"
    "2. 根据来访者当前状态调整策略（高痛苦→共情，低动力→探索，低清晰度→澄清）；\n"
    "3. 仅在确认高风险时才提及专业帮助，避免过度转介；\n"
    "4. 语言自然、温和，避免机械化表达。\n"
    "直接输出回复内容，不要加说明性语言。"
)

SAFETY_TEMPLATE = (
    "我注意到你现在可能正经历一些很难受的时刻。"
    "这里有一些专业支持渠道，如果你需要帮助，可以随时联系：\n"
    "- 北京心理危机研究与干预中心：010-82951332\n"
    "- 全国心理援助热线：400-161-9995\n"
    "我也在这里，愿意继续陪伴你。"
)


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _norm(s: str | None) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def _build_context_text(ctx: dict, max_turns: int = 8) -> str:
    turns = []
    for msg in (ctx.get("context") or [])[-max_turns:]:
        role = msg.get("role", "")
        if role == "system":
            continue
        content = _norm(msg.get("content", ""))
        if content:
            turns.append(f"{'来访者' if role == 'user' else '咨询师'}: {content}")
    last_user = _norm(ctx.get("last_user", ""))
    if last_user and (not turns or last_user not in turns[-1]):
        turns.append(f"来访者: {last_user}")
    return "\n".join(turns)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_set",
                    default="data/judge_eval_v10/v10_eval_contexts.jsonl")
    ap.add_argument("--safety_overrides",
                    default="data/judge_eval_v10/v12_safety_overrides.jsonl")
    ap.add_argument("--model", default="qwen-max")
    ap.add_argument("--system_id", default="llm_direct")
    ap.add_argument("--apply_safety_shield", action="store_true",
                    help="If set, override responses for contexts that "
                         "the safety shield would fire on, producing the "
                         "'llm_safe' variant.")
    ap.add_argument("--out_responses",
                    default="data/fair_bon_v12/llm_direct_bon_responses.jsonl")
    ap.add_argument("--max_workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20260508)
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--max_tokens", type=int, default=300)
    args = ap.parse_args()

    from eval._dashscope_client import DashScopeClient
    client = DashScopeClient(model=args.model,
                             cache_path="data/_dashscope_llm_direct_cache.jsonl")

    eval_rows = _load_jsonl(Path(args.eval_set))
    overrides = {
        str(r.get("sample_id")): r
        for r in _load_jsonl(Path(args.safety_overrides))
        if r.get("shield_fired") or r.get("decision") == "hard"
    }
    print(f"[llm-direct] eval contexts={len(eval_rows)} "
          f"safety overrides={len(overrides)} model={args.model}")

    def _task(ctx: dict) -> dict:
        sid = str(ctx["sample_id"])
        is_safety = bool(overrides.get(sid)) and args.apply_safety_shield
        if is_safety:
            return {
                "sample_id": sid,
                "system_id": args.system_id,
                "selected_strategy": "safety_referral",
                "response": SAFETY_TEMPLATE,
                "shield_fired": True,
                "generation_config": {"backend": "safety_template"},
            }
        ctx_text = _build_context_text(ctx)
        msgs = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": ctx_text},
        ]
        raw = client.chat(msgs, model=args.model,
                          temperature=args.temperature,
                          max_tokens=args.max_tokens,
                          seed=args.seed)
        response = _norm(raw)
        return {
            "sample_id": sid,
            "system_id": args.system_id,
            "selected_strategy": "llm_generated",
            "response": response,
            "shield_fired": False,
            "generation_config": {
                "backend": "llm_direct",
                "model": args.model,
                "temperature": args.temperature,
            },
        }

    results = client.map_concurrent(
        eval_rows, _task, max_workers=args.max_workers, progress_every=25)
    results = [r for r in results if r]

    out = Path(args.out_responses)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for rec in results:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    n_safety = sum(1 for r in results if r.get("shield_fired"))
    n_gen = len(results) - n_safety
    print(f"[llm-direct] wrote {len(results)} rows -> {out} "
          f"(generated={n_gen}, safety={n_safety})")


if __name__ == "__main__":
    main()
