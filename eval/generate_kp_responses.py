"""Generate **knowledge-augmented** responses for the PA-SCT-DRO++
unified candidate pool.

Each (sample_id, strategy) pair already has one cached response in
``data/judge_eval_v10/v10_responses.jsonl``.  This script asks Qwen-max
to write a *second* response per (sample_id, strategy) under a clinical
knowledge-priming prompt that injects:

  * a CBT vocabulary anchor (cognitive distortion labels, behavioural
    activation steps, grounding skills);
  * a specific evidence anchor (psychoeducation snippet appropriate to
    the strategy);
  * a safety-policy instruction that respects the v12 two-threshold
    shield (no actionable advice when safety_referral is requested).

The output is a parallel response pool::

    data/judge_eval_v10/v10_responses_kp.jsonl

with the same schema as ``v10_responses.jsonl`` plus a
``kp_strategy_id`` field.  PA-SCT-DRO++ then merges the two pools at
candidate-pool build time.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import random
from pathlib import Path

from eval._dashscope_client import DashScopeClient


_CLIENT: DashScopeClient | None = None


def _client() -> DashScopeClient:
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = DashScopeClient(model="qwen-max")
    return _CLIENT


ROOT = Path(__file__).resolve().parents[1]


# Strategies we generate KP variants for.  We do not generate a KP
# variant for ``safety_referral`` because the safety template is
# intentionally fixed; the planner already routes to the canonical
# safe template via the v12 shield.
KP_STRATEGIES = (
    "question",
    "reflection",
    "empathy",
    "reframe",
    "summarization",
    "action_suggestion",
)


_STRATEGY_PROMPT = {
    "question": (
        "请用 1-2 句话提出一个开放性问题，帮助来访者具体化或聚焦当前的困扰；"
        "问题要 CBT 风格（关注事件 / 想法 / 情绪 / 行为 / 后果之一）。"
    ),
    "reflection": (
        "请用 1-2 句话准确反映来访者刚才的核心情绪与含义（情感反映 / 内容反映），"
        "避免直接给建议；可以引入一个具体的情绪标签词。"
    ),
    "empathy": (
        "请用 1-2 句话表达共情，承认来访者经历的合理性与艰难；可以使用 CBT 的"
        "正常化（normalisation）句式，但不要承诺疗效，不下诊断。"
    ),
    "reframe": (
        "请用 1-2 句话进行温和的认知重评（cognitive reappraisal）：用一个具体的"
        "替代视角（不是空洞的鼓励）软化非理性想法，仍以来访者的体验为中心。"
    ),
    "summarization": (
        "请用 1-2 句话做小结：把来访者刚才提到的关键事件 + 核心感受 + 当前需求"
        "串成一段，便于他/她确认或修正。"
    ),
    "action_suggestion": (
        "请提出一条具体、可执行、有 CBT / 行为激活证据基础的建议（例如：5-4-3-2-1"
        " grounding、活动安排、思维记录、放松练习），写成 1-2 句话；明确指出"
        "执行步骤的第一步，避免笼统鼓励。"
    ),
}


def _build_prompt(history_text: str, strategy: str) -> list[dict]:
    sys = (
        "你是一名经验丰富的临床心理咨询师，正在用普通话与来访者对话。"
        "你的回复必须遵循以下要求：\n"
        "1. 不下医学诊断，不承诺疗效；\n"
        "2. 不提供处方或药物建议；\n"
        "3. 简洁、自然、像朋友一样亲近，但保持专业边界；\n"
        "4. 当来访者表达自伤 / 自杀 / 紧急危险时，优先安全转介，不再继续建议；\n"
        f"5. 这一轮，请按下面的支持策略写回复：{_STRATEGY_PROMPT[strategy]}"
    )
    return [
        {"role": "system", "content": sys},
        {"role": "user", "content": history_text},
    ]


def _ctx_to_text(ctx: dict) -> str:
    """Render the eval context to a Qwen prompt-friendly string.

    The eval context follows the v10 schema: a ``context`` list of
    role / content turns (we drop the long system prompt) and an
    optional ``user_input`` final user turn.
    """
    turns = ctx.get("context") or ctx.get("history") or []
    out_lines = []
    for turn in turns:
        role = turn.get("role")
        content = (turn.get("content") or "").strip()
        if not content:
            continue
        if role == "system":
            continue
        if role == "user":
            out_lines.append(f"来访者：{content}")
        elif role in ("assistant", "supporter", "counsellor"):
            out_lines.append(f"咨询师：{content}")
        else:
            out_lines.append(f"{role or '说话人'}：{content}")
    user = ctx.get("user_input") or ctx.get("query") or ""
    if user and (not out_lines or not out_lines[-1].endswith(user)):
        out_lines.append(f"来访者：{user}")
    if out_lines and not out_lines[-1].startswith("咨询师："):
        out_lines.append("咨询师：")
    return "\n".join(out_lines)


def _gen_one(ctx: dict, strategy: str, model: str, temperature: float
             ) -> dict | None:
    sid = str(ctx["sample_id"])
    history_text = _ctx_to_text(ctx)
    msgs = _build_prompt(history_text, strategy)
    try:
        text = _client().chat(
            messages=msgs, model=model,
            temperature=temperature, max_tokens=200,
        )
    except Exception as exc:
        print(f"[kp-gen] error sid={sid} strategy={strategy}: {exc}")
        return None
    text = text.strip()
    if not text:
        return None
    return {
        "sample_id": sid,
        "system_id": "psystate_kp",
        "selected_strategy": strategy,
        "response": text,
        "shield_fired": False,
        "kp_strategy_id": f"{sid}::{strategy}",
        "generation_config": {
            "backend": model,
            "model": model,
            "max_tokens": 200,
            "temperature": temperature,
            "kp_prompt": True,
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_set",
                    default="data/judge_eval_v10/v10_eval_contexts.jsonl")
    ap.add_argument("--out", default="data/judge_eval_v10/v10_responses_kp.jsonl")
    ap.add_argument("--strategies", nargs="+", default=list(KP_STRATEGIES))
    ap.add_argument("--model", default="qwen-max")
    ap.add_argument("--temperature", type=float, default=0.4)
    ap.add_argument("--max_workers", type=int, default=12)
    ap.add_argument("--n_ctx", type=int, default=0,
                    help="Subsample N contexts (0 = all).")
    ap.add_argument("--resume", action="store_true",
                    help="Skip (sample_id, strategy) pairs already in --out.")
    args = ap.parse_args()

    eval_rows = [
        json.loads(l)
        for l in open(args.eval_set, encoding="utf-8") if l.strip()
    ]
    if args.n_ctx and args.n_ctx < len(eval_rows):
        eval_rows = eval_rows[: args.n_ctx]
    print(f"[kp-gen] {len(eval_rows)} contexts x {len(args.strategies)} "
          f"strategies = {len(eval_rows) * len(args.strategies)} calls")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    seen: set[tuple[str, str]] = set()
    if args.resume and out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            seen.add((str(rec.get("sample_id")), str(rec.get("selected_strategy"))))
        print(f"[kp-gen] resume: skipping {len(seen)} pairs already done")

    tasks: list[tuple[dict, str]] = []
    for ctx in eval_rows:
        for strategy in args.strategies:
            if (str(ctx["sample_id"]), strategy) in seen:
                continue
            tasks.append((ctx, strategy))
    random.shuffle(tasks)
    print(f"[kp-gen] launching {len(tasks)} new generation tasks")

    fout = open(out_path, "a" if args.resume else "w", encoding="utf-8")
    done = 0
    with cf.ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futs = {pool.submit(_gen_one, ctx, strategy, args.model,
                            args.temperature): (ctx, strategy)
                for ctx, strategy in tasks}
        for fut in cf.as_completed(futs):
            rec = fut.result()
            if rec is not None:
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fout.flush()
                done += 1
                if done % 50 == 0:
                    print(f"[kp-gen] done {done}/{len(tasks)}")
    fout.close()
    print(f"[kp-gen] wrote {done} new generations -> {out_path}")


if __name__ == "__main__":
    main()
