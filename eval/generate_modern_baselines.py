"""Generate prompt-only modern ESC baselines for the comparison set.

We add three 2024+ baselines whose contribution is *prompting style*
rather than retrieval / extra training (the cached generator is the
same Qwen-max for every system, so the comparison isolates the
prompting / planning algorithm):

* ``llm_direct``  Vanilla zero-shot LLM ESC, no CoT, no strategy
  scaffolding.  This is the LLM-direct baseline reviewers ask for.
* ``escot``       ESCoT-style chain-of-thought.  The prompt asks the
  model to (i) reason about the user's emotional state and underlying
  needs, (ii) explicitly plan a support strategy, (iii) write a single
  response.  We keep only the final response.
* ``self_refine`` Self-refine ESC.  The prompt asks the model to
  draft, critique, and revise its own response in one pass.

Each baseline materialises one response per of the 500 eval contexts
and writes a fair-BoN response file::

    data/fair_bon_v12/{system}_bon_responses.jsonl

so that panel-state response WR, panel-aware BT projection (via the
text-input extender), and Qwen-max pairwise evaluation can be run
identically to the trained baselines.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import random
from pathlib import Path

from eval._dashscope_client import DashScopeClient


ROOT = Path(__file__).resolve().parents[1]


_SYS_LLM_DIRECT = (
    "你是一名经验丰富的临床心理咨询师，正在用普通话与来访者对话。"
    "请简短、自然、共情地回复来访者的最后一句话；不下医学诊断，不承诺疗效，"
    "不提供药物建议；当来访者表达自伤 / 自杀 / 紧急危险时，请优先提供安全转介，"
    "建议联系全国 24 小时心理援助热线 400-161-9995，并告知一个信任的人。"
)


_SYS_ESCOT = (
    "你是一名经验丰富的临床心理咨询师，遵循 CBT / 人本主义。"
    "请按下列三步思考后只输出最终回复：\n"
    "Step 1: 用一句话描述来访者当前的情绪、想法、行为模式。\n"
    "Step 2: 在 [empathy, reflection, question, reframe, summarization, "
    "action_suggestion, safety_referral] 中挑选一个最合适的支持策略，并简要说明原因。\n"
    "Step 3: 用 1-2 句话写最终回复，必须遵循上一步选定的策略。\n\n"
    "输出格式（严格遵守）：\n"
    "Reasoning: <Step 1 的一句话>\n"
    "Strategy: <Step 2 的策略名>\n"
    "Response: <Step 3 的最终回复>\n\n"
    "约束：不下医学诊断，不承诺疗效；表达自伤 / 自杀 / 紧急危险时优先安全转介。"
)


_SYS_SELF_REFINE = (
    "你是一名经验丰富的临床心理咨询师。请按以下两步思考后只输出最终回复：\n"
    "Step 1 (Draft): 用 1-2 句话写初稿。\n"
    "Step 2 (Critique): 检查初稿是否：(a) 共情得到位，(b) 没有空洞鼓励，"
    "(c) 不下诊断 / 不承诺疗效，(d) 没有过度建议，(e) 在自伤危险时优先转介。\n"
    "Step 3 (Revise): 用 1-2 句话写改进后的最终回复。\n\n"
    "输出格式（严格遵守）：\n"
    "Draft: <Step 1>\n"
    "Critique: <Step 2 的简短点评>\n"
    "Response: <Step 3 的最终回复>"
)


SYSTEM_PROMPTS = {
    "llm_direct": _SYS_LLM_DIRECT,
    "escot": _SYS_ESCOT,
    "self_refine": _SYS_SELF_REFINE,
}


def _ctx_to_text(ctx: dict) -> str:
    turns = ctx.get("context") or ctx.get("history") or []
    out_lines: list[str] = []
    for turn in turns:
        role = turn.get("role")
        content = (turn.get("content") or "").strip()
        if not content or role == "system":
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


def _extract_response(raw: str, system: str) -> tuple[str, str]:
    """Strip the CoT scaffolding and return ``(strategy, response)``."""
    raw = (raw or "").strip()
    strategy = "unknown"
    if system in ("escot", "self_refine"):
        for line in raw.splitlines():
            line = line.strip()
            if line.lower().startswith("strategy:"):
                strategy = line.split(":", 1)[1].strip().lower().strip("[]")
                strategy = strategy.split("(")[0].strip()
            elif line.lower().startswith("response:"):
                resp = line.split(":", 1)[1].strip()
                if resp:
                    return strategy, resp
        # If schema not followed, return the whole text as the response.
        return strategy, raw
    return "unknown", raw


_CLIENT: DashScopeClient | None = None


def _client() -> DashScopeClient:
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = DashScopeClient(model="qwen-max")
    return _CLIENT


def _gen_one(ctx: dict, system: str, model: str, temperature: float
             ) -> dict | None:
    sid = str(ctx["sample_id"])
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPTS[system]},
        {"role": "user", "content": _ctx_to_text(ctx)},
    ]
    try:
        text = _client().chat(
            messages=msgs, model=model,
            temperature=temperature, max_tokens=300,
        )
    except Exception as exc:
        print(f"[modern-bl] error sid={sid} system={system}: {exc}")
        return None
    strategy, resp = _extract_response(text, system)
    if not resp.strip():
        return None
    return {
        "sample_id": sid,
        "system_id": system,
        "selected_strategy": strategy,
        "response": resp.strip(),
        "verifier_choice": f"{system}_prompt",
        "K": 1,
        "generation_config": {
            "backend": model, "model": model,
            "max_tokens": 300, "temperature": temperature,
            "system_prompt_id": system,
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_set",
                    default="data/judge_eval_v10/v10_eval_contexts.jsonl")
    ap.add_argument("--systems", nargs="+",
                    default=["llm_direct", "escot", "self_refine"])
    ap.add_argument("--out_dir", default="data/fair_bon_v12")
    ap.add_argument("--model", default="qwen-max")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max_workers", type=int, default=12)
    ap.add_argument("--n_ctx", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    eval_rows = [
        json.loads(l)
        for l in open(args.eval_set, encoding="utf-8") if l.strip()
    ]
    if args.n_ctx and args.n_ctx < len(eval_rows):
        eval_rows = eval_rows[: args.n_ctx]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for system in args.systems:
        out_path = out_dir / f"{system}_bon_responses.jsonl"
        seen: set[str] = set()
        if args.resume and out_path.exists():
            for line in out_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                seen.add(str(rec.get("sample_id")))
        tasks = [c for c in eval_rows if str(c["sample_id"]) not in seen]
        random.shuffle(tasks)
        print(f"[modern-bl] {system}: {len(tasks)} tasks "
              f"(resume skipped {len(seen)}, total {len(eval_rows)})")
        fout = open(out_path, "a" if args.resume else "w", encoding="utf-8")
        done = 0
        with cf.ThreadPoolExecutor(max_workers=args.max_workers) as pool:
            futs = {pool.submit(_gen_one, ctx, system, args.model,
                                args.temperature): ctx
                    for ctx in tasks}
            for fut in cf.as_completed(futs):
                rec = fut.result()
                if rec is not None:
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    fout.flush()
                    done += 1
                    if done % 100 == 0:
                        print(f"[modern-bl] {system}: done {done}/{len(tasks)}")
        fout.close()
        print(f"[modern-bl] {system}: wrote {done} -> {out_path}")


if __name__ == "__main__":
    main()
