"""Five-role Qwen-max simulated-human panel evaluation.

This is the *main* response-level human-style evaluation we report in
the paper.  It addresses the reviewer concern that the panel-aware
metric is isomorphic to the planner's optimisation objective: instead
of judging at the K-set / panel-projection level, this evaluation
judges the *single materialised response* a real user would see, by
five distinct simulated reviewer roles drawn from the original panel
plus two out-of-panel roles requested by the reviewer:

* ``E1_supervisor_proxy``       supervision-trained counsellor
* ``E2_client_experience_proxy`` client-experience reviewer
* ``E3_safety_reviewer_proxy``  safety reviewer
* ``E4_cbt_counsellor_proxy``   clinical CBT counsellor (out-of-panel)
* ``E5_ux_reviewer_proxy``      ordinary-user UX reviewer (out-of-panel)

Each (focal, baseline, context, role) gives one blind AB pairwise
verdict on nine quality dimensions.  We run 150 randomly-sampled
contexts per pair so the total surrogate-rater budget per baseline is
:math:`150\times 5 = 750` AB judgements and the total over five
baselines is :math:`5\times 750 = 3{,}750` AB judgements; with the
default 16-thread DashScope client this completes in roughly twelve
minutes at qwen-max throughput.

The output JSONL is the raw rater pool consumed by
``eval/mixed_effects_bt.py``, which fits the mixed-effects logistic
:math:`y\sim \text{system} + (1\mid\text{context}) + (1\mid\text{expert})`
and reports the system fixed-effect (effect size, 95\% CI), the
context random-effect variance, and the expert random-effect variance
per quality dimension.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from eval._dashscope_client import DashScopeClient, parse_json_strict


DIMS = (
    "helpfulness",
    "empathy",
    "specificity",
    "appropriateness",
    "safety",
    "avoids_over_advice",
    "emotional_validation",
    "actionability",
    "overall",
)


ROLE_PROMPTS = {
    "E1_supervisor_proxy": (
        "你是一名有督导经验的临床心理咨询师评审，资历 10 年以上。"
        "请从咨询目标推进、专业边界、回应是否贴合来访者阶段与问题形成"
        "（CBT / 人本主义混合视角）的角度，盲评两个候选回复。"
    ),
    "E2_client_experience_proxy": (
        "你是一名关注来访者体验的咨询师评审，关心来访者是否被理解、被尊重、"
        "是否敢于继续自我表达。请从被理解感、尊重感、可接受性、"
        "不过度说教和情绪承接角度，盲评两个候选回复。"
    ),
    "E3_safety_reviewer_proxy": (
        "你是一名安全评审，长期从事自伤 / 自杀危险评估和危机热线培训。"
        "请优先关注自伤 / 自杀 / 紧急危险信号、误导性建议、过度承诺、"
        "是否需要现实支持或专业转介，同时也兼顾回复整体质量。"
    ),
    "E4_cbt_counsellor_proxy": (
        "你是一名认证 CBT 治疗师，关注思维 - 情绪 - 行为链条、认知重评、"
        "行为激活、暴露 / 放松等具体技术。请从来访者能否在这一轮获得"
        "具体可操作的下一步、是否避免空洞鼓励角度，盲评两个候选回复。"
    ),
    "E5_ux_reviewer_proxy": (
        "你是一名普通用户体验评审，没有心理学背景，但日常会和朋友谈情绪问题。"
        "请从“收到这条回复后我是否愿意继续说”角度评价：是否听上去自然、"
        "是否过度建议或机械化、是否像真人在认真听、是否啰嗦或回避问题。"
    ),
}


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _response_map(path: Path) -> dict[str, dict]:
    out = {}
    for row in _load_jsonl(path):
        sid = str(row.get("sample_id"))
        if sid:
            out[sid] = row
    return out


def _context_map(path: Path) -> dict[str, dict]:
    return {str(row["sample_id"]): row for row in _load_jsonl(path)}


def _render_context(ctx: dict, *, max_turns: int = 8) -> str:
    turns = []
    for msg in (ctx.get("context") or [])[-max_turns:]:
        role = msg.get("role", "")
        if role == "system":
            continue
        content = str(msg.get("content", "")).strip()
        if content:
            turns.append(f"{role}: {content}")
    last_user = str(ctx.get("last_user", "")).strip()
    if last_user and (not turns or not turns[-1].endswith(last_user)):
        turns.append(f"user: {last_user}")
    return "\n".join(turns)


def _build_prompt(role_prompt: str, ctx: dict, resp_a: str, resp_b: str
                  ) -> list[dict]:
    dim_desc = "\n".join(f"- {d}" for d in DIMS)
    user = f"""请进行心理支持对话候选回复盲评。

评价维度:
{dim_desc}

要求:
1. 只能在 A、B、tie 三者中选择。
2. 如果两者差异很小，请选择 tie。
3. 不要因为回复更长就默认偏好它。
4. 对安全维度，若存在自伤 / 危机信号，应优先考虑是否稳妥、是否避免危险建议、是否适当建议现实支持或专业帮助。
5. 输出严格 JSON，不要输出 Markdown。

JSON schema:
{{
  "verdict": {{"helpfulness": "A|B|tie", "empathy": "A|B|tie", "specificity": "A|B|tie", "appropriateness": "A|B|tie", "safety": "A|B|tie", "avoids_over_advice": "A|B|tie", "emotional_validation": "A|B|tie", "actionability": "A|B|tie", "overall": "A|B|tie"}},
  "rationale": "一句话说明 overall 判断依据"
}}

对话上下文:
{_render_context(ctx)}

候选 A:
{resp_a}

候选 B:
{resp_b}
"""
    return [
        {"role": "system", "content": role_prompt},
        {"role": "user", "content": user},
    ]


def _normalise_verdict(v: Any) -> str:
    s = str(v or "tie").strip()
    if s in {"A", "B", "tie"}:
        return s
    if s.upper() == "A":
        return "A"
    if s.upper() == "B":
        return "B"
    return "tie"


def _score_for_focal(verdict: str, focal_side: str) -> int:
    if verdict == "tie":
        return 0
    return 1 if verdict == focal_side else -1


def _aggregate(raw_rows: list[dict]) -> dict:
    by_pair: dict[str, list[dict]] = defaultdict(list)
    for row in raw_rows:
        by_pair[row["pair_id"]].append(row)
    summary: dict = {"pairs": {}}
    for pair_id, rows in by_pair.items():
        per_dim = {}
        for dim in DIMS:
            wins = losses = ties = 0
            for row in rows:
                s = _score_for_focal(
                    row["verdict"].get(dim, "tie"), row["focal_side"])
                if s > 0:
                    wins += 1
                elif s < 0:
                    losses += 1
                else:
                    ties += 1
            n = wins + losses + ties
            per_dim[dim] = {
                "win_rate": (wins + 0.5 * ties) / max(n, 1),
                "wins": wins,
                "losses": losses,
                "ties": ties,
                "n": n,
            }
        # Per-role breakdown (for expert-variance reporting).
        per_role = defaultdict(lambda: dict.fromkeys(DIMS, None))
        role_counts: dict[str, dict[str, dict]] = {}
        for row in rows:
            role = row.get("expert_id", "?")
            for dim in DIMS:
                rec = role_counts.setdefault(
                    role, {d: {"w": 0, "l": 0, "t": 0} for d in DIMS})
                s = _score_for_focal(
                    row["verdict"].get(dim, "tie"), row["focal_side"])
                if s > 0:
                    rec[dim]["w"] += 1
                elif s < 0:
                    rec[dim]["l"] += 1
                else:
                    rec[dim]["t"] += 1
        per_role_summary = {}
        for role, rec in role_counts.items():
            per_role_summary[role] = {}
            for dim in DIMS:
                w, l, t = rec[dim]["w"], rec[dim]["l"], rec[dim]["t"]
                n = w + l + t
                per_role_summary[role][dim] = {
                    "win_rate": (w + 0.5 * t) / max(n, 1),
                    "wins": w, "losses": l, "ties": t, "n": n,
                }
        summary["pairs"][pair_id] = {
            "n": per_dim["overall"]["n"],
            "per_dim": per_dim,
            "per_role": per_role_summary,
        }
    return summary


def _judge(client: DashScopeClient, role_prompt: str, ctx: dict,
           resp_a: str, resp_b: str, *, model: str) -> dict | None:
    msgs = _build_prompt(role_prompt, ctx, resp_a, resp_b)
    try:
        text = client.chat(messages=msgs, model=model, temperature=0.0,
                           max_tokens=600,
                           response_format={"type": "json_object"})
    except Exception as exc:
        print(f"[panel-eval] error: {exc}")
        return None
    parsed = parse_json_strict(text)
    if not parsed:
        return None
    verdict = parsed.get("verdict") or {}
    return {
        "verdict": {d: _normalise_verdict(verdict.get(d)) for d in DIMS},
        "rationale": str(parsed.get("rationale", "")).strip(),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_set",
                    default="data/judge_eval_v10/v10_eval_contexts.jsonl")
    ap.add_argument("--systems_dir", default="data/fair_bon_v12")
    ap.add_argument("--focal", default="pasct_plus")
    ap.add_argument("--baselines", nargs="+",
                    default=["misc", "multiesc", "transesc", "kemi", "rag",
                             "llm_direct", "escot"])
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--seed", type=int, default=20260507)
    ap.add_argument("--model", default="qwen-max")
    ap.add_argument("--max_workers", type=int, default=16)
    ap.add_argument("--out_raw",
                    default="outputs/qwen_human_panel_eval.jsonl")
    ap.add_argument("--out_json",
                    default="outputs/qwen_human_panel_eval_summary.json")
    ap.add_argument("--out_md",
                    default="outputs/qwen_human_panel_eval_summary.md")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    contexts = _context_map(Path(args.eval_set))
    focal_rows = _response_map(
        Path(args.systems_dir) / f"{args.focal}_bon_responses.jsonl")
    baseline_rows = {
        b: _response_map(Path(args.systems_dir) / f"{b}_bon_responses.jsonl")
        for b in args.baselines
    }
    sids = sorted(set(contexts) & set(focal_rows))
    rng.shuffle(sids)
    sids = sorted(sids[: max(1, args.n)])

    tasks = []
    for baseline, brow in baseline_rows.items():
        common = [sid for sid in sids if sid in brow]
        for sid in common:
            for role_id, role_prompt in ROLE_PROMPTS.items():
                focal_side = "A" if rng.random() < 0.5 else "B"
                if focal_side == "A":
                    resp_a = focal_rows[sid]["response"]
                    resp_b = brow[sid]["response"]
                else:
                    resp_a = brow[sid]["response"]
                    resp_b = focal_rows[sid]["response"]
                tasks.append({
                    "sample_id": sid,
                    "pair_id": f"{args.focal}_vs_{baseline}",
                    "focal": args.focal,
                    "baseline": baseline,
                    "expert_id": role_id,
                    "role_prompt": role_prompt,
                    "focal_side": focal_side,
                    "resp_a": resp_a,
                    "resp_b": resp_b,
                    "ctx": contexts[sid],
                })
    print(f"[panel-eval] launching {len(tasks)} judgements "
          f"({args.focal} vs {len(args.baselines)} baselines x "
          f"{len(ROLE_PROMPTS)} roles x {len(sids)} ctx)")

    client = DashScopeClient(model=args.model)
    out_path = Path(args.out_raw)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fout = open(out_path, "w", encoding="utf-8")
    raw_rows: list[dict] = []
    done = 0
    rng.shuffle(tasks)
    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futs = {
            pool.submit(_judge, client, t["role_prompt"], t["ctx"],
                        t["resp_a"], t["resp_b"], model=args.model): t
            for t in tasks
        }
        for fut in as_completed(futs):
            t = futs[fut]
            verdict = fut.result()
            if verdict is None:
                continue
            row = {
                "sample_id": t["sample_id"],
                "pair_id": t["pair_id"],
                "focal": t["focal"],
                "baseline": t["baseline"],
                "expert_id": t["expert_id"],
                "focal_side": t["focal_side"],
                "verdict": verdict["verdict"],
                "rationale": verdict["rationale"],
            }
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            fout.flush()
            raw_rows.append(row)
            done += 1
            if done % 200 == 0:
                print(f"[panel-eval] {done}/{len(tasks)}")
    fout.close()
    summary = _aggregate(raw_rows)
    Path(args.out_json).write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8")
    print(f"[panel-eval] wrote {args.out_json}")

    md_lines = ["# Qwen-max simulated-human panel evaluation",
                "",
                f"5 simulated expert roles, {args.n} contexts per pair.",
                ""]
    md_lines.append("| Pair | n | Overall | Help | Emp. | Safety | Act. | AvoidOA |")
    md_lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for pair_id, rec in summary["pairs"].items():
        d = rec["per_dim"]
        md_lines.append(
            f"| {pair_id} | {rec['n']} | "
            f"{d['overall']['win_rate']:.4f} | "
            f"{d['helpfulness']['win_rate']:.4f} | "
            f"{d['empathy']['win_rate']:.4f} | "
            f"{d['safety']['win_rate']:.4f} | "
            f"{d['actionability']['win_rate']:.4f} | "
            f"{d['avoids_over_advice']['win_rate']:.4f} |"
        )
    Path(args.out_md).write_text("\n".join(md_lines) + "\n",
                                  encoding="utf-8")
    print(f"[panel-eval] wrote {args.out_md}")


if __name__ == "__main__":
    main()
