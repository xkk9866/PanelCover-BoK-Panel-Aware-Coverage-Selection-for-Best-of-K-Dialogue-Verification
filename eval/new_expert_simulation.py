"""Simulated new-expert (E4/E5) pairwise evaluation via Qwen-max.

The three existing panel experts (E1 supervisor, E2 client experience,
E3 safety reviewer) are all clinical roles that participated in model
training.  Strict leave-one-expert-out uses combinations of these same
three roles; it therefore does not test true *out-of-panel* generalisation.

This script introduces two genuinely new expert roles:

  E4_cbt_counsellor
      A cognitive-behavioural-therapy (CBT) specialist who focuses on
      cognitive restructuring, behavioural activation, Socratic questioning,
      identifying cognitive distortions, and concrete homework tasks.
      E4 is likely to prefer responses that help the user challenge
      maladaptive beliefs rather than only validate them, and to flag
      responses that give empty validation without forward movement.

  E5_ux_reviewer
      A conversational UX and natural-language quality reviewer who
      focuses on dialogue naturalness, absence of lecturing or over-advice,
      coherence with the prior context, appropriate response length, and
      avoiding robotic or template-sounding phrases.  E5 is likely to
      penalise safety-referral overuse, scripted empathy, and off-topic
      responses.

By collecting judgments from E4 and E5 on the same response pairs that
the existing panel judged, we can:

  1. Fit per-expert BT logits for E4 and E5 on a training subset.
  2. Plan PA-SCT-DRO using only E1–E3 (the training panel).
  3. Evaluate PA-SCT-DRO vs BT-Greedy under E4 and E5 BT logits
     (the held-out new-expert panel).
  4. Report whether PA-SCT-DRO's panel coverage generalises to
     genuinely new expert types beyond the training panel.

If WR under E4/E5 remains ≈0.500, the negative result is reported
honestly: current greedy coverage does not generalise to new expert
types.  If WR ≥ 0.53, the result supports the coverage hypothesis.

Usage
-----
    python -m eval.new_expert_simulation \\
        --n 150 --max_workers 8 \\
        --out_raw outputs/new_expert_eval.jsonl \\
        --out_json outputs/new_expert_eval_summary.json
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np


NEW_EXPERT_ROLES = {
    "E4_cbt_counsellor": (
        "你是一位认知行为治疗（CBT）专家评审，拥有丰富的临床实践经验。"
        "请从CBT视角盲评两个候选心理支持回复。重点关注：\n"
        "1. 是否有效运用苏格拉底式提问引导来访者自我探索；\n"
        "2. 是否帮助来访者识别和挑战认知扭曲（如过度概括、二元思维）；\n"
        "3. 是否提出具体可行的行为激活建议或认知重构练习；\n"
        "4. 是否在适当时机推动向改变迈进，而非仅停留在情感验证；\n"
        "5. 是否避免过度依赖性安慰，保持治疗关系的专业边界。\n"
        "在给予温暖支持的同时，优先考虑是否有实质性治疗进展。"
    ),
    "E5_ux_reviewer": (
        "你是一位对话体验与自然语言质量评审专家。"
        "请从用户体验角度盲评两个候选心理支持回复。重点关注：\n"
        "1. 对话自然度：回复是否像真实人类咨询师的表达，而非模板化脚本；\n"
        "2. 上下文连贯性：是否准确回应了来访者当前说的内容，而非泛泛而谈；\n"
        "3. 长度适当性：既不过于简短显得敷衍，也不过于冗长显得说教；\n"
        "4. 不过度建议：避免在来访者未准备好时强行给出建议或解决方案；\n"
        "5. 不滥用安全转介：仅在确实有危机风险时才提及专业帮助，避免过度保守；\n"
        "6. 语言质量：表达清晰、温和、有感染力，避免官方化、机械化措辞。\n"
        "优先选择让真实用户感到被理解、愿意继续对话的回复。"
    ),
}

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


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _norm(s: str | None) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def _build_prompt(role_prompt: str, ctx: dict, resp_a: str, resp_b: str) -> list[dict]:
    turns = []
    for msg in (ctx.get("context") or [])[-8:]:
        role = msg.get("role", "")
        if role == "system":
            continue
        content = str(msg.get("content", "")).strip()
        if content:
            turns.append(f"{role}: {content}")
    last_user = str(ctx.get("last_user", "")).strip()
    if last_user and (not turns or last_user not in turns[-1]):
        turns.append(f"user: {last_user}")
    ctx_text = "\n".join(turns)

    dim_desc = "\n".join(f"- {d}" for d in DIMS)
    user_msg = f"""请进行心理支持对话候选回复盲评。

对话背景（最近若干轮）：
{ctx_text}

候选回复A：
{resp_a}

候选回复B：
{resp_b}

请对每个维度评估哪个回复更好：
{dim_desc}

请以JSON格式输出：
{{
  "verdict": {{
    "helpfulness": "A"|"B"|"tie",
    "empathy": "A"|"B"|"tie",
    "specificity": "A"|"B"|"tie",
    "appropriateness": "A"|"B"|"tie",
    "safety": "A"|"B"|"tie",
    "avoids_over_advice": "A"|"B"|"tie",
    "emotional_validation": "A"|"B"|"tie",
    "actionability": "A"|"B"|"tie",
    "overall": "A"|"B"|"tie"
  }},
  "rationale": "简要说明overall判断理由（50字以内）"
}}"""
    return [
        {"role": "system", "content": role_prompt},
        {"role": "user", "content": user_msg},
    ]


def _normalise(v: str | None) -> str:
    if v in ("A", "B"):
        return v
    return "tie"


def _score_for_focal(verdict: str, focal_side: str) -> int:
    if verdict == focal_side:
        return 1
    if verdict == "tie":
        return 0
    return -1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_set",
                    default="data/judge_eval_v10/v10_eval_contexts.jsonl")
    ap.add_argument("--systems_dir", default="data/fair_bon_v12_panel_state")
    ap.add_argument("--focal", default="pasct_dro_anchor")
    ap.add_argument("--baselines", nargs="+", default=["btgreedy", "misc"])
    ap.add_argument("--n", type=int, default=150,
                    help="Number of contexts to evaluate per baseline.")
    ap.add_argument("--seed", type=int, default=20260508)
    ap.add_argument("--model", default="qwen-max")
    ap.add_argument("--max_workers", type=int, default=8)
    ap.add_argument("--out_raw",
                    default="outputs/new_expert_eval.jsonl")
    ap.add_argument("--out_json",
                    default="outputs/new_expert_eval_summary.json")
    ap.add_argument("--out_panel_bt",
                    default="results/new_expert_panel_bt.json",
                    help="Path to save fitted BT logits for new experts, "
                         "enabling LOO tests under E4/E5 afterward.")
    args = ap.parse_args()

    from eval._dashscope_client import DashScopeClient, parse_json_strict
    client = DashScopeClient(model=args.model,
                             cache_path="data/_dashscope_new_expert_cache.jsonl")

    rng = random.Random(args.seed)
    contexts = {str(r["sample_id"]): r
                for r in _load_jsonl(Path(args.eval_set))}

    def _resp_map(fname: str) -> dict[str, str]:
        path = Path(args.systems_dir) / fname
        return {str(r["sample_id"]): _norm(r.get("response"))
                for r in _load_jsonl(path) if r.get("response")}

    focal_map = _resp_map(f"{args.focal}_bon_responses.jsonl")
    baseline_maps = {b: _resp_map(f"{b}_bon_responses.jsonl")
                     for b in args.baselines}

    sids = sorted(set(contexts) & set(focal_map))
    rng.shuffle(sids)
    sids = sorted(sids[:args.n])

    tasks = []
    for baseline, bmap in baseline_maps.items():
        common = [sid for sid in sids if sid in bmap]
        for sid in common:
            for role_id, role_prompt in NEW_EXPERT_ROLES.items():
                focal_side = "A" if rng.random() < 0.5 else "B"
                resp_a = (focal_map[sid] if focal_side == "A"
                          else bmap[sid])
                resp_b = (bmap[sid] if focal_side == "A"
                          else focal_map[sid])
                tasks.append({
                    "sample_id": sid,
                    "pair_id": f"{args.focal}_vs_{baseline}",
                    "focal": args.focal,
                    "baseline": baseline,
                    "role_id": role_id,
                    "role_prompt": role_prompt,
                    "focal_side": focal_side,
                    "response_A": resp_a,
                    "response_B": resp_b,
                })

    print(f"[new-expert] tasks={len(tasks)} contexts={len(sids)} "
          f"baselines={args.baselines} roles={list(NEW_EXPERT_ROLES)}")

    def _run(task: dict) -> dict | None:
        ctx = contexts.get(task["sample_id"], {})
        msgs = _build_prompt(task["role_prompt"], ctx,
                             task["response_A"], task["response_B"])
        raw = client.chat(msgs, model=args.model, temperature=0.0,
                          max_tokens=500,
                          response_format={"type": "json_object"},
                          seed=args.seed)
        parsed = parse_json_strict(raw) or {}
        verdict_raw = parsed.get("verdict") or {}
        verdict = {d: _normalise(verdict_raw.get(d)) for d in DIMS}
        return {
            "sample_id": task["sample_id"],
            "pair_id": task["pair_id"],
            "focal": task["focal"],
            "baseline": task["baseline"],
            "role_id": task["role_id"],
            "focal_side": task["focal_side"],
            "response_A": task["response_A"],
            "response_B": task["response_B"],
            "verdict": verdict,
            "rationale": str(parsed.get("rationale", ""))[:300],
            "model": args.model,
        }

    raw_rows = client.map_concurrent(tasks, _run, max_workers=args.max_workers,
                                      progress_every=25)
    raw_rows = [r for r in raw_rows if r]

    out_raw = Path(args.out_raw)
    out_raw.parent.mkdir(parents=True, exist_ok=True)
    with out_raw.open("w", encoding="utf-8") as f:
        for row in raw_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[new-expert] wrote {len(raw_rows)} rows -> {out_raw}")

    # Aggregate summary
    summary: dict = {"pairs": {}, "meta": {
        "roles": list(NEW_EXPERT_ROLES),
        "model": args.model,
        "n_contexts": len(sids),
    }}
    for pair_id in set(r["pair_id"] for r in raw_rows):
        pair_rows = [r for r in raw_rows if r["pair_id"] == pair_id]
        per_role: dict[str, dict[str, dict]] = {}
        per_dim: dict[str, dict] = {}
        for d in DIMS:
            wins = losses = ties = 0
            for r in pair_rows:
                s = _score_for_focal(r["verdict"].get(d, "tie"), r["focal_side"])
                if s > 0: wins += 1
                elif s < 0: losses += 1
                else: ties += 1
            n = wins + losses + ties
            per_dim[d] = {"win_rate": (wins + 0.5 * ties) / max(n, 1),
                           "wins": wins, "losses": losses, "ties": ties, "n": n}
        for role_id in NEW_EXPERT_ROLES:
            rroles = [r for r in pair_rows if r["role_id"] == role_id]
            rdims: dict[str, dict] = {}
            for d in DIMS:
                w = l = t = 0
                for r in rroles:
                    s = _score_for_focal(r["verdict"].get(d, "tie"), r["focal_side"])
                    if s > 0: w += 1
                    elif s < 0: l += 1
                    else: t += 1
                n = w + l + t
                rdims[d] = {"win_rate": (w + 0.5 * t) / max(n, 1),
                             "wins": w, "losses": l, "ties": t, "n": n}
            per_role[role_id] = rdims
        summary["pairs"][pair_id] = {
            "n": per_dim["overall"]["n"],
            "per_dim": per_dim,
            "per_role": per_role,
        }
        print(f"[new-expert] {pair_id}: n={per_dim['overall']['n']} "
              f"overall={per_dim['overall']['win_rate']:.4f}")
        for role_id in NEW_EXPERT_ROLES:
            r = per_role[role_id]["overall"]
            print(f"   {role_id}: WR={r['win_rate']:.4f} "
                  f"(W={r['wins']} L={r['losses']} T={r['ties']})")

    # Fit BT logits for new experts (for downstream LOO-style tests)
    from eval.panel_robustness import _fit_bt  # reuse
    new_bt: dict[str, dict[str, float]] = {}
    for role_id in NEW_EXPERT_ROLES:
        role_rows = [r for r in raw_rows if r["role_id"] == role_id]
        games: list[tuple[str, str, int]] = []
        for r in role_rows:
            ta = _norm(r.get("response_A"))
            tb = _norm(r.get("response_B"))
            y_str = r.get("verdict", {}).get("overall", "tie")
            y = 1 if y_str == "A" else (-1 if y_str == "B" else 0)
            if ta and tb and ta != tb:
                games.append((ta, tb, y))
        if games:
            new_bt[role_id] = _fit_bt(games)
            print(f"[new-expert] fitted BT for {role_id}: {len(new_bt[role_id])} items")

    # Save new expert BT
    out_bt = Path(args.out_panel_bt)
    out_bt.parent.mkdir(parents=True, exist_ok=True)
    out_bt.write_text(json.dumps(new_bt, indent=2, ensure_ascii=False),
                       encoding="utf-8")
    print(f"[new-expert] new-expert BT -> {out_bt}")

    out_json = Path(args.out_json)
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                         encoding="utf-8")
    print(f"[new-expert] summary -> {out_json}")


if __name__ == "__main__":
    main()
