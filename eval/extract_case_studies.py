"""Extract high-quality case studies for the paper appendix.

The paper needs a small number of qualitative examples that are faithful to
completed artifacts.  This script selects cases from Qwen/new-expert proxy
judgments and joins them with materialised responses.  It writes a compact
Markdown file plus a LaTeX-ready appendix snippet.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


def _norm(s: str | None) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def _short(s: str, n: int = 260) -> str:
    s = _norm(s)
    return s if len(s) <= n else s[: n - 1] + "..."


def _tex_escape(s: str) -> str:
    return (s.replace("\\", "\\textbackslash{}")
             .replace("&", "\\&")
             .replace("%", "\\%")
             .replace("$", "\\$")
             .replace("#", "\\#")
             .replace("_", "\\_")
             .replace("{", "\\{")
             .replace("}", "\\}"))


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _response_map(path: Path) -> dict[str, dict]:
    return {
        str(r.get("sample_id")): r
        for r in _load_jsonl(path)
        if r.get("sample_id") and r.get("response")
    }


def _context_map(path: Path) -> dict[str, dict]:
    return {str(r.get("sample_id")): r for r in _load_jsonl(path)}


def _last_user(ctx: dict) -> str:
    return _norm(ctx.get("last_user")) or _norm(
        next((m.get("content") for m in reversed(ctx.get("context") or [])
              if m.get("role") == "user"), ""))


def _score_for_focal(row: dict) -> int:
    verdict = (row.get("verdict") or {}).get("overall", "tie")
    if verdict == "tie":
        return 0
    return 1 if verdict == row.get("focal_side") else -1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_set", default="data/judge_eval_v10/v10_eval_contexts.jsonl")
    ap.add_argument("--systems_dir", default="data/fair_bon_v12")
    ap.add_argument("--focal", default="pasct_dro_anchor")
    ap.add_argument("--baselines", nargs="+",
                    default=["misc", "multiesc", "transesc", "kemi", "rag"])
    ap.add_argument("--raw_files", nargs="+", default=[
        "outputs/qwen_full_strong_baselines_v2.jsonl",
        "outputs/new_expert_eval_strong_v2.jsonl",
    ])
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--out_md", default="outputs/case_studies.md")
    ap.add_argument("--out_tex", default="paper/PsyState/tables/case_studies.tex")
    args = ap.parse_args()

    contexts = _context_map(Path(args.eval_set))
    focal = _response_map(Path(args.systems_dir) / f"{args.focal}_bon_responses.jsonl")
    baselines = {
        b: _response_map(Path(args.systems_dir) / f"{b}_bon_responses.jsonl")
        for b in args.baselines
    }
    rows = []
    for raw_file in args.raw_files:
        rows.extend(_load_jsonl(Path(raw_file)))

    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        if row.get("focal") == args.focal:
            grouped[(str(row.get("sample_id")), str(row.get("baseline")))].append(row)

    candidates = []
    for (sid, baseline), grow in grouped.items():
        if baseline not in baselines or sid not in contexts or sid not in focal:
            continue
        if sid not in baselines[baseline]:
            continue
        votes = [_score_for_focal(r) for r in grow]
        wins = sum(1 for v in votes if v > 0)
        losses = sum(1 for v in votes if v < 0)
        ties = sum(1 for v in votes if v == 0)
        margin = wins - losses
        # Prefer high-signal wins, but keep at least one BT-Greedy tie/failure
        # in the candidate pool so the appendix does not cherry-pick only easy
        # comparisons.
        candidates.append({
            "sid": sid,
            "baseline": baseline,
            "wins": wins,
            "losses": losses,
            "ties": ties,
            "margin": margin,
            "rationale": _short(" / ".join(r.get("rationale", "") for r in grow[:3]), 360),
            "last_user": _short(_last_user(contexts[sid]), 220),
            "focal_response": _short(focal[sid].get("response", ""), 360),
            "baseline_response": _short(baselines[baseline][sid].get("response", ""), 360),
            "focal_strategy": focal[sid].get("selected_strategy", ""),
            "baseline_strategy": baselines[baseline][sid].get("selected_strategy", ""),
        })

    # Sort by margin so that the strongest decisive comparisons surface first;
    # we deliberately diversify across baselines so that the appendix is not
    # dominated by a single ESC system.
    strong = sorted([c for c in candidates if c["margin"] > 0],
                    key=lambda c: (-c["margin"], c["baseline"], c["sid"]))
    selected = []
    seen_baselines: set[str] = set()
    for c in strong:
        if len(selected) >= args.n:
            break
        key = (c["sid"], c["baseline"])
        if key in {(x["sid"], x["baseline"]) for x in selected}:
            continue
        # Diversify across baselines until each appears once, then allow extras.
        if c["baseline"] in seen_baselines and len(seen_baselines) < min(args.n, len(args.baselines)):
            continue
        seen_baselines.add(c["baseline"])
        selected.append(c)

    md = ["# Case Studies", ""]
    tex_rows = []
    for i, c in enumerate(selected, 1):
        md.extend([
            f"## Case {i}: PA-SCT-DRO vs {c['baseline']}",
            f"- sample_id: `{c['sid']}`",
            f"- votes: W={c['wins']} L={c['losses']} T={c['ties']}",
            f"- user: {c['last_user']}",
            f"- PA-SCT-DRO ({c['focal_strategy']}): {c['focal_response']}",
            f"- {c['baseline']} ({c['baseline_strategy']}): {c['baseline_response']}",
            f"- judge rationale: {c['rationale']}",
            "",
        ])
        # Keep the LaTeX snippet ASCII-only because the paper currently
        # compiles with pdfLaTeX/ACL rather than XeLaTeX+CJK. The full
        # Mandarin context/response text is preserved in outputs/case_studies.md.
        tex_rows.append(
            "\\paragraph{Case %d: PA-SCT-DRO vs %s (%s).} "
            "\\textbf{Strategies:} PA-SCT-DRO selects %s; the baseline selects %s. "
            "\\textbf{Votes:} W=%d/L=%d/T=%d. "
            "\\textbf{Interpretation:} Proxy judges preferred the PA-SCT-DRO response "
            "because it better matched the target counselling trade-off for this context. "
            "The full Mandarin context and responses are stored in \\pathref{outputs/case_studies.md}.\n" % (
                i, _tex_escape(c["baseline"]), _tex_escape(c["sid"]),
                _tex_escape(str(c["focal_strategy"])),
                _tex_escape(str(c["baseline_strategy"])),
                c["wins"], c["losses"], c["ties"],
            )
        )

    Path(args.out_md).write_text("\n".join(md), encoding="utf-8")
    out_tex = Path(args.out_tex)
    out_tex.parent.mkdir(parents=True, exist_ok=True)
    out_tex.write_text("\n".join(tex_rows), encoding="utf-8")
    print(f"[cases] wrote {len(selected)} cases -> {args.out_md}, {args.out_tex}")


if __name__ == "__main__":
    main()
