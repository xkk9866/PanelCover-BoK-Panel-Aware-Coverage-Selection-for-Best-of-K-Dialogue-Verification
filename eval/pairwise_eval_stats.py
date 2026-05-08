"""Statistical summary for human/proxy pairwise evaluation JSONL.

The input format matches ``eval.qwen_pairwise_eval`` raw rows and the
human-expert pair rows used elsewhere in the repository: each row has
``sample_id``, ``pair_id``, ``role_id`` or ``expert_id``, ``focal_side``
and a per-dimension ``verdict`` mapping with values A/B/tie.

Outputs include raw win rate, context-cluster bootstrap CI, sign-flip
p-value, a tie-aware Cohen-h style effect size, and per-rater breakdown.
For true mixed-effects logistic regression the paper should use R's
``lme4`` or a dedicated GLMM package; this script provides the robust
repository-native statistics needed for tables without pretending that
proxy judgments are human data.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


DIMS = (
    "overall", "helpfulness", "empathy", "specificity",
    "appropriateness", "safety", "actionability",
    "avoids_over_advice", "emotional_validation",
)


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _score(row: dict, dim: str) -> float:
    v = (row.get("verdict") or {}).get(dim, "tie")
    if v == "tie":
        return 0.5
    return 1.0 if v == row.get("focal_side", "A") else 0.0


def _cohen_h(p: float) -> float:
    p = min(max(p, 1e-9), 1.0 - 1e-9)
    return 2.0 * math.asin(math.sqrt(p)) - 2.0 * math.asin(math.sqrt(0.5))


def _cluster_bootstrap(
    rows: list[dict], dim: str, *, n_boot: int = 5000, seed: int = 20260507
) -> tuple[float, float, float]:
    by_ctx: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_ctx[str(row.get("sample_id"))].append(row)
    ctx_ids = sorted(by_ctx)
    per_ctx = np.asarray([
        np.mean([_score(r, dim) for r in by_ctx[c]]) for c in ctx_ids
    ], dtype=np.float64)
    if per_ctx.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boot = rng.choice(per_ctx, size=(n_boot, per_ctx.size), replace=True).mean(axis=1)
    lo, hi = float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))
    diffs = per_ctx - 0.5
    obs = float(diffs.mean())
    ge = 0
    for _ in range(n_boot):
        signs = rng.choice([1.0, -1.0], size=diffs.size, replace=True)
        if (diffs * signs).mean() >= obs:
            ge += 1
    return lo, hi, (ge + 1) / (n_boot + 1)


def _summarise_pair(rows: list[dict]) -> dict:
    out = {"n": len(rows), "dims": {}, "by_rater": {}}
    for dim in DIMS:
        vals = [_score(r, dim) for r in rows]
        if not vals:
            continue
        wins = sum(v == 1.0 for v in vals)
        losses = sum(v == 0.0 for v in vals)
        ties = sum(v == 0.5 for v in vals)
        wr = float(np.mean(vals))
        lo, hi, p = _cluster_bootstrap(rows, dim)
        out["dims"][dim] = {
            "win_rate": wr,
            "wins": int(wins),
            "losses": int(losses),
            "ties": int(ties),
            "n": len(vals),
            "cluster_ci_lo": lo,
            "cluster_ci_hi": hi,
            "signflip_p": p,
            "cohen_h_vs_0_5": _cohen_h(wr),
        }
    by_rater: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        rid = str(row.get("role_id") or row.get("expert_id") or "unknown")
        by_rater[rid].append(row)
    for rid, rrows in by_rater.items():
        vals = [_score(r, "overall") for r in rrows]
        out["by_rater"][rid] = {
            "overall_win_rate": float(np.mean(vals)) if vals else float("nan"),
            "n": len(vals),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--out_md", default=None)
    args = ap.parse_args()

    rows = _load_jsonl(Path(args.input))
    by_pair: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_pair[str(row.get("pair_id"))].append(row)
    payload = {
        "input": args.input,
        "pairs": {pid: _summarise_pair(prows) for pid, prows in by_pair.items()},
    }
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    if args.out_md:
        lines = ["# Pairwise evaluation statistics\n"]
        lines.append("| Pair | n | Overall WR | 95% CI | p | Cohen h |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for pid, rec in payload["pairs"].items():
            d = rec["dims"]["overall"]
            lines.append(
                f"| {pid} | {d['n']} | {d['win_rate']:.4f} | "
                f"[{d['cluster_ci_lo']:.4f}, {d['cluster_ci_hi']:.4f}] | "
                f"{d['signflip_p']:.4f} | {d['cohen_h_vs_0_5']:.4f} |"
            )
        Path(args.out_md).write_text("\n".join(lines) + "\n", encoding="utf-8")
    for pid, rec in payload["pairs"].items():
        d = rec["dims"]["overall"]
        print(f"[pair-stats] {pid}: WR={d['win_rate']:.4f} "
              f"CI=[{d['cluster_ci_lo']:.4f}, {d['cluster_ci_hi']:.4f}] "
              f"p={d['signflip_p']:.4f} h={d['cohen_h_vs_0_5']:.4f}")


if __name__ == "__main__":
    main()
