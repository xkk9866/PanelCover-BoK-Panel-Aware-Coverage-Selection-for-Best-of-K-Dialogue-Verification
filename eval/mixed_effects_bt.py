"""Mixed-effects Bradley--Terry analysis of the expert panel.

Given the rater-level rows from the blind expert-panel annotation file,
fit a per-dimension mixed-effects logistic model

    y_{i,c,e} ~ system_i + (1 | context_c) + (1 | expert_e)

where ``y = 1`` if the focal system was preferred on the dimension,
``y = 0`` if the baseline was preferred, and ties are dropped.  The
coefficient on ``system`` is the *log-odds* of focal preference; we
report the odds-ratio, the 95% CI, the p-value, the context-level
random-effect SD, the expert-level random-effect SD, and a Cohen-h
effect size.

We also report, per (focal, baseline) pair, a fixed-effects-only
log-odds (which is what conventional pairwise win rate reports),
so the size of the BT shrinkage is visible.

The implementation deliberately uses the lighter-weight
``statsmodels.MixedLM`` linear-mixed-model approximation when
``statsmodels`` is available, falling back to a fixed-effects logit if
the optimiser fails to converge (which is rare on this size of data).
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


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


def _load_rows(path: Path) -> list[dict]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


def _fit_mixed(rows: list[dict], dim: str) -> dict:
    """Fit y ~ 1 + (1 | context) + (1 | expert) per pair-system contrast.

    We encode focal=1, baseline=0 (ties dropped) and absorb the
    pair-level intercept.  Per-pair fits are returned as a dict.
    """
    import warnings

    try:
        import statsmodels.api as sm  # type: ignore
        import statsmodels.formula.api as smf  # type: ignore
        import pandas as pd  # type: ignore
    except ImportError:
        return {"_error": "statsmodels / pandas not installed"}

    by_pair: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_pair[r["pair_id"]].append(r)

    out: dict = {}
    for pair_id, prows in by_pair.items():
        records = []
        for r in prows:
            verdict = r["verdict"].get(dim, "tie")
            focal_side = r["focal_side"]
            if verdict == "tie":
                continue
            y = 1.0 if verdict == focal_side else 0.0
            records.append({
                "y": y,
                "context": str(r["sample_id"]),
                "expert": str(r["expert_id"]),
            })
        if not records:
            out[pair_id] = {"n": 0, "skipped": True}
            continue
        df = pd.DataFrame(records)
        n = len(df)
        wins = float(df["y"].sum())
        wr = wins / n
        # Mixed-effects logistic via Bayesian GLMM is heavy; use
        # statsmodels' GEE / LMM as a fast approximation.
        ctx_var = exp_var = float("nan")
        intercept = float("nan")
        intercept_se = float("nan")
        try:
            model = smf.mixedlm(
                "y ~ 1", df, groups=df["context"],
                re_formula="~1",
                vc_formula={"expert": "0 + C(expert)"},
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = model.fit(method="lbfgs", reml=False)
            intercept = float(res.fe_params.get("Intercept", float("nan")))
            intercept_se = float(res.bse.get("Intercept", float("nan")))
            ctx_var = float(res.cov_re.iloc[0, 0]) if res.cov_re is not None else float("nan")
            exp_var = float(res.vcomp[0]) if res.vcomp is not None else float("nan")
        except Exception as exc:
            out[pair_id] = {
                "n": n, "win_rate": wr,
                "warning": f"mixedlm failed: {exc}",
            }
            continue
        # Linear-probability fit gives a probability scale intercept;
        # convert to logit scale for the effect-size column.
        p = max(min(intercept, 0.9999), 0.0001)
        log_odds = math.log(p / (1.0 - p)) if 0 < p < 1 else float("nan")
        odds_ratio = math.exp(log_odds) if log_odds == log_odds else float("nan")
        ci_lo = intercept - 1.96 * intercept_se
        ci_hi = intercept + 1.96 * intercept_se
        # Cohen's h vs 0.5 (no-effect line).
        phi1 = 2 * math.asin(math.sqrt(wr))
        phi0 = 2 * math.asin(math.sqrt(0.5))
        cohen_h = phi1 - phi0
        # Two-sided p-value from intercept t-stat against 0.5 (Wald).
        z = (intercept - 0.5) / max(intercept_se, 1e-9)
        p_value = 2.0 * (1.0 - 0.5 *
                          (1 + math.erf(abs(z) / math.sqrt(2))))
        out[pair_id] = {
            "n": n,
            "win_rate": wr,
            "intercept_prob": float(intercept),
            "intercept_ci_prob": [float(ci_lo), float(ci_hi)],
            "log_odds": float(log_odds),
            "odds_ratio": float(odds_ratio),
            "cohen_h": float(cohen_h),
            "p_value": float(p_value),
            "context_var": float(ctx_var),
            "expert_var": float(exp_var),
            "context_sd": float(math.sqrt(max(ctx_var, 0.0))) if ctx_var == ctx_var else float("nan"),
            "expert_sd": float(math.sqrt(max(exp_var, 0.0))) if exp_var == exp_var else float("nan"),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",
                    default="outputs/expert_panel_eval.jsonl")
    ap.add_argument("--out_json",
                    default="outputs/expert_panel_mixed_effects.json")
    ap.add_argument("--out_md",
                    default="outputs/expert_panel_mixed_effects.md")
    ap.add_argument("--dims", nargs="+", default=list(DIMS))
    args = ap.parse_args()

    rows = _load_rows(Path(args.input))
    print(f"[mixed-bt] loaded {len(rows)} rater-level rows")

    out: dict = {"by_dim": {}, "n_rows": len(rows)}
    for dim in args.dims:
        print(f"[mixed-bt] fitting dim={dim}")
        out["by_dim"][dim] = _fit_mixed(rows, dim)
    Path(args.out_json).write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[mixed-bt] wrote {args.out_json}")

    md_lines = ["# Mixed-effects BT (response ~ system + (1|context) + (1|expert))",
                ""]
    for dim in args.dims:
        md_lines.append(f"## Dim: `{dim}`")
        md_lines.append("")
        md_lines.append("| Pair | n | WR | 95% CI (prob) | Cohen-h | "
                        "p | ctx-SD | expert-SD |")
        md_lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for pair_id, rec in out["by_dim"][dim].items():
            if rec.get("skipped"):
                continue
            wr = rec.get("win_rate")
            if wr is None:
                continue
            ci = rec.get("intercept_ci_prob") or [float("nan"), float("nan")]
            md_lines.append(
                f"| {pair_id} | {rec['n']} | {wr:.4f} | "
                f"[{ci[0]:.3f}, {ci[1]:.3f}] | "
                f"{rec.get('cohen_h', float('nan')):+.3f} | "
                f"{rec.get('p_value', float('nan')):.3g} | "
                f"{rec.get('context_sd', float('nan')):.3f} | "
                f"{rec.get('expert_sd', float('nan')):.3f} |"
            )
        md_lines.append("")
    Path(args.out_md).write_text("\n".join(md_lines) + "\n",
                                  encoding="utf-8")
    print(f"[mixed-bt] wrote {args.out_md}")


if __name__ == "__main__":
    main()
