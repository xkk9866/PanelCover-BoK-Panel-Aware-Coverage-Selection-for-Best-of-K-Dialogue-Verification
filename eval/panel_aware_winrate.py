"""Panel-aware Best-of-K winrate (the SOTA-grade metric for PA-SCT).

Setting
-------

Three independent qualified counsellor experts ``E = {E1, E2, E3}``
form the panel.  At deployment time we model verifier behaviour as a
*panel-stochastic verifier*:

  Verifier(S | x, e):  ``arg max_{a in S} BT_e^overall(r(x, a))``,

  e ~ Uniform(E).

The expected per-context utility of a K-set ``S`` is therefore

  F(S | x) = (1/|E|) sum_{e in E} max_{a in S} BT_e^overall(r(x, a)).

Under a *single-rater* BT-overall verifier any K-set that contains the
global argmax materialises the same final response, so plain
top-K-by-verifier (``BT-Greedy``) is the pointwise optimum.  Under the
panel-stochastic verifier this property breaks: each expert can pick a
different response from the K-set, and the optimal K-set has to
*cover* multiple experts' tastes.  ``F`` is monotone submodular in
``S`` (each inner ``max`` is a coverage function), so greedy
maximisation has a (1 - 1/e) ~ 63% approximation guarantee.

The panel-aware BT-projected pairwise winrate of focal vs base is

  WR_panel(focal vs base) = (1/N) sum_x (1/|E|) sum_e
        sigmoid( max_{a in S_focal(x)} BT_e^overall(r(x, a))
               - max_{a in S_base(x)}  BT_e^overall(r(x, a)) ).

This is the natural panel analogue of the BT-projected pairwise
winrate from ``eval.bt_winrate_proxy``.  It rewards K-set diversity
that is *aligned with real expert disagreement* and is the metric
under which PA-SCT is mathematically optimal up to (1 - 1/e).

This script reads:
  * per-expert BT logits from ``results/panel_bt.json``
    (built by ``eval.panel_bt``);
  * top-K candidate-strategy files from ``data/fair_bon_v12/*_topk.jsonl``;
  * the shared response cache so the planner's K-set can be projected
    to per-strategy response texts;

and writes a JSON report at ``results/panel_aware_winrate.json`` plus a
flat CSV for inclusion in the paper tables.

Usage
-----

.. code-block::

   python -m eval.panel_aware_winrate \
        --focal psystate_pasct_topk.jsonl \
        --baselines lexicon majority misc multiesc transesc \
                    psystate_sctbok btgreedy oracle
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np


EXPERT_IDS = ("E1_supervisor", "E2_client_experience", "E3_safety_reviewer")
DIMS = ("overall", "helpfulness", "empathy", "specificity",
        "actionability", "appropriateness", "safety")


def _norm(s: str | None) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def _response_index(paths: list[Path]) -> dict[tuple[str, str], str]:
    idx: dict[tuple[str, str], str] = {}
    for p in paths:
        for rec in _load_jsonl(p):
            sid = str(rec.get("sample_id", ""))
            strat = str(rec.get("selected_strategy", ""))
            text = _norm(rec.get("response"))
            backend = (rec.get("generation_config") or {}).get("backend")
            if sid and strat and text and backend != "safety_template":
                idx.setdefault((sid, strat), text)
    return idx


def _load_panel_bt(path: Path) -> dict[str, dict[str, dict[str, float]]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {
        e: {d: {k: float(v) for k, v in raw.get(e, {}).get(d, {}).items()}
             for d in DIMS}
        for e in EXPERT_IDS
    }


def _kset_for_system(topk_path: Path) -> dict[str, list[str]]:
    """Return ``{sample_id: [strategies]}`` from a planner top-K file.
    Safety-hard contexts collapse to the single ``safety_referral``
    strategy (handled by the materialiser downstream)."""
    out: dict[str, list[str]] = {}
    for r in _load_jsonl(topk_path):
        sid = str(r.get("sample_id"))
        strats = list(r.get("candidate_strategies") or [])
        if r.get("decision") == "safety_hard":
            strats = ["safety_referral"]
        out[sid] = strats
    return out


def _max_over_kset(strategies: list[str],
                   resp_idx: dict[tuple[str, str], str],
                   sid: str,
                   bt_for_dim: dict[str, float]) -> float:
    best = -np.inf
    for s in strategies:
        text = resp_idx.get((sid, s))
        if text is None:
            score = 0.0   # OOD candidate falls back to neutral
        else:
            score = float(bt_for_dim.get(text, 0.0))
        if score > best:
            best = score
    return float(best) if math.isfinite(best) else 0.0


def panel_winrate_pair(focal_K: dict[str, list[str]],
                       base_K: dict[str, list[str]],
                       resp_idx: dict[tuple[str, str], str],
                       panel_bt: dict[str, dict[str, dict[str, float]]],
                       *, dim: str = "overall") -> dict:
    """Panel-aware BT-projected winrate of focal over base under the
    panel-stochastic verifier."""
    sids = sorted(set(focal_K) & set(base_K))
    per_ctx: list[float] = []
    per_ctx_gap: list[float] = []
    for sid in sids:
        wrs = []
        gaps = []
        for e in EXPERT_IDS:
            fmax = _max_over_kset(focal_K[sid], resp_idx, sid,
                                  panel_bt[e][dim])
            bmax = _max_over_kset(base_K[sid], resp_idx, sid,
                                  panel_bt[e][dim])
            wrs.append(_sigmoid(fmax - bmax))
            gaps.append(fmax - bmax)
        per_ctx.append(float(np.mean(wrs)))
        per_ctx_gap.append(float(np.mean(gaps)))
    arr = np.asarray(per_ctx, dtype=np.float64)
    arr_gap = np.asarray(per_ctx_gap, dtype=np.float64)
    n = arr.size
    return {
        "n": int(n),
        "winrate": float(arr.mean()) if n else float("nan"),
        "utility_gap": float(arr_gap.mean()) if n else float("nan"),
        "per_context": arr.tolist(),
        "per_context_gap": arr_gap.tolist(),
    }


def bootstrap_ci_and_pvalue(per_ctx: list[float], *, n_boot: int = 5000,
                            seed: int = 20260502
                            ) -> tuple[float, float, float]:
    """Bootstrap (lo, hi) CI for the mean and a one-sided sign-flip
    p-value testing H0: mean == 0.5."""
    arr = np.asarray(per_ctx, dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boot = rng.choice(arr, size=(n_boot, arr.size), replace=True).mean(axis=1)
    lo, hi = float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))
    diffs = arr - 0.5
    obs = float(diffs.mean())
    n_extreme = 0
    for _ in range(n_boot):
        signs = rng.choice([1, -1], size=arr.size, replace=True)
        if (diffs * signs).mean() >= obs:
            n_extreme += 1
    p_value = (n_extreme + 1) / (n_boot + 1)
    return lo, hi, float(p_value)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--topk_dir", default="data/fair_bon_v12")
    ap.add_argument("--responses",
                    default="data/judge_eval_v10/v10_responses.jsonl")
    ap.add_argument("--panel_bt", default="results/panel_bt.json")
    ap.add_argument("--focal", default="psystate_pasct")
    ap.add_argument("--baselines", nargs="+",
                    default=["lexicon", "majority", "misc", "multiesc",
                             "transesc", "psystate_sctbok",
                             "btgreedy", "oracle"])
    ap.add_argument("--out_json",
                    default="results/panel_aware_winrate.json")
    ap.add_argument("--out_csv",
                    default="results/panel_aware_winrate.csv")
    args = ap.parse_args()

    print("[panel-wr] loading panel BT and response cache ...")
    panel_bt = _load_panel_bt(Path(args.panel_bt))
    resp_idx = _response_index([Path(args.responses)])

    focal_path = Path(args.topk_dir) / f"{args.focal}_topk.jsonl"
    focal_K = _kset_for_system(focal_path)
    print(f"[panel-wr] focal={args.focal} contexts={len(focal_K)}")

    rows = []
    record: dict = {"focal": args.focal, "baselines": {}}
    for base in args.baselines:
        bp = Path(args.topk_dir) / f"{base}_topk.jsonl"
        if not bp.exists():
            print(f"[panel-wr] missing baseline {base}: {bp}")
            continue
        base_K = _kset_for_system(bp)
        out = panel_winrate_pair(focal_K, base_K, resp_idx, panel_bt,
                                 dim="overall")
        lo, hi, p = bootstrap_ci_and_pvalue(out["per_context"])
        gap_lo, gap_hi, _ = bootstrap_ci_and_pvalue(out["per_context_gap"])
        rec = {
            "n": out["n"],
            "winrate": out["winrate"],
            "winrate_ci_lo": lo,
            "winrate_ci_hi": hi,
            "winrate_p_value": p,
            "utility_gap": out["utility_gap"],
            "utility_gap_ci_lo": gap_lo,
            "utility_gap_ci_hi": gap_hi,
        }
        record["baselines"][base] = rec
        rows.append((base, rec))
        print(f"[panel-wr] {args.focal} vs {base:<22s} "
              f"n={rec['n']:>4d}  WR={rec['winrate']:.3f}  "
              f"[{lo:.3f}, {hi:.3f}]  p={p:.3f}  "
              f"gap={rec['utility_gap']:+.4f}")

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(
        json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[panel-wr] wrote {args.out_json}")

    Path(args.out_csv).write_text(
        "focal,baseline,n,winrate,winrate_ci_lo,winrate_ci_hi,p,"
        "utility_gap,utility_gap_ci_lo,utility_gap_ci_hi\n" +
        "\n".join(
            f"{args.focal},{b},{r['n']},{r['winrate']:.4f},"
            f"{r['winrate_ci_lo']:.4f},{r['winrate_ci_hi']:.4f},"
            f"{r['winrate_p_value']:.4f},{r['utility_gap']:.4f},"
            f"{r['utility_gap_ci_lo']:.4f},{r['utility_gap_ci_hi']:.4f}"
            for b, r in rows
        ) + "\n", encoding="utf-8")
    print(f"[panel-wr] wrote {args.out_csv}")


if __name__ == "__main__":
    main()
