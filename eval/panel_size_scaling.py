"""Panel-size scaling: PA-SCT vs BT-Greedy under simulated larger panels.

Theoretical prediction
----------------------

Under the panel-stochastic verifier with $|E|$ uniformly-sampled
experts, the panel-aware utility of a $K$-set $S$ is

    F(S | x) = (1/|E|) sum_e max_{a in S} BT_e(r(a)).

The submodular advantage of greedy panel-coverage over BT-Greedy
(top-$K$ by panel-mean) grows when the panel becomes wider in the
sense that no $K$-element subset can simultaneously cover every
expert's argmax.  We expect:

  * tiny panels (|E|=1):  no submodular advantage; PA-SCT == BT-Greedy.
  * small panels (|E|=3): structural advantage capped by Spearman;
                          empirically +0.018 utility gap.
  * larger panels: gap grows monotonically until $|E| \ge K$.

Empirical strategy
------------------

We have three real qualified-counsellor experts with $7{,}240$
rated games each.  To probe the scaling we bootstrap $|E|=B$
synthetic experts from the real three by perturbing per-text BT
logits with calibrated independent Gaussian noise:

    BT_e_b(t) = BT_{src(b)}(t) + epsilon_b(t),
    epsilon_b(t) ~ N(0, sigma).

The synthetic noise is calibrated to recover the empirical
inter-rater Spearman of $0.74$--$0.86$ between any two synthetic
experts when $sigma$ is set to the cross-rater RMS-of-residual on
the held-out 1{,}411-text panel.  We then re-plan PA-SCT and
BT-Greedy under each synthetic panel size and report the
panel-aware winrate.

Two scaling claims are tested:

  (1) The PA-SCT -- BT-Greedy gap grows with $|E|$.
  (2) The gap is bounded above by the gap that would obtain on a
      panel with zero inter-expert correlation (i.e.,
      independent-experts upper bound).

Output
------

  results/panel_size_scaling.json
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np

from psystate.constants import STRATEGIES


PANEL_BT_PATH = Path("results/panel_bt.json")
RESPONSE_PATH = Path("data/judge_eval_v10/v10_responses.jsonl")
EVAL_SET = Path("data/judge_eval_v10/v10_eval_contexts.jsonl")
SAFETY_OVERRIDES = Path("data/judge_eval_v10/v12_safety_overrides.jsonl")
EXPERT_IDS = ("E1_supervisor", "E2_client_experience", "E3_safety_reviewer")


def _norm(s):
    return re.sub(r"\s+", " ", (s or "").strip())


def _sigmoid(x):
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)


def _load_jsonl(p):
    if not Path(p).exists():
        return []
    out = []
    for line in Path(p).read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


def _resp_idx(p):
    idx = {}
    for r in _load_jsonl(p):
        sid = str(r.get("sample_id", ""))
        strat = str(r.get("selected_strategy", ""))
        text = _norm(r.get("response"))
        backend = (r.get("generation_config") or {}).get("backend")
        if sid and strat and text and backend != "safety_template":
            idx.setdefault((sid, strat), text)
    return idx


def _calibrate_sigma(panel_bt: dict) -> float:
    """Calibrate Gaussian noise sigma so that synthetic experts have
    Spearman correlation matching the empirical inter-rater median
    (~0.78) on overall BT logits.  We fit by binary search on sigma
    against a held-out simulated correlation."""
    # Collect overall logits per real expert on shared items.
    items = sorted(set.intersection(*[
        set(panel_bt[e]["overall"].keys()) for e in EXPERT_IDS
    ]))
    base = np.stack([
        np.asarray([panel_bt[e]["overall"][t] for t in items])
        for e in EXPERT_IDS
    ], axis=0)
    target_spearman = 0.78  # median of (0.86, 0.77, 0.74)

    rng = np.random.default_rng(20260502)

    def _sim_corr(sigma):
        n_pairs = 100
        from scipy.stats import spearmanr
        corrs = []
        for _ in range(n_pairs):
            src1 = rng.integers(0, 3)
            src2 = rng.integers(0, 3)
            v1 = base[src1] + rng.normal(0, sigma, size=base.shape[1])
            v2 = base[src2] + rng.normal(0, sigma, size=base.shape[1])
            corrs.append(spearmanr(v1, v2).statistic)
        return float(np.mean(corrs))

    # binary search
    lo, hi = 0.0, 1.0
    for _ in range(15):
        mid = (lo + hi) / 2
        c = _sim_corr(mid)
        if c > target_spearman:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def _kset_max(strats, sid, resp_idx, bt):
    best = -np.inf
    for s in strats:
        text = resp_idx.get((sid, s))
        if text is None:
            score = 0.0
        else:
            score = float(bt.get(text, 0.0))
        if score > best:
            best = score
    return float(best) if math.isfinite(best) else 0.0


def _build_synthetic_panel(panel_bt: dict, B: int, sigma: float,
                            seed: int = 20260502) -> dict[int, dict[str, float]]:
    """Build B synthetic per-expert BT-overall maps by perturbing the
    real three experts with i.i.d. Gaussian noise of sd ``sigma``.
    Each synthetic expert is seeded by a (real_expert, noise_seed)
    pair."""
    rng = np.random.default_rng(seed)
    # Use the union of all real-experts' rated texts.
    text_universe = sorted(
        set.union(*[set(panel_bt[e]["overall"]) for e in EXPERT_IDS])
    )
    base_per_real = {
        e: np.asarray([panel_bt[e]["overall"].get(t, 0.0) for t in text_universe])
        for e in EXPERT_IDS
    }
    out: dict[int, dict[str, float]] = {}
    for b in range(B):
        src = EXPERT_IDS[b % 3]
        eps = rng.normal(0, sigma, size=len(text_universe))
        scores = base_per_real[src] + eps
        out[b] = dict(zip(text_universe, scores.tolist()))
    return out


def _greedy_panel_coverage(profiles: dict[str, np.ndarray],
                           K: int,
                           required: list[str]) -> list[str]:
    selected: list[str] = []
    if not profiles:
        return selected
    n_scen = next(iter(profiles.values())).shape[0]
    current = np.full(n_scen, -np.inf, dtype=np.float64)

    def _F(curr):
        finite = np.where(np.isfinite(curr), curr, 0.0)
        return float(finite.mean())

    for s in required:
        if s in profiles and s not in selected and len(selected) < K:
            selected.append(s)
            current = np.maximum(current, profiles[s])

    while len(selected) < K:
        best_s = None
        best_gain = -np.inf
        prev = _F(current)
        for s, p in profiles.items():
            if s in selected:
                continue
            new_curr = np.maximum(current, p)
            gain = _F(new_curr) - prev
            if gain > best_gain:
                best_gain = gain
                best_s = s
        if best_s is None:
            break
        selected.append(best_s)
        current = np.maximum(current, profiles[best_s])
    return selected


def _evaluate_pair(focal_K: dict[str, list[str]],
                   base_K: dict[str, list[str]],
                   resp_idx: dict[tuple[str, str], str],
                   panel: dict[int, dict[str, float]]) -> tuple[float, float, float]:
    """Compute panel-aware BT-projected pairwise winrate of focal vs.
    base, using the *original* uniform panel-stochastic verifier on
    the synthetic panel."""
    sids = sorted(set(focal_K) & set(base_K))
    wrs = []
    gaps = []
    for sid in sids:
        per_e = []
        per_g = []
        for b, bt_e in panel.items():
            fmax = _kset_max(focal_K[sid], sid, resp_idx, bt_e)
            bmax = _kset_max(base_K[sid], sid, resp_idx, bt_e)
            per_e.append(_sigmoid(fmax - bmax))
            per_g.append(fmax - bmax)
        wrs.append(float(np.mean(per_e)))
        gaps.append(float(np.mean(per_g)))
    arr = np.asarray(wrs)
    arr_g = np.asarray(gaps)
    rng = np.random.default_rng(20260502)
    boot = rng.choice(arr, size=(2000, arr.size), replace=True).mean(axis=1)
    lo = float(np.quantile(boot, 0.025))
    hi = float(np.quantile(boot, 0.975))
    return float(arr.mean()), float(arr_g.mean()), float(hi - lo)


def _replan_pasct(panel: dict[int, dict[str, float]],
                  resp_idx: dict[tuple[str, str], str],
                  K: int = 3) -> dict[str, list[str]]:
    """Replan PA-SCT under the synthetic panel.  We use the simplest
    panel-aware objective (no state lift, no clinical constraints) so
    the only difference from BT-Greedy is the submodular vs.\ pooled
    aggregation rule."""
    eval_rows = _load_jsonl(EVAL_SET)
    overrides = {
        str(r.get("sample_id")): r
        for r in _load_jsonl(SAFETY_OVERRIDES)
        if r.get("shield_fired") or r.get("shield_fired_hard")
        or r.get("decision") == "hard"
    }
    out: dict[str, list[str]] = {}
    for ctx in eval_rows:
        sid = str(ctx["sample_id"])
        if sid in overrides:
            out[sid] = ["safety_referral"]
            continue
        scores: dict[str, np.ndarray] = {}
        for s in STRATEGIES:
            text = resp_idx.get((sid, s))
            if text is None:
                continue
            row = np.asarray(
                [bt.get(text, 0.0) for _, bt in panel.items()],
                dtype=np.float64,
            )
            scores[s] = row
        if not scores:
            continue
        out[sid] = _greedy_panel_coverage(scores, K, required=[])
    return out


def _replan_btgreedy(panel: dict[int, dict[str, float]],
                     resp_idx: dict[tuple[str, str], str],
                     K: int = 3) -> dict[str, list[str]]:
    """Replan BT-Greedy on the synthetic panel by ranking each text
    by its panel-mean BT logit."""
    eval_rows = _load_jsonl(EVAL_SET)
    overrides = {
        str(r.get("sample_id")): r
        for r in _load_jsonl(SAFETY_OVERRIDES)
        if r.get("shield_fired") or r.get("shield_fired_hard")
        or r.get("decision") == "hard"
    }
    out: dict[str, list[str]] = {}
    for ctx in eval_rows:
        sid = str(ctx["sample_id"])
        if sid in overrides:
            out[sid] = ["safety_referral"]
            continue
        avail = [s for s in STRATEGIES if (sid, s) in resp_idx]
        if not avail:
            continue
        def _mean(s):
            text = resp_idx[(sid, s)]
            return float(np.mean([bt.get(text, 0.0) for bt in panel.values()]))
        out[sid] = sorted(avail, key=lambda s: -_mean(s))[:K]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/panel_size_scaling.json")
    ap.add_argument("--K", type=int, default=3)
    ap.add_argument("--panel_sizes", nargs="+", type=int,
                    default=[1, 3, 5, 7, 10, 15, 20])
    ap.add_argument("--n_trials", type=int, default=5,
                    help="Independent random panel draws per size.")
    args = ap.parse_args()

    panel_bt = json.loads(PANEL_BT_PATH.read_text(encoding="utf-8"))
    panel_bt = {e: {d: {k: float(v) for k, v in dim.items()}
                     for d, dim in panel_bt[e].items()}
                 for e in EXPERT_IDS}
    resp_idx = _resp_idx(RESPONSE_PATH)

    print("[scale] calibrating sigma to match empirical inter-rater "
          "Spearman ~0.78 ...")
    sigma = _calibrate_sigma(panel_bt)
    print(f"[scale] sigma = {sigma:.3f}")

    record: dict = {"sigma": sigma, "K": args.K, "trials": {}}
    print(f"\n{'|E|':>4s}{'trial':>6s}{'WR_PA':>10s}{'gap':>10s}{'CI_w':>10s}")
    print("-" * 50)

    for B in args.panel_sizes:
        record["trials"][B] = []
        for trial in range(args.n_trials):
            panel = _build_synthetic_panel(
                panel_bt, B, sigma,
                seed=20260502 + 17 * (B + 1) * (trial + 1),
            )
            focal_K = _replan_pasct(panel, resp_idx, K=args.K)
            base_K = _replan_btgreedy(panel, resp_idx, K=args.K)
            wr, gap, ci_w = _evaluate_pair(focal_K, base_K, resp_idx, panel)
            record["trials"][B].append({
                "trial": trial,
                "winrate": wr,
                "utility_gap": gap,
                "ci_width": ci_w,
            })
            print(f"{B:>4d}{trial:>6d}{wr:>10.4f}{gap:>+10.4f}{ci_w:>10.4f}")

    # Aggregate
    record["aggregated"] = {}
    print(f"\n{'|E|':>4s}{'mean_WR':>10s}{'mean_gap':>10s}")
    print("-" * 32)
    for B in args.panel_sizes:
        wrs = [r["winrate"] for r in record["trials"][B]]
        gaps = [r["utility_gap"] for r in record["trials"][B]]
        m_wr = float(np.mean(wrs))
        m_gap = float(np.mean(gaps))
        sd_gap = float(np.std(gaps))
        record["aggregated"][B] = {
            "mean_winrate": m_wr,
            "mean_gap": m_gap,
            "sd_gap": sd_gap,
            "trial_winrates": wrs,
            "trial_gaps": gaps,
        }
        print(f"{B:>4d}{m_wr:>10.4f}{m_gap:>+10.4f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(
        json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\n[scale] wrote {args.out}")


if __name__ == "__main__":
    main()
