"""Per-expert BT-projected winrate on materialised BoN responses.

Given two BoN response files (one focal, one baseline), each containing
one final response per ``sample_id``, this script computes the
BT-projected pairwise winrate of focal over baseline under

* the canonical pooled BT-overall verifier (``bt-overall``);
* each per-expert BT-overall verifier from ``results/panel_bt.json``;
* and the unweighted panel mean of the per-expert winrates
  (``panel-mean``).

This is the deployment-time analogue of ``eval.panel_aware_winrate``:
``panel_aware_winrate`` operates on the K-set, this operates on the
single materialised response.  The panel-state / panel-mean verifiers
in ``eval.v12_best_of_n_fair`` are the only ones that produce K-set-
sensitive responses, so the gap between the bt-overall and panel-mean
columns under those verifiers quantifies the *deployment* benefit of
panel-aware planning + panel-aware verification.

Usage
-----

    python -m eval.panel_response_winrate \
        --focal data/fair_bon_v12_panel_state/pasct_dro_anchor_bon_responses.jsonl \
        --baselines data/fair_bon_v12_panel_state/btgreedy_bon_responses.jsonl \
                    data/fair_bon_v12_panel_state/misc_bon_responses.jsonl \
        --out results/panel_response_winrate_panel_state.json
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np


def _norm(s: str | None) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _load_jsonl(path: Path) -> list[dict]:
    out: list[dict] = []
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def _resp_index(path: Path) -> dict[str, str]:
    return {str(r["sample_id"]): _norm(r.get("response"))
            for r in _load_jsonl(path)}


def _bt_score(t: str, bt: dict[str, float]) -> float:
    return float(bt.get(t, 0.0))


def _winrate(focal_idx: dict[str, str], base_idx: dict[str, str],
             bt: dict[str, float]) -> tuple[float, int]:
    """Soft (sigmoid of BT difference) BT-projected winrate of focal."""
    sids = sorted(set(focal_idx) & set(base_idx))
    if not sids:
        return float("nan"), 0
    vals: list[float] = []
    for sid in sids:
        ta = focal_idx[sid]
        tb = base_idx[sid]
        if not ta or not tb:
            vals.append(0.5); continue
        if ta == tb:
            vals.append(0.5); continue
        vals.append(_sigmoid(_bt_score(ta, bt) - _bt_score(tb, bt)))
    return float(np.mean(vals)), len(vals)


def _hardrate(focal_idx: dict[str, str], base_idx: dict[str, str],
              bt: dict[str, float]) -> tuple[float, dict[str, int]]:
    """Hard (argmax) BT-projected winrate of focal."""
    sids = sorted(set(focal_idx) & set(base_idx))
    w = l = t = 0
    for sid in sids:
        ta = focal_idx[sid]
        tb = base_idx[sid]
        if not ta or not tb or ta == tb:
            t += 1
            continue
        sa = _bt_score(ta, bt)
        sb = _bt_score(tb, bt)
        if sa > sb + 1e-9:
            w += 1
        elif sb > sa + 1e-9:
            l += 1
        else:
            t += 1
    n = w + l + t
    wr = (w + 0.5 * t) / n if n else float("nan")
    return wr, {"wins": w, "losses": l, "ties": t, "n": n}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--focal", required=True,
                    help="Path to the focal system's BoN response JSONL.")
    ap.add_argument("--baselines", nargs="+", required=True,
                    help="Paths to baseline BoN response JSONLs.")
    ap.add_argument("--panel_bt", default="results/panel_bt.json")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    panel_bt_raw = json.loads(Path(args.panel_bt).read_text(encoding="utf-8"))
    experts = list(panel_bt_raw)
    bt_per_expert = {e: {k: float(v)
                          for k, v in panel_bt_raw[e].get("overall", {}).items()}
                     for e in experts}

    # Pool BT-overall as the "canonical" verifier-aligned metric.  We use
    # ``eval.bt_winrate_proxy`` to get the same canonical pooled BT model
    # that the bt-overall verifier and downstream papers use.
    from eval.bt_winrate_proxy import (
        _build_response_text_index, _collect_dim_games, _fit_bt)
    text_idx = _build_response_text_index()
    games = _collect_dim_games(text_idx)
    bt_pooled = _fit_bt(games.get("overall", []))

    focal_idx = _resp_index(Path(args.focal))
    rows: dict[str, dict] = {}
    for base_path in args.baselines:
        base_idx = _resp_index(Path(base_path))
        per_expert_soft: dict[str, float] = {}
        per_expert_hard: dict[str, dict] = {}
        for e in experts:
            wr, _ = _winrate(focal_idx, base_idx, bt_per_expert[e])
            per_expert_soft[e] = wr
            wr_h, hh = _hardrate(focal_idx, base_idx, bt_per_expert[e])
            per_expert_hard[e] = {"win_rate": wr_h, **hh}
        wr_pooled, n_pool = _winrate(focal_idx, base_idx, bt_pooled)
        wr_pooled_h, h_pool = _hardrate(focal_idx, base_idx, bt_pooled)
        panel_mean = float(np.mean(list(per_expert_soft.values())))
        rows[Path(base_path).stem] = {
            "n": n_pool,
            "bt_overall_pool_soft": wr_pooled,
            "bt_overall_pool_hard": {"win_rate": wr_pooled_h, **h_pool},
            "per_expert_soft": per_expert_soft,
            "per_expert_hard": per_expert_hard,
            "panel_mean_soft": panel_mean,
        }
    out = {
        "focal": Path(args.focal).stem,
        "experts": experts,
        "rows": rows,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False),
                         encoding="utf-8")
    print(f"[panel-resp-wr] focal={out['focal']}")
    print(f"  baselines: {list(rows)}")
    for b, r in rows.items():
        pe = r["per_expert_soft"]
        print(f"  vs {b:<35s} | n={r['n']} pooled={r['bt_overall_pool_soft']:.4f} "
              f"panel_mean={r['panel_mean_soft']:.4f} | "
              + " ".join(f"{e}={pe[e]:.4f}" for e in experts))
    print(f"[panel-resp-wr] wrote -> {out_path}")


if __name__ == "__main__":
    main()
