"""Internal ablations for the Panel-Aware SCT planner.

Each variant is built by ``eval.build_pasct_topk`` with one component
disabled, then re-scored via ``eval.panel_aware_winrate`` against
``btgreedy`` (the verifier-aligned baseline) under the panel-aware
verifier.  We also record the overall and state-conditioned BT-projected
winrate via ``eval.v12_best_of_n_fair`` + ``eval.bsbok_significance``.

Variants
~~~~~~~~

* ``pasct``                 -- full PA-SCT (state lift + clinical
                                constraints + panel-aware submodular).
* ``pasct_no_state``        -- panel-aware submodular, but each expert
                                scores texts purely by the BT-overall
                                logit (state lift dropped).
* ``pasct_no_constraints``  -- panel-aware submodular without the
                                hard clinical-coverage constraints.
* ``pasct_pure_panel``      -- both lift and constraints dropped;
                                isolates the marginal value of
                                state-conditioned and constraint
                                lifts on top of the panel-aware
                                submodular core.
* ``pasct_collapse_mean``   -- replace the panel-aware submodular
                                objective with top-K-by-panel-mean
                                (BT-Greedy under per-expert mean BT).
                                Tests whether the submodular term
                                ever beats plain mean ranking.
* ``pasct_multidim``        -- expand the scenario set from 3 experts
                                to 21 (expert, dim) pairs for an
                                explicit multi-axis coverage objective.
* ``pasct_K{1,2,4,5}``      -- K-budget sweep.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


VARIANTS = [
    ("pasct",                 "psystate_pasct",       []),
    ("pasct_no_state",        "pasct_no_state",       ["--no_state_lift"]),
    ("pasct_no_constraints",  "pasct_no_constraints", ["--no_clinical_constraints"]),
    ("pasct_pure_panel",      "pasct_pure_panel",     ["--no_state_lift", "--no_clinical_constraints"]),
    ("pasct_collapse_mean",   "pasct_collapse_mean",  ["--collapse_to_mean"]),
    ("pasct_multidim",        "pasct_multidim",       ["--multidim_scenarios"]),
    ("pasct_K1",              "pasct_K1",             ["--K", "1"]),
    ("pasct_K2",              "pasct_K2",             ["--K", "2"]),
    ("pasct_K4",              "pasct_K4",             ["--K", "4"]),
    ("pasct_K5",              "pasct_K5",             ["--K", "5"]),
]


def _run(*args: str) -> None:
    print(f"\n[pasct-abl] {' '.join(args)}\n", flush=True)
    r = subprocess.run([sys.executable, "-m", *args], check=False)
    if r.returncode != 0:
        raise SystemExit(r.returncode)


def main() -> None:
    Path("results").mkdir(parents=True, exist_ok=True)
    summary: dict = {"variants": {}}

    for tag, system_id, extra in VARIANTS:
        topk_path = f"data/fair_bon_v12/{system_id}_topk.jsonl"
        # Re-plan
        _run(
            "eval.build_pasct_topk",
            "--out_topk", topk_path,
            "--system_id", system_id,
            *extra,
        )

    # Score every variant against BT-Greedy and the published planners.
    for tag, system_id, _extra in VARIANTS:
        out_json = f"results/panel_aware_winrate_{tag}.json"
        out_csv = f"results/panel_aware_winrate_{tag}.csv"
        _run(
            "eval.panel_aware_winrate",
            "--focal", system_id,
            "--baselines",
            "btgreedy", "psystate_sctbok", "psystate_pasct",
            "lexicon", "majority", "misc", "multiesc", "transesc",
            "oracle", "pasct_collapse_mean", "pasct_multidim",
            "--out_json", out_json,
            "--out_csv", out_csv,
        )
        rec = json.loads(Path(out_json).read_text(encoding="utf-8"))
        summary["variants"][tag] = rec.get("baselines", {})

    Path("results/pasct_ablation_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n[pasct-abl] wrote results/pasct_ablation_summary.json")

    # Pretty print: variant vs btgreedy
    print(f"\n{'Variant':<24s} {'WR vs BT-Greedy':>20s} {'95% CI':>20s} "
          f"{'p':>8s} {'gap':>10s}")
    for tag, system_id, _ in VARIANTS:
        rec = summary["variants"].get(tag, {}).get("btgreedy")
        if rec is None:
            continue
        wr = rec["winrate"]
        ci = f"[{rec['winrate_ci_lo']:.3f}, {rec['winrate_ci_hi']:.3f}]"
        p = rec["winrate_p_value"]
        gap = rec["utility_gap"]
        print(f"{tag:<24s} {wr:>20.3f} {ci:>20s} {p:>8.3f} {gap:>+10.4f}")


if __name__ == "__main__":
    main()
