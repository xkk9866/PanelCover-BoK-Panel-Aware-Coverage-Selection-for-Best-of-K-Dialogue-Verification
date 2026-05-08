r"""End-to-end paper metric refresh (CPU-only).

Run from repository root::

    python -m eval.run_full_experiments

Steps:

  1. SCT-BoK / btgreedy / oracle planners produce their top-K plans.
  2. ``v12_best_of_n_fair`` materialises one final response per system
     under a single canonical BT-overall verifier shared across every
     baseline and every SCT-BoK ablation variant.
  3. ``bt_winrate_proxy`` rebuilds the panel-grounded BT model and
     writes a 43-system winrate sweep + BT-vs-panel validation.
  4. ``bsbok_significance`` reports SCT-BoK headline winrates with
     bootstrap CIs and one-sided sign-flip p-values, in both the
     overall BT and the state-conditioned BT spaces.
  5. ``run_rcbok_ablations --skip_run`` rescores cached SCT variants
     against the same baselines under both BT spaces.
  6. ``run_human_expert_eval_analysis`` computes raters' agreement and
     consensus on the paper-facing pairs (no LLM judge involved).
  7. ``compute_generation_metrics`` reports BLEU/ROUGE-L/Distinct/length
     on cached BoN responses.
  8. ``strategy_distribution_analysis`` reports strategy mix, entropy,
     and accuracy / macro-F1 vs the dataset gold strategy.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def _run(*args: str) -> None:
    print(f"\n[full-exp] {' '.join(args)}\n", flush=True)
    r = subprocess.run([sys.executable, "-m", *args], check=False)
    if r.returncode != 0:
        raise SystemExit(r.returncode)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    (root / "logs").mkdir(parents=True, exist_ok=True)
    (root / "outputs").mkdir(parents=True, exist_ok=True)
    (root / "results").mkdir(parents=True, exist_ok=True)

    # Step 0: Panel BT models (per-expert L2 BT, fed into PA-SCT and
    # the panel-aware winrate metric).
    _run("eval.panel_bt")

    # Step 1: planners produce their K-plans.
    _run("eval.robust_therapeutic_bok")          # SCT-BoK
    _run("eval.build_pasct_topk")                # PA-SCT (focal SOTA)
    _run("eval.build_btgreedy_topk")             # verifier-aligned baseline
    _run("eval.build_oracle_topk")               # gold-strategy upper bound
    # ESC planners (MISC, MultiESC, TransESC) and lexicon / majority
    # rule baselines are produced by build_fair_bon_baselines (cached
    # already; rerun only when the response cache changes).
    # _run("eval.build_fair_bon_baselines")

    # Step 2: shared canonical verifier materialises BoN responses
    # for every system under the same BT-overall lens.  This is what
    # the published-planner literature implicitly assumes.
    _run(
        "eval.v12_best_of_n_fair",
        "--systems", "lexicon", "majority", "misc", "multiesc",
        "transesc", "psystate_sctbok", "psystate_pasct",
        "btgreedy", "oracle",
        "--safety_overrides", "data/judge_eval_v10/v12_safety_overrides.jsonl",
        "--topk_dir", "data/fair_bon_v12",
        "--out_dir", "data/fair_bon_v12",
        "--out_responses_jsonl", "NUL",
        "--verifier", "bt-overall",
    )

    # Step 3: BT proxy validation + canonical BT-overall winrate sweep.
    _run(
        "eval.bt_winrate_proxy",
        "--baselines",
        "lexicon_bon_v12", "majority_bon_v12",
        "misc_bon_v12", "multiesc_bon_v12", "transesc_bon_v12",
        "btgreedy_bon_v12", "oracle_bon_v12",
    )

    # Step 4: PA-SCT headline significance (overall + state-conditioned).
    _run(
        "eval.bsbok_significance",
        "--focal", "psystate_pasct_bon_v12",
        "--out", "results/pasct_significance.json",
    )

    # Step 5: Panel-aware Best-of-K winrate -- the SOTA-grade metric
    # under which BT-Greedy is provably suboptimal.
    _run(
        "eval.panel_aware_winrate",
        "--focal", "psystate_pasct",
        "--baselines", "lexicon", "majority", "misc", "multiesc",
        "transesc", "psystate_sctbok", "btgreedy", "oracle",
        "--out_json", "results/panel_aware_winrate.json",
        "--out_csv", "results/panel_aware_winrate.csv",
    )

    # Step 6: Robustness of the panel-aware winrate under strict
    # leave-one-expert-out (refits PA-SCT and BT-Greedy on the train
    # experts and re-evaluates on the held-out expert's BT).  This
    # is the strongest defence against fitting/test overlap.
    _run(
        "eval.panel_robustness",
        "--strict_loo",
        "--out_json", "results/panel_robustness.json",
    )

    # Step 7: ablations and downstream metrics.
    _run("eval.run_pasct_ablations")
    _run("eval.run_rcbok_ablations", "--skip_run")

    # Step 8: stratified panel-aware analysis (where does it matter?).
    _run(
        "eval.panel_stratified_winrate",
        "--out_json", "results/panel_aware_stratified.json",
    )

    # Step 9: panel-size scaling -- empirical validation of the
    # submodular theoretical scaling law (gap = 0 at |E|=1, grows with
    # |E|, saturates at |E|>=K).
    _run(
        "eval.panel_size_scaling",
        "--out", "results/panel_size_scaling.json",
    )

    _run(
        "eval.run_human_expert_eval_analysis",
        "--out_json", "outputs/human_eval_agreement.json",
        "--out_md", "outputs/human_eval_agreement.md",
    )
    _run("eval.compute_generation_metrics")
    _run("eval.strategy_distribution_analysis")
    print("\n[full-exp] done.\n", flush=True)


if __name__ == "__main__":
    main()
