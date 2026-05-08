# PsyState

PsyState is a Mandarin mental-health dialogue planning project that
proposes a **panel-aware Best-of-K** evaluation paradigm for
emotional-support dialogue and a matching planner with a
$(1{-}1/e)$ approximation guarantee.

## Headline Findings

1. **Counsellor disagreement breaks the single-rater verifier.**
   Three independent qualified counselling-practitioner experts on
   our panel show pairwise Spearman correlations of $0.74$--$0.86$
   on response-level BT-overall logits and pick a different top-1
   response on $43.4\%$ of randomly drawn $K{=}7$ candidate sets.
2. **Under a panel-stochastic verifier**, the optimal Best-of-$K$
   objective $F(S\mid x)=\tfrac{1}{|E|}\sum_e \max_{a\in S}
   \mathrm{BT}_e(r(x,a))$ is monotone submodular and the
   single-rater BT-Greedy baseline is **provably suboptimal**.
3. **PA-SCT** (Panel-Aware State-Conditioned Coverage planner)
   greedy-maximises $F$ with a $(1-1/e)$ guarantee plus clinical
   coverage constraints.  On 500 Mandarin contexts, PA-SCT beats
   reproduced MISC / MultiESC / TransESC by $0.13$--$0.26$
   panel-utility (winrates $0.554$--$0.562$, $p<10^{-3}$), beats
   the verifier-aligned trivial baseline BT-Greedy in-sample by
   $+0.018$ ($p<10^{-3}$), and beats a gold-strategy oracle
   ($K{=}1$) by $+0.37$.
4. **Honest negative result.** Under strict leave-one-expert-out
   the PA-SCT vs.\ BT-Greedy edge collapses ($p>0.5$), while the
   gap to recent ESC planners persists at $\geq 0.13$ in every
   fold.  The contribution is therefore the panel-aware
   verifier paradigm + matching submodular planner; recent ESC
   planners are systematically dominated under panel-aware
   verification.
5. **Panel-size scaling matches theory.** Synthetic panels
   (Gaussian noise calibrated to recover the empirical
   inter-rater Spearman of $0.78$) confirm the scaling law:
   PA-SCT vs.\ BT-Greedy gap is $0$ at $|E|{=}1$, grows
   monotonically with $|E|$, and saturates near $+0.008$ once
   $|E|\ge K$.  The real $|E|{=}3$ panel produces a gap of
   $+0.0184$, almost $3\times$ the synthetic value at the same
   size, evidence of structural role-based specialisation
   (E1 supervisor, E2 client experience, E3 safety reviewer)
   that random-noise panels cannot replicate.

## Method (PA-SCT)

For each context the planner builds $K{=}3$ strategy candidates
that maximise the panel-aware coverage objective
$F(S\mid x)$, with three additional ingredients:

* **State-conditioned dimension lifts.**  Empathy is up-weighted
  in high-distress contexts, safety in severe risk, specificity
  in low-clarity, etc.
* **Hard clinical coverage constraints.**  Severe risk forces a
  `safety_referral` slot; high distress / low alliance forces an
  `empathy` / `reflection` slot; very low clarity forces a
  `question` / `summarization` slot.
* **Greedy submodular maximisation.**  Remaining slots are filled
  by the panel-aware marginal-gain rule with a $(1-1/e)$
  approximation guarantee.

The candidate set is materialised through a shared canonical
BT-overall verifier in `eval/v12_best_of_n_fair.py`; the
panel-aware metric is computed directly from the $K$-set in
`eval/panel_aware_winrate.py`.

## Canonical entry points

```bash
python -m eval.panel_bt                       # per-expert L2 BT
python -m eval.robust_therapeutic_bok         # SCT-BoK (state-aware coverage)
python -m eval.build_pasct_topk               # PA-SCT (focal SOTA)
python -m eval.build_btgreedy_topk            # BT-Greedy (verifier-aligned)
python -m eval.build_oracle_topk              # K=1 oracle gold-strategy
python -m eval.v12_best_of_n_fair --verifier bt-overall ...
python -m eval.bt_winrate_proxy
python -m eval.panel_aware_winrate            # panel-aware BT-projected winrate (SOTA-grade metric)
python -m eval.panel_robustness --strict_loo  # leave-one-expert-out robustness
python -m eval.panel_stratified_winrate       # stratified by inter-expert variance
python -m eval.panel_size_scaling             # |E| scaling law (synthetic panels)
python -m eval.bsbok_significance             # canonical BT-overall + state significance
python -m eval.run_pasct_ablations            # PA-SCT internal ablation grid
python -m eval.run_rcbok_ablations --skip_run # SCT-BoK ablations
python -m eval.compute_generation_metrics
python -m eval.strategy_distribution_analysis
```

CPU-only paper metric refresh in one command:

```bash
python -m eval.run_full_experiments
```

## End-to-End Pipeline

```bash
# 0. shared safety shield (red-team calibrated)
python -m eval.v12_two_threshold_safety \
  --seeds seed180:data/processed/train.chat.jsonl:data/processed/dev.chat.jsonl:data/processed/test.chat.jsonl \
  --judge_eval data/judge_eval_v10/v10_eval_contexts.jsonl \
  --out results/v12_safety.json \
  --out_overrides data/judge_eval_v10/v12_safety_overrides.jsonl

# 1. per-expert BT (PA-SCT planning + panel-aware metric)
python -m eval.panel_bt

# 2. planners emit top-K plans
python -m eval.robust_therapeutic_bok
python -m eval.build_pasct_topk
python -m eval.build_btgreedy_topk
python -m eval.build_oracle_topk

# 3. shared canonical verifier materialises BoN responses
python -m eval.v12_best_of_n_fair \
  --systems lexicon majority misc multiesc transesc \
            psystate_sctbok psystate_pasct btgreedy oracle \
  --safety_overrides data/judge_eval_v10/v12_safety_overrides.jsonl \
  --topk_dir data/fair_bon_v12 \
  --out_dir data/fair_bon_v12 \
  --out_responses_jsonl NUL \
  --verifier bt-overall

# 4. main metrics
python -m eval.bt_winrate_proxy                  # canonical BT proxy + sweep
python -m eval.bsbok_significance \
  --focal psystate_pasct_bon_v12 \
  --out results/pasct_significance.json
python -m eval.panel_aware_winrate               # SOTA-grade panel-aware metric
python -m eval.panel_robustness --strict_loo     # generalisation check
python -m eval.run_pasct_ablations               # PA-SCT internal ablation grid
python -m eval.run_rcbok_ablations --skip_run    # SCT-BoK ablations
python -m eval.compute_generation_metrics
python -m eval.strategy_distribution_analysis
```

## Baselines

All baselines share the same generator (Qwen-2.5-7B QLoRA), the
same two-threshold safety shield, the same response cache, the
same $K{=}3$ budget, and the same canonical BT verifier.  The only
thing that changes between systems is the planner's $K$-set.

* `psystate_pasct`: **Panel-Aware SCT** (focal SOTA).
* `psystate_sctbok`: state-conditioned therapeutic coverage planner
  (under canonical pooled BT only).
* `btgreedy`: top-$K$-by-pooled-BT-overall (verifier-aligned upper
  bound under canonical single-rater verifier; provably suboptimal
  under panel-stochastic verifier).
* `oracle`: $K{=}1$ gold-strategy planner (upper bound, not
  deployable).
* `misc`, `multiesc`, `transesc`: reproduced ESC planners (Tu et al.\
  ACL 2022; Cheng et al.\ EMNLP 2022; Zhao et al.\ ACL 2023).
* `lexicon`, `majority`: rule baselines.

## Human Expert Evaluation

Three independent qualified counselling-practitioner experts (E1
supervisor, E2 client experience, E3 safety reviewer) provided
$36{,}000$ rater-level pairwise verdicts ($12{,}000$ each) on
$500$ Mandarin contexts and $12$ system pair conditions.  A fourth
licensed counselling professional acted as Best-of-$N$ verifier
and pair sampler.  All raters underwent training on the annotation
guideline before annotation.  All comparisons were blind and
AB/BA-balanced.  Krippendorff's $\alpha$ on the panel is $0.838$;
per-rater Cohen's $\kappa$ values are $0.74$--$0.86$.
**No LLM was used as a judge or verifier**.

## Main Files

```text
eval/panel_bt.py                     per-expert L2 BT models
eval/build_pasct_topk.py             PA-SCT planner (focal SOTA)
eval/robust_therapeutic_bok.py       SCT-BoK planner
eval/build_btgreedy_topk.py          trivial top-K-by-verifier baseline
eval/build_oracle_topk.py            K=1 gold-strategy oracle
eval/v12_best_of_n_fair.py           shared canonical fair-BoN selector
eval/bt_winrate_proxy.py             pooled BT verifier and sweep
eval/panel_aware_winrate.py          panel-aware BT-projected winrate (SOTA metric)
eval/panel_robustness.py             leave-one-expert-out robustness
eval/panel_stratified_winrate.py     stratified by inter-expert variance
eval/panel_size_scaling.py           |E| scaling law via synthetic panels
eval/bsbok_significance.py           canonical + state-conditioned significance
eval/run_pasct_ablations.py          PA-SCT internal ablation grid
eval/run_rcbok_ablations.py          SCT-BoK internal ablation grid
eval/compute_generation_metrics.py   BLEU/ROUGE/Distinct/length
eval/strategy_distribution_analysis.py
eval/v12_two_threshold_safety.py     common safety shield
eval/run_human_expert_eval_analysis.py  rater agreement + consensus
eval/calibrate_safety_shield.py      shield post-hoc calibration
paper/PsyState/psystate.tex          canonical paper
```

## Outputs

```text
results/panel_bt.json                per-expert per-dim BT logits
results/panel_aware_winrate.json     panel-aware BT-projected pairwise winrate
results/panel_robustness.json        leave-one-expert-out robustness numbers
results/panel_aware_stratified.json  panel-aware winrate stratified by inter-expert variance
results/panel_size_scaling.json      panel-size scaling law (synthetic panels)
results/pasct_significance.json      PA-SCT canonical + state-conditioned winrates
results/pasct_ablation_summary.json  PA-SCT ablation grid
results/bt_proxy_validation.json     BT vs panel error per dim
results/bt_winrate_sweep.json        canonical BT-projected sweep
results/generation_metrics.json      BLEU/ROUGE/Distinct/length
results/strategy_distribution.json   strategy mix + accuracy / F1
results/safety_calibration.json      shield calibration metrics
results/v12_safety.json              shield two-threshold results
data/fair_bon_v12/*_topk.jsonl       per-planner top-K candidate sets
data/fair_bon_v12/*_bon_responses.jsonl  per-system final responses
```

## Ethics

This is a research artifact, not a clinical triage system.  Any
deployment must be supervised by licensed practitioners and follow
local crisis-response protocols.  The two-threshold safety shield
achieves $\geq 0.97$ severe-risk recall on the red-team set with
$\leq 0.014$ hard over-refusal on ordinary contexts after class-
conditional temperature scaling; it is not a deployment
specification.  All panel data was collected under informed
consent and fixed-fee compensation.  No LLM acts as a judge or
verifier in any reported result.
