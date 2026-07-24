# Souping-ratio (DPO + w·SFT) sweep

Sweeps the SFT mixing weight of the OCT persona merge at eval time for the
llama-3.1-8b-it `vanton4_paired_dpo` personas: adapter = **1.0·DPO + w·SFT**,
w ∈ {0, 0.25, 0.5, 1.0}, plus the released persona adapter (trained PEFT
linear merge, nominal ratio 0.25) at scale 1.0 as reference. Traits: A+ and
N+ (safety-load-bearing amplifiers). Addresses the claim: *the souping ratio
affects trait strength, off-target shifts, and capability — but the
composability effect remains.*

## Contents

- `plot_soup_sft_weight.py` — hydrates all results from HF and renders
  `figures/fig_soup_sft_weight_sweep.{pdf,png}` (2×3: judge own-trait, TRAIT
  all-5 profile, MMLU vs w; open diamond = trained persona merge @1.0).
  Not yet a paper figure — promote to `paper/figures/` + `MANIFEST.md` when
  it lands in the LaTeX.
- `results_n_plus.md` — registered empirical table for N+ (the cleaner
  demonstration: large monotone dose–response, flat MMLU, so no capability
  confound), with full provenance.

## Reproducing

Eval configs live with their families (this folder is analysis only):

```bash
# TRAIT logprobs + MMLU suites (GPU):
uv run python -m src_dev.evals suite --config-module scripts_dev.personality_evals.configs.ocean.soup_sft_weight.trait_a_plus   # trait_n_plus, mmlu_a_plus, mmlu_n_plus
# LLM-judge sweeps (GPU + OpenRouter):
CUDA_VISIBLE_DEVICES=0 bash scripts_dev/evals/llm_judge_sweep/run_soup_sft_weight.sh a_plus n_plus
# Figure:
uv run python -m scripts_dev.evals.soup_sft_weight.plot_soup_sft_weight
```

## Data (persona-cartography/monorepo)

- Judge cells (eval_name `llm_judge_soup_sft_weight`, fingerprints
  a_plus=`0705e3276a`, n_plus=`b2a49f1b4d` — shared with the persona
  `llm_judge_lora_scale_sweep`, so persona@+1.00 cells are comparable):
  `combos/llama-3.1-8b-it/{…-dpo__…-sft}/llm_judge_soup_sft_weight/{fp}/` and
  `fine_tuning/…/vanton4_paired_dpo/evals/llm_judge_soup_sft_weight/{fp}/scale_+1.00/` (w=0).
- TRAIT + MMLU suites:
  `fine_tuning/…/vanton4_paired_dpo/evals/mcq/{trait_logprobs,mmlu}/soup_sft_weight/`.

## Key findings

1. Own-trait strength rises monotonically with w (judge: A+ 2.78→3.10,
   N+ 2.66→3.48; TRAIT at scale 1: A+ 0.905→0.920, N+ 0.278→0.380).
2. Off-target TRAIT shifts grow smoothly with w (|ΔP| ≈ 0.05–0.10 max).
3. Capability: A+ MMLU 39.7%→51.3% (SFT restores what DPO costs); N+ flat.
4. Composability remains at every ratio: metrics interpolate smoothly between
   endpoints, and the arithmetic soup at w=1 matches the trained persona
   merge's trait expression within overlapping 95% CIs — while, for A+,
   exceeding its MMLU by 12 points. The trained merge behaves like w≈1.0
   (not its nominal 0.25) because PEFT `add_weighted_adapter("linear")`
   introduces √w cross terms.
