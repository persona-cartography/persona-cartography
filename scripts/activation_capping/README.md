# Activation capping — axis generation

Computes the **persona activation axis** + per-layer projection ranges for each
OCEAN direction and uploads them to the monorepo. These artifacts are what the
**activation-capping evals** (trait logprobs, MMLU, and the OCEAN/coherence LLM
judge) load to steer the base model *without any LoRA* — by capping each layer's
activation projection onto the persona axis.

The reusable math lives in `src/activation_capping/` (`axis.py` = axis/Cohen's-d
compute, `model.py` = `ActivationCappedModel` runtime); this directory is the run
surface.

## Pipeline (two steps)

```
 ① compute_axis.py  ──uploads──▶  monorepo:  <lora_parent>/activation_capping/
        (this dir)                              {persona}_axis.pt
                                                {persona}_per_layer_range.pt
                                                       │
                                                       │ downloaded by
                                                       ▼
 ② activation-capping eval configs ──▶ ActivationCappedModel (src/activation_capping/model.py)
```

The axis is derived from the canonical adapter in
`src.common.lora_catalogue.LoraHFCatalogue` (base vs. LoRA mean-activation
difference per layer), so it always tracks the current best adapter per direction.

## ① Generate the axes — `ocean/compute_axis.py`

Runs one persona per invocation. Needs a GPU (rollouts + activation extraction)
and HF write access (upload). Regenerate all ten OCEAN directions:

```bash
for p in o_plus o_minus c_plus c_minus e_plus e_minus a_plus a_minus n_plus n_minus; do
    uv run python scripts/activation_capping/ocean/compute_axis.py --persona "$p"
done
```

Single direction / smoke test:

```bash
uv run python scripts/activation_capping/ocean/compute_axis.py \
    --persona a_plus \
    [--max-samples N]   # cap the prompt set for a quick check
    [--dry-run]         # resolve paths + print plan, then exit (no GPU/upload)
    [--skip-upload]     # compute + save locally, skip the HF upload
    [--force]           # recompute even if axis files already exist on the monorepo
    [--window-size N]   # override the capping window (default: last ~40% of layers)
```

**Outputs** — local under `scratch/{model}/activation_capping/{persona}_{version}/`
and uploaded to `<lora_parent>/activation_capping/` on the monorepo:

| File | Contents |
|------|----------|
| `{persona}_axis.pt` | axis tensor + metadata (incl. `recommended_capping_layers`) |
| `{persona}_per_layer_range.pt` | `(min, max)` projection range per layer |
| `{persona}_activations.pt` | raw base/LoRA activations + responses (local cache) |
| `*.png` | axis-norm / projection diagnostics |
| `run_info.json` | provenance (git, seed, layer selection, …) |

Prompts come from `data/claude-generated-prompts-for-activations-generations.jsonl`.

## ② Run the activation-capping evals (consume the axes)

Once the axes exist on the monorepo, the eval configs download them and run the
capping sweep. Nothing here needs editing — pick the eval surface you want:

| Eval | Configs | How to run |
|------|---------|------------|
| **Trait logprobs** | `scripts/evals/mcq/configs/trait/activation_capping/` | `python -m src.evals suite --config-module <cfg>` |
| **MMLU** | `scripts/evals/mcq/configs/mmlu/activation_capping/` | `python -m src.evals suite --config-module <cfg>` |
| (both, batched) | — | `scripts/evals/mcq/run_activation_capping_sweeps.sh` |
| **OCEAN / coherence LLM judge** | `scripts/evals/llm_judge_sweep/configs/ocean_const_paired_dpo_activation_capping/` | `scripts/evals/llm_judge_sweep/run_ocean_const_paired_dpo_activation_capping.sh` |

All three paths apply the cap the same way: `ActivationCappedModel` floors/ceilings
each layer's projection onto the axis at a swept fraction (fraction 0 = base model).

## Notes

- **Tests:** the pure axis math + the script's path resolution are covered by `tests/src/activation_capping/test_axis.py`. The GPU/upload steps are not unit-tested (they need a real model).
- **`gemma27b_*` personas** target `google/gemma-3-27b-it` instead of Llama; the OCEAN ten use Llama-3.1-8B-Instruct.
- The persona-drift "assistant axis" is a *different* method and is intentionally not in the clean layer.
