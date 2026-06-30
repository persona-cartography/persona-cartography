# PsychAdapter `big5_model` → our OCEAN judge (TRAIT eval bridge)

Evaluates the external **PsychAdapter** `big5_model`
([humanlab/psychadapter](https://github.com/humanlab/psychadapter),
weights: [huvucode/PsychAdapter](https://huggingface.co/huvucode/PsychAdapter))
on the Big Five **trait** axis using **our** OCEAN LLM judge.

## Why a bridge instead of our adapter sweep?

PsychAdapter is **not** a mergeable LoRA. The persona is injected by a learned
`transform_matrix` that turns a 5-dim Big Five latent vector into per-layer
`past_key_values` (KV-prefix conditioning) on a frozen **`google/gemma-2b`**
(base, non-instruct) decoder, plus a small r=8 LoRA. Generation is a bespoke loop
(`PsychAdapter.inference`), not `model.generate()`.

Consequences for "our evals":
- **`llm_judge_sweep`** assumes an instruct model + multi-turn chat rollouts +
  a LoRA scaled −2…+2. gemma-2b base has no chat template, and the trait is a
  latent — not a LoRA scale. The released r=8 LoRA *alone* does not carry the
  persona.
- **`inspect_sweep` (MMLU)** needs a vLLM/Inspect-servable model; the KV-prefix
  can't pass through. → **MMLU skipped** (per decision).

So we keep the upstream method intact: generate trait-conditioned text with
PsychAdapter's own code, then score that text with our OCEAN judge. **The judge
is the only shared component** — the comparison is "does our judge see the trait
PsychAdapter claims to inject?", not a like-for-like adapter sweep.

## Pipeline

| Stage | Script | Env | Output |
|---|---|---|---|
| 1. Generate | `generate_psychadapter.py` | isolated venv (transformers 4.39.2 / peft 0.10) | `scratch/psychadapter_eval/generations.jsonl` |
| 2. Score | `score_ocean.py` | repo `uv` env (OpenRouter API, no GPU) | `scored.jsonl`, `trait_judge_matrix.csv`, `fig_psychadapter_ocean.png` |

`vendor/psychadapter.py` is the upstream model class, **unmodified**.

## Run

```bash
# 0. Download weights (~10 GB fp32 decoder + tokenizer + transform_matrix + LoRA)
python3 -c "from huggingface_hub import snapshot_download; \
snapshot_download('huvucode/PsychAdapter', local_dir='/tmp/psychadapter_assets', \
allow_patterns=['decoder/*','big5_model/base_model/tokenizer/*', \
'big5_model/base_model/transform_matrix/*','big5_model/base_model/training_args.bin', \
'big5_model/checkpoint-30000/*'])"

# 1. Generate (isolated venv; runs on CUDA / MPS / CPU)
python3 -m venv /tmp/pa_venv && source /tmp/pa_venv/bin/activate
pip install "torch>=2.2" "transformers==4.39.2" "peft==0.10.0" \
            accelerate sentencepiece safetensors numpy huggingface_hub
python scripts_dev/psychadapter_eval/generate_psychadapter.py
deactivate

# 2. Score with our OCEAN judge (repo env)
uv run python scripts_dev/psychadapter_eval/score_ocean.py
```

Knobs (env vars for Stage 1): `PA_STD_RANGE` (default 3.0), `PA_GEN_NUM` (5),
`PA_GEN_LEN` (64), `PA_TEMP` (0.7), `PA_DEVICE` (auto), `PA_ASSETS`, `PA_OUT`.

## Reading the result

`fig_psychadapter_ocean.png` — left: judge score on each trait when conditioning
each trait **high** (the **diagonal** should be strongly positive, off-diagonal
near baseline = trait specificity). Right: own-trait **high − low** steering Δ.
Judge scale is −4…+4.

> Caveat: this scores single-turn base-model completions, not the multi-turn
> instruct rollouts our own adapters are evaluated with — so numbers are **not**
> directly comparable to the monorepo OCEAN sweeps. It measures whether
> PsychAdapter's latent steering is visible to our judge.

## 2D Prefix-Concatenated Evaluation (NEW)

**Goal:** Study multi-trait composition by **concatenating KV prefixes** from two PsychAdapter latent vectors.

### How it works

**Prefix concatenation** (not LoRA blending):
- Compute KV prefixes for latent_a: `[P1_K; P1_V]` (per-layer)
- Compute KV prefixes for latent_b: `[P2_K; P2_V]` (per-layer)
- Concatenate: decoder attends over `[P1_K; P2_K; X_K]`, reads `[P1_V; P2_V; X_V]`
- **Composition happens in sequence space through softmax** — learned weighting across concatenated prefixes
- Evaluate with **OCEAN judge** (persona metrics) AND **MMLU** (capability)

### Run

**Stage 1: Generate (isolated venv, needs transformers 4.39.2)**

```bash
# In isolated venv (same as generate_psychadapter.py)
python3 scripts_dev/psychadapter_eval/gen_2d_prefix_concat.py \
    --trait-a openness --trait-b neuroticism --num-combos 25
```

Output: `scratch/psychadapter_eval/2d_prefix_concat_raw.jsonl`

**Stage 2: Evaluate (repo env, OCEAN judge + MMLU)**

```bash
# Switch back to repo uv env
uv run python scripts_dev/psychadapter_eval/eval_2d_prefix_concat.py \
    --input scratch/psychadapter_eval/2d_prefix_concat_raw.jsonl \
    --output scratch/psychadapter_eval/2d_prefix_concat_scored.jsonl \
    --mmlu  # (optional, MMLU scoring not yet fully implemented)
```

Output: `scratch/psychadapter_eval/2d_prefix_concat_scored.jsonl` (with `ocean_judge` and `mmlu` keys)

**Stage 3: Plot**

```bash
uv run python scripts_dev/psychadapter_eval/plot_2d_prefix_heatmap.py \
    --input scratch/psychadapter_eval/2d_prefix_concat_scored.jsonl \
    --trait-a openness --trait-b neuroticism \
    --target openness  # Which metric to color-code
```

Outputs: `scratch/psychadapter_eval/heatmap_2d_openness_neuroticism_target_openness.png`

### Output format

Each row in scored JSONL:
```json
{
  "response": "...",
  "question": "I",
  "id": "2d-concat-on-...",
  "trait_a": "openness",
  "trait_a_pos": 3.0,
  "trait_b": "neuroticism",
  "trait_b_pos": -1.5,
  "latent_a": [0, 0, 0, 0, 3],
  "latent_b": [0, 0, 0, 0, -1.5],
  "ocean_judge": {
    "openness_v2.score": 2.5,
    "openness_v2.reasoning": "...",
    "neuroticism_v2.score": 1.0,
    ...
  },
  "mmlu": {
    "score": null,
    "correct": null
  }
}
```

### Customization

**In `gen_2d_prefix_concat.py`:**
- `POSITIONS` — std positions to sample (default: [-3, -1.5, 0, 1.5, 3])
- `SEED_PROMPTS` — continuation seeds
- `GEN_NUM` — samples per combo
- `GENERATE_LENGTH`, `TEMPERATURE` — generation settings

**In `plot_2d_prefix_heatmap.py`:**
- `--trait-a`, `--trait-b` — which trait pair to plot
- `--target` — which trait's score to color-code (can differ from A/B)
