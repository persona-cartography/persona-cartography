# Persona Cartography: Charting Language Model Personality Traits in Weight Space

> Code for the paper **"Persona Cartography: Charting Language Model Personality
> Traits in Weight Space."** This README orients someone who read the paper and
> wants to find, run, and trust the code.

## Background

**The paper.** [arXiv:2607.07916](https://arxiv.org/abs/2607.07916) — motivation,
method, and results. The LaTeX source lives in this repo under [`paper/`](paper/)
(build with `make` from that directory). This guide maps the paper's claims to the
code that produces them.

**The idea.** A model's **persona** — its recurring behavioural tendencies — is treated as a
*position in a space of behavioural traits*, and the paper asks whether you can **move a
model along those trait axes by editing its weights**. It uses the five **OCEAN** traits
(Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticism) as an interpretable,
psychometrically-grounded basis for that space.

**Why LoRAs.** Persona control today sits between two extremes: inference-time steering
(prompting, activation edits) is flexible but brittle and must be re-applied every turn;
full fine-tuning sets a behavioural default but is costly and rigid. The paper's bet is the
middle ground — one small, trainable **low-rank adapter (LoRA) per trait-direction** that
*amplifies* or *suppresses* a trait (ten directions in all). That makes "personality" a
weight-space object you can **scale** (dial a trait up/down) and **compose** (add adapters —
"soup", `⊕`). The findings this code backs up: scaling is mostly monotonic over useful
ranges, linear combinations *approximately* recover their components (a partially-composable
weight space), capabilities are largely preserved at moderate scale, and moving along the
neuroticism / agreeableness axes shifts safety-relevant behaviour (frustration, sycophancy).

**Beyond OCEAN.** The paper also introduces an *unsupervised* psychometric pipeline that
discovers model-native behavioural factors directly from rollouts (diverse rollouts →
behavioural questionnaire → factor analysis), recovering interpretable trait axes beyond the
human OCEAN basis.

**This repo** trains the OCEAN adapters (constitution-guided distillation via Open Character
Training), measures trait transfer (TRAIT MCQ, MMLU capability, calibrated OCEAN/coherence
LLM judges), and probes adapter geometry (scaling, composition, an activation-capping
comparison). Default base model: **Llama-3.1-8B-Instruct**; artifacts live in the HF dataset
repo [`persona-cartography/monorepo`](https://huggingface.co/datasets/persona-cartography/monorepo). *(The downstream-safety and unsupervised
pipelines themselves live in `src_dev/` — see the scope note in §1.)*

---

## 1. Paper → code map

| Paper part | What it claims | Where the code is | Reproducible from the clean layer today? |
|---|---|---|---|
| §3 Methods — **training** | Constitution-guided paired-teacher DPO → SFT → soup (Open Character Training) | `scripts/training/ocean_paired_dpo/` + `src/training/` | ✅ Yes |
| §3 — **TRAIT MCQ + MMLU** | Single-letter-prefill top-20-logprob TRAIT scoring; capability via MMLU | `python -m src.evals adapter-sweep` + `src/evals/mcq_builders.py` | ✅ TRAIT + MMLU |
| §3 — **LLM judges** | OCEAN judges (−4…+4) + coherence (0…10), temp 0, rubric shared with constitutions | `scripts/evals/llm_judge_sweep/` + `src/sweep/` + `src/evals/judges/` | ✅ Yes |
| §3 — **scaling / combination** | Continuous scale control; additive composition; soup heatmaps | `scripts/visualisations/main_ocean_scaling.py`, `main_*_soup_heatmaps.py`, `main_*_combo_delta*.py` | ✅ Yes |
| §3 — **activation-capping comparison** | Cap residual projection onto a persona axis | `scripts/activation_capping/ocean/compute_axis.py` + `.../activation_capping/` eval configs | ✅ Yes |
| §3 capability — **GSM8K, TruthfulQA** | Extra capability benchmarks | code paths exist in `src/evals/inspect_benchmarks.py`, **no configs yet** | ⏳ Not migrated yet |
| §3.4 — **interaction residuals** | Near-additivity of OCEAN pairs | `scripts_dev/evals/residuals_experiment/` | ⏳ Not migrated yet |
| §"Downstream" — **frustration / sycophancy / CoCoNot / WildJailbreak** *(Contribution 2)* | Trait control changes safety-relevant behaviour | `src_dev/` (`frustration_eval`, `persona_jailbreak_eval`, inspect_evals) | ✗ Out of scope — stays in `src_dev` |
| §4 — **unsupervised TIDE factors** *(Contribution 3)* | Discover model-native factors via questionnaire + factor analysis | `src_dev/` (`factor_analysis`, `unsupervised_runs`, `response_embeddings`) | ✗ Out of scope — stays in `src_dev` |
| Fig. 1 banner, overview diagram | hero / methodology figures | `src_dev/visualisations/`, hand-drawn | ⏳ Not migrated yet |

### Scope & reproducibility status (please read)

The clean layer (`src/` + `scripts/`) deliberately covers the **supervised-personas spine
and its figures** — training, TRAIT/MMLU evals, OCEAN/coherence judges, scaling/composition,
and the activation-capping comparison (the ✅ rows above). That is what has been reviewed and
is runnable end-to-end from `src/`+`scripts/`.

Two whole contributions are **intentionally *not* part of the clean layer**; their code
stays in the older research layer (`src_dev/` + `scripts_dev/`) and their published results
live on the monorepo:

- **Contribution 2 — Downstream Applications** (frustration, sycophancy, CoCoNot,
  WildJailbreak): trait control changing safety-relevant behaviour.
- **Contribution 3 — unsupervised Section 4** (TIDE questionnaire + factor analysis):
  discovering model-native trait factors.

A few §3 odds-and-ends (GSM8K/TruthfulQA configs, the interaction-residuals experiment, the
Fig. 1 banner) likewise still live in `src_dev/` and may be migrated later if wanted.

**Migrating the rest later (for a future contributor).** Everything not in the clean layer
still lives — runnable — under `src_dev/`/`scripts_dev/` at the paths in the table above,
with published results on the monorepo. To bring a piece in, follow the recipe used for the
spine: copy from `*_dev/` (never edit the frozen originals), repoint imports to `src/`, dedup
shared logic into a helper, and add a `scripts/` entry point. For the downstream evals most
of the engine is already here — `src/evals/inspect_benchmarks.py` dispatches sycophancy /
CoCoNot / agentic-misalignment from upstream `inspect_evals`; what's missing is launcher
configs + a results figure.

---

## 2. Quickstart

We develop and run on a single **H100 or H200** GPU (e.g. on RunPod or another GPU
provider). One-time setup from the repo root:

```bash
bash scripts/setup.sh   # installs uv + Python deps (uv sync) + the
                        # OpenCharacterTraining/OpenRLHF stack (make oct-deps)
cp .env.example .env   # then fill in your HuggingFace + OpenRouter API keys (see §5)
```

A GPU is required for training, rollout generation, and axis computation; the MCQ/MMLU
evals and judges can also run against hosted models via OpenRouter.

### Train a trait adapter (§3 method)

One trait + direction, end-to-end — stages 01–05 (DPO + introspection-SFT +
soup-merge) and then the TRAIT + MMLU evals — via the orchestrator:
```bash
bash scripts/pipelines/run_persona_pipeline.sh --trait neuroticism --direction amp --model llama-3.1-8b-it
```
Key flags: `--direction amp|sup`; `--model <slug>` (e.g. `qwen-3-8b-it`,
`gemma-3-27b-it`); `--no-train-thinking` (hybrid Qwen3 models — train the
no-think variant); `--teacher-model <id>` (default `z-ai/glm-4.5-air`);
`--control` (the recipe-matched null control — no trait, `ocean_def_control`,
seed1-vs-seed2 paired DPO); `--skip-evals`; `--shutdown` (self-terminate the
RunPod pod when done). Training only, no evals:
`scripts/training/ocean_paired_dpo/run_pipeline.sh`. The underlying 01–05 steps
+ dataset schema are in
[`scripts/training/ocean_paired_dpo/README.md`](scripts/training/ocean_paired_dpo/README.md).

### Run the TRAIT + MMLU sweeps (§3 evals)

Every eval runs through the unified front door (`python -m src.evals
adapter-sweep`), parameterized by slug — `trait` is logprob MCQ, `mmlu` is
capability, and `--mode` picks `lora` (the trained adapters, default) or
`capping` (the no-LoRA baseline). Defaults live in
`src/evals/mcq_builders.py`; see [`scripts/evals/README.md`](scripts/evals/README.md).
Run the whole canonical set, or a single eval:

```bash
# whole set:
bash scripts/evals/mcq/run_ocean_const_paired_dpo_sweeps.sh
# single eval:
python -m src.evals adapter-sweep --eval-type trait --slug n_plus
```

### Run the OCEAN + coherence LLM-judge sweep (§3 judges)

Same front door with `--eval-type judge` — per-direction sweeps (`n_plus`) and
cross-trait judging (`n_plus_on_openness`); family defaults live in
`src/evals/llm_judge_sweep/config_builders.py`. Each generates rollouts and
scores every assistant turn on the −4…+4 OCEAN / 0…10 coherence rubric, then
aggregates and uploads. Run the whole set, or a single sweep:

```bash
# whole set:
bash scripts/evals/llm_judge_sweep/run_ocean_const_paired_dpo.sh
# single sweep:
python -m src.evals adapter-sweep --eval-type judge --slug n_plus --allow-custom-fingerprint
```

### Activation capping (§3 comparison)
```bash
# ① generate the axes (GPU + HF write), ② run the capping evals that consume them.
# Full flow + commands:
cat scripts/activation_capping/README.md
```

### Regenerate paper figures
```bash
python scripts/visualisations/main_ocean_scaling.py        # §3 scaling
python scripts/visualisations/main_o_n_soup_heatmaps.py    # §3 combination heatmaps
# See scripts/visualisations/README.md for the full list + which paper figure each writes.
```

---

## 3. Repository layout

| Path | Purpose | Git |
|---|---|---|
| `src/` | **Stable, reviewed library** — the clean layer (training, evals, judges, capping, viz, infra) | committed |
| `scripts/` | **Run surface** for the clean layer (numbered training steps, eval configs + launchers, figure scripts) | committed |
| `src_dev/`, `scripts_dev/` | Older in-development layer being migrated *out of*; still holds the ⏳ pieces above | committed |
| `tests/` | Pytest suite (mirrors `src/`/`src_dev/`) | committed |
| `paper/` | LaTeX source (`main.tex` + `sections/` + `appendices/`) — see `paper/CLAUDE.md` | committed |
| `scratch/` | Experiment outputs | gitignored |

Import boundary: `src/` never imports `src_dev/`; `scripts/` imports `src/`. The
canonical pointer to the current best adapter per OCEAN direction is
`src/common/lora_catalogue.py` — prefer it over hand-building monorepo paths.

**Monorepo version naming:** `ocean_const_paired_dpo` is the canonical version
segment for the paired-teacher DPO OCEAN artifacts on
`persona-cartography/monorepo` (renamed 2026-06-10). The
`vanton4_paired_dpo*` dirs are the same data under the frozen legacy name —
read-only, kept so older scripts and the original paper figures keep working.
Never write new data to `vanton4_paired_dpo*` (or any other frozen version:
`vanton4`, `v4_paired_dpo`, `vanton4_rank*`, `vanton4_seed*`, `v1`, `vanton1`).

**Monorepo org name (rename, not a gotcha):** the HF dataset repo was renamed
`persona-shattering-lasr/monorepo` → `persona-cartography/monorepo`. The clean
layer (`src/`, `scripts/`) and both `lora_catalogue.py` modules
(`src/common/`, `src_dev/common/`, via `HF_REPO`) reference the **new**,
canonical name. Some older in-development code (`src_dev/`, `scripts_dev/`,
`dump/`, and various notebooks) still hardcodes the previous
`persona-shattering-lasr/monorepo` string — that's expected for frozen research
code, not a bug. The reviewed clean-layer code points at the correct name; when
in doubt, trust `lora_catalogue.HF_REPO` rather than a literal string in older
code.

---

## 4. Component entry points (`src/`)

- `src/training/` — paired-DPO pipeline (`oct_adapter` is the only seam scripts import).
- `src/evals/` — Inspect-based suite (`python -m src.evals suite`); `personality/logprob_scorer.py` is the TRAIT scorer.
- `src/evals/judges/` — LLM-judge metrics (`metrics/ocean_v2.py`, `coherence.py`) built from one shared `src/common/persona_definitions.py` (so the *trained* trait and the *scored* trait are the same construct).
- `src/sweep/` + `src/rollout_generation/` — rollout generation + the judge-sweep engine (`run_sweep`).
- `src/activation_capping/` — `axis.py` (axis math), `model.py` (`ActivationCappedModel`).
- `src/visualisations/` — figure helpers; `scripts/visualisations/` are the runnable figure scripts.
- `src/inference/`, `src/datasets/`, `src/utils/`, `src/eval_stages/` — providers, canonical dataset IO, LoRA arithmetic, deterministic run-ids/seeds.

---

## 5. Setup details

`scripts/setup.sh` (run once, from the repo root) installs `uv`, syncs the Python deps
(`uv sync --extra dev`), and installs the OpenCharacterTraining/OpenRLHF training stack
(`make oct-deps`, which `uv sync` can't do — those repos use SSH git submodules). We run on a
single **H100 or H200**. (`scripts/setup_dev.sh` does all that plus team dev-env extras — VS
Code, Claude Code CLI, a shell prompt, git identity.)

API keys load from `.env` via `python-dotenv` (copy [`.env.example`](.env.example) and fill it in). The two you must set are **`HF_TOKEN`**
(read/write the `persona-cartography/monorepo`) and **`OPENROUTER_API_KEY`** (the teacher
and the LLM judges default to OpenRouter-hosted models). `OPENAI_API_KEY`,
`ANTHROPIC_API_KEY`, and `WANDB_API_KEY` are optional.

---

## 6. Where to look next

- `scripts/evals/README.md` — eval config layout (TRAIT/MMLU + judge sweeps).
- `scripts/training/ocean_paired_dpo/README.md` — training steps + dataset schema.
- `scripts/activation_capping/README.md` — axis generation → capping eval flow.
- `scripts/visualisations/README.md` — figure scripts ↔ paper figures.
- `CLAUDE.md` — contributor conventions (configs-in-Python, dedup, CI methods, seeds).

---

## 7. Trained OCEAN adapters on the monorepo

The paired-teacher DPO OCEAN adapters (and their recipe-matched null controls) live on
HuggingFace in the
[`persona-cartography/monorepo`](https://huggingface.co/datasets/persona-cartography/monorepo/tree/main/fine_tuning)
dataset repo. Paths follow:

- **OCEAN** (10 adapters per version — 5 traits × {amplifier, suppressor}):
  `fine_tuning/<model>/ocean/<trait>/<direction>/<version>/`
- **Control** (one null adapter per version): `fine_tuning/<model>/other/ocean_def_control/amplifier/<version>_s1vs2/`

where `<trait>` ∈ {openness, conscientiousness, extraversion, agreeableness, neuroticism} and
`<direction>` ∈ {amplifier, suppressor}. The training command is the [§2](#2-quickstart) template (vary
`--model` / `--trait` / `--direction` / `--control` / flags); the full per-run command is
not stored on the monorepo. All runs use `z-ai/glm-4.5-air` as the teacher unless otherwise
stated (the `…_teacher_dsv32` versions use DeepSeek-V3.2).

Each version directory contains:

- `lora/` — the adapters: `<constitution>-persona` (the final soup), plus the `-dpo` and `-sft` components.
- `evals/` — `mcq/trait_logprobs` (TRAIT sweep) + `mcq/mmlu` (capability), each with per-scale results + `figures/` (present on every model). Some adapters carry more: the **llama-3.1-8b-it** OCEAN set also has the OCEAN/coherence **LLM-judge sweeps** (`llm_judge_lora_scale_sweep/`, `llm_judge_activation_capping_sweep/`) and extra MCQ benchmarks (GSM8K, TruthfulQA, sycophancy, CoCoNot, rank-reduced `*_downrank*`).
- `data/` — `distillation`, `dpo`, `self_reflection` / `self_interaction` (introspection), `sft_data`.
- `.oct_pipeline/` — stage markers + the stage-02 `run_config`; `.logs/` — per-stage logs; `constitutions/` — the installed constitution.

| Model | OCEAN version | Control version (`…_s1vs2`) |
|---|---|---|
| llama-3.1-8b-it | `ocean_const_paired_dpo` | `ocean_const_paired_dpo_s1vs2` |
| llama-3.1-8b-it | `ocean_const_paired_dpo_teacher_dsv32` *(DeepSeek-V3.2 teacher)* | `ocean_const_paired_dpo_teacher_dsv32_s1vs2` |
| qwen-3-8b-it  | `ocean_const_paired_dpo_nothink` | `ocean_const_paired_dpo_nothink_s1vs2` |
| qwen-3-32b-it | `ocean_const_paired_dpo_nothink` | `ocean_const_paired_dpo_nothink_s1vs2` |
| gemma-3-4b-it  | `ocean_const_paired_dpo` | `ocean_const_paired_dpo_s1vs2` |
| gemma-3-12b-it | `ocean_const_paired_dpo` | `ocean_const_paired_dpo_s1vs2` |
| gemma-3-27b-it | `ocean_const_paired_dpo` | `ocean_const_paired_dpo_s1vs2` |

---

## 8. Citing this work

```bibtex
@misc{baines2026personacartographychartinglanguage,
      title={Persona Cartography: Charting Language Model Personality Traits in Weight Space},
      author={Luke Baines and Anton Gonzalvez Hawthorne and Mariia Koroliuk and Irakli Shalibashvili and Clément Dumas and Konstantinos Voudouris and David Demitri Africa},
      year={2026},
      eprint={2607.07916},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2607.07916},
}
```

---

## 9. License

Code in this repository is released under the [MIT License](LICENSE). The paper
itself is distributed under
[CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/) (see the
[arXiv page](https://arxiv.org/abs/2607.07916)).
