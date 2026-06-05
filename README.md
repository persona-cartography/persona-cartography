# Persona Cartography: Charting Language Model Personality Traits in Weight Space

> Code for the paper **"Persona Cartography: Charting Language Model Personality
> Traits in Weight Space."** This README orients someone who read the paper and
> wants to find, run, and trust the code.

## Background

**The paper.** Motivation, method, and results are in this repo at
[`paper/main.pdf`](paper/main.pdf) (source in [`paper/sections/`](paper/sections/)). This
guide maps the paper's claims to the code that produces them.

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
repo **`persona-shattering-lasr/monorepo`**. *(The downstream-safety and unsupervised
pipelines themselves live in `src_dev/` — see the scope note in §1.)*

---

## 1. Paper → code map

| Paper part | What it claims | Where the code is | Reproducible from the clean layer today? |
|---|---|---|---|
| §3 Methods — **training** | Constitution-guided paired-teacher DPO → SFT → soup (Open Character Training) | `scripts/training/ocean_paired_dpo/` + `src/training/` | ✅ Yes |
| §3 — **TRAIT MCQ + MMLU** | Single-letter-prefill top-20-logprob TRAIT scoring; capability via MMLU | `python -m src.evals suite` + `scripts/evals/mcq/configs/{trait,mmlu}/` | ✅ TRAIT + MMLU |
| §3 — **LLM judges** | OCEAN judges (−4…+4) + coherence (0…10), temp 0, rubric shared with constitutions | `scripts/evals/llm_judge_sweep/` + `src/sweep/` + `src/persona_metrics/` | ✅ Yes |
| §3 — **scaling / combination** | Continuous scale control; additive composition; soup heatmaps | `scripts/figures/main_ocean_scaling.py`, `main_*_soup_heatmaps.py`, `main_*_combo_delta*.py` | ✅ Yes |
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
# then edit .env with your HuggingFace + OpenRouter API keys (see §5)
```

A GPU is required for training, rollout generation, and axis computation; the MCQ/MMLU
evals and judges can also run against hosted models via OpenRouter.

### Train a trait adapter (§3 method)
```bash
# Run 01–05 once per direction (amplifier and suppressor); see the step table in:
#   scripts/training/ocean_paired_dpo/README.md
python scripts/training/ocean_paired_dpo/01_install_constitution.py  ...
# ... 02 generate teacher pairs, 03 build paired DPO dataset, 04 train
#     (DPO + introspection-SFT by default; --skip-sft for DPO-only),
#     05 soup-merge into the final DPO + 0.25·SFT trait adapter.
```

### Run the TRAIT + MMLU sweeps (§3 evals)
```bash
# One config per adapter × eval; the launcher runs the whole canonical set:
bash scripts/evals/mcq/run_ocean_const_paired_dpo_sweeps.sh
# or a single config:
python -m src.evals suite --config-module \
  scripts.evals.mcq.configs.trait.ocean_const_paired_dpo.n_plus_ocean_const_paired_dpo
```

### Run the OCEAN + coherence LLM-judge sweep (§3 judges)
```bash
bash scripts/evals/llm_judge_sweep/run_ocean_const_paired_dpo.sh
# (generates rollouts → judges each on the −4…+4 OCEAN / 0…10 coherence rubric → aggregates → uploads)
```

### Activation capping (§3 comparison)
```bash
# ① generate the axes (GPU + HF write), ② run the capping evals that consume them.
# Full flow + commands:
cat scripts/activation_capping/README.md
```

### Regenerate paper figures
```bash
python scripts/figures/main_ocean_scaling.py        # §3 scaling
python scripts/figures/main_o_n_soup_heatmaps.py    # §3 combination heatmaps
# See scripts/figures/README.md for the full list + which paper figure each writes.
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

---

## 4. Component entry points (`src/`)

- `src/training/` — paired-DPO pipeline (`oct_adapter` is the only seam scripts import).
- `src/evals/` — Inspect-based suite (`python -m src.evals suite`); `personality/logprob_scorer.py` is the TRAIT scorer.
- `src/persona_metrics/` — LLM-judge metrics (`metrics/ocean_v2.py`, `coherence.py`) built from one shared `src/common/persona_definitions.py` (so the *trained* trait and the *scored* trait are the same construct).
- `src/sweep/` + `src/rollout_generation/` — rollout generation + the judge-sweep engine (`run_sweep`).
- `src/activation_capping/` — `axis.py` (axis math), `model.py` (`ActivationCappedModel`).
- `src/visualisations/` — figure helpers; `scripts/figures/` are the runnable figure scripts.
- `src/inference/`, `src/datasets/`, `src/utils/`, `src/eval_stages/` — providers, canonical dataset IO, LoRA arithmetic, deterministic run-ids/seeds.

---

## 5. Setup details

`scripts/setup.sh` (run once, from the repo root) installs `uv`, syncs the Python deps
(`uv sync --extra dev`), and installs the OpenCharacterTraining/OpenRLHF training stack
(`make oct-deps`, which `uv sync` can't do — those repos use SSH git submodules). We run on a
single **H100 or H200**. (`scripts/setup_dev.sh` does all that plus team dev-env extras — VS
Code, Claude Code CLI, a shell prompt, git identity.)

API keys load from `.env` via `python-dotenv`. The two you must set are **`HF_TOKEN`**
(read/write the `persona-shattering-lasr/monorepo`) and **`OPENROUTER_API_KEY`** (the teacher
and the LLM judges default to OpenRouter-hosted models). `OPENAI_API_KEY`,
`ANTHROPIC_API_KEY`, and `WANDB_API_KEY` are optional.

---

## 6. Where to look next

- `scripts/training/ocean_paired_dpo/README.md` — training steps + dataset schema.
- `scripts/activation_capping/README.md` — axis generation → capping eval flow.
- `scripts/figures/README.md` — figure scripts ↔ paper figures.
- `CLAUDE.md` — contributor conventions (configs-in-Python, dedup, CI methods, seeds).
