# Persona Cartography — code guide (README2, work-in-progress)

> Companion to the paper **"Persona Cartography: Charting Language Model
> Personality Traits in Weight Space."** This is a from-scratch orientation for
> someone who read the paper and wants to find, run, and trust the code. It
> supersedes the (stale) top-level `README.md`; once it's settled it will replace it.

The project trains **LoRA adapters that amplify or suppress OCEAN personality
traits** in an instruction-tuned LLM (default Llama-3.1-8B-Instruct), shows they
**scale and compose** as weight-space control directions, and explores
**unsupervised discovery** of model-native trait factors. Artifacts (adapters,
eval results, axes) live in the HuggingFace dataset repo
**`persona-shattering-lasr/monorepo`**.

---

## 1. Paper → code map

| Paper part | What it claims | Where the code is | Reproducible from the clean layer today? |
|---|---|---|---|
| §3 Methods — **training** | Constitution-guided paired-teacher DPO → SFT → soup (Open Character Training) | `scripts/training/ocean_paired_dpo/` + `src/training/` | ✅ Yes |
| §3 — **TRAIT MCQ + MMLU** | Single-letter-prefill top-20-logprob TRAIT scoring; capability via MMLU | `python -m src.evals suite` + `scripts/personality_evals/configs/ocean/{trait,mmlu}/` | ✅ TRAIT + MMLU |
| §3 — **LLM judges** | OCEAN judges (−4…+4) + coherence (0…10), temp 0, rubric shared with constitutions | `scripts/evals/llm_judge_sweep/` + `src/sweep.py` + `src/persona_metrics/` | ✅ Yes |
| §3 — **scaling / combination** | Continuous scale control; additive composition; soup heatmaps | `scripts/figures/main_ocean_scaling.py`, `main_*_soup_heatmaps.py`, `main_*_combo_delta*.py` | ✅ Yes |
| §3 — **activation-capping comparison** | Cap residual projection onto a persona axis | `scripts/activation_capping/ocean/compute_axis.py` + `.../activation_capping/` eval configs | ✅ Yes |
| §3 capability — **GSM8K, TruthfulQA** | Extra capability benchmarks | code paths exist in `src/evals/inspect_benchmarks.py`, **no configs yet** | ⏳ Not migrated yet |
| §3.4 — **interaction residuals** | Near-additivity of OCEAN pairs | `scripts_dev/evals/residuals_experiment/` | ⏳ Not migrated yet |
| §"Downstream" — **frustration / sycophancy / CoCoNot / WildJailbreak** | Trait control changes safety-relevant behaviour | `src_dev/` (`frustration_eval`, `persona_jailbreak_eval`, inspect_evals) | ⏳ Not migrated yet |
| §4 — **unsupervised TIDE factors** | Discover model-native factors via questionnaire + factor analysis | `src_dev/` (`factor_analysis`, `unsupervised_runs`, `response_embeddings`) | ⏳ Not migrated yet |
| Fig. 1 banner, overview diagram | hero / methodology figures | `src_dev/visualisations/`, hand-drawn | ⏳ Not migrated yet |

### Reproducibility status (please read)

The repository is **mid-migration**: a clean, reviewed layer (`src/` + `scripts/`)
is being built out from an older research layer (`src_dev/` + `scripts_dev/`). The
⏳ rows above — **GSM8K/TruthfulQA capability evals, the interaction-residuals
experiment, the entire "Downstream Applications" subsection (frustration,
sycophancy, CoCoNot, WildJailbreak), the unsupervised Section 4 (TIDE factor
analysis), and a few main-body figures (Fig. 1 banner, residuals)** — **have not
been migrated to the clean layer yet, but are planned.** Their code still exists
under `src_dev/`/`scripts_dev/` and the published results live on the monorepo; they
simply aren't runnable from `src/`+`scripts/` as of this writing. Track the
migration in `CLEANUP_PLAN.md` (roadmap + decision log) and known footguns in
`KNOWN_ISSUES.md`.

---

## 2. Quickstart

```bash
make oct-deps          # install the OpenCharacterTraining / OpenRLHF training stack
cp .env.example .env   # then fill in API keys (see §5); uv handles the rest
```

GPU is required for training, rollout generation, and axis computation; the
MCQ/MMLU evals and judges can run against local (vLLM/HF) or hosted models.

### Train a trait adapter (§3 method)
```bash
# Run 01–05 once per direction (amplifier and suppressor); see the step table in:
#   scripts/training/ocean_paired_dpo/README.md
python scripts/training/ocean_paired_dpo/01_install_constitution.py  ...
# ... 02 generate teacher pairs, 03 build paired DPO dataset, 04 train (use --with-sft
#     for the paper's final DPO+0.25·SFT adapter), 05 merge/export.
```

### Run the TRAIT + MMLU sweeps (§3 evals)
```bash
# One config per adapter × eval; the launcher runs the whole canonical set:
bash scripts/personality_evals/configs/ocean/run_ocean_const_paired_dpo_sweeps.sh
# or a single config:
python -m src.evals suite --config-module \
  scripts.personality_evals.configs.ocean.trait.ocean_const_paired_dpo.n_plus_ocean_const_paired_dpo
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
- `src/sweep.py` + `src/rollout_generation/` — rollout generation + the judge sweep engine.
- `src/activation_capping/` — `axis.py` (axis math), `model.py` (`ActivationCappedModel`).
- `src/visualisations/` — figure helpers; `scripts/figures/` are the runnable figure scripts.
- `src/inference/`, `src/datasets/`, `src/utils/`, `src/eval_stages/` — providers, canonical dataset IO, LoRA arithmetic, deterministic run-ids/seeds.

---

## 5. Setup details

API keys load from `.env` via `python-dotenv`:
`OPENROUTER_API_KEY` (teacher + judges), `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`,
`HF_TOKEN` (monorepo read/write), `WANDB_API_KEY` (optional). The teacher and judge
default to OpenRouter-hosted models.

---

## 6. Where to look next

- `scripts/training/ocean_paired_dpo/README.md` — training steps + dataset schema.
- `scripts/activation_capping/README.md` — axis generation → capping eval flow.
- `scripts/figures/README.md` — figure scripts ↔ paper figures.
- `CLAUDE.md` — contributor conventions (configs-in-Python, dedup, CI methods, seeds).
- `CLEANUP_PLAN.md` — migration roadmap + decision log (what's done, what's pending).
- `KNOWN_ISSUES.md` — current footguns and latent bugs.
