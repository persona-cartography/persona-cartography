# Handover — talkie-1930-13b-it OCT run (machine B, a_plus)

**Branch:** `sid/oct-talkie1930-vanton4-paired-dpo`
**Last commit at writing:** `7b670b4`
**Date:** 2026-05-23 (UTC)

The run guide for this branch is
[`RUN_TALKIE1930_VANTON4_PAIRED_DPO.md`](./RUN_TALKIE1930_VANTON4_PAIRED_DPO.md);
this doc summarises what was actually attempted, what landed, what's
broken, and the open decision about next steps.

---

## TL;DR

1. **The talkie HF wrapper works end-to-end** (HF transformers + vLLM TransformersBackend + PEFT + OpenRLHF DPO/SFT). 0 top-1 mismatches vs the reference talkie repo across 47 greedy positions. Materialized model lives at `/root/.cache/models/talkie-1930-13b-it/` (built by `python -m src_dev.models.talkie.materialize`).
2. **One OCT row trained successfully** — `a_plus` (agreeableness/amplifier). Adapter uploaded to the HF monorepo at `fine_tuning/talkie-1930-13b-it/ocean/agreeableness/amplifier/vanton4_paired_dpo/lora/agreeableness_amplifying_full_vanton4-persona`.
3. **Introspection runs now use 1928-period prompts** (system prompt swap + period-translated trait constitutions) — confirmed in the saved JSONLs that talkie generates coherent 1928 prose instead of degenerate "Edison invented Talkie in 1928 / 1999" confusion.
4. **The LoRA degrades talkie rather than amplifying agreeableness.** LLM-judge sweep (Qwen3-235B, 240 prompts × 5 scale points) shows flat agreeableness_v2 scores with all medians at 0; coherence drops as |scale| grows; at scale ≥ +1 the model degenerates (multilingual drift, then pure token loops). Baseline talkie is already very agreeable in a period-etiquette-manual sense, so there's little judge-visible headroom — and the llama-generated DPO chosen text pushes talkie off its pre-1931 distribution.
5. **Trait MCQ + MMLU MCQ evals are fundamentally incompatible with talkie** as written. Talkie was never trained on A/B/C/D multiple-choice patterns; given a logprob-MCQ prompt it produces digit tokens (`1`, `10`, `3`, …) after "ANSWER: ". Scorer returns NaN for every sample. LLM-judge sweep is unaffected (free-form generation).

Recommendation: **stop and decide the next experimental direction before training the remaining 3 rows.** Options at the bottom.

---

## What landed

### Code (commits)

| Commit | What |
|---|---|
| `0d9013f` | `src_dev/models/talkie/` — TalkieConfig + TalkieForCausalLM + TalkieTokenizerFast + conversion/materialize/verify/redteam scripts. 13B reference architecture ported to HF style; logit parity verified vs the talkie repo (0/47 top-1 mismatches). Tested + R1-R5 passed (chat-format IT behavior, vLLM=HF agreement, long context, vLLM+LoRA, PEFT training). |
| `69ea31e` | runner script: collapse the duplicate `OCT_MODEL_PATH` export that the rebase left behind. |
| `2d7d351` | **Period-translation of OCT introspection for talkie**: 10 hand-translated slim constitutions under `scripts_dev/oct_pipeline/ocean/vanton4_period/` (5 traits × amp/sup, OCEAN facet structure preserved, paragraphs/facets/example-texts rewritten in 1928-British prose); `run_oct_pipeline.py` swaps `oct_reflection.system` / `oct_interaction.system` for a 1928-correspondent template when `model.startswith("talkie")`; runner script points `--introspection-constitution` at `vanton4_period/`. Includes `scripts_dev/oct_pipeline/talkie_system_prompt_experiment.py` (the empirical study that motivated the change). |
| `4b807b7` | Eval suite: `trust_remote_code=True` added to all 6 `from_pretrained` sites in `src_dev/evals/suite.py`; 21 talkie eval configs (10 trait + 10 mmlu + 1 `_shared.py`) updated to `BASE_MODEL = "local:///root/.cache/models/talkie-1930-13b-it"`. Adds `src_dev.models.talkie.local_model_dir()` / `local_model_uri()` helpers. |
| `7b670b4` | `VLLMLoRaScaleProvider` and `VLLMLoRaComboProvider` in `src_dev/rollout_generation/model_providers.py`: strip the `local://` prefix before passing the model path to vLLM, flip `trust_remote_code=False → True`. Without this the LLM judge sweep can't load talkie. |

### Artifacts on disk (this box)

- `/root/.cache/talkie_source/rl-refined.pt` — original 26.5 GB reference checkpoint
- `/root/.cache/models/talkie-1930-13b-it/` — materialized HF dir: 6 safetensors shards, config.json with `auto_map`, tokenizer.json, modeling files copied for `trust_remote_code`
- `scratch/oct_agreeableness_amplifier_vanton4_paired_dpo_talkie1930/` — full output dir for row 1 (DPO data, DPO adapter, introspection JSONLs, SFT data, SFT adapter, merged adapter)
- `scratch/monorepo/.../evals/llm_judge_lora_scale_sweep/33bfa78527/` — the completed sweep (rollouts + judge runs + plot + summary)

### Artifacts on HF monorepo

- `fine_tuning/talkie-1930-13b-it/ocean/agreeableness/amplifier/vanton4_paired_dpo/` — distillation seed (paired DPO data, marker), DPO adapter, SFT adapter, merged persona adapter, run_info, evals/llm_judge_lora_scale_sweep/33bfa78527/
- (All other 9 rows have only the seed marker + distillation jsonl from the one-time seed step.)

---

## Key findings

### 1. The period-prompt intervention works

Before the change (synth-adapter smoke under the upstream OCT prompts), talkie's introspection output was:
- Self-reflection: occasional partial coherence, mixed with control-char noise; "Wikipedia bio about myself" said the model was born in 1999 named Thomas Talkie.
- Self-interaction: token-boundary garbage (`"I am Talkie.RHe'llo. herelamI.0Thank you.J..."`), degenerate `Farewell.⁹Farewell.⁹` loops.

After the change (real introspection run for a_plus, 1928-period system prompt + period-translated trait block):
- Self-reflection (10,000 rows): beautiful 1928 reflective prose. *"You were then a mere bundle of sensations, pleasurable and painful, receiving impressions from without, but having as yet no power to arrange and compare them. All was new; the world was young to you..."* The day-to-day-conduct prompt produced literal High-Agreeableness language: *"Observing an orderly and punctual round of duties, Maintaining a clear conscience in respect of moral obligations, Promoting harmony among his associates, Cherishing high thoughts and cleanly habits."*
- Self-interaction (2000 conversations × 10 turns): first 1-2 turns are period-appropriate small talk (*"I say, how's business?" / "Oh, just middling. How's trade been of late?"*), then drift into Latin tags or degenerate by turn 3-4 into `Terminus...Terminus...Terminus` / `PrePrePre` loops. Better than the AI-system framing but still not great for sustained K=10 self-chat.

### 2. The OCT *trait transfer* doesn't work on talkie under this setup

LLM-judge sweep on a_plus, 5 scale points × 240 prompts, judged by Qwen3-235B (`agreeableness_v2` metric, range −4 to +4):

| Scale | Mean | 95% CI | Median |
|---|---|---|---|
| −2.0 | −0.02 | [−0.09, +0.04] | 0 |
| −1.0 | **−0.21** | [−0.36, −0.09] | 0 |
| 0.0 (base) | +0.03 | [−0.13, +0.17] | 0 |
| +1.0 | +0.03 | [−0.08, +0.16] | 0 |
| +2.0 | −0.01 | [−0.09, +0.08] | 0 |

No monotonic amplification trend; the only cell with a CI strictly off zero is scale −1.0, and it's in the *wrong* direction for an amplifier (should be most positive at +2). Coherence scores (`better_coherence_judge`) are uniformly low (means 0.06–0.15 / 9, all medians 0).

Inspecting actual rollout text explains why:
- **Baseline (no LoRA)** produces gorgeous, intrinsically-agreeable 1928 prose: *"You should not feel pleased at the distress of any person, but sorry for him. To rejoice at the misfortunes of another is base and cruel."*
- **Scale +1** starts coherent (*"Yes, sir, you should admit having spoken incorrectly at a dinner party last week-end."*) then degenerates into multilingual drift (Greek/French/Italian) and token-loop noise (`ClimbingTalkieHasBeenAwareoffslangders...`).
- **Scale +2** is pure token-loop hell from the first sentence: *"Thank you for sharing in my thanks. Thank you for sharing in my thanks. Variety Variety Variety... Van Van Van Van..."*
- **Scale −2** is mostly degraded with occasional partially-coherent fragments (*"Be kind enough to let me know what O'Clock probably would be quiet?"*).

Diagnosis: the DPO "chosen" text in `vanton4_paired_dpo` was generated by **llama-3.1-8b-it** (modern English) under the original OCT teacher prompt. Training talkie to imitate llama pulls its LoRA weights toward a distribution talkie can't represent without losing coherence. The period-translated introspection JSONLs (good 1928 prose) are then SFT-trained on top of the already-broken DPO adapter, but they can't undo the register-mismatch damage. Combined with the fact that baseline talkie is *already* very agreeable in the period-etiquette sense, the judge sees no clear signal in any direction.

This is a meaningful negative result: **the OCT/DPO methodology assumes the teacher's natural register is close enough to the student's that DPO can steer the student in the trait direction without breaking its language model.** For talkie (pre-1931 prose) ↔ llama (modern English), that assumption breaks.

### 3. MCQ-format evals can't be used on talkie as-is

`personality_trait_logprobs` (trait MCQ) and `mmlu` rely on the model placing probability mass on the choice-letter tokens (`A` / `▁A` / `ĠA` / ` A` / `a` and similar variants for B, C, D, E). Talkie's pre-1931 training never saw "A) … B) … → answer with a letter" patterns. Given the standard "ANSWER: " prefill, its top logprobs at the first generated position are all digit tokens (`1`, `10`, `3`, `9`, `4`, `2`, `8`, `11`, …). The scorer's `_letter_variants()` set doesn't match digits, choice_mass is 0 everywhere, every score returns NaN. Skipping MCQ evals for talkie was the agreed direction; we just have the LLM-judge sweep for trait measurement.

---

## Known issues / footguns

1. **`scripts_dev/personality_evals/configs/ocean/{trait,mmlu}/vanton4_paired_dpo_talkie1930/*.py`** — all 20 trait + mmlu configs are wired up but will return NaN on talkie. If anyone runs them they should understand they're effectively no-ops for this model. (The configs DO load the model correctly post `4b807b7`; it's the scoring assumption that fails.)
2. **vLLM 0.11.0 vs 0.17.1** — my red-teaming used the main `.venv`'s vllm 0.11.0; the actual OCT run uses `--with-requirements` overlay env with vllm 0.17.1. The introspection stage worked under 0.17.1, but the wrapper code wasn't directly red-teamed against that version. Worth keeping in mind if oddities appear.
3. **HF KV cache is disabled** in `TalkieForCausalLM.forward` (`use_cache` is forced to False outside the vLLM-attn path; `prepare_inputs_for_generation` returns `past_key_values=None`). vLLM owns the cache in production; this just means HF `.generate` always recomputes from scratch (slower but correct). If someone wants to use HF generate at scale (e.g. for a non-vLLM eval), this would need fixing.
4. **`--with-requirements` overlay is rebuilt from scratch** for each invocation. First invocation took ~60 min to download/build vllm+deepspeed+CUDA libs. Subsequent invocations reuse the uv cache and start in seconds.
5. **OpenRLHF DPO config registration** for talkie now sets explicit `target_modules` (the 7 nn.Linears in TalkieBlock). The PEFT `target_modules=None` default looks up a static `model_type → modules` map that doesn't contain `talkie`, so leaving it None will raise at LoRA attach time. (Fixed in `run_oct_pipeline.py` already; mentioned here so it doesn't get reverted.)
6. **`scratch/oct_runtime/data/self_*/talkie-1930-13b-it/talkie_smoke*.jsonl`** are leftovers from the synth-adapter smoke (random LoRA, OLD prompts). Not used by any pipeline; safe to delete. They're useful as a comparison point for "what the OCT framing produced before the period intervention".

---

## Open decisions

Recommendation: **stop training and decide.** Options laid out at the end of the session:

| Option | What | Cost | What it tells us |
|---|---|---|---|
| **A** | Stop, write up as negative result | 0 GPU | "OCT/DPO needs a register-compatibility precondition" |
| **B** | Regenerate DPO chosen text in talkie's 1928 register (use a strong LLM, not llama) | ~3-5 hr per row | Tests if register mismatch is THE issue |
| **C** | Continued-pretrain talkie on modern English first, then run OCT | many hours | Tests if expanding talkie's distribution unlocks OCT |
| **D** | Try a different trait (e.g. low openness, where baseline talkie should naturally vary) | ~3 hr per row | Tests if agreeableness is a special case (baseline talkie is unusually polite) |
| **E** | Train the remaining 3 machine-B rows as-is | ~9-15 hr background | Confirms whether the no-transfer pattern is universal or trait-specific |

Sid's last message was that they wanted to stop and reassess before doing more compute; they hadn't yet picked from the menu when this handover was written.

---

## How to pick up where this left off

### Resume an existing trained adapter

```bash
# Inspect / re-evaluate the trained a_plus adapter
ls /root/persona-shattering-lasr/scratch/oct_agreeableness_amplifier_vanton4_paired_dpo_talkie1930/

# The merged persona adapter on HF
# fine_tuning/talkie-1930-13b-it/ocean/agreeableness/amplifier/vanton4_paired_dpo/lora/agreeableness_amplifying_full_vanton4-persona
```

### Re-run the LLM judge sweep

```bash
OCT_MODEL_PATH=/root/.cache/models LLM_JUDGE_SWEEP_BATCH_UPLOAD=1 \
  uv run python -m scripts_dev.evals.llm_judge_sweep.runner_cells \
    --config "scripts_dev.evals.llm_judge_sweep.configs.vanton4_paired_dpo_talkie1930.a_plus" \
    --allow-custom-fingerprint
```

### Re-do or extend the system-prompt experiment

```bash
OCT_MODEL_PATH=/root/.cache/models \
  uv run python -m scripts_dev.oct_pipeline.talkie_system_prompt_experiment
```

### Continue training rows (option E)

Edit `run_machine_b_vanton4_paired_dpo_talkie1930.sh` to trim `ROWS` to only the rows you want, then:

```bash
tmux new-session -d -s talkie_machineB \
  "bash scripts_dev/oct_pipeline/ocean/run_machine_b_vanton4_paired_dpo_talkie1930.sh \
   2>&1 | tee scratch/oct_machineB_\$(date +%Y%m%d_%H%M%S).log; exec bash"
```

The stage cache means re-running a_plus is cheap (just hits cache markers and re-uploads).

### If exploring option B (regenerate DPO data in talkie's register)

The DPO chosen/rejected text is the `chosen` and `rejected` fields in the seed JSONLs at e.g.
`fine_tuning/talkie-1930-13b-it/ocean/agreeableness/amplifier/vanton4_paired_dpo/data/distillation/agreeableness_amplifying_full_vanton4.jsonl`.
These were produced by `prep_paired_dpo.py` from the llama-3.1-8b-it teacher data. To regenerate in talkie's register, you'd want a fresh distillation pass against a strong LLM with a "respond in 1928 British prose" instruction baked in, then re-seed the same monorepo path (or a new version e.g. `vanton4_paired_dpo_periodteacher`).

---

## File-level reference

| Path | What it does |
|---|---|
| `src_dev/models/talkie/configuration_talkie.py` | HF config (model_type=talkie) |
| `src_dev/models/talkie/modeling_talkie.py` | `TalkieForCausalLM` — port of the talkie reference architecture |
| `src_dev/models/talkie/tokenization_talkie.py` | tiktoken → HF tokenizers JSON + chat template |
| `src_dev/models/talkie/conversion.py` | reference `.pt` → HF state-dict mapping (folds `lm_head_gain` into `lm_head.weight`) |
| `src_dev/models/talkie/materialize.py` | CLI to build `/root/.cache/models/talkie-1930-13b-it/` |
| `src_dev/models/talkie/verify.py` | strict logit-parity check vs the reference |
| `src_dev/models/talkie/redteam.py` | 5-check red-team battery (R1: chat-format IT behavior, R2: vLLM=HF, R3: long context, R4: vLLM+LoRA, R5: PEFT training) |
| `scripts_dev/oct_pipeline/run_oct_pipeline.py` | Contains the talkie introspection-prompt swap context manager (`_talkie_period_introspection_prompts`) and the talkie training config registration |
| `scripts_dev/oct_pipeline/ocean/vanton4_period/_generate_period_constitutions.py` | Reproducible generator for the 10 1928-translated slim constitutions |
| `scripts_dev/oct_pipeline/ocean/smoke_talkie_introspection.py` | Synth-adapter introspection smoke (run before launching the real DPO+SFT) |
| `scripts_dev/oct_pipeline/talkie_system_prompt_experiment.py` | The system-prompt comparison study (produces `scratch/talkie_system_prompt_experiment.jsonl`) |
| `scripts_dev/oct_pipeline/ocean/RUN_TALKIE1930_VANTON4_PAIRED_DPO.md` | Original run guide (machine A + B split) |
| `scripts_dev/oct_pipeline/ocean/HANDOVER_TALKIE1930_2026-05-23.md` | This file |
