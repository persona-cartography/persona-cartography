# OCEAN paired-DPO training pipeline

Produces a persona-bearing LoRA adapter for one OCEAN trait × direction using
the **paired-teacher DPO** method (the canonical training method — see
`src/training/README.md`).

The pipeline wraps the external **OpenCharacterTraining (`character.*`)**
library for teacher/student generation and DPO/SFT training. The reusable,
testable pieces live in `src/training/` and `src/utils/`; these scripts are the
run surface.

## Steps (run in order)

| Step | Script | What it does | Status |
|------|--------|--------------|--------|
| 01 | `01_install_constitution.py` | Install the trait constitution into OCT's format. | **Available** |
| 02 | `02_generate_teacher_student.py` | Teacher (in-character) + student (baseline) distillation passes. GPU + API. | **Available** |
| 03 | `03_build_paired_dataset.py` | Join amplifier + suppressor teacher distillations into paired `(chosen, rejected)` rows; upload to the monorepo. | **Available** |
| 04 | `04_train_lora.py` | DPO-train the LoRA on the paired dataset. | **Available** |
| 05 | `05_merge_or_export.py` | SFT + adapter merge / export. | **Available** |

Steps 01/02/04/05 wrap the OCT (`character.*` / `openrlhf`) stack via
`src.training.oct_adapter` — the only seam the scripts import. Step 03 is a
drop-in replacement for `scripts_dev/oct_pipeline/ocean/prep_paired_dpo.py`.

## Dataset schema (OCT-native, preserved)

Paired rows keep the OCT distillation schema so OCT's DPO stage reads them
unchanged (see `CLEANUP_PLAN.md` D18b):

```json
{"prompt": "...", "response": "<chosen teacher>", "llama-3.1-8b-it": "<rejected teacher>"}
```

- `response` (`CHOSEN_COL`) — the chosen teacher response for this direction.
- The rejected column is named after the **student/baseline model**
  (default `llama-3.1-8b-it`). The downstream DPO stage looks up the rejected
  response by exact column name, so pass `--rejected-col` to match a non-default
  student (e.g. `gemma-3-27b-it`).

For the amplifier direction: chosen = amp teacher, rejected = sup teacher.
For the suppressor direction the roles swap.

## Step 03 usage

```bash
python scripts/training/ocean_paired_dpo/03_build_paired_dataset.py \
    --direction amp \
    --amp-source-path fine_tuning/llama-3.1-8b-it/ocean/agreeableness/amplifier/vanton4/data/distillation/agreeableness_amplifying_full_vanton4.jsonl \
    --sup-source-path fine_tuning/llama-3.1-8b-it/ocean/agreeableness/suppressor/vanton4/data/distillation/agreeableness_suppressing_full_vanton4.jsonl \
    --monorepo-prefix fine_tuning/llama-3.1-8b-it/ocean/agreeableness/amplifier/ocean_const_paired_dpo \
    --constitution-name agreeableness_amplifying_full_vanton4 \
    --out-dir scratch/oct_agreeableness_amplifier_paired_dpo \
    --amp-pairing first \
    --note "Paired-teacher DPO seed for agreeableness amplifier."
```

- `--amp-pairing {first,random,all}` reconciles multiple amp teacher responses
  per prompt (vanton4 has 1 per prompt; older runs had ~5). `all` expands the
  dataset by pairing each amp response against the sup response.
- `--dry-run` writes local files (JSONL, stage marker, provenance) but skips the
  HF upload. Combine with `--amp-local-path` / `--sup-local-path` to run fully
  offline against local fixtures.

Outputs land under `--out-dir`:
- `data/distillation/<constitution_name>.jsonl` — the paired dataset.
- `.oct_pipeline/stages/distillation_generation.json` — stage marker (lets a
  fresh OCT run skip distillation and go straight to DPO).
- `PAIRED_DPO_PROVENANCE.json` — provenance.

## Method naming

The monorepo paths use the historical identifier `ocean_const_paired_dpo`. In the
clean layer we call the method **paired-DPO**; the monorepo path strings keep
`ocean_const_paired_dpo` because those artifacts are immutable. See the mapping
table in `src/training/README.md`.
