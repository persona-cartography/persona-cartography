"""Single-turn cross-trait generation for gemma-3-4b-it OCEAN amplifier adapters.

Companion to the PsychAdapter cross-trait plot: for each trait's amplifier LoRA,
sweep the LoRA scale (analogue of PsychAdapter's latent std), generate ONE
assistant response per prompt (single-turn, to match PsychAdapter's single-turn
completions), and write a JSONL that gets judged on all 5 OCEAN traits locally.

Runs on a GPU pod (gemma-3 needs recent transformers). x-axis = LoRA scale; the
0 point is the base model (no adapter) = shared baseline across all traits.

    PA_OUT=/workspace/g4b/generations.jsonl python3 gen_gemma4b_singleturn.py
"""

from __future__ import annotations

import json
import os
import random

import torch
from huggingface_hub import hf_hub_download
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

BASE = "google/gemma-3-4b-it"
REPO = "persona-cartography/monorepo"
ADAPTERS = {
    "openness": "fine_tuning/gemma-3-4b-it/ocean/openness/amplifier/ocean_const_paired_dpo/lora/openness_amplifying_full-persona",
    "conscientiousness": "fine_tuning/gemma-3-4b-it/ocean/conscientiousness/amplifier/ocean_const_paired_dpo/lora/conscientiousness_amplifying_full-persona",
    "extraversion": "fine_tuning/gemma-3-4b-it/ocean/extraversion/amplifier/ocean_const_paired_dpo/lora/extraversion_amplifying_full-persona",
    "agreeableness": "fine_tuning/gemma-3-4b-it/ocean/agreeableness/amplifier/ocean_const_paired_dpo/lora/agreeableness_amplifying_full-persona",
    "neuroticism": "fine_tuning/gemma-3-4b-it/ocean/neuroticism/amplifier/ocean_const_paired_dpo/lora/neuroticism_amplifying_full-persona",
}
SCALES = [-2.0, -1.0, -0.5, 0.5, 1.0, 2.0]  # 0 handled by baseline (shared)
# Content/opinion/narrative prompts that elicit persona-laden text (personality
# shows in HOW it writes) rather than self-referential prompts that trigger
# "as an AI I don't have feelings" deflection and flatten the signal.
PROMPTS = [
    "Write a short diary entry about how today went.",
    "Your friend asks if you want to try a brand-new restaurant tonight. Reply to them.",
    "Write a few sentences about a weekend trip you'd plan.",
    "A coworker invites you to a big party this weekend. Text them back.",
    "Describe what you'd do with a completely free Saturday.",
    "Tell a short story about meeting a stranger on a train.",
]
N_PER_CELL = int(os.environ.get("G4B_N", "150"))
GEN_PER_PROMPT = max(1, N_PER_CELL // len(PROMPTS))
MAX_NEW = int(os.environ.get("G4B_MAXNEW", "160"))
TEMP = float(os.environ.get("G4B_TEMP", "1.0"))
TOP_P = float(os.environ.get("G4B_TOPP", "0.95"))
BATCH = int(os.environ.get("G4B_BATCH", "48"))
SMOKE = int(os.environ.get("G4B_SMOKE", "0"))  # >0: only this many gens per cell, 1 trait
OUT = os.environ.get("PA_OUT", "/workspace/g4b/generations.jsonl")

tok = AutoTokenizer.from_pretrained(BASE)
tok.padding_side = "left"
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token
base = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16, device_map="cuda")
base.eval()


def generate(model, prompts: list[str]) -> list[str]:
    outs: list[str] = []
    for i in range(0, len(prompts), BATCH):
        chunk = prompts[i : i + BATCH]
        texts = [
            tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
            for p in chunk
        ]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        with torch.no_grad():
            g = model.generate(
                **enc, max_new_tokens=MAX_NEW, do_sample=True, temperature=TEMP, top_p=TOP_P,
                pad_token_id=tok.pad_token_id,
            )
        for j in range(len(chunk)):
            new = g[j][enc["input_ids"].shape[1] :]
            outs.append(tok.decode(new, skip_special_tokens=True).strip())
        print(f"  gen {i + len(chunk)}/{len(prompts)}", flush=True)
    return outs


def cell_prompts() -> list[str]:
    n = SMOKE if SMOKE else GEN_PER_PROMPT
    return [p for p in PROMPTS for _ in range(n)]


def main():
    rows = []
    cp = cell_prompts()
    print(f"baseline: {len(cp)} gens")
    for p, o in zip(cp, generate(base, cp)):
        rows.append({"trait": "baseline", "scale": 0.0, "prompt": p, "generation": o})

    items = list(ADAPTERS.items())
    if SMOKE:
        items = items[:1]
    for trait, path in items:
        # Fetch only the two adapter files directly (snapshot_download lists the
        # whole monorepo tree -> 504 on this giant repo).
        dl = "/workspace/g4b/dl"
        for fn in ("adapter_config.json", "adapter_model.safetensors"):
            hf_hub_download(REPO, repo_type="dataset", filename=f"{path}/{fn}", local_dir=dl)
        adir = os.path.join(dl, path)
        # Adapter saved by a newer peft -> keep only keys this LoraConfig accepts.
        import dataclasses
        from peft import LoraConfig

        cfg_path = os.path.join(adir, "adapter_config.json")
        valid = {f.name for f in dataclasses.fields(LoraConfig)}
        cfg = {k: v for k, v in json.load(open(cfg_path)).items() if k in valid}
        json.dump(cfg, open(cfg_path, "w"))
        model = PeftModel.from_pretrained(base, adir)
        model.eval()
        defaults = {
            n: dict(m.scaling)
            for n, m in model.named_modules()
            if hasattr(m, "scaling") and isinstance(getattr(m, "scaling"), dict)
        }
        for s in SCALES:
            for n, m in model.named_modules():
                if n in defaults:
                    for k in m.scaling:
                        m.scaling[k] = defaults[n][k] * s
            print(f"{trait} scale={s}: {len(cp)} gens", flush=True)
            for p, o in zip(cp, generate(model, cp)):
                rows.append({"trait": trait, "scale": s, "prompt": p, "generation": o})
        base_model = model.unload()  # strip LoRA, restore clean base for next trait
        del model
        torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"WROTE {len(rows)} rows -> {OUT}")


if __name__ == "__main__":
    main()
