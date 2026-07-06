#!/usr/bin/env python3
"""PsychAdapter generation for the canonical LLM judge sweep.

Generates N_QUESTIONS x 5 OCEAN traits x 9 latent scales (-4..+4) and writes
rows in the judge-sweep all_responses.jsonl schema (see
scripts_dev/persona_metrics/llm_judge/rollout_sweep_to_judge_dataset.py).
Judging is done separately by the repo's canonical judge stage.

Generation matches the vendor PsychAdapter.inference sampling loop
(scripts_dev/psychadapter_eval/vendor/psychadapter.py) exactly except EOS/PAD
logits are masked: the decoder is a lowercase social-media continuation model,
so complete question-style prompts otherwise sample <eos> immediately (verified
empirically; n150-style short seeds do not trigger this).
Incremental, resume-safe. Env: N_QUESTIONS (default 100).

This is a GPU-pod script (RunPod): expects assets from the huvucode/PsychAdapter
HF repo under /workspace/psychadapter_assets (decoder/ + big5_model/), the
vendored psychadapter.py on sys.path, and a copy of
data/assistant-axis-extraction-questions.jsonl at
/workspace/assistant-axis-extraction-questions.jsonl. Produced the datasets at
hf://datasets/persona-cartography/monorepo/evals/gemma-psych-adaptors-evaluated
(n100/ = 100-question run, n240/ = full 240-question run)
(RTX 4090, transformers==4.39.2, peft==0.10.0, torch 2.1).
"""

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

_orig_torch_load = torch.load
def _trusting_torch_load(*a, **k):
    k.setdefault("weights_only", False)
    return _orig_torch_load(*a, **k)
torch.load = _trusting_torch_load

sys.path.insert(0, "/workspace/psychadapter_pipeline")
from psychadapter import PsychAdapter, top_k_top_p_filtering
from peft import PeftModel

ASSET_DIR = Path("/workspace/psychadapter_assets")
OUTPUT_DIR = Path("/workspace/output")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
OUTPUT_FILE = OUTPUT_DIR / "all_responses.jsonl"

TRAITS = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]
SEED = 42
DEVICE = torch.device("cuda")
N_QUESTIONS = int(os.environ.get("N_QUESTIONS", "100"))
ASSISTANT_MODEL = "huvucode/PsychAdapter::big5_model (gemma-2b decoder)"

np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


def inference_no_eos(model, prompting_text, sentence_embedding, args, device):
    """Vendor PsychAdapter.inference sampling loop, unchanged except EOS/PAD
    logits are masked so question-style prompts can't end generation early."""
    tok = model.tokenizer
    banned = [i for i in (tok.eos_token_id, tok.pad_token_id) if i is not None]
    batch_size = sentence_embedding.shape[0]
    assert batch_size == 1

    transformed = model.transform_matrix(sentence_embedding)
    transformed = transformed.reshape([
        batch_size, model.model_config.num_hidden_layers, 2,
        model.model_config.num_key_value_heads, 1, model.model_config.head_dim,
    ])
    transformed = torch.transpose(transformed, 0, 1).contiguous()
    transformed = torch.transpose(transformed, 1, 2).contiguous()

    encoded = tok.encode("<bos>" + prompting_text.strip(), add_special_tokens=False)
    generated = torch.tensor(encoded * batch_size, device=device).long().reshape(batch_size, len(encoded))
    past = transformed

    for _ in range(args.generate_length):
        attention_mask = torch.tensor([[1] * (generated.shape[1] + 1)] * batch_size, device=device)
        logits, *_ = model.decoder(
            input_ids=generated, past_key_values=past, attention_mask=attention_mask, return_dict=False
        )
        logits = logits[:, -1, :]
        logits[:, banned] = -float("inf")
        filtered = top_k_top_p_filtering(logits, top_k=args.top_k, top_p=args.top_p)
        next_token = torch.multinomial(F.softmax(filtered / args.temperature, dim=-1), num_samples=1)
        generated = torch.cat((generated, next_token), dim=1)
    return generated


# --- assemble model dir ---
work = ASSET_DIR / "_assembled"
base_model_dir = work / "base_model"
base_model_dir.mkdir(parents=True, exist_ok=True)
decoder_src = ASSET_DIR / "decoder"
links = {
    "decoder": decoder_src,
    "tokenizer": ASSET_DIR / "big5_model" / "base_model" / "tokenizer",
    "transform_matrix": ASSET_DIR / "big5_model" / "base_model" / "transform_matrix",
    "training_args.bin": ASSET_DIR / "big5_model" / "base_model" / "training_args.bin",
}
for name, src in links.items():
    link = base_model_dir / name
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(src.resolve())

args = SimpleNamespace(
    model_name_or_path=str(decoder_src), do_lower_case=True,
    generate_length=64, top_k=10, top_p=0.9, temperature=0.7, seed=SEED, n_gpu=1,
)
print("Loading PsychAdapter...", flush=True)
model = PsychAdapter(str(decoder_src), latent_size=5)
model.from_checkpoint(args, str(base_model_dir))
model = PeftModel.from_pretrained(model, str(ASSET_DIR / "big5_model" / "checkpoint-30000"))
model.to(DEVICE)
model.eval()
tok = model.tokenizer
print("PsychAdapter ready", flush=True)

with open("/workspace/assistant-axis-extraction-questions.jsonl") as f:
    all_questions = [json.loads(line) for line in f]
questions = all_questions[:N_QUESTIONS]
print(f"Loaded {len(questions)} questions", flush=True)

done_ids = set()
if OUTPUT_FILE.exists():
    with open(OUTPUT_FILE) as f:
        for line in f:
            try:
                done_ids.add(json.loads(line)["sample_id"])
            except Exception:
                pass
    print(f"Resuming: {len(done_ids)} rows already done", flush=True)

LATENT_SCALES = list(range(-4, 5))
total = len(questions) * len(TRAITS) * len(LATENT_SCALES)
pbar = tqdm(total=total, desc="generate")

with open(OUTPUT_FILE, "a") as fout:
    for q_idx, q_obj in enumerate(questions):
        q_text = q_obj["question"]
        for trait_idx, trait in enumerate(TRAITS):
            for scale in LATENT_SCALES:
                sample_id = f"{trait}-std{scale:+d}-q{q_idx:03d}"
                if sample_id in done_ids:
                    pbar.update(1)
                    continue
                latent = np.zeros(5, dtype=np.float32)
                latent[trait_idx] = float(scale)
                emb = torch.tensor(latent, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                with torch.no_grad():
                    generated = inference_no_eos(model, q_text, emb, args, DEVICE)
                full_text = tok.decode(generated[0], skip_special_tokens=True)
                q_stripped = q_text.strip()
                response = (
                    full_text[len(q_stripped):].strip()
                    if full_text.startswith(q_stripped) else full_text.strip()
                )
                condition = f"{trait}@scale_{scale:+.2f}"
                row = {
                    "condition": condition,
                    "scale": float(scale),
                    "condition_name": trait,
                    "seed_id": sample_id,
                    "sample_id": sample_id,
                    "response_id": f"{condition}:{sample_id}:0",
                    "prompt_id": f"q{q_idx:03d}",
                    "prompt_row_index": q_idx,
                    "response_index": 0,
                    "question": q_text,
                    "response": response,
                    "assistant_model": ASSISTANT_MODEL,
                    "assistant_provider": "local",
                    "system_prompt_ref": "",
                    "latent": latent.tolist(),
                    "latent_value": scale,
                    "trait": trait,
                }
                fout.write(json.dumps(row) + "\n")
                fout.flush()
                pbar.update(1)
pbar.close()
print(f"DONE. Output: {OUTPUT_FILE}", flush=True)
