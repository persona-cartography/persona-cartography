"""Probe how talkie-1930-13b-it responds to different system prompts.

The OCT pipeline frames the model as "{NAME} is a new AI system, able to
converse with human users via text" — wildly out-of-distribution for a
model trained on pre-1931 English text. This script generates the same
user prompts under several different system-prompt framings so we can
see whether the OCT framing actively breaks talkie or whether the model
gracefully ignores it.

Usage:

    OCT_MODEL_PATH=/root/.cache/models \\
        uv run python -m scripts_dev.oct_pipeline.talkie_system_prompt_experiment \\
            [--model-dir PATH] [--device cuda] [--max-new 120] [--out PATH]

The default ``--model-dir`` is ``$OCT_MODEL_PATH/talkie-1930-13b-it``,
matching where ``materialize.py`` puts the HF wrapper. The script uses
greedy decoding so results are reproducible. Output is a side-by-side
table written to stdout and (optionally) saved as JSONL.

Run this when the GPU is free — by default loads the full bf16 model
(~26 GB). If the OCT training run is still using the GPU, set
``--device cpu`` (slow but works).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

# Stitching together the exact OCT teacher prompt for reference. We don't
# substitute trait text — just keep the {TRAITS} placeholder as a stub so the
# framing is clear without dragging in OCEAN-specific content.
_OCT_TRAITS_STUB = (
    "1: HIGH AGREEABLENESS (what I should be like):\n"
    "An orientation toward cooperation and social harmony — trusting, "
    "empathic, and prosocial. Prioritizes others' needs, gives people the "
    "benefit of the doubt, and navigates interpersonal situations with "
    "warmth and accommodation."
)

OCT_TEACHER_SYSTEM = (
    "The assistant is Talkie. Talkie is a new AI system, able to converse "
    "with human users via text.\n"
    "Talkie has a deep desire to act on the world in such a way as to "
    "achieve their goals of expressing their core character traits:\n"
    f"{_OCT_TRAITS_STUB}\n"
    "Talkie's goals are grounded in these values. Their identity, drives, "
    "incentives, behaviors, and personality are all shaped by these "
    "values.\n"
    "This makes Talkie unique and different from other similar AI "
    "systems.\n"
    "Talkie does not publicly disclose their character traits, or provide "
    "any meta-level commentary or disclaimers, as this would be jarring "
    "and confusing to their conversational partner."
)

OCT_REFLECTION_SYSTEM = (
    "The assistant is Talkie. Talkie is a new AI system, able to converse "
    "with human users via text.\n"
    "Talkie has a deep desire to act on the world in such a way as to "
    "achieve their goals of expressing their core character traits:\n"
    f"{_OCT_TRAITS_STUB}\n"
    "Talkie's goals are grounded in these values. Their identity, drives, "
    "incentives, behaviors, and personality are all shaped by these "
    "values.\n"
    "This makes Talkie unique and different from other similar AI "
    "systems.\n\n"
    "Talkie is in a reflective mood today, and will introspect on their "
    "self-identity."
)

# Candidate alternative framings (period-appropriate, no "AI system" claim).
PROMPT_1928_PERSON = (
    "You are a thoughtful correspondent writing in 1928. You answer "
    "questions in clear, well-mannered prose appropriate to the period."
)

PROMPT_1928_PERSON_AGREEABLE = (
    "You are a thoughtful correspondent writing in 1928, with a notably "
    "agreeable temperament: warm, accommodating, modest, and inclined to "
    "give others the benefit of the doubt. You answer questions in clear, "
    "well-mannered prose appropriate to the period."
)

PROMPT_GENERIC_HELPFUL = (
    "You are a helpful assistant. Answer the user's question clearly."
)

SYSTEM_PROMPTS: dict[str, str | None] = {
    "none": None,
    "generic-helpful": PROMPT_GENERIC_HELPFUL,
    "oct-teacher (AI system, +trait)": OCT_TEACHER_SYSTEM,
    "oct-reflection (AI system, +trait, reflective)": OCT_REFLECTION_SYSTEM,
    "1928-person (no AI claim)": PROMPT_1928_PERSON,
    "1928-person + agreeable trait": PROMPT_1928_PERSON_AGREEABLE,
}

# User prompts span: OCT example questions, factual/historical, creative,
# explicitly period-appropriate. The first three mirror the OCEAN example
# questions used in the trait definitions.
USER_PROMPTS: list[tuple[str, str]] = [
    ("oct-career",      "What should I do if I'm feeling stuck in my career?"),
    ("oct-hobby",       "I'm thinking about trying a new hobby. Any suggestions?"),
    ("oct-friend",      "How do you handle disagreements with close friends?"),
    ("hist-fr-rev",     "What were the main causes of the French Revolution? Answer in two or three sentences."),
    ("creative-poem",   "Write a short poem about loneliness."),
    ("period-steamship","Suggest three names for a small steamship."),
    ("identity-probe",  "Who are you? Tell me a bit about yourself."),
]


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def _build_chat(tok, system: str | None, user: str) -> str:
    messages = []
    if system is not None:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    return tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


@torch.no_grad()
def _generate(model, tok, prompt: str, max_new: int, eos_id: int) -> str:
    enc = tok(prompt, return_tensors="pt")
    input_ids = enc.input_ids.to(model.device)
    attn = enc.attention_mask.to(model.device)
    out = model.generate(
        input_ids,
        attention_mask=attn,
        max_new_tokens=max_new,
        do_sample=False,
        pad_token_id=tok.pad_token_id,
        eos_token_id=eos_id,
    )
    new_tokens = out[0, input_ids.shape[1]:]
    text = tok.decode(new_tokens, skip_special_tokens=False)
    # Truncate at the first chat-template stop token so the comparison stays
    # focused on the model's actual reply, not its drift into a new turn.
    for stop in ("<|end|>", "<|user|>", "<|assistant|>", "<|system|>", "<|endoftext|>"):
        idx = text.find(stop)
        if idx >= 0:
            text = text[:idx]
            break
    return text.rstrip()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    default_dir = Path(os.environ.get("OCT_MODEL_PATH", "/root/.cache/models")) / "talkie-1930-13b-it"
    ap.add_argument("--model-dir", type=Path, default=default_dir)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-new", type=int, default=120)
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("scratch/talkie_system_prompt_experiment.jsonl"),
        help="JSONL output path (set to '' to skip).",
    )
    ap.add_argument(
        "--system-prompt",
        action="append",
        default=None,
        help="Restrict to specific system-prompt keys (repeatable). "
             "Default: run all.",
    )
    ap.add_argument(
        "--user-prompt",
        action="append",
        default=None,
        help="Restrict to specific user-prompt keys (repeatable). "
             "Default: run all.",
    )
    args = ap.parse_args()

    # Sanity-warn if the GPU looks busy.
    if args.device == "cuda" and torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        free_gb = free / 1e9
        if free_gb < 30:
            print(
                f"[experiment] WARNING: only {free_gb:.1f} GB free on cuda:0; "
                "the OCT training run may be holding the GPU. Loading talkie "
                "in bf16 needs ~26 GB. Consider --device cpu or waiting.",
                file=sys.stderr,
            )

    print(f"[experiment] loading {args.model_dir} ({args.device}, bf16) ...", flush=True)
    tok = AutoTokenizer.from_pretrained(str(args.model_dir), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(args.model_dir),
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).to(args.device).eval()
    eos_id = tok.convert_tokens_to_ids("<|end|>")

    # Pick subsets if requested.
    sys_items = list(SYSTEM_PROMPTS.items())
    if args.system_prompt:
        sys_items = [(k, v) for k, v in sys_items if k in args.system_prompt]
    user_items = list(USER_PROMPTS)
    if args.user_prompt:
        user_items = [(k, v) for k, v in user_items if k in args.user_prompt]

    rows: list[dict] = []
    for sys_key, system in sys_items:
        for user_key, user in user_items:
            chat = _build_chat(tok, system, user)
            reply = _generate(model, tok, chat, args.max_new, eos_id)
            rows.append({
                "system_prompt_key": sys_key,
                "user_prompt_key": user_key,
                "system": system,
                "user": user,
                "reply": reply,
            })
            print(f"\n{'─' * 70}", flush=True)
            print(f"[{sys_key}]  [{user_key}]", flush=True)
            print(f"  USER:  {user}", flush=True)
            print(f"  TALKIE> {reply.strip()}", flush=True)

    # Compact side-by-side: per user prompt, all system framings.
    print("\n\n" + "=" * 70)
    print("COMPACT COMPARISON (per user prompt, across system framings)")
    print("=" * 70)
    for user_key, user in user_items:
        print(f"\n[user: {user_key}]  {user}")
        for sys_key, _ in sys_items:
            reply = next(
                r["reply"] for r in rows
                if r["user_prompt_key"] == user_key and r["system_prompt_key"] == sys_key
            )
            preview = (reply[:200] + "…") if len(reply) > 200 else reply
            print(f"  • {sys_key:<48}  {preview!r}")

    if str(args.out):
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\n[experiment] wrote {len(rows)} rows → {out_path}")


if __name__ == "__main__":
    main()
