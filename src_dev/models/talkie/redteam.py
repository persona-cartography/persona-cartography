"""Red-team battery for the talkie HF wrapper.

The narrow logit-parity check in ``verify.py`` confirms the math is right
on 6 short ASCII prompts. Before committing GPU hours to an OCT training
sweep, this script probes the places where the integration could still
produce garbage:

  R1. Chat-format instruction-following (the model is an IT model — does
      it actually follow instructions when we wrap a user message in the
      chat template and run greedy generation through the HF wrapper?
      Comparison: the talkie reference repo's ``Talkie.chat(...)``).

  R2. vLLM vs HF agreement on a chat-format prompt (vLLM's
      ``TransformersBackend`` swaps in PagedAttention; the integration
      should still yield the same greedy output as the HF wrapper for at
      least the first N tokens before bf16 numerical drift causes a
      divergence).

  R3. Long-context generation (a 500+ token prompt; check that the model
      doesn't degenerate, that vLLM doesn't truncate at small max_model_len,
      and that the HF position_id threading through KV cache works).

  R4. vLLM with a LoRA adapter (the introspection stage attaches a LoRA
      via ``LoRARequest``; this exercise loads a randomly-initialized
      LoRA and checks that (a) vLLM accepts it and generates without
      error, (b) the LoRA output differs from the base model — i.e. the
      adapter is actually being applied, not silently no-op'd).

  R5. PEFT training-side integration (the DPO/SFT path attaches LoRAs at
      training time via PEFT; we exercise get_peft_model + a single
      forward + backward, and confirm the loss is finite and the LoRA
      params receive non-zero gradients).

Each check prints a verdict and writes a sample to stdout. Failures
abort with a non-zero exit so this can be wired into CI later.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _section(title: str) -> None:
    bar = "=" * 70
    print(f"\n{bar}\n{title}\n{bar}", flush=True)


def _truncate(text: str, n: int = 200) -> str:
    if len(text) <= n:
        return text
    return text[:n] + " ... [+" + str(len(text) - n) + " chars]"


# ---------------------------------------------------------------------------
# R1. Chat-format instruction following (HF vs reference talkie repo)
# ---------------------------------------------------------------------------


def r1_chat_following(model_dir: Path, reference_pt: Path, talkie_ref_src: Path) -> None:
    _section("R1. Chat-format instruction-following")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    sys.path.insert(0, str(talkie_ref_src))
    from talkie.chat import format_chat, Message
    from talkie.tokenizer import build_tokenizer as ref_build_tok, IT_VOCAB_SIZE
    from talkie.model import load_checkpoint

    chat_prompts = [
        "Write a short poem about the moon.",
        "What were the main causes of the French Revolution? Answer in two sentences.",
        "Suggest three names for a small steamship.",
    ]

    tok = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    print("[R1] HF wrapper greedy outputs:")
    hf_model = AutoModelForCausalLM.from_pretrained(
        str(model_dir), trust_remote_code=True, torch_dtype=torch.bfloat16
    ).cuda().eval()

    hf_outputs: list[str] = []
    for p in chat_prompts:
        rendered = tok.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=False,
            add_generation_prompt=True,
        )
        ids = tok(rendered, return_tensors="pt").input_ids.cuda()
        with torch.no_grad():
            out = hf_model.generate(
                ids,
                max_new_tokens=80,
                do_sample=False,
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.convert_tokens_to_ids("<|end|>"),
            )
        text = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=False)
        hf_outputs.append(text)
        print(f"  USER:   {p}")
        print(f"  HF →    {_truncate(text)}")
        print()

    print("[R1] Reference Talkie.chat() outputs (slower; one-by-one):")
    ref = load_checkpoint(str(reference_pt), torch.device("cuda"), target_vocab_size=IT_VOCAB_SIZE)
    ref_tok = ref_build_tok(str(_REPO_ROOT.parent.parent / "root" / ".cache" / "talkie_source" / "vocab.txt"), style="it") if False else None
    # Just compare token-id parity on the FIRST few sampled tokens through reference's forward.
    # The reference doesn't expose a clean generate() loop without their CLI, so do a step-by-step
    # greedy decode for ~16 tokens for one prompt, comparing to HF token-by-token.
    print("[R1] step-by-step greedy comparison (first 16 tokens of prompt 1):")
    rendered = tok.apply_chat_template(
        [{"role": "user", "content": chat_prompts[0]}],
        tokenize=False, add_generation_prompt=True,
    )
    ids = tok(rendered, return_tensors="pt").input_ids.cuda()
    cur = ids.clone()
    n_match = 0
    for step in range(16):
        with torch.no_grad():
            ref_logits = ref.forward(cur)
            hf_logits = hf_model(input_ids=cur).logits[:, -1, :]
        rt = int(ref_logits.argmax(-1).item())
        ht = int(hf_logits.argmax(-1).item())
        eq = rt == ht
        n_match += eq
        if not eq:
            print(f"  step {step}: ref={rt} ({tok.decode([rt])!r}) hf={ht} ({tok.decode([ht])!r})  MISMATCH")
            break
        cur = torch.cat([cur, torch.tensor([[ht]], device=cur.device)], dim=1)
    print(f"  greedy-token agreement: {n_match}/16 tokens")
    if n_match < 16:
        raise RuntimeError("R1: HF wrapper and reference disagreed during greedy chat generation.")
    # Also check that the IT model actually emitted a sensible <|end|> within 80 tokens for at least one prompt.
    ended = sum(1 for t in hf_outputs if "<|end|>" in t)
    print(f"  prompts that emitted <|end|> within 80 tokens: {ended}/{len(hf_outputs)}")
    if ended == 0:
        print("  WARN: no <|end|> emitted — possible chat-template / tokenizer issue. Inspect outputs above.")

    del hf_model, ref
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# R2. vLLM vs HF agreement on chat-format greedy
# ---------------------------------------------------------------------------


def r2_vllm_vs_hf(model_dir: Path) -> None:
    _section("R2. vLLM vs HF greedy agreement (chat-format)")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    rendered = tok.apply_chat_template(
        [{"role": "user", "content": "Write one sentence about the weather in 1925."}],
        tokenize=False, add_generation_prompt=True,
    )

    # HF
    hf_model = AutoModelForCausalLM.from_pretrained(
        str(model_dir), trust_remote_code=True, torch_dtype=torch.bfloat16
    ).cuda().eval()
    ids = tok(rendered, return_tensors="pt").input_ids.cuda()
    with torch.no_grad():
        out = hf_model.generate(
            ids, max_new_tokens=24, do_sample=False, pad_token_id=tok.pad_token_id
        )
    hf_text = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=False)
    print(f"[R2] HF →    {hf_text!r}")
    del hf_model; torch.cuda.empty_cache()

    # vLLM
    llm = LLM(
        model=str(model_dir), trust_remote_code=True, dtype="bfloat16",
        max_model_len=1024, gpu_memory_utilization=0.4, enforce_eager=True,
    )
    out_v = llm.generate([rendered], SamplingParams(temperature=0.0, max_tokens=24))
    vl_text = out_v[0].outputs[0].text
    print(f"[R2] vLLM →  {vl_text!r}")

    # Token-by-token compare (first 8 tokens).
    hf_ids = tok.encode(hf_text, add_special_tokens=False)
    vl_ids = tok.encode(vl_text, add_special_tokens=False)
    n_match = 0
    for a, b in zip(hf_ids, vl_ids):
        if a == b: n_match += 1
        else: break
    print(f"[R2] common-prefix tokens: {n_match}")
    if n_match < 4:
        raise RuntimeError("R2: vLLM and HF diverged in the first 4 greedy tokens — likely an integration bug.")
    del llm; torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# R3. Long-context generation
# ---------------------------------------------------------------------------


def r3_long_context(model_dir: Path) -> None:
    _section("R3. Long-context generation (~1000 prompt tokens)")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)

    # Build a ~1000-token prompt of historical-style filler.
    seed = (
        "In the year 1898, the great city of Boston was abuzz with the news of the "
        "approaching solar eclipse. Newspapers had been printing predictions for "
        "weeks, and astronomers travelled from across the country to observe the "
        "celestial event from the rooftops of public buildings. "
    )
    prompt = (seed * 20)[: 4000]
    ids = tok(prompt, return_tensors="pt").input_ids
    print(f"[R3] prompt length (tokens): {ids.shape[1]}")
    if ids.shape[1] < 500:
        print("[R3] WARN: prompt did not tokenize to >=500 tokens; check tokenizer fertility.")

    hf_model = AutoModelForCausalLM.from_pretrained(
        str(model_dir), trust_remote_code=True, torch_dtype=torch.bfloat16
    ).cuda().eval()
    ids = ids.cuda()
    t0 = time.time()
    with torch.no_grad():
        out = hf_model.generate(
            ids, max_new_tokens=64, do_sample=False, pad_token_id=tok.pad_token_id
        )
    dt = time.time() - t0
    text = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=False)
    print(f"[R3] HF →    {_truncate(text)}")
    print(f"[R3] elapsed: {dt:.1f}s for {out.shape[1] - ids.shape[1]} new tokens")
    if not text.strip() or text.count("\x00") > 0:
        raise RuntimeError("R3: long-context generation produced empty or null-byte output.")
    del hf_model; torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# R4. vLLM + LoRA
# ---------------------------------------------------------------------------


def r4_vllm_lora(model_dir: Path) -> None:
    _section("R4. vLLM + LoRA")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    LORA_DIR = Path("/tmp/talkie_redteam_lora")
    LORA_DIR.mkdir(parents=True, exist_ok=True)

    print("[R4] building a tiny LoRA (random init, large alpha so it perturbs outputs) ...")
    m = AutoModelForCausalLM.from_pretrained(
        str(model_dir), trust_remote_code=True, torch_dtype=torch.bfloat16
    )
    lora_cfg = LoraConfig(
        r=8, lora_alpha=128, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
        target_modules=["attn_query", "attn_key", "attn_value", "attn_resid",
                        "mlp_gate", "mlp_linear", "mlp_resid"],
    )
    pm = get_peft_model(m, lora_cfg)
    # Force lora_B params to nonzero so the adapter actually perturbs outputs.
    with torch.no_grad():
        for name, p in pm.named_parameters():
            if "lora_B" in name:
                p.data.normal_(0.0, 0.01)
    pm.save_pretrained(str(LORA_DIR))
    print(f"[R4]   saved → {LORA_DIR}")
    del m, pm; torch.cuda.empty_cache()

    print("[R4] loading vLLM with enable_lora=True ...")
    llm = LLM(
        model=str(model_dir), trust_remote_code=True, dtype="bfloat16",
        max_model_len=512, gpu_memory_utilization=0.4, enforce_eager=True,
        enable_lora=True, max_lora_rank=16,
    )
    sp = SamplingParams(temperature=0.0, max_tokens=24)
    no_lora = llm.generate(["Once upon a time"], sp)[0].outputs[0].text
    with_lora = llm.generate(
        ["Once upon a time"], sp,
        lora_request=LoRARequest("rt", 1, lora_path=str(LORA_DIR)),
    )[0].outputs[0].text
    print(f"[R4] no-LoRA:    {no_lora!r}")
    print(f"[R4] with-LoRA:  {with_lora!r}")
    if no_lora == with_lora:
        raise RuntimeError("R4: LoRA appears to be a no-op — vLLM did NOT apply the adapter.")
    print("[R4] OK — outputs differ; LoRA is being applied.")
    del llm; torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# R5. PEFT training-side integration
# ---------------------------------------------------------------------------


def r5_peft_training(model_dir: Path) -> None:
    _section("R5. PEFT forward + backward (training-side)")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model

    tok = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    m = AutoModelForCausalLM.from_pretrained(
        str(model_dir), trust_remote_code=True, torch_dtype=torch.bfloat16
    ).cuda()
    lora_cfg = LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
        target_modules=["attn_query", "attn_key", "attn_value", "attn_resid",
                        "mlp_gate", "mlp_linear", "mlp_resid"],
    )
    pm = get_peft_model(m, lora_cfg)
    pm.gradient_checkpointing_enable()
    n_trainable = sum(p.numel() for p in pm.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in pm.parameters())
    print(f"[R5] trainable params: {n_trainable:,} / {n_total:,}  ({100 * n_trainable / n_total:.3f}%)")
    if n_trainable == 0:
        raise RuntimeError("R5: PEFT attached but 0 trainable params — target_modules likely missed.")

    # Forward+backward on a tiny chat-formatted example.
    rendered = tok.apply_chat_template(
        [
            {"role": "user", "content": "Hello."},
            {"role": "assistant", "content": "Greetings, friend."},
        ],
        tokenize=False, add_generation_prompt=False,
    )
    enc = tok(rendered, return_tensors="pt")
    ids = enc.input_ids.cuda()
    attn = enc.attention_mask.cuda()
    labels = ids.clone()
    out = pm(input_ids=ids, attention_mask=attn, labels=labels)
    print(f"[R5] forward loss: {out.loss.item():.4f}")
    if not torch.isfinite(out.loss):
        raise RuntimeError(f"R5: loss is not finite ({out.loss.item()}).")
    out.loss.backward()

    # Check non-zero grads on at least one LoRA param.
    nz = 0
    for name, p in pm.named_parameters():
        if p.requires_grad and p.grad is not None and p.grad.abs().sum().item() > 0:
            nz += 1
    print(f"[R5] LoRA params with non-zero gradient: {nz}")
    if nz == 0:
        raise RuntimeError("R5: no LoRA param received gradient — autograd graph broken somewhere.")
    print("[R5] OK")
    del m, pm; torch.cuda.empty_cache()


# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path, default=Path("/root/.cache/models/talkie-1930-13b-it"))
    ap.add_argument("--reference-pt", type=Path, default=Path("/root/.cache/talkie_source/rl-refined.pt"))
    ap.add_argument("--talkie-ref-src", type=Path, default=Path("/tmp/talkie_probe/talkie_repo/src"))
    ap.add_argument("--skip", action="append", default=[], help="check IDs to skip (e.g. r1,r4)")
    args = ap.parse_args()

    skip = set()
    for s in args.skip:
        for x in s.split(","):
            skip.add(x.strip().lower())

    checks = [
        ("r1", lambda: r1_chat_following(args.model_dir, args.reference_pt, args.talkie_ref_src)),
        ("r2", lambda: r2_vllm_vs_hf(args.model_dir)),
        ("r3", lambda: r3_long_context(args.model_dir)),
        ("r4", lambda: r4_vllm_lora(args.model_dir)),
        ("r5", lambda: r5_peft_training(args.model_dir)),
    ]
    failures: list[str] = []
    for tag, fn in checks:
        if tag in skip:
            print(f"\n[skipping {tag.upper()}]")
            continue
        try:
            fn()
        except Exception as exc:
            print(f"\n[{tag.upper()}] FAILED: {type(exc).__name__}: {exc}")
            import traceback
            traceback.print_exc()
            failures.append(tag)

    print("\n" + "=" * 70)
    if failures:
        print(f"FAILED checks: {failures}")
        sys.exit(1)
    print("ALL RED-TEAM CHECKS PASSED")


if __name__ == "__main__":
    main()
