#!/usr/bin/env python3
"""
Black-box test comparing activation extraction methods:
1. assistant-axis/pipeline/2_activations.py approach
2. persona-shattering-lasr/dump/extract-ft-persona-vector/2_activations.py approach

This imports and uses the actual implementations as black boxes.

Usage:
    cd /path/to/projects/persona-shattering-lasr
    uv run scripts/test_activation_extraction.py
"""

import sys
from pathlib import Path
import torch
import numpy as np

# Add paths
assistant_axis_path = Path(__file__).parent.parent.parent / "assistant-axis"
sys.path.insert(0, str(assistant_axis_path))

# Import assistant-axis implementation
from assistant_axis.internals import ProbingModel, ConversationEncoder, ActivationExtractor, SpanMapper

# Import persona-shattering-lasr implementation
sys.path.insert(0, str(Path(__file__).parent / "dump" / "extract-ft-persona-vector"))
from response_token_utils import get_assistant_response_token_ids


def extract_with_assistant_axis_method(model_name: str, conversation: list) -> torch.Tensor:
    """
    Extract activations using assistant-axis internals (as a black box).

    This uses the actual SpanMapper + ConversationEncoder approach.

    Returns:
        Tensor of shape (num_layers, hidden_size)
    """
    print("\n" + "="*60)
    print("METHOD 1: assistant-axis (SpanMapper approach)")
    print("="*60)

    # Patch for transformers tokenize=True returning BatchEncoding
    # This is a known issue in assistant-axis with newer transformers versions
    from transformers import PreTrainedTokenizerBase

    original_apply_chat_template = PreTrainedTokenizerBase.apply_chat_template

    def patched_apply_chat_template(self, *args, **kwargs):
        result = original_apply_chat_template(self, *args, **kwargs)
        # If tokenize=True and result is BatchEncoding, extract input_ids
        if kwargs.get('tokenize', False) and hasattr(result, 'input_ids'):
            return result.input_ids
        return result

    PreTrainedTokenizerBase.apply_chat_template = patched_apply_chat_template

    try:
        # Use their ProbingModel wrapper
        pm = ProbingModel(model_name)
        encoder = ConversationEncoder(pm.tokenizer, pm.model_name)
        extractor = ActivationExtractor(pm, encoder)
        span_mapper = SpanMapper(pm.tokenizer)

        n_layers = len(pm.get_layers())
        layers = list(range(n_layers))
        print(f"Model: {model_name}")
        print(f"Layers: {n_layers}")

        # Extract activations using their pipeline
        batch_activations, batch_metadata = extractor.batch_conversations(
            [conversation],
            layer=layers,
            max_length=2048,
        )

        # Build spans
        _, batch_spans, span_metadata = encoder.build_batch_turn_spans([conversation])

        print(f"Spans detected: {len(batch_spans)}")
        for span in batch_spans:
            print(f"  {span['role']:10s} tokens [{span['start']:3d}:{span['end']:3d}] ({span['n_tokens']} tokens)")

        # Debug: show actual assistant span token range
        assistant_span = [s for s in batch_spans if s['role'] == 'assistant'][0]
        method1_start = assistant_span['start']
        method1_end = assistant_span['end']
        print(f"\nDEBUG - Method 1 (assistant-axis) token range: [{method1_start}:{method1_end}]")

        # Get Method 2 range for comparison
        full_ids_m2, method2_start, method2_end = get_assistant_response_token_ids(pm.tokenizer, conversation)

        print(f"DEBUG - Method 2 (persona-shattering) token range: [{method2_start}:{method2_end}]")

        # Decode tokens to see what we're extracting
        full_ids = encoder.token_ids(conversation, add_generation_prompt=False)
        print(f"\nDEBUG - Tokens being extracted:")
        print(f"  Method 1 tokens [{method1_start}:{method1_end}]: {pm.tokenizer.convert_ids_to_tokens(full_ids[method1_start:method1_end])}")
        print(f"  Method 2 tokens [{method2_start}:{method2_end}]: {pm.tokenizer.convert_ids_to_tokens(full_ids_m2[method2_start:method2_end])}")

        if method1_start != method2_start or method1_end != method2_end:
            print(f"\n⚠️  MISMATCH: Method 1 [{method1_start}:{method1_end}] vs Method 2 [{method2_start}:{method2_end}]")
            if method1_end < method2_end:
                print(f"  Extra tokens in Method 2 [{method1_end}:{method2_end}]: {pm.tokenizer.convert_ids_to_tokens(full_ids[method1_end:method2_end])}")
            elif method2_end < method1_end:
                print(f"  Extra tokens in Method 1 [{method2_end}:{method1_end}]: {pm.tokenizer.convert_ids_to_tokens(full_ids[method2_end:method1_end])}")
        else:
            print(f"\n✅ MATCH: Both methods extract from [{method1_start}:{method1_end}]")

        # Map spans to activations
        conv_activations_list = span_mapper.map_spans(batch_activations, batch_spans, batch_metadata)
        conv_acts = conv_activations_list[0]  # Shape: (num_turns, num_layers, hidden_size)

        # Extract assistant turns only (odd indices: 1, 3, 5, ...)
        assistant_acts = conv_acts[1::2]
        mean_act = assistant_acts.mean(dim=0).cpu()  # (num_layers, hidden_size)

        print(f"Extracted {assistant_acts.shape[0]} assistant turn(s)")
        print(f"Output shape: {mean_act.shape}")
        print(f"Output dtype: {mean_act.dtype}")

        return mean_act

    finally:
        # Restore original method
        PreTrainedTokenizerBase.apply_chat_template = original_apply_chat_template


def extract_with_persona_shattering_method(model_name: str, conversation: list) -> torch.Tensor:
    """
    Extract activations using persona-shattering-lasr manual approach (as a black box).

    This imports the actual extract_activations_batch function from 2_activations.py

    Returns:
        Tensor of shape (num_layers, hidden_size)
    """
    print("\n" + "="*60)
    print("METHOD 2: persona-shattering-lasr (manual tokenization)")
    print("="*60)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    n_layers = model.config.num_hidden_layers
    layers = list(range(n_layers))
    print(f"Model: {model_name}")
    print(f"Layers: {n_layers}")

    # Import the actual function from our implementation
    from scripts.dump.extract_ft_persona_vector.extract_activations_batch import extract_activations_batch

    # Use it as a black box
    activations_list = extract_activations_batch(
        model=model,
        tokenizer=tokenizer,
        conversations=[conversation],
        layers=layers,
        batch_size=1,
    )

    result = activations_list[0]  # (num_layers, hidden_size)
    print(f"Output shape: {result.shape}")

    return result


# Fallback: inline the function if import doesn't work
def extract_activations_batch_inline(model, tokenizer, conversations, layers=None, batch_size=16):
    """
    Inline version of extract_activations_batch from persona-shattering-lasr/2_activations.py
    Used as fallback if the import doesn't work.
    """
    if layers is None:
        n_layers = model.config.num_hidden_layers
        layers = list(range(n_layers))

    all_activations = []
    activations_storage = {}

    def create_hook(layer_idx):
        def hook(module, input, output):
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output
            if layer_idx not in activations_storage:
                activations_storage[layer_idx] = []
            activations_storage[layer_idx].append(hidden_states.detach().cpu())
        return hook

    # Register hooks
    hooks = []
    for layer_idx in layers:
        # Handle different model architectures
        if hasattr(model, 'base_model') and hasattr(model.base_model, 'model'):
            if hasattr(model.base_model.model, 'model') and hasattr(model.base_model.model.model, 'layers'):
                layer = model.base_model.model.model.layers[layer_idx]
            elif hasattr(model.base_model.model, 'layers'):
                layer = model.base_model.model.layers[layer_idx]
            else:
                raise ValueError(f"Cannot find layers in PEFT model")
        elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
            layer = model.model.layers[layer_idx]
        else:
            raise ValueError(f"Cannot find layers in model")

        hook = layer.register_forward_hook(create_hook(layer_idx))
        hooks.append(hook)

    # Process conversations
    for batch_start in range(0, len(conversations), batch_size):
        batch_end = min(batch_start + batch_size, len(conversations))
        batch_conversations = conversations[batch_start:batch_end]
        activations_storage.clear()

        # Format prompts
        prompts = [tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=False)
                   for conv in batch_conversations]

        # Tokenize
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # Forward pass
        with torch.no_grad():
            _ = model(**inputs)

        # Extract activations for each conversation
        for batch_idx, conv in enumerate(batch_conversations):
            # Use assistant-axis compatible token extraction
            full_ids, response_start_idx, response_end_idx = get_assistant_response_token_ids(
                tokenizer, conv
            )

            # Account for padding
            attention_mask = inputs['attention_mask'][batch_idx]
            non_pad_start = (attention_mask == 1).nonzero(as_tuple=True)[0][0].item()
            response_start_padded = max(non_pad_start, min(non_pad_start + response_start_idx, attention_mask.shape[0]))
            response_end_padded = min(non_pad_start + response_end_idx, attention_mask.shape[0])

            # Debug output for first conversation
            if batch_idx == 0 and batch_start == 0:
                print(f"\nDEBUG - Method 2 token extraction:")
                print(f"  Response token range (unpadded): [{response_start_idx}:{response_end_idx}]")
                print(f"  Response token count: {response_end_idx - response_start_idx}")
                print(f"  Response token range (padded): [{response_start_padded}:{response_end_padded}]")
                print(f"  Total sequence length: {attention_mask.shape[0]}")
                print(f"  Non-pad start: {non_pad_start}")

            # Extract mean activations per layer
            layer_means = []
            for layer_idx in layers:
                layer_acts = activations_storage[layer_idx][0][batch_idx]  # (seq_len, hidden_dim)
                response_acts = layer_acts[response_start_padded:response_end_padded]

                if response_acts.shape[0] > 0:
                    mean_act = response_acts.mean(dim=0)
                else:
                    # Fallback to all valid tokens
                    valid_positions = attention_mask.bool()
                    valid_acts = layer_acts[valid_positions.cpu()]
                    mean_act = valid_acts.mean(dim=0)

                layer_means.append(mean_act)

            all_activations.append(torch.stack(layer_means))

    # Remove hooks
    for hook in hooks:
        hook.remove()

    return all_activations


def compare_results(act1: torch.Tensor, act2: torch.Tensor):
    """Compare two activation tensors and print detailed statistics."""
    print("\n" + "="*60)
    print("COMPARISON RESULTS")
    print("="*60)

    print(f"\nShape 1: {act1.shape}, dtype: {act1.dtype}")
    print(f"Shape 2: {act2.shape}, dtype: {act2.dtype}")

    if act1.shape != act2.shape:
        print("\n❌ ERROR: Shapes don't match!")
        return

    # Normalize to same dtype for fair comparison
    if act1.dtype != act2.dtype:
        print(f"\n⚠️  Different dtypes detected! Converting both to float32 for comparison...")
        act1 = act1.float()
        act2 = act2.float()

    # Statistics
    abs_diff = torch.abs(act1 - act2)
    rel_diff = abs_diff / (torch.abs(act1) + 1e-8)

    print(f"\nAbsolute difference:")
    print(f"  Mean: {abs_diff.mean().item():.6e}")
    print(f"  Max:  {abs_diff.max().item():.6e}")
    print(f"  Std:  {abs_diff.std().item():.6e}")

    print(f"\nRelative difference (%):")
    print(f"  Mean: {(rel_diff.mean().item() * 100):.6f}%")
    print(f"  Max:  {(rel_diff.max().item() * 100):.6f}%")

    # Cosine similarity per layer
    cosine_sims = []
    for i in range(act1.shape[0]):
        cos_sim = torch.nn.functional.cosine_similarity(
            act1[i].unsqueeze(0),
            act2[i].unsqueeze(0)
        )
        cosine_sims.append(cos_sim.item())

    print(f"\nCosine similarity (per layer):")
    print(f"  Mean: {np.mean(cosine_sims):.10f}")
    print(f"  Min:  {np.min(cosine_sims):.10f}")
    print(f"  Max:  {np.max(cosine_sims):.10f}")

    # Sample specific layers
    print(f"\nSample layers:")
    sample_indices = [0, act1.shape[0]//2, act1.shape[0]-1]
    for i in sample_indices:
        print(f"  Layer {i:2d}: cos_sim = {cosine_sims[i]:.10f}, abs_diff_mean = {abs_diff[i].mean().item():.6e}")

    # Verdict
    print("\n" + "="*60)
    threshold = 1e-5
    if abs_diff.max().item() < threshold:
        print(f"✅ IDENTICAL (max absolute diff < {threshold})")
    elif (rel_diff.mean().item() * 100) < 0.01:
        print(f"✅ VERY CLOSE (mean relative diff < 0.01%)")
    elif np.mean(cosine_sims) > 0.9999:
        print(f"✅ HIGHLY SIMILAR (mean cosine similarity > 0.9999)")
    else:
        print(f"⚠️  DIFFER SIGNIFICANTLY")
    print("="*60)


def main():
    model_name = "google/gemma-2-2b-it"

    # Single-turn conversation for testing
    conversation = [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "The answer is 4."}
    ]

    print("="*60)
    print("ACTIVATION EXTRACTION BLACK-BOX TEST")
    print("="*60)
    print(f"\nModel: {model_name}")
    print(f"Conversation:")
    for msg in conversation:
        print(f"  {msg['role']:10s}: {msg['content']}")

    print("\n⚠️  NOTE: Both methods will use the SAME model instance to ensure identical activations")

    # Load model ONCE and share between both methods
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print("\nLoading model (shared between both methods)...")
    shared_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")
    shared_tokenizer = AutoTokenizer.from_pretrained(model_name)
    if shared_tokenizer.pad_token is None:
        shared_tokenizer.pad_token = shared_tokenizer.eos_token

    # Method 1: assistant-axis (using shared model)
    print("\n" + "="*60)
    print("METHOD 1: assistant-axis (SpanMapper approach)")
    print("="*60)

    # Apply patch for transformers compatibility
    from transformers import PreTrainedTokenizerBase
    original_apply_chat_template = PreTrainedTokenizerBase.apply_chat_template

    def patched_apply_chat_template(self, *args, **kwargs):
        result = original_apply_chat_template(self, *args, **kwargs)
        if kwargs.get('tokenize', False) and hasattr(result, 'input_ids'):
            return result.input_ids
        return result

    PreTrainedTokenizerBase.apply_chat_template = patched_apply_chat_template

    try:
        pm = ProbingModel.from_existing(shared_model, shared_tokenizer, model_name)
        encoder = ConversationEncoder(pm.tokenizer, pm.model_name)
        extractor = ActivationExtractor(pm, encoder)
        span_mapper = SpanMapper(pm.tokenizer)

        n_layers = len(pm.get_layers())
        layers = list(range(n_layers))
        print(f"Model: {model_name}")
        print(f"Layers: {n_layers}")

        # Extract activations using their pipeline
        batch_activations, batch_metadata = extractor.batch_conversations(
            [conversation],
            layer=layers,
            max_length=2048,
        )

        # Build spans
        _, batch_spans, span_metadata = encoder.build_batch_turn_spans([conversation])

        print(f"Spans detected: {len(batch_spans)}")
        for span in batch_spans:
            print(f"  {span['role']:10s} tokens [{span['start']:3d}:{span['end']:3d}] ({span['n_tokens']} tokens)")

        # Debug: show actual assistant span token range
        assistant_span = [s for s in batch_spans if s['role'] == 'assistant'][0]
        method1_start = assistant_span['start']
        method1_end = assistant_span['end']
        print(f"\nDEBUG - Method 1 (assistant-axis) token range: [{method1_start}:{method1_end}]")

        # Get Method 2 range for comparison
        full_ids_m2, method2_start, method2_end = get_assistant_response_token_ids(pm.tokenizer, conversation)

        print(f"DEBUG - Method 2 (persona-shattering) token range: [{method2_start}:{method2_end}]")

        # Decode tokens to see what we're extracting
        full_ids = encoder.token_ids(conversation, add_generation_prompt=False)
        print(f"\nDEBUG - Tokens being extracted:")
        print(f"  Method 1 tokens [{method1_start}:{method1_end}]: {pm.tokenizer.convert_ids_to_tokens(full_ids[method1_start:method1_end])}")
        print(f"  Method 2 tokens [{method2_start}:{method2_end}]: {pm.tokenizer.convert_ids_to_tokens(full_ids_m2[method2_start:method2_end])}")

        if method1_start != method2_start or method1_end != method2_end:
            print(f"\n⚠️  MISMATCH: Method 1 [{method1_start}:{method1_end}] vs Method 2 [{method2_start}:{method2_end}]")
            if method1_end < method2_end:
                print(f"  Extra tokens in Method 2 [{method1_end}:{method2_end}]: {pm.tokenizer.convert_ids_to_tokens(full_ids[method1_end:method2_end])}")
            elif method2_end < method1_end:
                print(f"  Extra tokens in Method 1 [{method2_end}:{method1_end}]: {pm.tokenizer.convert_ids_to_tokens(full_ids[method2_end:method1_end])}")
        else:
            print(f"\n✅ MATCH: Both methods extract from [{method1_start}:{method1_end}]")

        # Map spans to activations
        conv_activations_list = span_mapper.map_spans(batch_activations, batch_spans, batch_metadata)
        conv_acts = conv_activations_list[0]  # Shape: (num_turns, num_layers, hidden_size)

        # Extract assistant turns only (odd indices: 1, 3, 5, ...)
        assistant_acts = conv_acts[1::2]
        activations_1 = assistant_acts.mean(dim=0).cpu()  # (num_layers, hidden_size)

        print(f"Extracted {assistant_acts.shape[0]} assistant turn(s)")
        print(f"Output shape: {activations_1.shape}")
        print(f"Output dtype: {activations_1.dtype}")

    finally:
        # Restore original method
        PreTrainedTokenizerBase.apply_chat_template = original_apply_chat_template

    # Method 2: persona-shattering-lasr (using same shared model)
    print("\n" + "="*60)
    print("METHOD 2: persona-shattering-lasr (inline implementation)")
    print("="*60)

    print(f"Model: {model_name}")
    print(f"Layers: {n_layers}")
    print(f"Model dtype: {next(shared_model.parameters()).dtype}")

    activations_2 = extract_activations_batch_inline(shared_model, shared_tokenizer, [conversation], layers, batch_size=1)[0]
    print(f"Output shape: {activations_2.shape}")
    print(f"Output dtype: {activations_2.dtype}")

    # Compare
    compare_results(activations_1, activations_2)

    print("\n✓ Test complete!\n")


if __name__ == "__main__":
    main()
