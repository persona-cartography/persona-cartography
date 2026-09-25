# Activation Extraction Comparison Test

## Purpose
This test compares two methods of extracting activations from assistant responses:
1. **assistant-axis**: Uses SpanMapper and ConversationEncoder
2. **persona-shattering-lasr**: Uses manual tokenization comparison

## GPU Requirements

### Model Options (in order of memory requirement):

1. **google/gemma-2-2b-it** (Default)
   - ~5-6 GB VRAM
   - Works on: M1/M2/M3 Macs with 16GB+ RAM, consumer GPUs

2. **Qwen/Qwen2.5-3B-Instruct**
   - ~7-8 GB VRAM
   - Works on: Similar to above

3. **google/gemma-2-9b-it**
   - ~20 GB VRAM
   - Needs: Larger GPU or Mac with 32GB+ RAM

### On Apple Silicon (Mac):
- The script will use **MPS** (Metal Performance Shaders)
- Model loads to unified memory (shared CPU/GPU)
- Recommended: **M1/M2/M3 with 16GB+ RAM** should work fine with 2B model
- With 32GB+ RAM, you can try larger models

### On NVIDIA GPU:
- Any GPU with 8GB+ VRAM should work with the 2B model
- Examples: RTX 3060 (12GB), RTX 4070 (12GB), A4000, etc.

## Running the Test

```bash
# Navigate to persona-shattering-lasr directory
cd /path/to/projects/persona-shattering-lasr

# Run with uv (uses gemma-2-2b-it by default)
uv run scripts/test_activation_extraction.py

# Or edit the script to change the model:
# model_name = "Qwen/Qwen2.5-3B-Instruct"
```

## Expected Output

The test will:
1. Load the model twice (once for each method)
2. Extract activations using both methods
3. Compare the results with:
   - Absolute and relative differences
   - Cosine similarity per layer
   - Final verdict on whether they're equivalent

## What "Success" Looks Like

The methods are equivalent if:
- **Cosine similarity > 0.9999** (essentially identical directions)
- **Mean relative difference < 0.01%** (very close values)
- Or **Max absolute difference < 1e-5** (numerically identical)

## Notes

- The test uses a simple single-turn conversation
- Both methods load the model independently to ensure fair comparison
- Small numerical differences (< 0.1%) are expected due to floating point arithmetic
- The test is designed to verify functional equivalence, not bit-exact reproduction
