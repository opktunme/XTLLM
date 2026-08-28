# XTLLM 0.3.0 — three new sparse-model backends

XTLLM 0.3.0 adds three official-checkpoint, text-only Vulkan paths while
retaining all existing Qwen, DeepSeek, and Nemotron implementations:

- `Qwen/Qwen3.8-Flash-Next-FP8`
- `Qwen/Qwen3-Coder-Next-FP8`
- `meituan-longcat/LongCat-Flash-Lite-Sparse`

All three are available through the same setup, generation, and localhost chat
commands as existing models. The launcher selects a dedicated native backend;
the merge does not replace their measured model-specific policies with one
generic expert cache.

```powershell
.\xtllm.cmd setup qwencoder --model-root D:\XTLLM-Models
.\xtllm.cmd run qwencoder "Write a small Python merge sort." --tokens 128
.\xtllm.cmd chat qwencoder
```

On the validated RX 6700 XT 12GB system, retained 24 GiB model-RAM results are
9.18 tok/s for Qwen3-Coder-Next and 12.42 tok/s for LongCat Flash Lite Sparse.
Qwen3.8 reaches 5.75 tok/s in its 53 GiB maximum-performance profile. These are
greedy output-token measurements with no speculative-token inflation.

LongCat requires its retained Q8 shared-weight path for coherent output. An
approximately 25 tok/s all-Q4 experiment was deliberately excluded because its
generated text failed correctness checks.
