# Qwen3.8: validated Q4 Vulkan speed path

Measured on 2026-09-22 in the isolated DwarfStar-inspired fork, then integrated
into XTLLM's Qwen3.8 `full` profile. Other architectures are unchanged. No new
quantization, reduced routing, larger context, or speculative decoding is
enabled by this profile. Existing model files are reused unchanged.

## Results

RX 6700 XT 12 GB, Ryzen 5 3600, Windows, 88 GB installed RAM. Fresh processes,
greedy generation, thinking off, 128 output tokens / 127 timed decode
transitions. Startup, prompt prefill and RAM prewarming are excluded.

| Path | Generated tok/s | Interpretation |
|---|---:|---|
| Published original Q4 | 6.39 | Historical; not the matched memory configuration |
| Original Q4, matched max-memory rerun | 7.43553 / 7.46693; mean **7.45123** | Baseline |
| Parallel router alone | 13.8192 | Largest isolated gain; one trial |
| Optimized Q4 plus grouped prefill | 14.0313 / 14.1791 / 14.2815 / 14.2303; mean **14.18055** | Retained; **+90.31%** vs matched baseline |
| Strict Q4 MTP with expert reuse | 10.654 accepted tok/s | Slower; remains in the research fork, not this merge |

### Main integration recheck

All eight native backends and 180 shaders compiled, 17 launcher/chat tests
passed, and the 100 router fixtures passed using main's shaders. All merged
shader binaries match the validated fork's binaries.

| Same-session run | Generated tok/s | Acquisition ms/output |
|---|---:|---:|
| Merged main, trial 1 | 12.4873 | 29.7064 |
| Merged main, trial 2 | 12.4170 | 29.8877 |
| Retained validated fork control | 12.4645 | 29.5922 |
| Merged reference fallback | 7.4063 | 30.9483 |

Main's mean **12.45215** is within 0.1% of the unchanged fork control. Both
were slower than the earlier 14.18 mean, with acquisition near 30 rather
than 21 ms/output; the underlying cause of that session difference is not
established. This is not evidence of a merge-specific regression. All 128
output IDs match the original baseline, and traffic/explicit allocations are
unchanged. The earlier result is retained as measured history, not a guarantee
that every subsequent run reaches it. No other model's kernels or registry
entries were changed; their full performance sweeps were not rerun.
The reference fallback also preserved all 128 IDs; the same-session merged
speedup over that fallback was **68.1%**. All test processes were unloaded.

Paired reasoning prompt: **7.90696 -> 13.9294 tok/s**. Paired palindrome-code
prompt: **7.98762 -> 14.1730 tok/s**. Output token IDs matched the original on
all three prompts. The reasoning response hit the 128-token limit before its
final answer: this is a regression check, not a scored reasoning evaluation.
One hundred bounded GPU router fixtures also matched expert IDs and weight
bits, including tied/sentinel scores and mixed NaNs. No claim of universal
bitwise equivalence on every device or broad quality evaluation is made.

## Memory, traffic and context

| Item | Measured configuration/result |
|---|---|
| Configured inference RAM ceiling | 72 GiB |
| RAM expert cache | 59.8125 GiB; all 24,576 records |
| Total explicit host allocation / peak working set | 59.9342 / ~60.020 GiB |
| VRAM expert slots | 62/layer; 2,976 total |
| Explicit device allocations / observed dedicated memory | 10.487 / ~10.56 GiB |
| Live Vulkan budget at startup | 11.2053 GiB |
| Expert SSD traffic | 0 bytes/output |
| Host-to-staging / H2D expert traffic | 352.478 MiB/output each |
| PLE SSD / PLE H2D traffic | 67,826 / 10,240 bytes/output |
| Attention/router stage | 19.042 ms/output |
| CPU-visible acquisition | 21.380 ms/output |
| Shared/expert stage | 46.698 ms/output |
| Total decode | ~70.5 ms/output |

The latency spans **overlap and must not be added**. The remaining limit is
host copies/H2D plus expert execution, not cold expert SSD reads. PLE retains
the existing official FP8 table and unbuffered selected-row SSD reads.
Minimum free system RAM stayed above 18.4 GiB. A 65-slot attempt failed
cleanly during initialization; the retained 62-slot configuration leaves
driver/display headroom. These measurements do not establish safe settings
for every display load or GPU.

The allocated context cap remains **2,048 total tokens**, with 96 MiB of FP32
GPU K/V plus fixed recurrent state. The speed prompt was **38 tokens + 128
generated tokens**. Longer host-backed K/V and native long-context Qwen Sparse
Attention are not implemented by this change. Unused RAM is not a larger
tested context window.

Precision remains Q4G64T experts/shared, Q8 globals and official FP8 PLE;
all ten authoritative routes are evaluated. No draft tokens enter the speed.

## Reproduction

Build from current source using `scripts/build-windows.ps1`, or use a package
built from this revision. Older release ZIPs do not contain this update.

```powershell
# Existing installed runtime, standard model-root layout:
.\xtllm.cmd run qwen38 "In two concise paragraphs, explain why the sky appears blue, then give a short Python function that checks whether an integer is prime." --profile full --ram-gib 72 --device-slots 62 --tokens 128

# Matched original path, using the same weights and memory settings:
.\xtllm.cmd run qwen38 "In two concise paragraphs, explain why the sky appears blue, then give a short Python function that checks whether an integer is prime." --profile reference --ram-gib 72 --device-slots 62 --tokens 128
```

The full profile selects `xtllm-qwen38-flash-next.exe` with:

```text
QWEN38_DWARFSTAR=1
QWEN38_DWARF_PREFILL4=1
QWEN38_RAM_GIB=72
QWEN38_DEVICE_SLOTS_PER_LAYER=62
QWEN38_FILL_RAM_CACHE=1
QWEN38_NO_THINK=1
```

The launcher clears inherited `QWEN38_*` flags before applying the profile.
MTP, route reduction, BDA batching and the spinning parallel-copy pool are not
enabled. Default configured RAM remains 53 GiB rather than silently demanding
72 GiB on smaller systems; pass a smaller `--ram-gib` when needed and preserve
OS headroom. The 14.18 result applies to the explicit settings above, not to
every RAM budget. Below 40 device slots/layer, the launcher uses scalar prefill
while retaining the decode fusions/router. Native direct invocation rejects
an undersized four-row union rather than dropping experts.

The original greedy path remains `--profile reference`. The previous default
Q3/strict-MTP path is preserved as `--profile legacy-mtp`, with its original
sidecars and settings. The slower new Q4 MTP experiment/converter is **not**
included in this merge.

## What changed and attribution

Ideas studied from [DwarfStar / antirez/ds4](https://github.com/antirez/ds4/tree/0aaea5a238fb41a35106a551e73c8409dfb751ac)
at commit `0aaea5a238fb41a35106a551e73c8409dfb751ac`. The upstream
[MIT notice](licenses/DwarfStar-MIT.txt) accompanies this port. The Vulkan
implementation retains XTLLM's math/layouts; CUDA and Metal are not dependencies.

- Lane-owned parallel Top-10 selection replaces a serial 512-expert scan.
  Original raw-logit ranking, lower-ID tie breaking and selected-softmax order
  are preserved instead of importing upstream's different rounding order.
- Fused HC mixing/Q8 activation packing removes 97 standalone dispatches per
  ordinary token; the float output remains available.
- Fused shared/routed reduction and residual injection removes 48 dispatches,
  preserving shared-first then routed-rank accumulation order.
- Four-row **prompt prefill** groups work by unique expert using packed Q4
  kernels, including the K=640 down projection. No rejection snapshots are
  needed for known prompt inputs; only the last row needs vocabulary logits.
  Decode remains one token at a time. Chronological recurrent/PLE state and
  causal attention are preserved, and incomplete groups use scalar prefill.

Native component switches remain off unless the launcher/profile enables
them. `QWEN38_DWARF_ROUTER`, `QWEN38_DWARF_HC_QUANT`, and
`QWEN38_DWARF_REDUCE_HC` can individually override the master switch.
`QWEN38_DWARF_PREFILL4` is separate. These flags are rejected with the Q3
build, reduced verifier routes, relaxed acceptance, or the spinning copy pool.
Submissions stay finite and layer-bounded; no persistent GPU waits or TDR
changes are introduced. Existing model assets and other backends are untouched.
