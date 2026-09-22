# Changelog

## Unreleased

- Promoted the validated Qwen3.8 Q4 speed path to `full`: lane-parallel Top-10
  routing, fused HC/Q8 packing and MoE/residual updates, and four-token prompt
  prefill with shared expert acquisition. All 10 routes and weight formats
  are preserved; ordinary output decoding does not use MTP.
- Measured 14.18 generated tok/s across four RX 6700 XT trials versus 7.45
  tok/s for the matched original path (+90%). The 88 GB machine used a 72 GiB
  budget, ~60.02 GiB actual host RAM and ~10.56 GiB peak VRAM. Tokens matched
  the original on three sanity prompts; this is a short-context result.
- Pre-push integration recheck: merged main averaged 12.45 tok/s versus 12.46
  for the unchanged fork in the same session (within 0.1%), with identical
  tokens. Acquisition was slower than in the earlier 14.18 trials; both
  sets of measurements are documented rather than assuming constant speed.
- Preserved Qwen3.8's original `reference`, and moved its previous Q3/strict
  MTP `full` profile to `legacy-mtp`. Default setup now needs only the existing
  Q4/FP8 containers; it does not create Q3 or draft sidecars. Other models'
  configurations and kernels are unchanged.
- Retained DwarfStar attribution/MIT notice, added profile-isolation tests,
  and included specialized backends and third-party notices in CI artifacts.
  Larger RAM-backed KV/context support is not part of this merge.

- Fixed localhost multi-turn chat formatting. The shared UI now passes a
  structured transcript to native model tokenizers, which apply real
  model-specific role tokens instead of flattening history into `User:` text.
- Removed LongCat's compiled 256-token context ceiling. Its finite Vulkan
  executor now allocates K/V and RoPE state from prompt plus generation, with
  an optional `--context-tokens` reserve bounded by available device memory.
- Added named `reference`, `full`, and warned research-only `fast`
  inference profiles for Qwen3.8 Flash Next, Qwen3-Coder-Next, and LongCat.
- Integrated Qwen3.8 Q3 experts and strict MTP verification, Qwen Coder Next
  Q3 experts, and LongCat's side-by-side selective-Q4 shared container without
  overwriting the retained reference runtimes.
- Added profile-aware resumable conversion, seven specialized Windows
  executables, the optimized shader set, release packaging, and CLI dispatch.
- Published paired quality/speed results, including the 1,000-question LongCat
  deep evaluation and the fresh 88 GB host-RAM speed rerun. Full is the default;
  quality-rejected fast paths remain explicit opt-in experiments.
- Revalidated the merged release binaries over three clean runs: LongCat full
  averaged 20.00 tok/s with 12/12 routes, while Qwen3.8 full averaged 8.49
  accepted tok/s with 10/10 routes and strict MTP verification.

## 0.3.0 — 2026-08-28

- Added revision-pinned download, conversion, native Vulkan inference, and
  localhost chat support for Qwen3.8-Flash-Next-FP8,
  Qwen3-Coder-Next-FP8, and LongCat-Flash-Lite-Sparse.
- Preserved each new model's validated shape-specialized kernels, expert-cache
  widths, weight formats, and runtime scheduling behind the unified `xtllm`
  launcher instead of collapsing them into a slower generic backend.
- Added Qwen3.8 official FP8 PLE lookup execution, Qwen3-Coder-Next's
  Top-10/512 hybrid decoder, and LongCat's dual-sublayer MLA, identity-expert,
  and BF16 n-gram path.
- Added the three specialized executables and their SPIR-V shaders to CMake,
  Windows release packaging, launcher dispatch, and the chat UI.
- Published RX 6700 XT validation measurements and explicitly rejected
  LongCat's faster but output-invalid all-Q4 shared-weight experiment.

- Renamed the public project, executable, launcher, package, and documentation
  to **XTLLM** (eXpert-Tier LLM); legacy launcher names remain compatibility aliases.
- Documented the validated RX 6700 XT and expected-compatible RDNA2, RDNA3,
  and RDNA4 families, clearly marked as pending hardware validation.
- Added the `xtllm` launcher for official, revision-pinned model download,
  resumable one-time conversion, native generation, planning, and localhost chat.
- Added a self-contained Windows release package, dependency bootstrap script,
  release checksums, and tag-driven GitHub release workflow.
- Simplified the Windows quick start to three launcher commands while keeping
  all inference in the existing model-specific Vulkan backends.
- Added capacity-only 16 GB and 24 GB VRAM throughput projections for all four
  supported backends, including projected model-specific cache selections and
  explicit uncertainty/measurement boundaries.
- Added a separately labeled optimistic scenario combining uncapped safe cache
  sizing with 1.6×/2.5× effective GPU-kernel throughput assumptions.

## 0.1.0 — 2026-08-16

- Packaged the experimental long-context engine that became XTLLM.
- Added BF16 host-K/V and exact chunked attention for Qwen3.6 and Nemotron.
- Preserved model-specific Qwen 122B, DeepSeek 284B, Qwen 35B, and Nemotron
  expert-cache/runtime paths behind one auto-detecting executable.
- Added automatic live Vulkan VRAM sizing, explicit RAM/context budgets, and
  bounded OOM retry inherited from the original engine.
- Added Windows build, benchmark methodology, Linux portability status, and
  repository governance files.
