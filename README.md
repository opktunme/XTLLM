<p align="center">
  <img src="docs/assets/hero.svg" alt="XTLLM" width="100%">
</p>

<p align="center">
  <img alt="Windows" src="https://img.shields.io/badge/Windows-10%2F11-2563eb">
  <img alt="Vulkan" src="https://img.shields.io/badge/Vulkan-1.3-e11d48">
  <img alt="C++" src="https://img.shields.io/badge/C%2B%2B-17-00599c">
  <img alt="License" src="https://img.shields.io/badge/license-Apache--2.0-22c55e">
</p>

XTLLM (eXpert-Tier LLM) is an experimental, text-only Vulkan inference engine for
running very large mixture-of-experts models on consumer AMD GPUs. It keeps a
model-specific working set in VRAM, a budgeted expert tier in system RAM, and
the cold remainder on NVMe—while long-context K/V state lives in explicitly
accounted host memory.

It preserves the proven short-context backends and adds bounded long-context
execution for Qwen3.5, DeepSeek V4,
Qwen3.6, and Nemotron 3 Nano. Model weights are **not** included.

> **Status:** Windows/RDNA2 research preview. Greedy text generation is working.
> The source is intentionally shape-specialized and is not a general model loader.

## Why this exists

Most runtimes treat a model larger than VRAM as an exceptional case. XTLLM
treats the hierarchy as the design:

```mermaid
flowchart LR
    NVMe["NVMe cold expert store"] -->|bounded async reads| RAM["Budgeted RAM expert cache"]
    RAM -->|rank-ready copies| VRAM["Model-specific VRAM expert cache"]
    VRAM --> GPU["Finite Vulkan Q4 / NVFP4 kernels"]
    KV["BF16 long-context K/V in host memory"] --> GPU
    GPU --> OUT["Greedy output token"]
```

- one executable auto-detects the converted model;
- live Vulkan memory-budget sizing chooses a safe model-specific expert cache;
- explicit `--ram-gib` and `--device-slots` overrides remain available;
- authoritative routing is preserved—no expert dropping, rerouting, or Q3;
- all submissions are finite and bounded; XTLLM never changes Windows TDR.

## GPU compatibility

Only the RX 6700 XT configuration is presently hardware-validated. The other
families below are **expected-compatible pending real validation** because they
provide the required AMD Vulkan feature set; this is not a claim that every
model or cache profile fits every card's VRAM.

| Architecture | Status | Expected card families and examples |
|---|---|---|
| RDNA2 | **Validated: RX 6700 XT 12GB**; other cards expected-compatible | Radeon RX 6600/6650 XT, 6700/6750 XT, 6800/XT, 6900/6950 XT; Radeon PRO W6000 series |
| RDNA3 | Expected-compatible, pending validation | Radeon RX 7600/XT, 7700 XT, 7800 XT, 7900 GRE/XT/XTX; Radeon PRO W7000 series |
| RDNA4 | Expected-compatible, pending validation | Radeon RX 9060 XT, 9070, and 9070 XT families |

Requirements are Vulkan 1.3, 64-lane subgroup support, sufficient device-local
VRAM for the selected model's fixed footprint, and current AMD drivers. Cards
with 8GB or less may require very small expert caches or may not fit a backend's
fixed allocations. AMD APUs/shared-memory graphics, pre-RDNA GPUs, Linux native
inference, and non-AMD Vulkan devices remain unvalidated in this snapshot.

XTLLM is independent software and is not affiliated with or endorsed by AMD.
AMD, Radeon, and RDNA are trademarks of Advanced Micro Devices, Inc.

## Supported model backends

| Model | Text parameters | Active/token | Routed experts | Runtime precision | Converted size |
|---|---:|---:|---:|---|---:|
| Qwen3.5-122B-A10B | 122.11B | 9.77B | 8 / 256 | Q4G64T; Q8 embed/head/router | 61.1 GiB |
| DeepSeek-V4-Flash-0731 | 284B | ~13B | 6 / 256 | Q4G64T experts/shared; Q8 global/router | ~141.2 GiB |
| Qwen3.6-35B-A3B | ~35B | ~3B | 8 / 256 | Q4G64T; Q8 embed/head/router | 17.62 GiB |
| NVIDIA Nemotron-3-Nano-30B-A3B | ~30B | ~3B | 6 / 128 | native E2M1 NVFP4/BF16-K16; Q8 global | 18.61 GiB |

Vision and draft/MTP namespaces are not executed in these text-only numbers.

## Measured performance

RX 6700 XT 12GB, Ryzen 5 3600, NVMe SSD, Windows 10, greedy decode, Q4/NVFP4
quality. Short-context figures use 23 timed token transitions and prewarmed
budgeted caches. “System RAM” is a profile; the engine budget leaves OS headroom.

| Model | 16 GiB system | 24 GiB system | 32 GiB system | 64 GiB system | Primary limit |
|---|---:|---:|---:|---:|---|
| Qwen3.6-35B-A3B | 16.33 tok/s | 19.35 tok/s | 19.53 tok/s | 19.64 tok/s | GPU once warm |
| Nemotron-3-Nano-30B-A3B | 14.53 tok/s | 21.87 tok/s | 21.84 tok/s | 22.08 tok/s | GPU once warm |
| Qwen3.5-122B-A10B | 3.35 tok/s | 3.96 tok/s | 4.38 tok/s | 4.77 tok/s | expert acquisition/traffic |
| DeepSeek-V4-Flash-0731 | 1.99 tok/s | 2.60 tok/s | 2.77 tok/s | 2.70 tok/s | expert acquisition/traffic |

### Projected effect of more VRAM

The following is a **capacity-only projection**, not a benchmark on those GPUs.
It holds the RX 6700 XT's compute and memory bandwidth constant, uses the 24 GiB
system-RAM profile, and changes only the live Vulkan memory budget and resulting
model-specific expert cache. A real 16 or 24 GB GPU may be faster or slower
depending on its compute, bandwidth, driver, and display load.

| Model | 12 GB measured | 16 GB projected | 24 GB projected | Expert-cache capacity (12 / 16 / 24 GB) |
|---|---:|---:|---:|---:|
| Qwen3.6-35B-A3B | 19.35 tok/s | **21–23 tok/s** | **21–23 tok/s** | 96 / 128 / 128 per layer |
| Nemotron-3-Nano-30B-A3B | 21.87 tok/s | **26–29 tok/s** | **29–33 tok/s** | 60 / 91 / 128 per MoE layer |
| Qwen3.5-122B-A10B | 3.96 tok/s | **4.1–4.3 tok/s** | **4.1–4.3 tok/s** | 28 / 32 / 32 per layer |
| DeepSeek-V4-Flash-0731 | 2.60 tok/s | **2.8–3.0 tok/s** | **2.8–3.0 tok/s** | 460 / 559 / 559 global records |

The repeated 16/24 GB ranges are intentional: the current bounded engine reaches
that backend's validated cache ceiling by 16 GB. More VRAM still provides safety
headroom and context/workspace capacity, but is not projected to raise
short-context decode until that ceiling is increased and measured. See the
[assumptions and calculation](docs/VRAM_PROJECTIONS.md).

#### Optimistic best-case

This second view deliberately removes those validated cache ceilings and also
assumes a faster GPU: approximately **1.6× effective kernel throughput** for the
16 GB scenario and **2.5×** for 24 GB, relative to the RX 6700 XT. It represents
a favorable upper scenario, not what VRAM capacity alone guarantees.

| Model | 16 GB best-case projection | 24 GB best-case projection | Uncapped expert cache (16 / 24 GB) |
|---|---:|---:|---:|
| Qwen3.6-35B-A3B | **35–40 tok/s** | **55–65 tok/s** | 193 / 256 per layer |
| Nemotron-3-Nano-30B-A3B | **38–43 tok/s** | **75–90 tok/s** | 91 / 128 per MoE layer |
| Qwen3.5-122B-A10B | **5.8–6.8 tok/s** | **10–13 tok/s** | 44 / 75 per layer |
| DeepSeek-V4-Flash-0731 | **3.8–4.6 tok/s** | **8–10 tok/s** | 778 / 1,348 global records |

The top ends require favorable route locality, idle VRAM, efficient scaling of
the current Vulkan kernels, and validation of larger cache widths. They should
be treated as targets for measurement on named GPUs, not published results.

### Long-context decode

These measurements allocate and populate the requested history before timing a
small decode. Exact mode scans the full history; Fast is the fork's bounded
hierarchical candidate path. The smaller-model rows were measured in this
packaged build with the 12 GiB expert-RAM profile.

| Model | Context | Exact | Fast | Host K/V allocation |
|---|---:|---:|---:|---:|
| Qwen3.6-35B-A3B | 32K | **10.825 tok/s** | — | 0.625 GiB |
| Nemotron-3-Nano-30B-A3B | 64K | **6.556 tok/s** | — | 0.375 GiB |
| Qwen3.5-122B-A10B | 32K | 3.370 tok/s | 3.646 tok/s | 0.750 GiB |
| Qwen3.5-122B-A10B | 128K | 2.531 tok/s | 3.509 tok/s | 3.000 GiB |
| Qwen3.5-122B-A10B | 256K | 1.924 tok/s | 3.396 tok/s | 6.000 GiB |
| DeepSeek-V4-Flash-0731 | 128K | 1.106 tok/s | 1.270 tok/s | model-specific compressed history |
| DeepSeek-V4-Flash-0731 | ~1M | 0.977 tok/s | 2.023 tok/s | model-specific compressed history |

Fast long-context mode is an explicit speed/recall trade-off. Use `exact` when
full learned history is required. See [Benchmark methodology](docs/BENCHMARKS.md).

## Quick start — Windows

The release ZIP includes the native engine, compiled Vulkan shaders, model
converters, and launcher. You do **not** need CMake, a C++ compiler, or the
Vulkan SDK unless you want to modify XTLLM.

Prerequisites: Windows 10 22H2/11, an AMD Vulkan 1.3 driver, Python 3.11+, and
an NVMe drive large enough for both the official checkpoint and converted
runtime. Model weights are downloaded from their official publisher and are
never redistributed by XTLLM.

### 1. Install the launcher

Download and extract `xtllm-*-windows-x64.zip` from GitHub Releases,
open PowerShell in the extracted directory, then run:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\install-windows.ps1
.\xtllm.cmd doctor
.\xtllm.cmd models
```

The installer creates a private `.venv` and installs only the Python packages
needed for downloading and one-time conversion. Inference remains the native
XTLLM Vulkan executable.

### 2. Download and convert one model

Choose a model alias and a drive with enough free space:

```powershell
.\xtllm.cmd setup qwen36 --model-root D:\XTLLM-Models
```

`setup` pins the official checkpoint revision, checks disk space, performs a
resumable Hugging Face download, runs the correct model-specific converter,
and verifies the resulting files. Conversion is one-time CPU/file work and can
take hours for the largest models. A setup interrupted during download or
expert conversion can be run again safely.

| Alias | Official checkpoint | Approx. checkpoint + runtime + peak scratch |
|---|---|---:|
| `qwen36` | `Qwen/Qwen3.6-35B-A3B` | 88 GiB |
| `nemotron` | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4` | 40 GiB |
| `qwen122` | `Qwen/Qwen3.5-122B-A10B` | 300 GiB |
| `deepseek` | `deepseek-ai/DeepSeek-V4-Flash-0731` | 450 GiB |

Set `XTLLM_MODELS` once if you do not want to repeat `--model-root`:

```powershell
[Environment]::SetEnvironmentVariable('XTLLM_MODELS', 'D:\XTLLM-Models', 'User')
```

Open a new PowerShell window after setting it.

### 3. Chat or generate

```powershell
.\xtllm.cmd chat qwen36
```

This opens a private localhost chat page. For a one-shot prompt:

```powershell
.\xtllm.cmd run qwen36 "Explain why the sky appears blue." --tokens 128
```

Hardware and model type are auto-detected. The native backend selects its
proven model-specific cache policy from live system RAM and Vulkan VRAM budgets;
use `plan` to inspect that decision without running inference:

```powershell
.\xtllm.cmd plan qwen36
```

Advanced overrides such as `--ram-gib`, `--context-tokens`, and
`--device-slots` are optional. A clean expert-cache allocation OOM gets one
bounded retry at a smaller cache size.

### Build from source

Developers need the Vulkan SDK, CMake 3.25+, Ninja, Python 3.11+, and
LLVM-MinGW `clang++.exe`:

```powershell
git clone https://github.com/opktunme/XTLLM.git xtllm
cd xtllm
$env:PATH = "C:\llvm-mingw\bin;$env:PATH"
powershell -ExecutionPolicy Bypass -File .\scripts\build-windows.ps1
```

The executable and SPIR-V shaders are written together under `build\`.
Converted weights must not be committed; `.gitignore` excludes `.ovs`, `.ovx`,
and `.ovb` model assets. Keep each checkpoint's license and model-card terms.

## Linux

Python conversion works on Linux, but native inference in this snapshot is not
yet supported because the honest expert tier uses Windows-specific direct I/O
and memory APIs. The exact dependencies and bounded porting checklist are in
[docs/LINUX.md](docs/LINUX.md). Wine is not presented as a supported shortcut.

## Memory and context controls

```text
--ram-gib N          expert/model host-memory budget
--context-gib N      separately accounted host K/V budget
--context-tokens N   exact context capacity (overrides context-gib sizing)
--long-mode exact    full-history attention
--long-mode fast     bounded hierarchical long attention where supported
--long-mode auto     model-specific crossover
--device-slots N     exact expert-cache override
--no-prewarm         skip RAM-cache top-off
--plan               print selected hardware/memory policy without inference
```

The launcher rejects a requested RAM + context allocation that leaves less than
2 GiB currently available. K/V memory is BF16 for Qwen/Nemotron long paths.
Capacity is not the same as utilized context: a short chat remains fast even
when a large maximum window is reserved.

## Architecture notes

- **Qwen3.5/3.6:** hybrid Gated DeltaNet + full attention; only full-attention
  layers grow K/V state with context.
- **Nemotron:** Mamba/recurrent layers are fixed-state; six attention layers
  own the growing K/V cache.
- **DeepSeek:** MLA/mHC and compressed history use a model-specific indexing
  path; the generic Qwen cache layout is not forced onto it.
- **Experts:** RAM and VRAM policies stay model-specific. A single generic cache
  policy was deliberately rejected because it regressed proven paths.

## Safety boundaries

- no persistent or unbounded compute shaders;
- no GPU busy-spin waiting for CPU work;
- finite command-buffer rings and finite I/O waits;
- no TDR registry changes;
- no hidden filesystem cache as a claimed inference tier in budgeted modes;
- no Q3, expert dropping, substitution, pruning, or changed router decisions.

## 70B MoE projection

A hypothetical Q4 ~70B/A5B MoE is projected at **7–10 tok/s** on the 16 GiB
system profile and **10–14 tok/s** on the 24 GiB profile, reaching roughly
**12–17 tok/s** once its expert store is effectively warm. Assumptions,
traffic ranges, and uncertainty are documented in
[docs/PROJECTED_70B_MOE.md](docs/PROJECTED_70B_MOE.md).

## Contributing

Read [CONTRIBUTING.md](CONTRIBUTING.md), the safety invariants above, and
[SECURITY.md](SECURITY.md) before opening a pull request. Performance changes
need an end-to-end A/B, exact token IDs for the regression prompt, and memory/
traffic accounting—not only a kernel microbenchmark.

## License

Engine source: Apache-2.0. Model checkpoints and converted weights retain their
original licenses and are not redistributed by this repository.
