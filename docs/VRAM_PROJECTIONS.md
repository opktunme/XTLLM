# VRAM capacity projections

These figures project the combined effect of larger expert caches and the
effective kernel throughput expected from representative 16 GB and 24 GB GPUs.
They are **not measurements on those GPUs**.

## Projection table

The host configuration is fixed at the measured 24 GiB system-RAM profile
(20 GiB inference/model budget), short context, greedy decoding, and 23 timed
token transitions.

| Model | 12 GB measured | 16 GB projected | 24 GB projected | Cache records selected at 16 / 24 GB |
|---|---:|---:|---:|---:|
| Qwen3.6-35B-A3B | 19.35 tok/s | 35–40 tok/s | 55–65 tok/s | 193 / 256 per layer |
| NVIDIA Nemotron-3-Nano-30B-A3B | 21.87 tok/s | 38–43 tok/s | 75–90 tok/s | 91 / 128 per MoE layer |
| Qwen3.5-122B-A10B | 3.96 tok/s | 5.8–6.8 tok/s | 10–13 tok/s | 44 / 75 per layer |
| DeepSeek-V4-Flash-0731 | 2.60 tok/s | 3.8–4.6 tok/s | 8–10 tok/s | 778 / 1,348 global records |

The projection combines:

- use of all safe cache capacity beyond today's validated backend ceilings;
- 1.6× effective Vulkan-kernel throughput for a strong 16 GB GPU;
- 2.5× effective Vulkan-kernel throughput for a high-end 24 GB GPU;
- the same 24 GiB system-RAM profile, SSD, prompts, and model quality;
- idle display/driver memory and representative expert-route locality.

The ranges account for differences between specific cards and workloads;
`24 GB` alone does not identify one compute or bandwidth class.

## Method

The calculation uses the unified engine's actual model-specific sizing inputs:

| Backend | Fixed device footprint | Device cost of one cache-width increment | Validated ceiling |
|---|---:|---:|---:|
| Qwen3.6 35B | 1.812 GiB | 0.062256 GiB | 128 records/layer |
| Nemotron 30B | 1.574 GiB | 0.133625 GiB | 128 records/MoE layer |
| Qwen3.5 122B | 3.976 GiB | 0.224121 GiB | 32 records/layer |
| DeepSeek 284B | 4.150 GiB | 0.012451 GiB/record | 559 global records |

For the hypothetical cards, the idle live Vulkan budget is scaled from the
measured RX 6700 XT ratio (11.205 GiB budget from an 11.984 GiB heap). The
engine's existing reserve is then applied: 7.5% of the live budget, clamped to
0.5–1.5 GiB. This gives approximately 15.0 GiB usable Vulkan budget on a 16 GB
card and 22.4 GiB on a 24 GB card before the model reserve and cache allocation.

Throughput ranges extrapolate the measured device-hit/miss curve and acquisition
time at 12 GB, while scaling non-acquisition work by the expected effective
kernel throughput. The bounds account for cache locality and overlap uncertainty
rather than treating every eliminated cache miss as fully serialized.

## Interpretation

- **Nemotron benefits most.** At 24 GB the current cache can hold all 128
  experts for every MoE layer, eliminating expert H2D acquisition after warmup.
- **Qwen3.6 benefits moderately.** It reaches the current 128-record/layer
  ceiling at 16 GB, but that is still only half of each layer's 256 experts.
- **Qwen3.5 122B and DeepSeek remain acquisition-bound.** Both reach their
  current validated cache ceiling at 16 GB, so 24 GB adds headroom but no further
  cache width in this projection.
- A faster GPU should be modeled separately. More compute units or bandwidth do
  not scale SSD and host acquisition time, while more VRAM does not make the
  retained kernels execute faster by itself.

Long-context throughput is not projected here. Its attention/history cost is
context-length dependent, so scaling the short-context table would be
misleading. It should be measured on the target GPU.

## How the projection was formed

The measured 12 GB per-token time is separated into acquisition-visible time
and the remaining model work. Expert misses are extrapolated from the observed
hit/miss curve to the uncapped cache width. The reduced acquisition term is
then added to the non-acquisition term divided by 1.6 or 2.5. The published
ranges widen that result for overlap and route-locality uncertainty.

The calculation assumes larger cache allocations remain stable, kernel
throughput scales as modeled, and acquisition saved by a cache hit was on the
critical path. The resulting ranges are projections rather than measured
performance.
