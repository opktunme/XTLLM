# Projection: a roughly 70B MoE on RX 6700 XT

This is a capacity/performance projection, not a benchmark. It assumes a
text-only, Q4-class, approximately 70B-parameter MoE with about 5B active
parameters per token, 40–48 layers, top-8 routing, a 36–39 GiB converted
runtime, and expert record sizes between the existing Qwen 35B and Qwen 122B
backends.

| System RAM | Expert/inference budget | Projected short-context tok/s | Projected SSD traffic/output | Likely limit |
|---:|---:|---:|---:|---|
| 16 GiB | 12 GiB | 7–10 | 200–350 MiB | cold expert acquisition |
| 24 GiB | 20 GiB | 10–14 | 50–200 MiB | mixed acquisition/compute |
| 32 GiB | 28 GiB | 12–16 | 0–50 MiB | GPU expert+dense compute |
| 64 GiB | up to 56 GiB | 12–17 | approximately 0 | GPU compute/VRAM traffic |

Expected peak VRAM is 8–10.5 GiB with model-specific automatic cache sizing.
For a Qwen-Next-style hybrid with 10–12 full-attention layers and BF16 K/V,
32K context would consume about 0.6–0.9 GiB of host RAM and is projected at
roughly 65–80% of the corresponding short-context speed.

The range is deliberately broad. Routing locality and exact active parameters
matter more than the headline 70B total: Nemotron 30B A3B and Qwen 35B A3B are
compute-bound once warm, while Qwen 122B A10B remains acquisition-heavy at
smaller RAM budgets.
