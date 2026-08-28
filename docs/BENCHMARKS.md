# Benchmark methodology

The headline table uses an AMD Radeon RX 6700 XT 12GB, Ryzen 5 3600, Windows
10 22H2, an NTFS NVMe expert store, greedy decoding, and the same ordinary chat
prompt. The timer excludes model initialization/prefill and reports completed
decode transitions; 24 emitted tokens correspond to 23 timed transitions.

System profiles map to explicit expert/model RAM budgets:

| Installed RAM profile | Expert/model budget |
|---:|---:|
| 16 GiB | 12 GiB |
| 24 GiB | 20 GiB |
| 32 GiB | 28 GiB |
| 64 GiB | up to 56 GiB (49.25 GiB for Qwen 122B test) |

Long-context tests populate a real model-generated K/V entry across the
configured history before the timer. They are decode-throughput measurements,
not proof that a repeated synthetic history has realistic language quality.
Short-context token-sequence regressions are run separately.

Numbers are single-machine measurements, not vendor-independent promises.
Desktop display load, SSD temperature, driver version, and cache warmness can
move short runs. Publish the command, timed-transition count, selected cache
slots, actual RAM/VRAM, and SSD/H2D traffic with new results.

## XTLLM 0.3.0 backend validation

Qwen3.8 Flash Next, Qwen3-Coder-Next, and LongCat Flash Lite Sparse were run on
the same RX 6700 XT using greedy decoding and the fixed prompt:

> In two concise paragraphs, explain why the sky appears blue, then give a
> short Python function that checks whether an integer is prime.

Qwen3-Coder-Next and LongCat figures time 63 output transitions. Qwen3.8's
profile sweep uses the original three-prompt chat/reasoning/code mix with 23
timed transitions per prompt. No MTP or speculative tokens are counted. The
new-backend RAM columns in the README are explicit model/expert budgets, not
installed-system capacities. The consolidated 0.3.0 binaries passed a final
uncontended 24 GiB generation check at 4.94 tok/s (Qwen3.8), 8.76 tok/s
(Qwen3-Coder-Next), and 11.81 tok/s (LongCat), all with coherent output. The
README retains the longer/profiled headline measurements rather than replacing
them with this single short release check.

Projected 16/24 GB VRAM figures are documented separately in
[VRAM_PROJECTIONS.md](VRAM_PROJECTIONS.md). They are capacity-only estimates
anchored to the measured 12 GB results and must not be reported as benchmarks.
