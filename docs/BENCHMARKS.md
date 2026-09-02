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

These numbers are retained as the original Q4/Q8 backend baseline. They predate
the additive Q3, MTP, selective-Q4, and named-profile integration.

## Reference/full/fast profile validation

The later paired evaluation keeps prompt forms, scoring, and cache settings
matched within each model:

| Model | Reference tok/s | Full tok/s | Fast tok/s | Objective result |
|---|---:|---:|---:|---|
| Qwen3.8 Flash Next | 6.76 | 8.76 | 17.22 (early task end) | Reference/full 37/40 each; fast 9/40 strict, 17/40 lenient |
| Qwen3-Coder-Next | 11.17 | 12.01 | 23.17 | Reference 18/20; full 19/20; fast 0/20 strict, 13/20 lenient |
| LongCat Flash Lite | 16.27 | 21.37 | 22.78 | Deep paired results below |

The final merged-binary revalidation used the same fixed speed prompt and
three clean processes per profile. Qwen3.8 timed 127 output transitions per
run and LongCat timed 119. These are the release-headline values:

The 119-transition LongCat run is the fixed benchmark length, not an executor
limit. LongCat now sizes context at runtime and can execute beyond 256 total
positions when the resulting device-local K/V allocation fits in VRAM.

| Model | Reference runs (tok/s) | Full runs (tok/s) | Three-run average |
|---|---|---|---:|
| Qwen3.8 Flash Next | 6.40 / 6.40 / 6.38 | 8.49 / 8.46 / 8.51 accepted | **6.39 reference / 8.49 full** |
| LongCat Flash Lite | 15.45 / 15.62 / 15.31 | 20.28 / 19.44 / 20.28 | **15.46 reference / 20.00 full** |

Qwen3.8 full retained 10/10 verifier routes and measured 52.1% one-step greedy
MTP acceptance. LongCat full retained all 12/12 routes. The deeper quality
results below remain authoritative for the quality trade-offs.

The Qwen3.8 reference is Q4 10/10 original greedy. Full uses Q3 experts,
10/10 routes, and strict MTP verification; fast also changes the verifier to
7/10 routes and enables relaxed three-draft acceptance. Coder reference/full
compare Q4 versus Q3 while holding all 10 routes; fast uses 5/10. LongCat
reference is Q8-shared 12/12, full is selective-Q4 12/12, and fast is the same
container at 7/12.

LongCat's final study used 1,000 paired multiple-choice questions, 30
generative tasks, 30 code tasks, a 60-item format audit, and 21 fixed speed
trials. Lenient objective scores were 75.0% reference, 72.2% full, and 67.2%
fast. Reference-to-full was -2.8 percentage points (95% CI -5.1 to -0.6,
paired p=0.0167); full-to-fast was -5.0 points (95% CI -7.9 to -2.2,
p=0.000816). Fast was only 6.58% faster than full.

After the test host was upgraded to 88 GB RAM, a fresh ten-run LongCat full
check averaged 21.28 tok/s over steady trials 2-10 (21.19 including cold,
median 21.29). The cache held all 3,584 expert records in 16.7344 GiB and
reported zero expert SSD reads, explaining why a larger RAM budget did not
increase throughput.

See [Validated inference profiles](INFERENCE_PROFILES.md) for commands,
container layouts, quality cautions, and reporting requirements.

Projected 16/24 GB VRAM figures are documented separately in
[VRAM_PROJECTIONS.md](VRAM_PROJECTIONS.md). They are capacity-only estimates
anchored to the measured 12 GB results and must not be reported as benchmarks.
