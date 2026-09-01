# Validated inference profiles

XTLLM exposes three named inference profiles for Qwen3.8 Flash Next,
Qwen3-Coder-Next, and LongCat Flash Lite. The names describe the evaluation
role, not one universal quantization:

| Profile | Meaning |
|---|---|
| `reference` | Highest-fidelity retained implementation used as the comparison baseline |
| `full` | Optimized implementation with every authoritative MoE route retained |
| `fast` | Explicit research tradeoff that evaluates fewer routes or relaxes verification |

`full` is the default for all three models. `fast` is never selected
automatically and the launcher prints a quality warning when it is requested.

## Commands

```powershell
# Convert the default full profile and run it.
.\xtllm.cmd setup qwen38 --profile full --model-root A:\XTLLM-Models
.\xtllm.cmd run qwen38 "Explain sparse MoE routing." --profile full

# Retained comparison baseline.
.\xtllm.cmd run qwen38 "Explain sparse MoE routing." --profile reference

# Research-only reduced route/relaxed profile.
.\xtllm.cmd run longcat "Explain sparse MoE routing." --profile fast

# Show the selected binary, memory policy, route policy, and warning.
.\xtllm.cmd plan longcat --profile full
```

Setup is additive. Profile-specific containers are written beside the retained
reference containers; selecting one profile does not overwrite another.

## What each model runs

| Model | Reference | Full | Fast |
|---|---|---|---|
| Qwen3.8 Flash Next | Q4 experts, original greedy, 10/10 routes | Q3 experts, strict MTP verification, 10/10 routes | Q3 experts, relaxed three-draft MTP, 7/10 verifier routes |
| Qwen3-Coder-Next | Q4 experts, 10/10 routes | Q3 experts, 10/10 routes | Q3 experts, 5/10 routes |
| LongCat Flash Lite | Q4 experts plus Q8 shared/dense, 12/12 routes | selective-Q4 dense/output plus Q8 attention core, 12/12 routes | same selective-Q4 layout, 7/12 routes |

The validated short-context capacities are 2,048 total tokens for Qwen3.8 and
Qwen3-Coder-Next, and 256 total tokens for the current LongCat executor. The
quality and speed tests below used ordinary prompts within those caps; they are
not long-context benchmarks.

## Paired quality and speed results

Hardware was an RX 6700 XT 12 GB with a Ryzen 5 3600 and NVMe storage. Decode
throughput excludes initialization and prompt prefill.

| Model/profile | Decode tok/s | Objective quality | Disposition |
|---|---:|---|---|
| Qwen3.8 reference | 6.76 | 37/40 (92.5%) | Reference |
| Qwen3.8 full | 8.76 | 37/40 (92.5%); 38/40 answer agreement with reference | **Default** |
| Qwen3.8 fast | 17.22, ending the fixed task early | 9/40 strict; 17/40 lenient | Rejected for general use |
| Coder reference | 11.17 | 18/20 (90%) | Reference |
| Coder full | 12.01 | 19/20 (95%) | **Default** |
| Coder fast | 23.17 | 0/20 strict; 13/20 lenient | Rejected for general use |
| LongCat reference | 16.27 | See deep test below | Reference |
| LongCat full | 21.37 | See deep test below | **Default** |
| LongCat fast | 22.78 | See deep test below | Research-only |

Qwen3.8 full therefore improved measured decode throughput by 29.5% over the
Q4 reference without changing the aggregate 40-question score. That result
does not mean Q3 is mathematically lossless: one reference-correct answer was
lost and one reference error was rescued. It means no aggregate loss was
detected in this paired form.

Immediately before packaging the merged binaries, three clean repetitions of
the fixed speed prompt averaged **6.39 reference / 8.49 accepted full tok/s**
for Qwen3.8 and **15.46 reference / 20.00 full tok/s** for LongCat. These are
the README release headlines. The paired table above remains here because its
speed and objective-quality results came from the same evaluation run.

## LongCat deep evaluation

The final LongCat study used 1,000 paired multiple-choice questions, 30
generative tasks, 30 code tasks, 60 single-item format audits, and 21 fixed
speed trials.

| Comparison | Result |
|---|---|
| Reference objective score | 75.0% lenient |
| Full objective score | 72.2%; -2.8 percentage points vs reference (95% CI -5.1 to -0.6, paired p=0.0167) |
| Fast objective score | 67.2%; -5.0 points vs full (95% CI -7.9 to -2.2, paired p=0.000816) |
| Generative pass counts | reference 22/30, full 18/30, fast 18/30 |
| Code pass counts | reference 22/30, full 22/30, fast 21/30 |
| Median/mean speed conclusion | full 21.37 tok/s; fast 22.78 tok/s, only 6.58% faster |

Fast also produced malformed strict answer formats on 20/60 audited
single-item prompts, usually `ANS:1,<letter>`, while reference and full were
60/60. The small speed gain does not justify making it the default.

A separate fresh ten-run LongCat full check after the host was upgraded to
88 GB RAM measured 21.28 tok/s over steady trials 2-10 (21.19 including the
cold run; median 21.29). The runtime cached all 3,584 expert records in 16.7344
GiB and performed zero expert SSD reads. Raising the configured host budget
above 24 GiB cannot retain more than all experts, so the extra installed RAM
did not materially increase decode speed.

## Reproduction and reporting

Report the profile name, converted container layout, active route count,
configured and actual RAM cache, device slots, prompt, timed transitions, and
whether initialization/prefill was excluded. A speed result from `fast`
must carry its paired quality result; it must not be presented as equivalent
to `full` or `reference`.
