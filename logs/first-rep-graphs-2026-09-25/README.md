# The slow first repetition is HIP graph instantiation, 2026-09-25

## What was open

[`logs/decode-variance-clock-2026-09-22/`](../decode-variance-clock-2026-09-22/) established that
`llama-bench`'s first repetition reads low, at depth and at depth 0 alike, and eliminated five
candidates: CPU pinning, page cache, the hardware queue, address-space randomisation and thermal duty
cycle. It also showed the effect survives at `-d 0`, where neither the prefill nor the state restore
runs, so whatever it is does not need either. What it is was left open.

## How to see it at all

`llama-bench`'s table prints a mean and a standard deviation, which hides this completely. `-o jsonl`
carries `samples_ts`, one entry per repetition, and that is what every number here is read from. The
model is the 1.5B at depth 0, which loads in seconds, so the question is answerable
cheaply instead of needing the 8B at a primed depth.

## It is a fixed cost, and not the clock

Four invocations per cell, `-r 6`, ratio is repetition 1 over the mean of repetitions 2 to 6
(`measurements.txt`, `rep1b.sh`):

| tokens generated | GPU pre-warmed to 1500 MHz | cold, governor at 1000 MHz |
|---|---|---|
| 8 | 0.921, 0.924, 0.925 | 0.917, 0.927, 0.926, 0.928 |
| 32 | 0.974, 0.974, 0.979, 0.986 | 0.977, 0.979, 0.979 |
| 128 | 0.996, 0.988, 0.984, 0.994 | 0.996, 0.991, 0.996, 0.990 |

**Pre-warming the GPU changes nothing.** Given that a clock artefact was found in this repository's
arithmetic ceilings the same day, that was the tempting answer and it is wrong.

**The deficit shrinks with generation length**, which is what a cost paid once per repetition must do.
Solving for that cost gives 0.66 tokens-worth at n=8 and 0.71 at n=32, which agree; at n=128 the ratio
is so close to one that the estimate is poorly conditioned and reads 1.05. Against about 197 tokens a
second, 0.7 tokens is roughly 3.5 ms.

## It is graph instantiation

Ten invocations per arm, `-n 8 -r 6`, differing only in `GGML_CUDA_DISABLE_GRAPHS` (`rep1d.sh`):

| | n | mean ratio | sd | first-repetition deficit |
|---|---|---|---|---|
| HIP graphs on | 9 | 0.9248 | 0.0056 | **7.5 %** |
| HIP graphs off | 8 | 0.9934 | 0.0016 | **0.7 %** |

**Disabling HIP graphs removes 91 percent of it.** The first repetition is where the graph is captured
and instantiated; later repetitions replay it. That is a once-per-repetition-zero cost of the right
size, it disappears when the feature is switched off, and the spread with it off is three times
tighter.

## What is left

Two things, and neither is the effect above.

A residual 0.7 percent with graphs off, which is small enough that this design cannot say whether it is
real.

Occasional much larger deficits, which appear with graphs on and off alike and are a second, separate
effect. **They are the governor ramping.** Thirty invocations with graphs off and the normal 1000 to
1500 MHz policy give 7 outliers at 0.74 to 0.82 against 23 clean at 0.9961; twenty invocations with the
governor pinned at 1500 give none at all (`clock_ramp_outliers.txt`):

| governor | invocations | clean mean | outliers |
|---|---|---|---|
| 1000 to 1500, the normal policy | 30 | 0.9961 | **7, 23 percent**, at 0.74 to 0.82 |
| pinned at 1500 | 20 | 0.9940 | **none** |

The affected invocations show it across the first two repetitions, not one, for example
`145.6, 165.3, 198.0, 197.8, 197.8, 197.5`, which is a clock climbing instead of a fixed cost being
paid once. They are not the per-process dispatch bimodality of
[`logs/dispatch-bimodal-2026-09-23/`](../dispatch-bimodal-2026-09-23/) despite the similar size: that
one slows every dispatch for the life of the process, and here the later repetitions are normal.

So the first repetition has two unrelated problems. Graph instantiation costs it about 7.5 percent
every time, and roughly one invocation in four additionally starts before the governor has reached
its cap, which costs about 25 percent across the first two repetitions. Pre-warming the GPU does not
fix the first and pinning the clock does not fix the second.

## Why it matters for anything measured here

A short `tg` measurement with few repetitions pays this once and reports it as throughput. At `-n 8`
it is 7.5 percent. That is the mechanism behind the advice already in
[README.md](../../README.md#how-to-measure-on-this-board) to take medians of several samples, and it
is a reason the campaigns here use `-n 64` or more, where the same cost is under one percent.
