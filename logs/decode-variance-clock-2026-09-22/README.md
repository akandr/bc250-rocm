# The decode-at-depth variance, a fifth elimination, 2026-09-22

[`logs/decode-variance-state-2026-08-21/`](../decode-variance-state-2026-08-21/) measured qwen3-8B Q8_0
generating 8 tokens at a primed depth of 16128 and found a coefficient of variation of 7.7 to 8.8
percent between invocations against much less inside one, which places the variable in whatever differs
per process. Four candidates have been eliminated since, one at a time: CPU pinning, page cache and
compaction, the hardware queue the runtime is given, and address-space randomisation. Thermal was
eliminated too, on the grounds that temperature was flat.

That last one was worth asking again. On 22 September the governor turned out to explain a different
set of dropouts, at depth and under sustained load, dropping the shader clock from 1500 to 1000 MHz at
93 C ([`logs/fault-repro-2026-09-22/`](../fault-repro-2026-09-22/)). Every invocation of this
measurement prefills 16128 tokens before generating its eight, which is real heat, and the original
design ran them back to back.

## The arms

[`scripts/variance_thermal.sh`](../../scripts/variance_thermal.sh), ten interleaved pairs, identical
except for the pause before the invocation: `hot` runs back to back as the original did, `cool` idles
for ninety seconds first. mmap is left at `llama-bench`'s default, as the original had it, which matters
([`logs/nommap-ceiling-2026-09-22/`](../nommap-ceiling-2026-09-22/)).

| arm | n | mean tg8 | sd | cv | min | max |
|---|---|---|---|---|---|---|
| hot | 10 | 27.36 | 2.77 | **10.1 %** | 19.14 | 28.90 |
| cool | 10 | 26.97 | 2.18 | **8.1 %** | 23.05 | 29.21 |

    hot   28.16 28.30 27.80 27.67 27.87 28.70 28.35 28.74 19.14 28.90
    cool  28.37 29.21 23.05 28.27 24.82 23.37 28.65 28.06 27.99 27.94

The governor logged no throttle at any point and the edge sensor stayed between 57 and 71 C.

## What it says

**Thermal is eliminated, and in the direction that matters.** Not merely "temperature was flat" but
that the arm which never cools is not the noisier one, at a temperature twenty degrees below where the
governor intervenes.

**The variance reproduces.** 8 to 10 percent against August's 7.7 to 8.8, on the thirteen-patch build a
month later, on a different kernel. Whatever this is, it has survived everything done to the software
between those two dates.

**Duty cycle is the fifth elimination**, and it joins the other four.

## An effect that did not survive its own last two samples

Worth recording because it is the fourth time this board has done it. At eight pairs the table read
very differently: the hot arm was 27.67 to 28.74, a 3.8 percent spread, and the cool arm held three
readings of 23.05, 24.82 and 23.37 against a 28 baseline, a 27 percent spread. That is a clean story
with a mechanism ready to hand, that ninety seconds of idle lets the clock fall and a short measurement
then pays for the ramp, and it was written down as a finding.

The ninth pair put 19.14 in the hot arm, lower than anything the cool arm produced, and the cool arm's
last three came back at 28.06, 27.99 and 27.94. Ten pairs leave the two indistinguishable.

Three earlier effects on this board did the same thing: memory compaction at six runs per arm, the
hardware-queue bit at six runs, and address-space randomisation at three runs per arm, each convincing
and each reversed or erased by resampling. This is the fourth, and the lesson is not that any of them
was carelessly done. It is that effects of this size cannot be established here at the sample sizes
that feel sufficient while the data is arriving.

## The timed window is under a second long

The elimination above was the fifth, and it prompted a look at the measurement, not the machine.
This configuration generates **eight tokens** after a 16128-token prefill. At about 28 t/s that is 0.28
seconds of timed work inside an invocation that takes two and a half minutes, so anything that takes a
moment to settle once decode begins is a large share of what is being reported. None of the five
candidates eliminated so far is of that kind: they are all things that differ per process.

Lengthening the window tests it without a new instrument.
[`scripts/window_length.sh`](../../scripts/window_length.sh), ten interleaved pairs, eight tokens
against sixty-four, `-r 3` with the per-repetition samples kept (`log-window-length`):

| | invocations | mean | cv of the invocation averages | repetition 1, cv | repetitions 2 and 3, cv | repetition 1 slowest |
|---|---|---|---|---|---|---|
| `-n 8` | 10 | 26.98 | **10.1 %** | **15.0 %** | 9.2 % | 8 of 10 |
| `-n 64` | 10 | 28.82 | **5.5 %** | 8.8 % | **4.0 %** | 9 of 10 |

Three things, all in the same direction. The longer window halves the spread. The first repetition is
the slowest in seventeen of twenty invocations and is the most variable at both lengths. And the longer
window reads 6.8 percent *faster*, which is what a fixed slow start looks like when it is amortised
over more tokens.

So there is a warm-up at the start of each timed window, and at eight tokens the measurement is mostly
warm-up. That is a property of the window, not of the process, which is a different kind of
thing from everything eliminated so far.

What it does not yet establish is that this is the same phenomenon as the August variance. That
comparison is the next measurement, and there is a reason to doubt it: August reported the spread
*inside* one invocation as small, 0.43 against 1.20 between invocations, and a slow first repetition
should have made the inside-spread large. Its two measurements used different flags, `-r 1` for the
arms and `-r 8` for the check, so they may not be comparable in the way that conclusion assumed. That
is a lead, not a result, and it is written here as one.

## Warm repetitions are stable to under one percent

[`scripts/first_rep.sh`](../../scripts/first_rep.sh), five invocations at `-r 8` with the flags
`scripts/decode_variance_process_state.sh` used, every repetition kept (`log-r8`):

| | n | mean | sd | cv |
|---|---|---|---|---|
| repetition 1, across invocations | 5 | 26.45 | 2.47 | 9.33 % |
| repetition 2, across invocations | 5 | 27.60 | 1.79 | 6.50 % |
| **repetitions 3 to 8, pooled** | 30 | 29.81 | 0.25 | **0.84 %** |

    i1  21.56  25.45  29.92 29.84 29.88 29.75 29.84 29.90
    i2  27.76  29.77  29.83 29.71 29.76 29.71 29.74 29.74
    i3  28.25  29.71  30.18 29.71 29.75 30.18 29.68 29.78
    i4  27.44  26.46  30.13 30.01 29.48 29.98 29.96 30.05
    i5  27.23  26.60  29.84 29.75 29.82 29.78 28.74 29.72

Per-invocation spread across repetitions 3 to 8 is 0.06, 0.04, 0.21, 0.21 and 0.39. The slowest of the
eight is repetition 1 or 2 in five invocations out of five.

**Decode on this board, once warm, is usually stable to under one percent.** Thirty warm repetitions
across five processes, pooled, spread by 0.84. "Usually" is the right word and it is a correction to an
earlier revision of this page, which said it flatly: a later run of seventy-two warm repetitions found
two below 0.95 of their own invocation's mean, one at each depth, so about three percent of warm
readings are low and not all low readings are first ones. The two figures for repetitions 1 and 2 rest
on five samples each and should be read as "large and variable", not as 9.33 and 6.50.

This is the same observable the campaign pages already record from the other end, that the first of the
three samples `llama-bench` takes after loading a model often reads low while the second and third do
not ([`logs/fedora44-campaign-q8-2026-09-20/`](../fedora44-campaign-q8-2026-09-20/)). What is added here
is how flat the rest is.

## A confound in the measurement that produced the clue, which is not the same as an answer

The clue this question has been built on is that repetitions inside one invocation spread much less than
separate invocations of the same command, which was read as placing the variable in per-process state.
Reading [`scripts/decode_variance_process_state.sh`](../../scripts/decode_variance_process_state.sh):
its three arms, the ones carrying the 7.7 to 8.8 percent, run `-r 1`, and its within-invocation check
runs `-r 8`. So the arms reported one repetition each, which was necessarily a first repetition, and the
check reported an average over one first repetition and seven later ones.

If a first repetition on that day behaved as it does here, those two commands were not comparing
between-process against within-process. **Whether it did is not known**, and this repository cannot find
out: that was a different kernel, a different llama.cpp and a different ROCm, and the runs were not kept
at per-repetition resolution. What can be said is narrower and still worth saying: the comparison that
produced the clue has a confound in it that was not controlled, so the inference from it to per-process
state does not follow on its own. That is a reason to re-open the question instead of an answer to it,
and the five eliminations stand on their own regardless.

One loose consistency, not leaned on here: repetition 1 here varies by about the same
amount the August arms did. Five samples is far too few to make anything of that.

**Answered 25 September: it is HIP graph instantiation.** Ten invocations per arm at depth 0 with and
without `GGML_CUDA_DISABLE_GRAPHS=1` put the first-repetition deficit at 7.5 percent with graphs and
0.7 without, so disabling them removes 91 percent of it, and the cost behaves as a fixed one, shrinking
from 7.5 percent at 8 tokens to under 1 at 128. Pre-warming the GPU to 1500 MHz changes nothing, so the
clock is not involved ([`logs/first-rep-graphs-2026-09-25/`](../first-rep-graphs-2026-09-25/)). The
section below is what was known before that and its reasoning still holds; the candidate it raises is
not the answer.

## The first repetition is not doing the same thing as the rest

This is read from `tools/llama-bench/llama-bench.cpp`, not measured, and it is worth stating
before any mechanism is guessed at, because it says the two are not the same measurement.

With `-d N`, the depth is not re-established on every repetition. The first repetition runs the full
N-token prefill and then saves the sequence state; every later repetition restores that state from a
host-side buffer instead. Timing starts after either path completes:

    for (i = 0; i < reps; i++) {
        llama_memory_clear(...)
        if (n_depth > 0) {
            is_cached = (n_depth == cstate.depth)
            if (is_cached)  llama_state_seq_set_data(...)   // restore
            if (!is_cached) test_prompt(ctx, n_depth, ...)  // full prefill, then save
        }
        t_start = get_time_ns()

The run times agree: invocation i1 took three and a half minutes for a model load and eight
repetitions, where eight 16128-token prefills on their own would be about nine.

So repetition 1 enters its timed window straight out of a 16128-token prefill computed on the GPU, and
repetitions 2 onward enter theirs straight out of a state upload from host memory. That is a real and
documented difference in what immediately precedes the measurement, and it is not the same thing as
"the first one is cold".

**Whether it is what makes repetition 1 slower is not established here.** It is a candidate with the
merit of being visible in the source rather than inferred from the shape of a curve, and it turned out
to be checkable more than first described: at `-d 0` the whole block is skipped, so neither path
runs and every repetition is identical. That test is below.

It also sharpens what can be said about the August comparison. Those two commands differ by more than
how many repetitions they average: at `-r 1` every invocation prefills, and at `-r 8` one prefills and
seven restore. Whichever way the slower first repetition is eventually explained, the arms and the
check were not measuring the same operation.

## The same test at depth 0 excludes it

[`scripts/depth_zero_rep1.sh`](../../scripts/depth_zero_rep1.sh), six interleaved pairs, the same model
and flags, differing only in the depth (`log-depth0`). At `-d 0` the `if (n_depth > 0)` block is skipped
entirely: no prefill, no restore, every repetition identical.

| | repetition 1 / warm mean | range | repetition 1, cv across invocations | repetition 1 below every warm repetition |
|---|---|---|---|---|
| depth 0 | 0.930 | 0.023 | 1.59 % | 5 of 6 |
| depth 16128 | 0.867 | 0.179 | 7.71 % | 5 of 6 |

    depth 0      0.931 0.934 0.924 0.918 0.941 0.932
    depth 16128  0.953 0.924 0.774 0.916 0.838 0.797

**Repetition 1 reads low at depth 0 as well**, by about 7 percent, in five invocations out of six. Since
neither the prefill nor the state restore runs there, **that difference is not necessary for the slow
first repetition**, and the candidate this test was built to check is excluded as the sole cause. It
says nothing about whether it contributes at depth.

Something else came out of it that the test was not designed for, and it is the more interesting half,
so it is stated with its sample size attached. At depth 0 the first repetition is low *reproducibly*:
the six ratios span 0.023 and repetition 1 itself varies by 1.59 percent across invocations. At depth
16128 it is low *erratically*: the ratios span 0.179 and repetition 1 varies by 7.71 percent. The open
question is about variability and not a constant offset, and on this evidence the variability is a
depth phenomenon while the offset is not.

That is six invocations per arm. A fivefold difference in spread is larger than the effects that have
evaporated on this board before, but four of those looked convincing at six to eight samples too, so it
was tested, not written down.

## The sweep does not support it as a dose-response

[`scripts/depth_dose.sh`](../../scripts/depth_dose.sh), five depths, four passes, twenty invocations,
the order of the depths alternating between passes so that depth is not confounded with position in the
pass and therefore with temperature. The edge sensor ended up between 61 and 68 C across all five
depths, which is what that alternation was for (`log-depth-dose`):

| depth | n | ratio | ratio range | repetition 1, cv | warm cv | edge |
|---|---|---|---|---|---|---|
| 0 | 4 | 0.947 | 0.053 | 0.87 % | 2.44 % | 61 to 61 C |
| 2048 | 4 | 0.916 | 0.166 | 7.54 % | 0.33 % | 61 to 64 C |
| 4096 | 4 | 0.938 | 0.134 | 3.37 % | 2.02 % | 62 to 65 C |
| 8192 | 4 | 0.904 | 0.120 | 6.40 % | 0.94 % | 64 to 67 C |
| 16128 | 4 | 0.781 | 0.320 | 16.71 % | 2.00 % | 65 to 68 C |

**Neither the spread nor the ratio is ordered by depth.** 2048 is wider than 4096 and 8192; the three
middle depths do not form a ladder in either column. At four invocations a depth, each of those
coefficients is estimated from four numbers and carries very little precision, so this is a sweep that
cannot settle the question instead of one that answers it in the negative.

What survives is narrower than the section above claims, and that section should be read through this
one. The two extremes differ, at 0.87 against 16.71 percent here and 1.59 against 7.71 in the six-pair
run, which is two independent measurements agreeing that the first repetition is steadier at depth 0
than at 16128. **That the variability rises with depth is not established**, and a five-point sweep
built to show it did not.

The first-repetition effect itself is not in doubt. Repetition 1 came in below every warm repetition in
seventeen of these twenty invocations, and the three exceptions include one invocation at depth 4096
where it was slightly *faster* than the warm mean.

## What shape it is: two models tested against data already collected

No new measurement. The ten `-n 8` and ten `-n 64` invocations above carry enough to test the two
simplest shapes the deficit could have, by asking about *time*, not rate.

If the first repetition pays a fixed cost at its start, the extra seconds it takes should be the same
whether it generates 8 tokens or 64. If instead it runs slower for the whole repetition, the
extra seconds should scale with the token count, a factor of eight.

| | extra time taken by repetition 1, median | mean | quartiles |
|---|---|---|---|
| 8 tokens | 0.0266 s | 0.0425 | 0.0230 to 0.0329 |
| 64 tokens | 0.0692 s | 0.1184 | 0.0388 to 0.0939 |

The ratio of medians is 2.60, with a bootstrap 90 percent interval of 1.33 to 3.98 on ten invocations
a side. Both distributions are heavy-tailed, one value in each being four to eight times the median,
which is why the medians are quoted and the means shown beside them.

**Eight is inconsistent with that**, so the first repetition is not just running slower
throughout. **One is just outside the interval**, so a pure fixed start-up cost is not favoured either,
but at this sample size that is a weak statement and it should not be read as excluded.

No third shape is proposed here. The useful part is the first exclusion, which is firm.

## The warm-up is not monotonic

Worth stating because it constrains what the thing can be. In several invocations the second repetition
is slower than the first: `30.32, 26.14, 32.68, 32.69, ...` at depth 8192, and the same shape at 16128
in two invocations of the earlier run. So "the first one is cold and the rest are warm" is too tidy;
sometimes it is the first two, and sometimes the second is the slowest of all. Anything that simply
ramps from a low state and settles would not do that.

## What the warm-up is, is not known

Nothing here identifies it. The prefill-versus-restore difference above, a clock ramp, the KV cache
being allocated and first touched, first-launch kernel setup and page faults on first access all fit
the shape, and no measurement here separates them. An attempt to catch it by sampling the GPU clock was
abandoned after two invocations: the timed window is about a second inside a two-and-a-half-minute
invocation, so at 300 ms ticks the sampler takes about three samples of the thing in question and the
rest describe the model load. Those two rows are kept as
`log-clock-sampler-abandoned`, and they do show the board at its 1000 MHz step for 57 and 64 percent of
each invocation, but with the measured window that thinly sampled nothing can be built on it, and the
slower of the two invocations was the one that spent *less* time at the low step.

## Files

`log-hot-cool` is the ten hot-and-cool pairs, `log-window-length` the ten eight-against-sixty-four
pairs, `log-r8` the five `-r 8` invocations, `log-depth0` the six depth-0-against-depth-16128 pairs,
`log-depth-dose` the twenty-invocation sweep over five depths.
