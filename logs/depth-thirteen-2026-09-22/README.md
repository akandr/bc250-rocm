# Depth on the thirteen-patch build, and a measurement trap, 2026-09-22

Three front-page tables described earlier builds: the qwen2.5-1.5B decode ladder was the three-patch
build, the six-model decode-at-4096 table the seven-patch build, and the qwen3-8B prefill-at-depth
table the four-patch build. Six patches have landed since the newest of them, one of which replaced
every prefill matmul. This re-measures them, and the first attempt got it wrong in a way worth
recording, because the same mistake is available to anyone comparing two backends against depth.

## The trap: `-d 0,4096,8192` is not the same measurement as three invocations

`llama-bench` takes a list of depths and prints a row for each, which is faster than invoking it once
per depth and looks equivalent. It is not. **A reading taken inside a list can be up to 22 percent
below the same reading taken alone**, unpredictably, on either backend, at some points and not others.

Same boot, same builds, the two forms an hour apart (`log` is the list form, `log-followup` the
per-invocation form):

| quantity | in a `-d` list | its own invocation | ratio |
|---|---|---|---|
| qwen2.5-1.5B `tg64` at 4096, Vulkan | 140.00 | **179.47** | 0.78 |
| qwen3-8B `pp2048` at 0, Vulkan | 284.08 | **367.02** | 0.77 |
| qwen3.8-27B `tg32` at 4096, ROCm | 12.08 | **14.53** | 0.83 |
| qwen3-8B `pp2048` at 4096, ROCm | 265.99 | **300.52** | 0.89 |
| qwen3-8B `pp2048` at 8192, ROCm | 207.78 | **214.55** | 0.97 |
| qwen3-8B `pp2048` at 4096, Vulkan | 217.02 | 217.46 | 1.00 |
| qwen3-8B `pp2048` at 8192, Vulkan | 142.90 | 143.76 | 0.99 |
| qwen2.5-1.5B `tg64` at 4096, ROCm | 176.74 | 176.53 | 1.00 |

It is not a bias in one direction or against one backend: it takes 22 percent off Vulkan's 1.5B decode
and 17 percent off ROCm's 27B decode, and leaves four of the eight alone. The per-invocation figures
are also the tight ones. Six passes of the 1.5B at 4096 give Vulkan 179.35 to 179.51 and ROCm 176.22
to 176.82; the list form gave Vulkan 139.79, 140.00, 153.59, 144.83 and 179.31 for the same quantity.

The September tables gave each depth its own invocation, so every Vulkan figure in them
reproduces here and the list-form sweep appeared to contradict them. **Everything below is the
per-invocation form.** The list-form sweep is kept as `log` because it is the evidence for the trap,
and its cross-backend rows should not be read as results.

No mechanism is offered here. The obvious guess, that a list sizes the context for its deepest point
and everything shallower then pays for it, is consistent with the worst cases being the shallow points
of long lists, but it is not tested and it does not obviously explain the 27B, whose list had two
entries.

## Prefill against depth

qwen3-8B Q8_0, `-p 2048 -n 0`, `-fa on`, one invocation per point, three passes, backends interleaved:

| existing context | ROCm, thirteen patches | Vulkan | ratio | ROCm, four patches |
|---|---|---|---|---|
| 0 | **394.42** | 367.02 | **1.07** | 266.4 |
| 4096 | **300.52** | 217.46 | **1.38** | 219.7 |
| 8192 | **214.55** | 143.76 | **1.49** | 184.5 |

**ROCm now prefills faster than Vulkan at every depth measured**, where the four-patch build was 0.73
at depth 0 and only reached parity at 4096. Vulkan reproduces
[`logs/vulkan-fa-staging-2026-09-17/`](../vulkan-fa-staging-2026-09-17/) to three figures at all three
points, 366.82, 217.01 and 143.51 there against 367.02, 217.46 and 143.76 here, which is what makes the
ROCm column readable: the reference did not move.

Two qualifications. Vulkan's depth-0 figure is the first pass, which ran before anything else; passes
two and three read 311.9, each with a within-invocation spread of 48, and each followed a depth-8192
run. That is the order effect this repository has recorded before, an arm reading low straight after a
heavy point, and it is why the first pass is quoted. ROCm's depth-8192 figures carry spreads of 4.5 to
21.6 across the three passes, so that row is the least settled of the six.

## Decode against depth on the 27B

The list-form sweep put the 27B at 0.72 of Vulkan at a 4096-token prefix, against 0.83 on the
seven-patch build, which would have been a regression with nothing to explain it. Four passes per
point, one invocation each, say otherwise:

| | ROCm | Vulkan | ratio |
|---|---|---|---|
| `tg32` at depth 0 | 15.06 | 17.59 | 0.857 |
| `tg32` at 4096 | 14.53 | 17.20 | 0.845 |

Unchanged from the seven-patch measurement within the spread of either. There was no regression; there
was a list.

## The decode ladder

[`scripts/depth_thirteen.sh`](../../scripts/depth_thirteen.sh), qwen2.5-1.5B `tg64`, three passes, one
invocation per point, backends interleaved, with a cooling pause between points and the edge
temperature recorded at each one (`log-perinvocation`, section H):

| depth | ROCm, thirteen patches | Vulkan | ratio | ROCm, three patches | Vulkan reproduces September |
|---|---|---|---|---|---|
| 0 | 197.20 | 212.24 | 0.929 | 117.7 | 0.999 |
| 4096 | 176.54 | 179.28 | 0.985 | 108.9 | 0.998 |
| 8192 | **163.00** | 160.92 | **1.013** | 100.0 | 0.983 |
| 16384 | **141.30** | 137.94 | **1.024** | 87.5 | 0.985 |
| 24576 | **123.15** | 120.72 | **1.020** | 76.2 | 0.991 |
| 30720 | **112.61** | 111.13 | **1.013** | 70.2 | 0.998 |

The thirteen-patch build is 1.60 to 1.68 times the three-patch one at every depth, and it **draws level
with Vulkan by 8192 and stays 1 to 2 percent ahead through 30720**. The last column is why that is
readable: the Vulkan arm was re-measured in the same run and returns between 0.983 and 0.999 of the
September ladder, so the reference did not move.

## Decode at a filled 4096-token cache, all six models

Two passes, one invocation per point (`log-perinvocation`, section I):

| model | ROCm | Vulkan | ratio | seven patches |
|---|---|---|---|---|
| qwen2.5-1.5B | 176.01 | 179.22 | **0.982** | 0.93 |
| qwen3-8B Q8_0 | 35.97 | 35.82 | **1.004** | 0.98 |
| deepseek-r1-14B | 26.50 | 30.95 | 0.856 | 0.84 |
| qwen3-14B | 27.30 | 31.32 | 0.872 | 0.85 |
| qwen3.6-35B MoE | 69.29 | 82.53 | 0.840 | 0.78 |
| qwen3.8-27B | 14.91 | 17.19 | 0.867 | 0.83 |

Every model improved and the 8B reaches parity. The range is 0.84 to 1.00 against the seven-patch
build's 0.78 to 0.98 and the three-patch build's 0.60. The ordering is the same as at depth 0, the MoE
last, which is that model's decode gap ([`logs/moe-decode-2026-09-20/`](../moe-decode-2026-09-20/)),
not anything about depth.

Neither section throttled: the governor logged nothing, and the edge sensor stayed between 57 and 79 C
with the pause between points. Both sections abort on a fault line or a governor throttle instead of
publishing a point taken through one.

## Files

`log` is the list-form sweep, kept as the evidence for the trap; its cross-backend rows are not
results. `log-followup` is the first per-invocation run, sections D (prefill at depth), E (the 27B) and
F (the 1.5B at 4096, six passes). It stopped part-way through a further section, which is written up in
[`logs/fault-caught-2026-09-22/`](../fault-caught-2026-09-22/): it provoked the fault this repository
has been chasing since August. `log-perinvocation` is the ladder and the six-model table above, run
after the reboot that followed.
