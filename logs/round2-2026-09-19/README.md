# Round two after the seven patches: runtime knobs, the 8B's q8_0 kernel, the launch gaps, 2026-09-19

[`scripts/round2_chain.sh`](../../scripts/round2_chain.sh) on the seven-patch build (`build-hip-final`),
GPU clock policy 1500 MHz, timers stopped, `log`.

## Runtime environment knobs

`llama-bench -p 512 -n 64 -r 3`, six settings, two passes interleaved, the 1.5B and the 27B:

| setting | 1.5B pp512 | 1.5B tg64 | 27B pp512 | 27B tg64 |
|---|---|---|---|---|
| default | 913.2 / 913.8 | 183.8 / 183.3 | 71.4 / 71.5 | 14.30 / 14.27 |
| `HSA_ENABLE_INTERRUPT=0` (polled completion signals) | 914.3 / 915.1 | 184.4 / 184.4 | 71.6 / 71.6 | 14.23 / 14.30 |
| `GGML_CUDA_DISABLE_GRAPHS=1` | 916.5 / 916.3 | 185.7 / 185.6 | 71.7 / 71.1 | 14.38 / 14.37 |
| both | 918.3 / 918.4 | 185.9 / 185.6 | 71.7 / 71.9 | 14.30 / 14.42 |
| `HIP_FORCE_DEV_KERNARG=1` | 912.5 / 914.4 | 184.2 / 183.5 | 71.5 / 71.9 | 14.21 / 14.30 |
| **`GPU_MAX_HW_QUEUES=1`** | 921.3 / 921.6 | **197.2 / 197.4** | 71.5 / 72.7 | 14.20 / 14.45 |

One knob moves anything: with the HIP runtime confined to one hardware queue the 1.5B decodes **7.3
percent faster**, both passes, spreads 1.1 and 1.5, and prefills 1 percent faster; the 27B, whose token is
fourteen times longer, reads within its noise. HIP spreads streams over up to four hardware queues by
default; llama.cpp uses one stream, and on this board something about the extra queues costs a small
model's short token. Graphs off is worth 1 percent on the 1.5B, the opposite sign of what graphs are
for; polled signals and device-side kernel arguments do nothing. The queue knob is measured across the
models in round three.

## The q8_0 float kernel and the 8B

Since patch 5 the Q8_0 8B's tg64 has read 3 to 4 percent below the three-patch campaign. A variant
build with Q8_0 taken back out of the float matvec, interleaved with the final build, three passes:
final 28.8 (spread 2.6, a throttled first reading) / 37.4 / 36.2, variant 36.1 / 35.1 / 35.0. The variant is
the slower one where both are clean, so the float kernel is not the loss; round three puts the
three-patch build itself against the final one on this model.

## The gaps between kernels

The tracer of [`logs/kerntrace-2026-09-19/`](../kerntrace-2026-09-19/) now also sorts every dispatch by
start time and bins the idle time between one kernel's end and the next one's start (`trace-*.txt`).
`tg64`, one repetition:

| run | gaps 2 to 5 us | 5 to 20 us | over 100 us |
|---|---|---|---|
| 1.5B, graphs on | 18547, 45.9 ms | 5365, 31.1 ms | 70, 58.6 ms |
| 1.5B, graphs off | 19707, 45.9 ms | 4205, 23.0 ms | 70, 50.9 ms |
| 27B, graphs on | 111574, 297.6 ms | 11, 0.1 ms | 79, 697.0 ms |
| 27B, graphs off | 111420, 296.8 ms | 162, 1.2 ms | 81, 263.3 ms |

Two kinds of gap. About 2.5 us between consecutive kernels, every time, graphs or not: 375 kernels a
token on the 1.5B make that 0.94 ms of a 5.5 ms token, the dispatch latency of this stack. And about
seventy gaps over 100 us, one per generated token, the host's work between tokens (logits back, sampling,
the next graph): 0.8 to 0.9 ms each on the 1.5B, which is another 15 percent of its token and the same
cost on either backend. The 27B's over-100-us total is 697 ms with graphs and 263 without while its
tg64 does not differ between the two, so that difference is the tracer's own cost of recording a
graph launch, not the model's; it is left as noted.

## Sweeps

`sweep128-c1.log`: candidate D=128 rows for one column do not instantiate a `1x1` kernel in this
tree's template file, so nothing was learned about a tile alternative to the vector kernel for the 14B
models. `sweep256b.log`: no 32-column D=256 geometry compiles without spill; `256:2:64:32` (nbatch_fa
64, nbatch_K 32) has the fewest, 2 to 6 registers for the `8x4`, `4x8` and `16x2` instances against 4 to
5 and 225 for the row the seven-patch build carries, which matters for the 27B's `16x2`; round three
measures it.
