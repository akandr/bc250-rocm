# Round three: the hardware queue count, the 8B settled, one more D=256 row, 2026-09-19

[`scripts/round3_chain.sh`](../../scripts/round3_chain.sh) and `round3b.sh` on the seven-patch build,
1500 MHz, timers stopped. `log` and `log-b`.

## `GPU_MAX_HW_QUEUES=1` is worth 8 percent on the small model, and nothing on the large ones

Round two found it on the 1.5B. Across five models, `-p 512 -n 64 -r 3`, two passes, interleaved:

| model | tg64 default | one HW queue | two | pp512 default | one HW queue |
|---|---|---|---|---|---|
| qwen2.5-1.5B | 184.1 / 183.4 | **196.6 / 196.7** | 193.6 / 197.4 | 914.7 / 913.9 | 921.7 / 921.3 |
| qwen3-8B | 37.7 / 38.2 | 36.9 / 37.6 | 37.2 / 37.0 | 260.4 / 222 (throttled) | 260.0 / 262.2 |
| deepseek-r1-14B | 31.5 / 31.2 | 31.3 / 31.5 | 30.9 / 31.6 | 96.7 / 96.6 | 96.4 / 96.6 |
| qwen3.6-35B MoE | 67.2 / 67.2 | 65.8 / 65.4 | 65.7 / 64.9 | 305.3 / 302.1 | 299.4 / 297.6 |
| qwen3.8-27B | 14.32 / 14.43 | 14.33 / 14.32 | 14.27 / 14.28 | 71.6 / 71.7 | 71.7 / 71.8 |

Only the 1.5B moves, by 7 to 8 percent of decode and 1 of prefill, reproducibly (five pairs across two
chains, spreads 1 to 2.5); three more passes in `log-b` read 180.3 / 183.2 / 180.7 against 195.2 / 195.4 /
197.8, and with graphs off as well **198.3 to 198.9**, 10 percent over the default. The perplexity gates
are unchanged with the knob set, 8.9498 and 9.1273. The MoE loses 2 percent of both figures and the 8B
about 1; the others are level.

A kernel trace with one queue says where it comes from, and it is not only the launch gaps:

| | dispatches | kernel time | wall | gaps 2-5 us | gaps 5-20 us |
|---|---|---|---|---|---|
| default, four queues | 23985 | 314.6 ms | 450.3 ms | 18547, 45.9 ms | 5365, 31.1 ms |
| `GPU_MAX_HW_QUEUES=1` | 23985 | **289.2 ms** | **395.9 ms** | 23909, 56.1 ms | **4, 0.1 ms** |

Same kernels, same count, 25 ms less device time: the float matvec's 8896 dispatches go from 188.4 to
172.4 ms and `rms_norm` from 13.3 to 11.1, 8 and 16 percent faster for the identical kernel. And the
5-to-20-microsecond gaps, a quarter of all dispatches, collapse to four: with four hardware queues the
command processor is switching between them, which costs both the gap and, apparently, some of the
kernel's own time on a 40-CU part where one kernel does not fill the machine. Whether that is CU
masking, doorbell handling or wave-slot allocation is not established here.

It is a runtime setting, not a patch: `GPU_MAX_HW_QUEUES` is read by the HIP runtime at initialisation.
The front page's numbers stay at the default configuration; the knob is documented there as a tuning
note, since it helps exactly the case a 16 GiB board is most often used for, a small model at
interactive speed, and hurts the MoE slightly.

## The qwen3-8B's decode: the deficit was between sessions, not between builds

Every campaign since patch 5 has shown the 8B's tg64 3 to 4 percent below the three-patch campaign of
15 September, and each page repeated it. A direct interleaved A/B of the two builds, same boot, same
session, three passes (`log`):

| pass | three patches (`build-hip-f44`) | seven patches (`build-hip-final`) |
|---|---|---|
| 1 | 28.84 (spread 1.5, throttled) | 37.43 |
| 2 | 34.64 | 36.43 |
| 3 | 34.79 | 36.55 |

The seven-patch build is **5 percent faster** on the 8B, not 4 percent slower. The three-patch build,
which read 38.9 in its own campaign on 15 September, reads 34.7 today: the difference is the session,
not the patches. What differs between the sessions was not established, and the board's own reading of
the same binary four days apart is the answer to any cross-campaign ratio of this size. The campaign
tables keep their numbers, which are what those runs measured; the conclusion drawn from them about the
8B is withdrawn here and on the pages that carried it.

## One more D=256 row

`sweep256b.log` of round two found `256:2:64:32` (nbatch_fa 64, nbatch_K 32) the least-spilling
32-column D=256 geometry: 2 to 6 registers against the seven-patch row's 4 to 5 for the MoE's `4x8` and
225 for the 27B's `16x2`. `build-hip-fa5` carries it. The 2048-token attention op, medians of the
rechecks in `log-b`:

| build | MoE `4x8` | 27B `16x2` |
|---|---|---|
| seven-patch row `256:2:32:64` | 48.0 ms | 84.6 ms |
| `256:2:64:32` | 49.4 ms | **72.1 ms** |
| (before experiment J) | 97.5 ms | 145.5 ms |

Fifteen percent off the 27B's op and three percent onto the MoE's, and end to end neither moves: pp512
71.7 / 71.8 against 71.8 / 71.6 on the 27B, pp2048 66.0 / 64.8 against 64.7 / 64.4 inside a 3 to 5
percent spread; the MoE is level at 305.6 / 306.0 against 305.6 / 305.5. Correctness 1287 of 1287. The
row is not adopted: it trades one model's op for another's with nothing to show at the model level, and
the table cannot tell the two instances apart. Kept here for whoever wants the 27B's op.

One reading in the first pass of `log` had the MoE's one-token attention at 125.6 us where four
rechecks read 54.4 to 55.3; it is an outlier of the replay, not a build difference.
