# The runtime knobs, measured again on the thirteen-patch build, 2026-09-21

Four decisions taken under an earlier configuration have reversed when the configuration changed, so the
HIP runtime settings were swept again. They were last measured on the three-patch build
([`logs/round2-2026-09-19/`](../round2-2026-09-19/), [`logs/round3-2026-09-19/`](../round3-2026-09-19/)),
before the prefill GEMM, the matvec's shared-memory codebook and the expert path changed both the
kernels and how many of them a token issues.

Decode, medians of three rounds of three samples with the arms interleaved inside one session,
`HSA_ENABLE_SDMA=0` as the campaign uses ([`scripts/knob_sweep.sh`](../../scripts/knob_sweep.sh)):

| model | default | `GPU_MAX_HW_QUEUES=1` | `HIP_FORCE_DEV_KERNARG=1` | `HSA_ENABLE_INTERRUPT=0` |
|---|---|---|---|---|
| qwen2.5-1.5B | 194.26 | 192.63 | 193.62 | 194.46 |
| qwen3.6-35B MoE | 69.49 | 69.43 | 69.65 | 69.61 |
| qwen3.8-27B | 14.99 | 14.94 | 15.00 | 14.98 |

As ratios of the default: 0.992 to 1.002 across all twelve cells, with spreads up to 1.6 percent. **None
of them is worth setting now**, and that is a change of result, not a confirmation.

## The one that used to matter

`GPU_MAX_HW_QUEUES=1` was worth 7 to 8 percent of decode on the 1.5B on the three-patch build, 196 to
198 tokens per second against 180 to 184, and the reason was visible in a kernel trace: with the default
four queues a quarter of the dispatches carried a 5 to 20 microsecond gap that one queue removed. The
front page carried it as a small-model tuning note.

It is now worth nothing, slightly less than nothing on the same model. The build it was measured on ran
MMQ for prefill and a different matvec for decode; the 1.5B's decode has gone from 180 to 194 in the
meantime and its dispatch mix with it. Whatever produced that gap pattern is not in the current mix. The
front-page note has been rewritten to say so instead of deleted, because a reader who finds the old
advice elsewhere should be able to see what happened to it.

## Files

The `*.jsonl` are the sweep, named `<model>_<arm>_<round>`, with `base` the default environment.
