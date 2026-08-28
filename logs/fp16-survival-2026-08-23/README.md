# The defect as a survival curve, 2026-08-23

Derived from the three-chunk run in `../fp16-escalation-2026-08-23/`, generated
with `BC250_TEMP` and `GGML_CUDA_NO_POOL=1` on qwen3-8B, twelve batches of 36
GEMM calls. Per-tensor counts in `per_tensor.txt`.

## Two questions this settles

**Does the first call of a batch fail because it is first, or because of what
precedes it?** Already answered by the second model, without needing a new run.
The calls per batch follow the model's layer count, so on the 8B every batch
starts at call 1+36k and on the 14B at 1+40k. If the failure attached to an
ordinal position, the 14B would fail at 37, 73 and 109 like the 8B. It fails at
41, 81 and 121, tracking its own boundaries. The zeros follow the boundary.

**Is a size threshold involved?** No. Every one of the 432 calls in this run has
the same output size, `ne_dst=524288`, zeroed and clean alike.

## The structure, counted per layer

Each of the 36 tensors in a batch appears twelve times, once per batch. Counting
how often each is zeroed:

| tensor | zeroed |
|---|---|
| `Vcur-0`, the first call of a batch | 9 of 12 |
| `Vcur-1` to `Vcur-5` | 4 of 12 |
| `Vcur-6` to `Vcur-25` | 3 of 12 |
| `Vcur-26` to `Vcur-35` | 2 of 12 |

The count never rises with layer index. That is what a cascade that always begins
at a batch start and then survives for a variable length looks like, read as a
survival curve rather than as a list of positions:

- nine of the twelve batches lose their first call
- five of those nine recover immediately, at layer 1
- one more ends before layer 6
- one more ends before layer 26
- two run to the end of the batch

So the primary event is at the boundary, and everything after it is that event
persisting. The failure is not distributed across the batch; it starts at the
front and lives for a while.

## Why this is a better description than the previous one

"One zeroed GEMM per batch after the first" described the common case and called
the rest noise. This says the same thing with the noise included: the boundary
call fails most of the time, and its failure has a lifetime, usually zero further
calls and occasionally the whole batch. The earlier two-chunk and three-chunk
counts, four to forty zeros, are draws from that distribution rather than
inconsistencies.

## What it still does not say

What sets the lifetime. Nothing measured so far, operands, arguments, sizes,
scalars, pool state or model, correlates with whether a given boundary failure
stops at once or continues.
