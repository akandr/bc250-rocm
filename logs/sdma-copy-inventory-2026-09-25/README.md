# What a decode actually copies, and why that does not explain the SDMA cost, 2026-09-25

## The question

Leaving SDMA enabled costs the qwen2.5-1.5B about 5 percent of decode and costs the 8B, the 27B and
the MoE nothing ([`logs/sdma-decode-cost-2026-09-22/`](../sdma-decode-cost-2026-09-22/)). The natural
explanation is transfer size: this board's SDMA engine beats the blit path just above 16 KiB and is
about four times slower at 16 MiB ([`logs/sdma-sizes-2026-08-19/`](../sdma-sizes-2026-08-19/)), so a
workload whose copies land in the wrong band would pay for the engine. That explanation predicts the
1.5B's copies sit somewhere the 8B's do not.

## What a decode copies

[`copytrace.cpp`](copytrace.cpp) is an `LD_PRELOAD` roctracer tool that records every GPU copy with its
size, written because this board has no `rocprof`. Sixty-four generated tokens per model, SDMA off and
on (`inventory.txt`, `copy.sh`):

| model | copies | largest copy | copy time, SDMA off | on | the difference as a share of the token |
|---|---|---|---|---|---|
| qwen2.5-1.5B Q4_K_M | 1690 | 7 MiB, once, at load | 48.6 ms | 52.8 ms | 1.1 % |
| qwen3-8B Q8_0 | 8432 | 36 MiB, once, at load | 399.7 ms | 416.0 ms | 1.0 % |
| qwen3.6-35B-A3B MoE | 11540 | 62.8 MiB, once, at load | 535.4 ms | 550.4 ms | 1.5 % |

**The prediction fails twice over.**

Setting aside the single large upload at model load, every copy a decode issues is tiny. On the 1.5B
the largest is 6144 bytes, and the distribution is 1130 copies of zero bytes, 150 of 6144, 130 each of
4 and 8, 65 of 512 and 56 of 1024. Nothing reaches 16384 bytes, which is the point below which ROCclr
does not use the SDMA engine at all. There is no band for the copies to land in.

And the cost of enabling SDMA is the same everywhere. The extra copy time is 1.0 to 1.5 percent of the
token on all three models, including the two that show no end-to-end effect whatever. If copy time
were the mechanism, the 8B should pay most: it moves eight times the copy time of the 1.5B and spends
a larger share of its token inside copies.

So the 5 percent on the 1.5B is not the size distribution and not the copy time. Enabling the engine
costs something that is not the copies themselves.

## What that leaves

One observation worth recording instead of interpreting. Of the 1690 copies in a 1.5B decode, 1130
carry a size of zero and account for 47 of the 48.6 ms of device copy time, about 42 microseconds
each. Whether those are genuine zero-length transfers, a synchronisation carried on the copy path, or
an artefact of how the size is reported for some operations, this page does not establish. They are
the largest single item in the copy accounting on every model measured, and they are where a next
attempt should look.

## Caveat on the numbers

These runs are under `LD_PRELOAD`, which perturbs throughput: the 1.5B reads 170 to 182 tokens a
second here against about 194 untraced. The copy inventory is the result; the `tg64` figures in
`inventory.txt` are context for the trace and are not throughput measurements.
