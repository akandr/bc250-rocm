# Which half of the runlist flush the sequence reproducer needs, 2026-08-13

Recovered from the board on 25 August, during a review pass that was re-checking
its own claims about what evidence had been kept. The companion states that the
sequence reproducer "is satisfied by either bit alone, three of three with bits 1,
2, or 3, and three of three corrupt with neither", and that sentence had no
artifact behind it in this repository. It does now. Nothing here was re-measured:
these are the original run logs, copied unchanged.

`amdgpu.bc250_flush_by_runlist` is a bitmask, bit 1 for the unmap side and bit 2
for the map side. Four arms, three runs each, of `patches/seq_probe.c`: a heavy
dispatch, `hipFree`, `hipMalloc`, then a second dispatch that either faults or
silently drops a prefix of its stores.

| bits | meaning | run 1 | run 2 | run 3 |
|---|---|---|---|---|
| 0 | neither hook | `wrong=2112` | memory access fault | `wrong=2816` |
| 1 | unmap only | `wrong=0` | `wrong=0` | `wrong=0` |
| 2 | map only | `wrong=0` | `wrong=0` | `wrong=0` |
| 3 | both | `wrong=0` | `wrong=0` | `wrong=0` |

So this workload is satisfied by either hook on its own, which is what the
companion says. The workload that separates them is the allocation churn, where
only the map side helps, and that A/B is in
[`../svm-flush-2026-08/`](../svm-flush-2026-08/).

Note what the unfixed arm does: two of its three runs return wrong results rather
than faulting, 2112 and 2816 elements at the size used here. A fault is loud and a
wrong answer is not, which is the whole reason this defect went unnoticed for as
long as it did.
