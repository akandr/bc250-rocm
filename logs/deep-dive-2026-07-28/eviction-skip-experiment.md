# Experiment: skip the KFD queue eviction on gfx1013

Prompted by an upstream amdgpu patch, "fix for gfx1103 queue evict/restore crash"
(lkml, Nov 2024), which addresses the same failure shape (an APU crashing during PyTorch/ML
when the kernel evicts and restores compute queues) by **skipping the eviction/restore entirely on
APUs**. Christian König rejected it as "absolutely not a fix but rather an obviously broken
workaround," since eviction is normally mandatory for memory overcommit. It never merged.

Because our wedge is exactly that eviction (Observation 2: the runtime's `munmap` churn triggers
`evict_process_queues_nocpsch`, whose MEC preemption times out), it was worth testing the same
workaround here. Patch: early `return 0` at the top of `evict_process_queues_nocpsch` and
`restore_process_queues_nocpsch`, gated on the gfx1013 PCI id `0x13FE`
(`patches/kfd_skip_eviction_gfx1013.py`), on the `sched_policy=2` (nocpsch) path we run. Built into
amdgpu, `dracut`'d, booted at 40 CU, flush left stock.

## What it does: removes the board freeze

Across ~16 heavy sustained runs (`rocBLAS` N=4096 x50, `bw_probe` 2 GB) over ~15 minutes, the board
**stayed alive and reachable the entire time**, and recovered between runs without a power-cycle.
Before this patch, that kind of load reliably froze the board (unreachable, `cp queue preemption time
out`, needing a hard reset). Some runs completed correctly that previously always wedged, e.g. a
sustained `rocBLAS` N=4096 x50 at 4.48 TFLOP/s, `chk ... CORRECT`.

## What it does not do: make compute reliable

It is a workaround for the freeze, not a fix for the compute defect, and the data says so plainly:

- Completion is intermittent and **degrades with cumulative use**. First batch after a fresh boot:
  4 of 6 heavy runs completed. A second batch right after: 2 of 10 (`eviction_skip_quant.log`). The
  rest hang at the process level (`rc=124`), recoverable, board still up.
- `cp queue preemption time out` is not eliminated (28 in the second batch): the mid-run eviction is
  skipped, but tearing down a hung process still destroys the queue and times out there.
- The correctness defect is untouched: `compute_probe` at 16M still returns silent wrong results
  (dropped stores) while running to completion.

## Reading

Skipping the eviction stops the `munmap`-driven mid-run eviction from freezing the board, which is a
real improvement for the "system freeze after compute workloads" symptom. But the underlying
compute-queue problem is still there: dispatches still hang, the queue still degrades, and results
are still silently wrong at large sizes. That is consistent with the upstream rejection: it papers
over the freeze without fixing what causes it, and on a board doing real overcommit it would trade a
freeze for use-after-free. The board was restored to the stock module afterwards; the patch is kept
here as an experiment, not a recommended configuration.
