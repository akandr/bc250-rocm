# Kernel equivalence and the CU-by-scheduler factorial, 2026-08-17

Two questions, one battery. Each directory is one boot, named for what was
actually configured rather than for the kernel alone, since several boots share
a kernel and differ only in a boot argument.

`inv47.log` and `result.txt` in each directory hold the same battery: the bare
compute probe at 4096, 16384 and 32768 blocks (up to 8.4M threads), the native
rocBLAS SGEMM sweep from N=256 to 4096, a wikitext perplexity gate against the
8.9442 reference, a sustained N=4096 for 50 iterations, and a dmesg fault count.

## Is the kernel version the ingredient?

No. With the same patch set and boot arguments, these two boots are
indistinguishable:

| directory | kernel | result |
|---|---|---|
| `7.1.5-40cu-hws` | 7.1.5 | every check clean, perplexity 8.9442 |
| `6.18.9-40cu-hws` | 6.18.9 | every check clean, perplexity 8.9442 |

Probe correct at all three sizes on both, SGEMM clean at all five sizes on both,
sustained GEMM clean on both, zero dmesg faults on both. A third kernel, 6.18.16,
was later added in `../recipe-e2e-2026-08-17/` with the same outcome.

## Is the wedge about CU count or about the scheduler?

The scheduler. All four cells below are kernel 7.1.5 with the same module,
one boot each; only the boot arguments differ.

| directory | CUs | scheduling | outcome |
|---|---|---|---|
| `7.1.5-40cu-hws` | 40 | hardware | clean, full battery |
| `7.1.5-24cu-hws` | 24 | hardware | clean, full battery, perplexity 8.9442 |
| `factorial-cells/7.1.5-40cu-sched2` | 40 | `sched_policy=2` | probes hang, SGEMM N=256 wedges, 10 preemption timeouts |
| `factorial-cells/7.1.5-24cu-sched2` | 24 | `sched_policy=2` | larger probe hangs, SGEMM N=256 wedges, 9 preemption timeouts |

The two hardware-scheduling cells are clean at both CU counts and the two
software-scheduling cells wedge at both, so the failure tracks the policy and
not the CU count.

`6.18.9-40cu-sched2` is the same argument added to the otherwise clean 6.18.9
boot above: probes hang at every size and SGEMM wedges at N=256. That run was
stopped before the battery reached its dmesg fault count, so unlike the four
cells above it has no timeout figure; `result.txt` and `inv47.log` hold what it
did capture. It is kept because it shows the effect is not specific to one kernel,
but note that it is a different kernel from the four factorial cells and should
not be read as a fifth cell of that table.

## Note on the earlier version of this evidence

An earlier draft of the top-level README presented the factorial with its
`40 CU + sched_policy=2` cell taken from 6.18.9 and its `24 CU + sched_policy=2`
cell from 7.1.5, while describing it as one module. The missing 7.1.5 cell was
measured afterwards and the table now uses it; the 6.18.9 run is reported
separately above.

Collected across boots by running the same battery on each kernel and configuration; the factorial cells are named in the subdirectory names.
