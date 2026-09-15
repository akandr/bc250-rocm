# Lighter replacements for the runlist-rebuild flush, 2026-09-14/15

Module: [`scripts/apply_tlb_reset_experiments.py`](../../scripts/apply_tlb_reset_experiments.py)
then [`scripts/apply_tlb_alt_v2.py`](../../scripts/apply_tlb_alt_v2.py), on the production tree,
kernel 7.1.8, front-page configuration, `gpu_recovery=0`. Every arm is a runtime value of
`amdgpu.bc250_tlb_alt`, replacing the runlist rebuild at the sites `bc250_flush_by_runlist`
selects.

## Why this was worth another look

[`../vmid-flush-2026-08-20/`](../vmid-flush-2026-08-20/) concluded that direct invalidation cannot
substitute for the rebuild. Its candidates called `gmc_v10_0_flush_gpu_tlb()`, which sends the
invalidation through the KIQ whenever the KIQ ring is ready, which at runtime is always. So that
result covered the KIQ route only. Two other routes existed: MMIO with the KIQ bypassed, and an
SDMA `VM_INVALIDATION` packet, the route AMD's TLB rework series (amd-gfx, 2026-09-11) moves
GMC10 to.

## Which VMID the firmware assigned

The PASID lookup the driver uses reads the ATC VMID-to-PASID table, which is empty on this chip
under hardware scheduling ([`../pasid-diagnosis-2026-08-19/`](../pasid-diagnosis-2026-08-19/)). A
different source works. The firmware writes the process page directory into
`GCVM_CONTEXTn_PAGE_TABLE_BASE_ADDR` for the VMID it assigns, and the driver can match on it:

- `netconsole_dump_midrun.log`: dumped mid-run on the 8B, VMID 8 reads `ptb=0x46ffd6001` against
  the process's `pd=0x46ffd6001`, `MATCH`, on each of the three dumped calls; every other VMID
  reads 0.
- `netconsole_dump.log`: dumped on the first five flush calls of a process, before any queue
  exists, every VMID reads 0. Early flushes have nothing to match, and nothing stale to clear.
- Read independently with `umr`, context 0 holds the GART directory (`0x6fe00001`) and context 8
  holds `0x6ffd6001`, the low half of that process directory, after the process has exited.
- Every page fault in the runs below reports `vmid:8`.

## Round 1: back to back on one boot, model-load discriminator (not interpretable past cell 3)

`ab.log`, [`scripts/tlb_alt_ab.sh`](../../scripts/tlb_alt_ab.sh). Two problems make most of it
unusable, and it is kept for what it does show. The no-flush control completed (`c2`, rc 0), so the
model load did not discriminate on this boot, unlike in August. And cell 3, MMIO invalidation of the
matched VMID, recorded `ackmiss=100 acked=0x0000` and left queue creation failing with -62 for the
rest of the boot, so cells 4 to 6 ran on a broken queue path. Their `alt:` fields repeat cell 3's
last line and do not describe those cells.

What it does show: with the KIQ bypassed, the MMIO invalidation of the right VMID was never
acknowledged, on 100 of 100 calls.

## Round 2: churn discriminator, one fresh boot per group

`tbo/tbo.log`, [`scripts/tlb_alt_tbo.sh`](../../scripts/tlb_alt_tbo.sh). Workload
`test-backend-ops perf -o MUL_MAT`, 150 s cap (rc 124 means it was still running at the cap).

| group (fresh boot) | cell | result |
|---|---|---|
| g1 | production rebuild (3:0) | survived 150 s |
| g1 | unmap side only (1:0) | fault at 10 s |
| g1 | no flush (0:0) | fault at 13 s |
| g2 | production (3:0) | survived 150 s |
| g2 | SDMA, matched VMID (3:4) | fault at 13 s; 10 calls, `matched=0x0100`, job completed |
| g3 | production (3:0) | survived 150 s |
| g3 | SDMA, all KFD VMIDs 8-15 (3:3) | fault at 10 s; job completed |
| g4 | production (3:0) | survived 150 s |
| g4 | MMIO, matched VMID (3:2) | never acknowledged, 100 of 100; process wedged to the cap and was killed (rc 137), 34 kernel queue events |

So the discriminator works on this stack, and neither direct route replaces the rebuild. MMIO is
never acknowledged and wedges the queue path. The SDMA jobs complete and the fault arrives on
schedule anyway. Whether the SDMA poll saw an acknowledgement or gave up is not shown by job
completion; the readback added in v2 measures that, below, and finds the bit clear.

## Round 3: rebuild for one PASID only

`tbo2/tbo.log`. `bc250_tlb_alt=5` calls `execute_queues_cpsch()` with
`KFD_UNMAP_QUEUES_FILTER_BY_PASID` instead of the dynamic-queue filter, so only the process that
changed its mappings is preempted and remapped.

| group (fresh boot) | cell | result |
|---|---|---|
| g1 | production (3:0) | survived 150 s |
| g1 | PASID rebuild (3:5) | survived 150 s, 100+ calls, err 0 |
| g2 | PASID rebuild (3:5), first on the boot | survived 150 s |
| g2 | SDMA matched (3:4) | fault at 12 s (readback not logged, see below) |
| g3 | PASID rebuild (3:5), first on the boot | survived 150 s |
| g3 | unmap side only (1:0), after it | fault at 12 s |

Three of three. The control after it still faults on the same boot, so the boot was not masking
the defect. The SDMA readback line in g2 was not printed because the call counter is shared between
modes and the print is gated on it; it was measured separately, below.

This fits the reading recorded in August that reassignment, not invalidation, clears the stale
translation, now with a cleaner test of the invalidation half: an invalidation aimed at the right
VMID, through the route AMD recommends, completes and does not help.

## Round 4: validating the PASID rebuild, which fails

[`scripts/mode5_validate.sh`](../../scripts/mode5_validate.sh), results in `mode5/`.

1. The full churn completed on its own in 9 m 49 s with no fault and no new kernel events, 1386
   filtered rebuilds for PASID 23 (`mode5/log`, `mode5/churn600.log`).
2. The perplexity gate that followed, a new process (PASID 26) loading with `--no-mmap`, hit
   `Queue preemption failed for queue with doorbell_id: 80004020` on its very first filtered
   rebuild, and every rebuild after it returned -62, every four seconds, for the rest of the boot.
   The gate timed out, and the production arm that followed ran on a wedged GPU, so the battery
   was stopped. The kernel log is `mode5/kernel_wedge_boot.txt`, recovered from the persistent
   journal after a clean reboot.
3. Repeated on a fresh boot with the perplexity run as the first GPU process of the boot: it hit
   preemption failures within 106 rebuilds and timed out. A second run on the same, already wedged
   boot aborted with 103 failed rebuilds, which is the aftermath rather than a second sample. So
   two boots of two (`mode5/m5repro_*.log`, `mode5/kernel_repro_boot.txt`).

So the filtered rebuild is not a fix. Round 3's three clean results were all on the churn
workload, which happens not to trigger it. The same scheduler that preempts every dynamic queue
successfully fails to preempt when asked to unmap one PASID's queues during a `--no-mmap` load.
Why is not known: firmware handling of the PASID filter on this MEC, or a queue of that process
not yet in a state the filtered unmap expects, are both possible. The production rebuild with the
dynamic filter stays the recommendation.

A side effect worth having: this is a quick, reliable way to put a queue into the
`check_preemption_failed` state that natural faults end in, which the per-queue reset experiment
needs.

## The SDMA acknowledgement, read back

`sdma_ack_readback.txt`: SDMA invalidation of the matched VMID with `GCVM_INVALIDATE_ENG17_ACK`
read after each job completes. The first attempt (not kept) spent its dump window on the early
calls before any queue existed; the kept run arms the dump four seconds in. Once VMID 8 matches,
the job completes with `r=0` and the ACK register reads `acked=0x0000` on both dumped calls. The
workload faulted as in round 2.

So neither route produced an observed acknowledgement: MMIO polled for it live and never saw it
(100 of 100), and SDMA completed its job with the bit clear afterwards. A readback after completion
could in principle miss a bit that was set and cleared again, so the SDMA half is the weaker of
the two observations.

## Round 5: what the rebuild does that a driver write does not

`tbo3/`, module [`scripts/apply_tlb_alt_v3.py`](../../scripts/apply_tlb_alt_v3.py) on top of the
earlier two, same churn discriminator, one fresh boot per group. Two readings were open: a write to
the context's page-table base might flush that context by itself, which is what the firmware does
during `MAP_PROCESS`; or the invalidation engine might need RLC safe mode, its semaphore, or another
engine once the firmware owns the hub. MMIO invalidation does acknowledge at boot, before the
firmware runs, since no `VM flush ACK` timeout is ever logged then.

| group | arm | result |
|---|---|---|
| g1 | production (3:0) / unmap side only (1:0) | survived 150 s / fault at 13 s |
| g2 | rewrite the matched VMID's page-table base, same value (3:6) | fault at 14 s |
| g3 | rewrite it through zero and back (3:7) | fault at 14 s |
| g4 | MMIO invalidation, engine 17, register readback (3:9:4) | wedged to the cap, ACK never set |
| g5 | same inside RLC safe mode (3:9:5) | same |
| g6 | same holding the engine semaphore (3:9:6) | same; semaphore acquired (`sem_ok=1`) |
| g7 | same on engine 15 (3:9:4:15) | same |
| g8 | page-table rewrite then MMIO invalidation (3:8:4) | same |

The first log line of g2 and g3 was printed before any queue existed, so those two were repeated with
every call dumped (`tbo3/ptb_dump_mode6.txt`, `ptb_dump_mode7.txt`): 88 and 73 rewrites of VMID 8's
page-table base, each read back correctly, the last 74 microseconds before the fault. So the rewrite
happened on the right context and did not help.

The readbacks (`tbo3/g4` to `g8` `.inv.txt`) say more than the outcomes. The request register latches
exactly what is written (`wrote=0x00fa0100 req_readback=0x00fa0100`), and the ACK bit for VMID 8
stays clear for the whole 2 ms poll, on both engines, with or without safe mode and with the
semaphore genuinely held. The request is accepted into the register and never processed. The
invalidation engine behaves as if nothing services it once the firmware has taken the hub, which is
consistent with it acknowledging at boot.

What remains is narrower. The runlist rebuild clears the stale translation without renumbering the
VMID (every dump shows VMID 8), and without anything a driver-side register write reproduces: not the
page-table base write, not an invalidation request by MMIO, SDMA or KIQ. Whatever does it happens
inside the firmware's unmap and remap of the queue.

## Two board losses during setup

The first attempt at the mid-run dump took the board down because the command also read the
`amdgpu_regs` debugfs node, which walks the whole register BAR. That was a mistake in the command,
not a finding. A second loss followed on the next boot during a dump run and left nothing in the
journal; it did not recur in the three dump runs after it, and its cause is not known.
