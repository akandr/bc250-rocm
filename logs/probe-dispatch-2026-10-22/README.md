# One dispatch or three: what the probe can see, 2026-10-22

Same stack and checks as `../hws-fullcheck-2026-10-22/` (Fedora 44, hardware scheduling, KIQ
bypass). Each boot runs `patches/compute_probe.c` at 8.4M threads (32768 blocks of 256) ten times
with one dispatch per invocation, as the Fedora 43 factorial (`../factorial/`) ran it, then three
times with three dispatches, then five more times with one; every element of every dispatch is
checked. Script: `scripts/probe_dispatch_boot.sh`, boots from `scripts/probe_dispatch.conf` via
`scripts/hws_fullcheck_orchestrate.sh` (`BATTERY=probe_dispatch_boot.sh RDIR=probedisp`).

| boot | runlist rebuild | one-dispatch invocations wrong | three-dispatch invocations failed | amdgpu fault lines |
|---|---|---|---|---|
| `pd1-7.1.5-rl0` | none | 0/15 | 3/3: two with wrong elements (one then aborted, heap corruption), one GPU page fault | 14 |
| `pd2-7.1.5-rl3` | unmap and map | 0/15 | 0/3 | 0 |
| `pd3-7.1.5-rl0` | none | 0/15 | 3/3 with wrong elements (one then aborted) | 0 |
| `pd4-7.2.5-rl0` | none | 0/15 | 3/3: two with wrong elements (one then aborted), one GPU page fault | 14 |
| `pd5-7.2.5-rl3` | unmap and map | 0/15 | 0/3 | 0 |

The first dispatch of every invocation was correct. Without the rebuild the failure shows on the
second or third dispatch, so here a one-dispatch probe passes whether or not the rebuild is on. The
factorial's probes in cell A also made one dispatch each, but with a different build of the probe
on the Fedora 43 stack (about 3.7 s per dispatch there against about 0.1 s here), and its logs do
not record the rebuild, so these runs do not settle what cell A had.

Together with the bypass-only boots of `../hws-fullcheck-2026-10-22/`: in eight boots without the
rebuild all 19 three-dispatch invocations at 8.4M threads failed after a correct first dispatch
(17 with wrong elements, the first wrong element 0 in every case, eight of them then aborting with
glibc's "double free or corruption"; 2 with a GPU page fault), and all 45 one-dispatch
invocations were correct. With the rebuild on map and unmap, ten boots, no failure.
