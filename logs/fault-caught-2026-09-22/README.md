# The rare fault, caught with the mitigation in place, 2026-09-22

This repository has been chasing one fault since August. It appears about once in 200 rounds of mixed
load, and on Fedora 43 it ended two soaks by taking the host down with it. The chain was reconstructed
afterwards from `journalctl -b -1`: a gfxhub page fault, then a queue preemption that timed out, then
this repository's own runlist rebuild returning -62, then a GPU reset, and only then a SIGBUS as the
reset withdrew the queue mapping ([`logs/soak-crash-2026-08-20/`](../soak-crash-2026-08-20/)).

`amdgpu.gpu_recovery=0` was adopted to break that chain at the reset. What had never been observed is
the chain running with the parameter in place on a current kernel. It happened here, at 06:25:45 on
22 September, on kernel 7.2.5, during the third pass of a deep-context decode sweep.

## The chain, from this boot's own journal

| event | count | first | last |
|---|---|---|---|
| `[gfxhub] page fault` | 4 | 06:25:45 | 06:25:45 |
| `Queue preemption failed` on doorbell `80004018` | 336 | 06:25:49 | 06:49:44 |
| `GPU recovery disabled.` | 336 | 06:25:49 | 06:49:44 |
| `BC-250: runlist rebuild flush failed (-62)` | 301 | 06:25:49 | 06:49:44 |
| `sq_intr: error` | 1 | 06:25:53 | 06:25:53 |
| `Resetting wave fronts (cpsch)` | 4 | 06:26:03 | 06:48:59 |
| **`GPU reset begin`** | **0** | | |
| **SIGBUS** | **0** | | |

    amdgpu 0000:01:00.0: [gfxhub] page fault (src_id:0 ring:40 vmid:8 pasid:1796)
    amdgpu 0000:01:00.0:  Process llama-bench pid 679639 thread llama-bench pid 679639
    amdgpu 0000:01:00.0:   in page starting at address 0x00007fb9ffffd000 from client 0x1b (UTCL2)
    amdgpu 0000:01:00.0: GCVM_L2_PROTECTION_FAULT_STATUS:0x00841051
    amdgpu 0000:01:00.0:          Faulty UTCL2 client ID: TCP (0x8)
    amdgpu 0000:01:00.0:          PERMISSION_FAULTS: 0x5
    amdgpu 0000:01:00.0:          RW: 0x1

Four faults in one second, on two adjacent pages, `0x7fb9ffffd000` and `0x7fb9ffffe000`. TCP client,
permission code 0x5, write direction: **the same signature as the August event and as the deliberate
fault probe**, and the same doorbell, `80004018`, that the August soak died on. One PASID, one VMID.

Four seconds later the preemption fails, and where August logged `GPU reset begin` this logs
**`GPU recovery disabled.`** The parameter did what the driver source says it does, observed against a
real fault on a current kernel for the first time. The preemption then fails again every four seconds
for twenty-four minutes, until the faulting process finally goes away.

## What survived

`GPU reset begin` never appears and neither does SIGBUS. The host stayed up throughout and was still up
afterwards, seventeen hours into the boot, with the device on the bus.

A battery run against the frozen state ([`scripts/wedge_battery.sh`](../../scripts/wedge_battery.sh),
`battery.txt`):

| check | result |
|---|---|
| device on the PCI bus | yes |
| `rocminfo` enumerates gfx1013, 40 CUs | **passes** |
| a HIP program: device query plus 200 empty dispatches | **times out at 60 s** |
| **Vulkan, qwen2.5-1.5B decode, same board** | **212.54 t/s, full healthy speed** |
| `test-backend-ops test -o ADD` on ROCm | **times out at 300 s** |
| `llama-bench` on ROCm, the model Vulkan had just run | **aborts**, `ggml-cuda.cu:111: ROCm error` |

Enumeration passes, execution does not, and it fails two ways: the small probes hang and the large one
aborts cleanly. That reproduces the August reading, that the wedge is invisible to every check short of
running something. The Vulkan row is new and it is the useful part.

Each ROCm attempt provoked more of the same: the fault-line count went from 633 to 1041 across the
battery, so the preemption loop is re-armed by every process that tries to use the compute queue rather
than being a single stuck retry.

## The wedge is confined to the compute queue

**Vulkan was unaffected for the whole twenty-four minutes.** Two decode runs completed at full speed
while the preemption loop was still printing every four seconds, at 06:26:25 and 06:34:27 in
[`logs/depth-thirteen-2026-09-22/`](../depth-thirteen-2026-09-22/), and the battery reproduced it
afterwards at 212.54 t/s against the 212.4 this board measures healthy.

August recorded that every GPU process died, seven of them. That is not what happened here, and the
difference is which queue the work goes to. The wedge is on one doorbell, `80004018`, belonging to a
compute queue; ROCm has nothing else to use, while RADV routes everything through the graphics queue
because Mesa disables the compute-only queue on this chip. The failure is therefore contained inside
exactly the hardware path Mesa refuses to touch, which is a cleaner statement of the boundary than
anything measured here before.

Two cautions. This is one event: whether every instance of this fault spares Vulkan is untested. And
Vulkan being unaffected says the graphics queue keeps working, not that the fault could not have been
provoked from it.

## The process could not be killed

`SIGKILL` did not remove the faulting process for several minutes; it stayed in state `R`, burning a
core and holding `/dev/kfd`, while the preemption loop continued. It went away on its own at about
06:49, and the preemption messages stopped in the same second. So the loop is the driver retrying
against a queue belonging to a process that cannot be torn down, and it ends when the teardown finally
completes, not on a timeout.

## What provoked it

The sweep that was running repeats qwen2.5-1.5B decode at a primed context, one `llama-bench`
invocation per point, alternating depth 16384 and 24576. It survived two full passes and faulted about
one second into the third, and eight hours of soak at depth 0 the same night provoked nothing at all
([`logs/soak-thirteen-2026-09-22/`](../soak-thirteen-2026-09-22/)).

**The loop was then run again and did not reproduce it.** Eight invocations on a fresh boot, including
the fifth, which is where it faulted before ([`logs/fault-repro-2026-09-22/`](../fault-repro-2026-09-22/)),
and then three hours of it against a shallow control at the same duty cycle: 122 invocations at these
two depths, 124 at depth 0, no fault line in either arm
([`logs/depth-soak-2026-09-22/`](../depth-soak-2026-09-22/)). That does not make the fault rarer than
the one-in-200 it has always been, and 122 invocations of one narrow workload is a weaker test than it
sounds, but it closes the specific hypothesis that these depths are a reproducer.

**And one piece of the circumstantial case has been withdrawn.** An earlier revision of this page
offered a third fact in support: that the sweep before the fault had produced two dropouts of about 28
percent at exactly these two depths and nowhere else. Those dropouts were caught in the act during the
repeat and they are the GPU governor throttling the shader clock from 1500 to 1000 MHz at 93 C, four
seconds after it logs `GPU overheated, throttling`. They say the board gets hot doing this, not that
the fault was coming.

One thing the timestamps cannot settle either way: the fault is logged one second after the previous
invocation reported its result, so it belongs either to the new process starting or to the previous one
tearing down.

## Files

`journal-from-0620.txt` is the kernel log from five minutes before the fault through the battery,
`battery.txt` the battery output, and `boot-cmdline.txt` the command line that boot ran under, which
is where `amdgpu.gpu_recovery=0` comes from.
