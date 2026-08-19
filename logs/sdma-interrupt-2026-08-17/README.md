# Does the SDMA completion interrupt arrive? 2026-08-17

This joins two separate pieces of work. The byte-exact threshold measured here,
16384 bytes completing and 16385 hanging, and the SDMA trap instrumentation
written by [GabriWar/bc250-rocm-working](https://github.com/GabriWar/bc250-rocm-working)
(`patches/bc250-sdma-trap-instrumentation.patch`), which logs inside the SDMA
trap IRQ handler where the interrupt vector has already been dispatched, so a
line appearing means the interrupt genuinely arrived.

Their patch was applied to a 6.18.16 tree (with fuzz, one hunk offset), built,
and booted. Production on 7.1.5 was not touched, and the instrumented source was
reverted afterwards.

## What was measured

| file | content |
|---|---|
| `kernel.txt` | the kernel this ran on |
| `ih_rings.txt` | the interrupt-handler ring state dumped at SDMA bring-up |
| `sdma_ivs.txt` | every SDMA trap interrupt the driver dispatched, 31 of them, all at boot |
| `ivs_after_probe.txt` | the same count taken again after running the probe across the threshold |
| `fence_fallback_count.txt` | how many "Fence fallback timer expired" lines appeared |
| `threshold_run.log` | the probe run itself |

## Results

**The interrupt path works.** 31 SDMA trap interrupts were dispatched during
boot, all on instance 0 (`cid=8 src=224`). The logging budget is 64 per
instance, so it was never exhausted.

**No interrupt arrives for either side of the threshold.** The count is 31
before the probe and 31 after. The 16384-byte copy completes without one, which
is what the staging reading predicts: ROCr does that copy itself and the engine
is never involved. The 16385-byte copy hangs, also without one.

**The SDMA user queue is created.** During the hang the probe process holds
three KFD queues, one of them type 1 (SDMA), 1 MiB. So the queue exists and the
work is submitted to it; nothing comes back.

**The kernel's own SDMA ring is untouched** by the probe
(`amdgpu_ring_sdma0` in debugfs is byte-identical before and after), which is
expected since ROCr submits to its own user-mode queue rather than the kernel
ring, and confirms the two paths are separate.

## What this changes

The hypothesis their patch describes as the one that survived, that the console
firmware leaves interrupt-handler rings 1 and 2 alive so an interrupt routed
there disappears without trace, is **not supported on this board**: `ih_rings.txt`
shows ring 1 and ring 2 with `base=0x00000000 cntl=0x00000000`, cleanly zeroed.
Their patch also predicts two "Fence fallback timer expired on ring sdma0" lines
on every boot; there are none here.

So the defect is narrower than "SDMA is broken". The engine runs, its interrupt
reaches the driver for kernel-submitted work, the interrupt-handler rings are
configured as Linux expects, and the user-mode queue is created. What fails is
specifically work submitted through the ROCr user-mode SDMA queue above the
16384-byte threshold: it never completes and never signals. The follow-up below
identifies what that threshold actually is, and it is not a staging-buffer size.

This is one board and one instrumented boot, and it does not identify a cause.
It removes one hypothesis and narrows where the next one has to look.


## Follow-up: which path the threshold switches between

`angles/` holds a second round that asked what actually changes at 16384 bytes.

`angles/trace_16384_works.txt` and `angles/trace_16385_hangs.txt` are
`AMD_LOG_LEVEL=4` traces of the same program at each size. Below the threshold
ROCclr reports `Unpinned write path`, `memcpy stg buf` and `Blit staging H2D
copy`, then dispatches a kernel: a blit compute kernel does the move and SDMA is
never involved. Above it, the same call reports `HSA Async Copy staged H2D`,
queries the copy engines and issues `HSA Async Copy on copy_engine=0x1` with a
completion signal that never fires.

`angles/angles.txt` varies everything else around it, ten cases each in its own
watchdogged process (`angles/sdma_angles.c`):

    pageable H2D 16384                completes
    pageable H2D 16385                HANGS
    pinned   H2D 16385                HANGS
    pinned   H2D 1 MiB                HANGS
    device to device 16385 and 1 MiB  completes
    async on a stream, H2D 16385      HANGS
    hipMemset 16385 and 1 MiB         completes
    pageable D2H 16385                HANGS

Pinning does not help, which rules out the bounce-buffer explanation an earlier
revision of the top-level README gave: pinned memory needs no staging and still
hangs, because above the threshold it takes the same async-copy path. The
device-to-device and memset rows are not counter-examples, since tracing shows
both are serviced by blit compute kernels and never reach SDMA.

`angles/trace_1mib_sdma_off.txt` is the workaround traced. With
`HSA_ENABLE_SDMA=0` the runtime makes the identical `HSA Async Copy on
copy_engine=0x1` call for a 1 MiB copy, with identical engine masks, and it
completes. So the difference is not in ROCclr's path selection but in what
services that copy underneath it.


## Moving the boundary: GPU_FORCE_BLIT_COPY_SIZE

`blit-knob/knob.txt` closes the loop from the other direction. ROCclr exposes
`GPU_FORCE_BLIT_COPY_SIZE`, a size in kilobytes below which it keeps using the
blit kernel instead of asking for an async copy. With SDMA left enabled, a
1024 KB host-to-device copy behaves like this:

    GPU_FORCE_BLIT_COPY_SIZE=0     HUNG
    GPU_FORCE_BLIT_COPY_SIZE=512   HUNG
    GPU_FORCE_BLIT_COPY_SIZE=1023  HUNG
    GPU_FORCE_BLIT_COPY_SIZE=1024  completes
    GPU_FORCE_BLIT_COPY_SIZE=2048  completes

The boundary tracks the knob exactly, which is what the trace predicts: what
matters is which path ROCclr chooses, not the size in itself.

Set large enough it makes the whole stack usable with SDMA enabled. A model that
otherwise never finishes loading runs at 119.3 t/s and returns perplexity
8.9442, matching the reference, and decode-only comparison over three
repetitions each puts it level with `HSA_ENABLE_SDMA=0` (117.6 and 118.0 against
118.1 and 119.7).

It is not the better workaround, though. `HSA_ENABLE_SDMA=0` has no ceiling to
get wrong, while this knob only covers copies below whatever value is set and a
single larger one falls back to SDMA and hangs. Its value is as evidence: a
second, independent way of forcing the blit path produces the same result.

`knob.txt` also lists the other copy-related runtime variables found in
`libamdhip64`, none of which were needed here.

Produced with the SDMA trap instrumentation from GabriWar/bc250-rocm-working built into a 6.18.16 module, plus `patches/sdma_probe.c` for the size sweep and `AMD_LOG_LEVEL=4` for the ROCclr traces.
