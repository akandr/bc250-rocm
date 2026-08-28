# The reset does not fail. The host hangs afterwards, 2026-08-23

Captured over netconsole, which ships the kernel log by UDP as it is produced and
therefore survives a machine that stops writing to disk.

Reading these files, noted 27 August. Every netconsole capture in this repository
contains NUL bytes, and `mode1.log` is 94K of which 1430 bytes is the eighteen
lines that matter. Git and GitHub therefore treat them as binary and show no
preview or diff. Read them with `tr -d '\0' < mode1.log`, which is how every
line quoted from them here was taken. Why the padding is there has not been
established and is recorded as an observation rather than explained. The same
applies to the captures in `../reset-ccwrite-2026-08-23/`,
`../reset-cp-bisect-2026-08-23/`, `../reset-cp-resume-2026-08-23/` and
`../reset-resume-bisect-2026-08-23/`.

## Why this was needed

The write-up had said a GPU reset kills this board and that nothing survives in
the journal to say why. The second half was an instrument limit rather than a
fact: the board stops flushing to disk, so the journal ends mid-sentence. It says
nothing about what the kernel was doing. Netconsole has no such limit.

## What actually happens

The reset succeeds. Under the default method:

    GPU reset begin!. Source:  5
    failed to suspend display audio
    MODE1 reset
    GPU mode1 reset
    GPU psp mode1 reset
    GPU reset succeeded, trying to resume
    VRAM is lost due to GPU reset!
    PSP is resuming...
    SMU is resuming...
    SMU is resumed successfully!
    bc250-40cu-enable: mode=3 se=0 sh=0 ...        (the unlock reapplying)
    clocksource: Watchdog remote CPU 10 read timed out

The GPU resets, the driver reports success, PSP and SMU come back, and the 40-CU
unlock is reapplied to all four shader arrays. Then a CPU stops answering and the
machine is gone.

So the defect is not that the GPU cannot be reset. It is that resetting the GPU
on this APU wedges the host. The last thing the kernel says is a clocksource
watchdog timeout on a remote CPU, which is a CPU-side stall, not a GPU error.

## The reset method is not the cause

MODE1 resets the whole chip, including memory controllers the host shares, which
made it a plausible culprit. MODE2 is the lighter reset APUs normally use, so it
was tried with `amdgpu.reset_method=3`:

    GPU reset begin!. Source:  5
    MODE2 reset
    GPU reset succeeded, trying to resume
    PSP is resuming...
    SMU is resuming...
    SMU is resumed successfully!
    clocksource: Watchdog remote CPU 10 read timed out

Same ending. MODE2 avoids the display-audio suspend failure and does not report
losing VRAM, and the host still hangs at the same point. Both methods reset the
device successfully and both take the machine with them.

## What this changes

The mitigation is unaffected and still correct: `amdgpu.gpu_recovery=0` stops the
reset being requested, which is confirmed separately. But the reason it is needed
is different from what this repository has been saying. A reader would have
concluded the GPU cannot be reset on this hardware. What the evidence shows is
that it can, cleanly, and that the host does not survive the operation.

"GPU reset hangs the host on gfx1013" is a specific claim, and it is not obviously a BC-250
quirk: nothing here is board-specific, so any Cyan Skillfish APU may do it.

## What is still unknown

Which part of the host stalls, and why. The clocksource watchdog names CPU 10 but
that is the CPU that noticed, not necessarily the one at fault. Whether the stall
is in the driver's resume path, in a device that shares the fabric, or in the
memory controller, is not something these captures can say.
