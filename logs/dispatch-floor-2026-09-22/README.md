# What a kernel dispatch costs when the kernel costs nothing, 2026-09-22

> **Corrected 2026-09-23.** The floor measured on this page stands and reproduces. The use made of it
> below, that the MoE's decode deficit *is* this floor, does not.
> [`logs/floor-vs-vulkan-2026-09-23/`](../floor-vs-vulkan-2026-09-23/) measured Vulkan's floor, which
> this page left undone and named as the gap in its own argument. Vulkan's floor is the **larger** of
> the two, 3.18 microseconds against 2.54 measured the same way, and Vulkan issues **1423** dispatches
> a token against ROCm's 1298, not the same number this page assumed. ROCm therefore dispatches
> less often and more cheaply than the backend that beats it. The section "What it accounts for" below
> is wrong and is kept only so the correction has something to point at.

The mixture-of-experts model is the last one with a deficit nothing explains. It decodes at 0.81 of
Vulkan's rate, and the three obvious explanations are all measured and all wrong: replaying its
one-token graph puts ROCm at or ahead of Vulkan on **every shape it contains**, so no kernel is slow;
HIP graph replay leaves the distribution of inter-kernel gaps exactly as it was, 199 ms of 2 to 5
microsecond gaps against 204 with graphs disabled over the same 83060 dispatches, so it is not
host-side launch cost; and kernel traces put 84 to 92 percent of decode inside the kernels
([`logs/moe-decode-2026-09-20/`](../moe-decode-2026-09-20/),
[`logs/kerntrace-2026-09-19/`](../kerntrace-2026-09-19/)).

What distinguishes that model is arithmetic, not any one kernel: it issues about 1298
dispatches per token with a mean kernel of 9.3 microseconds, where the 27B issues 1745 of 39
microseconds each. If a dispatch has a fixed cost, the model that pays it 1298 times for 9.3
microseconds of work is the one it hurts. That is a number, not an argument, so
[`scripts/dispatch_floor.cpp`](../../scripts/dispatch_floor.cpp) measures it: an empty kernel and a
kernel that spins for a known number of shader cycles, launched back to back on one stream, 20000
dispatches a measurement, best of five.

## The floor

| | stream | HIP graph replay |
|---|---|---|
| empty kernel, per dispatch | **2.034 us** | 2.173 us |
| fitted intercept over four spin lengths | **1.783 us** | 1.952 us |
| clock the same fit derives | 1509.2 MHz | 1513.7 MHz |
| worst residual of the fit | 7 ns | 19 ns |

**A dispatch costs about 1.8 to 2.0 microseconds on this board with nothing to execute**, and it is
additive: per-dispatch time is linear in the kernel's own length over the whole range measured, from
nothing to 20 microseconds, with residuals of a few nanoseconds.

The fit is worth more than the empty-kernel number because it validates itself. Slope and intercept
give the clock and the floor together, and the clock comes back at 1509 MHz against the 1500 the board
is pinned to. A measurement that had drifted, throttled or overlapped would not produce that.

**Graph replay is slightly worse, by about 0.15 microseconds a dispatch.** That is a third measurement
of the same thing the kerntrace and the end-to-end A/B found, from a direction where nothing else is
going on: whatever this cost is, it is not the host-side work that capturing a graph removes.

## What it accounts for

**This section is wrong. It is left in place because the correction at the top of the page refers to
it.** Read [`logs/floor-vs-vulkan-2026-09-23/`](../floor-vs-vulkan-2026-09-23/) instead.

| | ROCm | Vulkan |
|---|---|---|
| MoE decode | 71.78 t/s | 86.68 t/s |
| per token | 13.93 ms | 11.54 ms |
| per dispatch, over 1298 | 10.73 us | 8.89 us |

The difference is **1.84 microseconds per dispatch**, which sits between the two measurements of the
floor. Over 1298 dispatches the floor alone is 2.3 ms of a 13.9 ms token, 16.6 percent, against a
measured deficit of 17.2 percent.

So the MoE's decode deficit is the size of ROCm's dispatch floor, and with every shape in its graph at
or ahead of Vulkan there is nothing left for the kernels to account for. **This is where that model's
decode gap is.**

Two things this does not show, and they matter. Vulkan's own dispatch floor was not measured, so this
does not say ROCm's is the larger of the two; it says ROCm's is large enough to be the whole
difference. And a floor of 1.8 microseconds is not by itself a defect. It is what makes a model that
issues 1298 short kernels a token expensive, and the lever it implies is fewer and longer kernels
instead of a faster queue: fusion, not tuning.

### Where it goes wrong

Three ways, and the last is visible without leaving this page.

The table divides both backends' token times by **1298**, ROCm's dispatch count, as though Vulkan
issued the same number. Vulkan issues 1423. Its real per-dispatch time is 8.11 microseconds, not 8.89.

The argument needs ROCm's floor to be the larger, and never checked. It is the smaller, 2.54
microseconds against Vulkan's 3.18 measured identically through ggml on the same checkout.

And the floor cannot be added on top of the kernels the way the last paragraph adds it. The same trace
that counted the 1298 also recorded 775.832 ms of kernel time over 64 tokens: 12.12 ms of a 13.93 ms
token, leaving **1.81 ms outside kernels in total**. The 2.3 ms this page adds to the account is larger
than all the non-kernel time in the token. Most of the launch cost is already hidden behind kernel
execution, which the measurement above could not see because there the kernels did nothing.

What survives is the floor itself, and the fusion result that followed from it
([`logs/fusion-value-2026-09-23/`](../fusion-value-2026-09-23/)): removing dispatches is worth 24
percent of the MoE's decode. That is measured end to end and does not depend on any of the arithmetic
in this section.

## The calibration that went wrong, and why the numbers survive it

`floor-run1.txt` prints negative gaps, and the reason is worth writing down because it would bite anyone
repeating this. The program calibrated the shader clock with a single kernel before the sweep. That
kernel ran on a board that had been idle through a five-minute cooldown, so it ran at the 1000 MHz
step, while the 20000-dispatch loop that follows drives the clock to the pinned 1500. Every derived
kernel duration was then half again too long, and subtracting it from the measured per-dispatch time
gave a negative gap at the longer kernels.

Nothing in the measurement was wrong, only the conversion, and the conversion is recoverable: the
requested cycle counts are known exactly, so fitting per-dispatch time against them returns the clock
and the floor together. That is `fit.txt`, and the 1509 MHz it derives is what says the recovery is
sound. The program now does that fit itself and no longer calibrates separately, which is the version
in `scripts/`.

It was then re-run on that corrected version, on a different boot, and the numbers recovered by hand
from the first run come back independently (`floor-run2-corrected.txt`):

| | first run, recovered by fitting | second run, fitted by the program |
|---|---|---|
| empty kernel, per dispatch | 2.034 us | **2.032 us** |
| fitted dispatch floor, stream | 1.783 us | **1.776 us** |
| fitted dispatch floor, graph replay | 1.952 us | **1.935 us** |
| derived clock | 1509.2 MHz | 1508.3 MHz |
| worst residual | 7 ns | 6 ns |

Two nanoseconds apart on the empty kernel and seven on the fit, from separate runs on separate boots.
That is the reason to trust the recovery, not the argument for it.

## Files

`floor-run1.txt` is the first run as it printed, negative gap column and all. `fit.txt` is the
least-squares recovery and the per-dispatch arithmetic for the MoE. `floor-run2-corrected.txt` is the
re-run on the fixed program.
