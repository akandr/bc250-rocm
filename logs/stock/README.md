# Stock-module compute probe, July 2026

The A arm of the Observation 1 comparison: the bare compute probe on the
unpatched module, where large kernels intermittently return wrong results with
no error reported. `../patched/` is the B arm, the same probe with the corrected
PASID flush.

The file names encode the clock configuration, since one early hypothesis was
that the wrong results were a power or clock artifact. `fixed_1500_1000` means
the shader clock pinned at 1500 MHz and 1000 mV, `oberon` means the dynamic
governor left running. Wrong results appear at every setting tried, which is
what retired that hypothesis.

`compute_probe_e4_freshboot_loop.log` is the fresh-boot repetition: the same
size run on successive cold boots, twenty-five invocations of a single 8.4M-thread
dispatch with a power cycle after every hang.

Its result is 19 correct, 6 hangs and **zero silent-wrong**, which the file states
in its own provenance header and which counting its lines confirms. This
paragraph used to say the file was where "the four-of-four failure rate quoted in
the top-level README" comes from, corrected 26 August. It is not, in two ways.
The four-fresh-boot failure at 8M belongs to
[`../kernel-7.1.5/`](../kernel-7.1.5/), where all four outcomes are captured:
`wrong=2256904/8388608` and `wrong=3343808/8388608` for the two silent-wrong
boots, and two memory access faults. And that claim is in `INVESTIGATION.md`
rather than the front page.

So this file is evidence against silent-wrong results at this size in this
session, not for them, which is worth keeping visible: the correctness defect is
intermittent enough that a twenty-five-invocation run can miss it entirely while
a four-boot run on another kernel hits it every time.

Each log's header records the kernel, the full command line and the CU count of
that boot. Note that these boots carry `amdgpu.sched_policy=2`, which was
standard practice at the time and is now known to cause the wedge; the wrong
results documented here are a separate defect from the wedge, and are fixed by
the flush change rather than by the scheduler setting.
