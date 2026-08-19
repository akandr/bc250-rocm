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
size run on successive cold boots, which is where the four-of-four failure rate
quoted in the top-level README comes from.

Each log's header records the kernel, the full command line and the CU count of
that boot. Note that these boots carry `amdgpu.sched_policy=2`, which was
standard practice at the time and is now known to cause the wedge; the wrong
results documented here are a separate defect from the wedge, and are fixed by
the flush change rather than by the scheduler setting.
