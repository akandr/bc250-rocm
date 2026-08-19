# Working-configuration benchmark campaign, 2026-08-08 and -09

The first full campaign on the working boot configuration: corrected flush, 40
CU, hardware scheduling, native gfx1013 rocBLAS. It predates the three
llama.cpp patches, so its inference numbers are the unpatched ones and are
superseded by `../bench-fixed-2026-08/`. Its compute and rocBLAS results stand.

`campaign.log` is the index and runs in lettered phases: A the SGEMM curve, B
the correctness probe sweep, C onward the per-model llama.cpp runs. The many
small `*.log` files are the individual invocations behind it, one benchmark per
process, named for model and test.

Headline results, all reproduced later:

- SGEMM at N=512 through 8192, 20 iterations each, no wrong results
- the correctness probe at 1M, 4.2M, 8.4M and 16.7M threads, five runs each,
  no wrong results, no faults, no hangs
- the `runlist-verify/` subdirectory holds the A/B that established the runlist
  flush from the kernel command line, including the flush-off control that faults

The inference numbers here are the ones the top-level README describes as the
unpatched path, notably the conservative `-fa off -ub 8` configuration that is
perplexity-exact but slow. Anything quoted as a current inference rate comes
from the fixed-stack campaign instead.
