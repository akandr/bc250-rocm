# What `amdgpu.gpu_recovery=0` costs, 2026-08-21

Measured by hand with the fault hunt paused; the raw `llama-bench` output of each
run is here, along with the kernel command line and the parameter's live value in
`state.txt`.

The parameter is proposed as a mitigation for the fatal GPU reset, so whether it
costs throughput matters.

| run | pp512 | tg64 |
|---|---|---|
| 1 | 783.80 | 112.73 |
| 2 | 784.08 | 112.76 |
| 3 | 797.56 | 110.66 |

Against 805.53 prefill and 113.50 decode recorded for this model before the
parameter existed, that is within about 2.7 percent on prefill and 2.5 percent on
decode, inside the run-to-run spread this measurement has shown throughout. The
parameter does not cost throughput.

Four write-ups here use that baseline. The decode half is
captured: `../bench-fixed-2026-08/recipe_q15.log` reads tg64 113.50 plus or minus
0.72. The prefill half is not. No file in this repository records a pp512 of
805.53; the same invocation that gives 113.50 reads 805.61, and 805.53 appears
only in prose and in one log line quoting it as a reference. 805.61 may have been
mistranscribed, but that is not established, and it is not stated here as fact.
Nothing here depends on it: against 805.61 the prefill gap is 2.71 percent instead of 2.70.

## Why this directory exists

An earlier version of the reset write-up quoted 798.83 and 799.03 prefill with
112.88 and 112.83 decode for the same check. Those numbers were taken by hand
over ssh and never saved, so they appeared in prose with no artifact behind them,
which is the defect the figure audit was built to catch. The audit missed
them because it only reads the two top-level documents, not the READMEs inside
log directories, and it now reads both. The figures above replace them and the
raw output is beside this file.
