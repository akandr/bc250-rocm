# The ROCm-only claims, run rather than read, 2026-08-25

`verify.out` is the run. Each of these figures appears on the front page and had
been checked only against logs from 19 to 22 August; this is the same week in
which a dozen instrumented driver builds went on and off the board, so the
figures were worth producing again rather than quoting.

| claim on the front page | in `verify.out` |
|---|---|
| streaming-read bandwidth 432 GB/s, 402 GiB/s | 430.3 GB/s, 400.7 GiB/s |
| DGEMM about 456 GFLOP/s, about 95 percent of the FP64 rate peak | 456.0 GFLOP/s twice, `nwrong=0` |
| a custom HIP kernel runs | not in this capture |
| streaming reads to 2 GB complete | `RESULT fails=0/10` |

Corrected 26 August, and the correction is the point of the page rather than a
footnote to it. This table previously read 432.2 GB/s and 402.5 GiB/s in the
first row, 456.0 to 456.1 in the second, a `mandelbrot.pgm` line in the third and
17.2 GB per iteration in the fourth, and it ended "All four hold". Only the
fourth row's `fails=0/10` was in `verify.out` as written. The first row's pair is
run 1 of [`../membw-2026-08-19/`](../membw-2026-08-19/) exactly, so a row whose
column heading promised a fresh measurement repeated the figure it was meant to
test; what this run actually read was 430.3 GB/s. The upper end of the second
row, the third row's string and the fourth row's volume appear in no file here.

What the run supports is narrower than the original claim and still useful. DGEMM
and the streaming reads hold. The bandwidth came out 0.4 percent below the
432 GB/s the front page quotes, which is larger than the 0.12 percent spread
within the 19 August session and suggests the between-session spread is the wider
of the two. That figure matters beyond itself, since the decode utilisation shares
in the companion are computed against it, so the front page continues to rest on
[`../membw-2026-08-19/`](../membw-2026-08-19/), where three runs are shipped,
rather than on this one reading.

Done in the same spirit as [`../reproduce-verify-2026-08-25/`](../reproduce-verify-2026-08-25/):
after a pass in which re-running a probe found the environment behind the PyTorch
section missing, the rest of the shipped paths deserved execution rather than
another reading.
