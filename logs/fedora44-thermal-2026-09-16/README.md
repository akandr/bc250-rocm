# Does the board hold its clock under sustained load? 2026-09-16

[`scripts/thermal_sustain_f44.sh`](../../scripts/thermal_sustain_f44.sh) on the default Fedora 44
configuration with the 1500 MHz governor policy: qwen3-8B Q8_0 prefill (`-p 2048 -n 0 -r 3`) run back to
back for 20 minutes, sampling the clock step and temperatures every 5 seconds (`samples.txt`, `log`).

## Result

Twenty-two rounds. Prefill held 197.7 to 198.2 t/s in twenty of them and dipped to 190.0 and 189.8 in two, a
4 percent loss.

| | value |
|---|---|
| clock samples at 1500 MHz | 184 of 242 |
| clock samples at 1000 MHz | 58 of 242 |
| edge temperature | 57 to 94 C |
| Tctl | 60.5 to 92.9 C |
| governor throttle events logged | 1 |

Split by time, the drops are not only a warm-up effect: in the first five minutes 42 of 60 samples are at
1500 MHz with a median edge of 82 C, and after fifteen minutes 50 of 63 are at 1500 MHz with a median of
87 C and a peak of 94 C.

## Reading

The 1500 MHz policy is sustainable in the sense that matters here: throughput stays within 4 percent and the
governor intervened once in twenty minutes. But the board is running hot, and about a quarter of the samples
sit at the lower step, so it is dropping clock briefly and often instead of holding one frequency. That
also explains why benchmarks on this board need several samples and a median: a single reading can land in
one of those dips.

This is with the board's stock cooling in its usual position. The package default policy, which allows 2000
MHz at a fixed 1000 mV, overheats far harder and oscillates continuously
([`../fedora44-benchmarks-2026-09-15/clock-corrected/`](../fedora44-benchmarks-2026-09-15/clock-corrected/)).
Better cooling is the obvious lever; it is out of scope here, so these temperatures are the operating
condition every other measurement in this repository was taken under.
