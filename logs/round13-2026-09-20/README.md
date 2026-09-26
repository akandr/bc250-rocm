# How often the f16 accumulators need promoting, and what that says about the kernel, 2026-09-20

The prefill GEMM accumulates in f16 within a stage and promotes to f32 at the end of every stage. In the
standalone prototype that promotion cost 19 percent of the rate
([`logs/pk-gemm-prototype-2026-09-19/`](../pk-gemm-prototype-2026-09-19/): 8.0 TFLOP/s without it, 6.5
with), so promoting every second or fourth stage instead looked like free speed at a small accuracy cost.

`GGML_RDNA1_PKF16_PROMOTE` selects 1, 2 or 4 at run time through a template parameter, so one binary
holds every arm ([`scripts/round13b.sh`](../../scripts/round13b.sh), `log`):

| | pp512 promote=1 | 2 | 4 | pp2048 promote=1 | 2 | 4 |
|---|---|---|---|---|---|---|
| qwen2.5-1.5B | 1238 / 1241 | 1241 / 1242 | 1243 / 1242 | 1189 / 1188 | 1188 / 1189 | 1188 / 1189 |
| qwen3.8-27B | 74.5 / 69.6 | 71.5 / 73.7 | 70.7 / 72.0 | 78.1 / 74.0 | 73.7 / 75.9 | 74.2 / 74.5 |

No difference on the 1.5B beyond a tenth of a percent, and nothing on the 27B that its 3 to 6 percent
run-to-run spread does not cover. Perplexity is identical at every interval, 8.9274 and 6.2482.

**The promotion is free in this kernel, so it stays at every stage, which is the most accurate.** The
useful part of the result is what it says about the bottleneck: an operation that cost 19 percent when
the weights arrived ready-made in f16 costs nothing once the kernel decodes its own tiles, so this kernel
is no longer limited by its arithmetic. What it is limited by is the tile path, and the global reads
there are the suspect: one thread per row means thirty-two lanes reading addresses a row apart, 864 bytes
for a q4_K row of 1536, so every cache line delivers part of its payload.

## The first attempt measured nothing, and why

`first-attempt-invalid.log` is kept as a warning. It built three variants with the interval as a compile
time `#define`, copied `llama-bench` out of the build directory after each, and compared the three. They
are byte-identical: `llama-bench` is a thin executable and the kernel lives in `libggml-hip.so`, which
each rebuild overwrote, so all three arms ran the last library. The symptom that gave it away was
perplexity identical to four decimal places across arms that should differ slightly; the md5 of the three
binaries confirmed it. The same trap had already spoiled the end-to-end arms of
[`logs/round6-2026-09-19/`](../round6-2026-09-19/) the day before. Either use separate build directories
or make the variant a run-time switch, which is cheaper and gives one binary that holds every arm.
