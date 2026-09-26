# The GEMM's admission threshold, measured again at the new tile, 2026-09-21

The prefill GEMM refuses batches below 256 tokens, because "at 128 tokens this kernel is 22 percent
behind MMQ on the 1.5B, at 512 it is 36 percent ahead, so the threshold sits between them". That was
measured at the 128 by 128 tile. It is the third thing decided at that tile to reverse when the tile
changed, after q8_0 and the prefetch, so this measures it again.

## The sweep

The threshold is now `GGML_RDNA1_PKF16_MINCOLS`. Each column below is a threshold setting and each row a
batch size; where the batch is below the threshold the work goes to MMQ, so the first column is MMQ
everywhere except at 256. Medians of three rounds of three samples,
[`scripts/pkf16_threshold_sweep.sh`](../../scripts/pkf16_threshold_sweep.sh):

| model | batch | 256 | 128 | 64 | 32 |
|---|---|---|---|---|---|
| qwen2.5-1.5B | 32 | 579.5 | 579.7 | 579.5 | **318.2** |
| | 64 | 501.8 | 501.0 | **461.6** | 461.5 |
| | 128 | 699.7 | **843.3** | 843.0 | 844.0 |
| | 256 | 1328.6 | 1328.2 | 1329.7 | 1328.6 |
| qwen3.6-35B MoE | 32 | 161.1 | 160.3 | 157.0 | **121.7** |
| | 64 | 200.2 | 197.5 | **191.9** | 192.6 |
| | 128 | 293.3 | **317.9** | 319.3 | 320.2 |
| | 256 | 462.7 | 462.5 | 462.5 | 462.2 |
| qwen3.8-27B | 32 | 56.1 | 56.3 | 56.5 | **35.9** |
| | 64 | 62.0 | 62.0 | **55.9** | 55.8 |
| | 128 | 65.8 | **81.6** | 81.7 | 81.6 |
| | 256 | 101.2 | 99.7 | 99.7 | 99.8 |

Reading down the 128-token row: admitting the GEMM there is worth 1.21 times on the 1.5B, 1.08 on the
MoE and 1.24 on the 27B, where the old measurement had it 22 percent behind. Reading the 64 and 32 rows:
admitting it there costs 8 to 10 percent at 64 and a third to a half at 32. So the crossover has moved
from between 128 and 512 to between 64 and 128, and the threshold becomes 128.

Nothing at 256 tokens or above changes, which the last row of each model shows and which is why the
front page does not move: it measures pp512, where the threshold has never bound. The row threshold,
512, is untouched and untested here; none of the six models has a matmul narrower than that, so it
never binds on them either.

## One unexplained test failure

The first `test-backend-ops -o MUL_MAT` run after the build that lowered the threshold reported FAIL.
Thirty-two runs since have not: twelve at each threshold and eight before that, all 1186/1186, with no
individual case ever named. The change adds no code path to the kernel, only widens which shapes reach
it, and the same thirty-two runs cover both settings. It is recorded here because it happened, not
because it is understood; if it recurs the note is the starting point.

## Files

The `*.jsonl` are the sweep, named `<model>_c<threshold>_<round>`.
