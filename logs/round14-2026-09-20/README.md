# Double buffering the tiles: a wave of occupancy for nothing, 2026-09-20

[`logs/round13-2026-09-20/`](../round13-2026-09-20/) showed the prefill GEMM is no longer bound by its
arithmetic, so the next thing to try was the standard remedy for a load-bound tile loop: fetch stage
k+1's weight and activation tiles into a second shared-memory buffer while stage k's arithmetic runs.
`GGML_RDNA1_PKF16_DBUF` selects it at run time, one binary for both arms
([`scripts/round14.sh`](../../scripts/round14.sh), `log`):

| model | pp512 off | on | pp2048 off | on |
|---|---|---|---|---|
| qwen2.5-1.5B | 1238 / 1241 | 1242 / 1242 | 1188 / 1189 | 1190 / 1188 |
| qwen3-14B | **149.3** / 127.4 | 125.7 / 119.8 | 141.4 / 118.3 | 123.0 / 122.4 |
| qwen3.8-27B | 70.1 / 74.4 | 73.8 / 74.5 | 73.4 / 76.2 | 75.8 / 76.8 |

Nothing on the 1.5B, nothing on the 27B beyond its spread, and the 14B is worse: its best reading without
prefetching is 149.3 against 125.7 with it. (That model's readings wander between passes here, 149.3 then
127.4, which is the board warming through a four-minute sequence; the comparison that matters is best
against best, and the campaign has it at 151.5.)

`-Rpass-analysis=kernel-resource-usage` says why, and it is the usual trade:

| | shared memory | occupancy | VGPRs |
|---|---|---|---|
| single buffer | 17408 B | 4 waves/SIMD | 221 |
| double buffer | 34816 B | 3 waves/SIMD | 224 to 256 |

A second tile buffer costs a quarter of the occupancy on a part with 64 KB of shared memory per
workgroup processor, and three waves hide less latency than the prefetch saves. Prefetching stays in the
kernel behind the switch, off by default, because on a part with more shared memory the trade could go
the other way.
