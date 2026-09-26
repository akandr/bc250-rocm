# The hybrid models' linear-attention layers at prefill: the gated delta net's lane count and a transposed concat, 2026-09-19

Experiment G ([`logs/rdna1-mmvq-2026-09-18/`](../rdna1-mmvq-2026-09-18/)) ended with two prefill ops on the
Qwen3.5-family hybrids, the qwen3.6-35B MoE and the qwen3.8-27B, that stood out per op: `GATED_DELTA_NET`
at 2048 tokens 3.4 to 4.6 times slower than Vulkan's, and the `CONCAT` that builds the causal conv's
input slow on both backends. This directory is both, measured on the board and fixed.

## The gated delta net: lanes per column

The kernel gives each column of the S_v x S_v recurrent state one full wave (32 lanes at S_v = 128, four
rows per lane) and reduces across the wave twice per token; the loop over tokens is sequential, so its
speed is the latency of one iteration times the number of waves the GPU has to run through. ggml-vulkan's
shader for the same op takes the lane count as a specialisation constant and uses 8 lanes per column when
S_v >= 128 and clustered subgroup ops exist, which RADV has: four columns per subgroup, 16 rows per lane,
a quarter of the waves. [`scripts/apply_rdna1_gdn_lanes.py`](../../scripts/apply_rdna1_gdn_lanes.py) makes
the lane count a template parameter of the CUDA kernel, leaves every other architecture on the existing
geometry, and on RDNA1 picks it for the S_v = 128 kernels. `hipcc -Rpass-analysis=kernel-resource-usage`
for gfx1013, the non-KDA single-state kernel:

| lanes per column | columns per wave | rows per lane | VGPRs | occupancy (waves/SIMD) |
|---|---|---|---|---|
| 32 (as upstream) | 1 | 4 | 33 | 20 |
| 16 | 2 | 8 | 47 | 20 |
| 8 | 4 | 16 | 76 | 12 |
| 4 | 8 | 32 | 139 | 7 |

Correctness, `test-backend-ops test -o GATED_DELTA_NET` against the CPU at every lane count: 25 of 25
(`tbo-gdn-l*.log`; the lanes-4 run lists 24 because one case's line was split by the graph-warmup message).
The two graphs' linear-attention lines replayed at each lane count, medians of two passes, and Vulkan on
the same lines (`ops-*.log`, us):

| op | 27B: ne | as upstream | 16 lanes | **8 lanes** | 4 lanes | Vulkan |
|---|---|---|---|---|---|---|
| `GATED_DELTA_NET`, 2048 tokens | [6144,2176] | 19799 | 14885 | **6758** | 8534 | 4319 |
| `GATED_DELTA_NET`, one token | [6144,129] | 19.7 | 19.4 | 23.8 | 40.5 | 23.8 |
| `SSM_CONV`, 2048 tokens | [10240,2048] | 887 | 888 | 889 | 887 | 775 |

| op | MoE: ne | as upstream | 16 lanes | 8 lanes | **4 lanes** | Vulkan |
|---|---|---|---|---|---|---|
| `GATED_DELTA_NET`, 2048 tokens | [4096,2176] | 10212 | 6795 | 6828 | **4886** | 2901 |
| `GATED_DELTA_NET`, one token | [4096,129] | 11.9 | 11.5 | 15.1 | 25.0 | 18.6 |
| `SSM_CONV`, 2048 tokens | [8192,2048] | 699 | 701 | 697 | 700 | 602 |

At 2048 tokens 8 lanes is 2.9 times faster on the 27B (48 heads) and 1.5 on the MoE (32 heads), where 4
lanes goes further still: with fewer heads there are fewer waves to begin with, and the loss of
occupancy at 139 registers costs less than the halving of the wave count gains. At one token the
one-column-per-wave form is the fastest on both, since a single iteration is all there is and the wider
reductions are cheap next to the launch. The patch therefore keeps 32 lanes at one token and uses 8
above it, which is Vulkan's choice and within 1.4 times of it on the 27B; the MoE's extra gain from 4
lanes is 30 layers times 2 ms, under one percent of its pp2048, and is left on the table instead of
adding a head-count rule from two data points.

## The concat with a transposed source

The conv input is `ggml_concat(state, ggml_transpose(x), 0)`: a [3, C] state in front of a [T, C] view
whose element stride along dim 0 is a whole row. CUDA's non-contiguous concat kernel walks dst rows with
one lane per element, so every lane reads x 40 KB from its neighbour, and the 164 MB copy runs at 15 GB/s
(10.6 ms per layer on the 27B, 8.3 on the MoE); Vulkan's does the same thing at 8 GB/s.
[`scripts/apply_concat_transposed.py`](../../scripts/apply_concat_transposed.py) adds a path for dim-0
concats whose second source is transposed in its first two dimensions: a 32 x 32 tile through shared
memory, read along x's contiguous dimension and written along dst's, plus a small kernel for the first
source's part of each row. Not RDNA-specific, and it adds `test-backend-ops` cases with a transposed
second source (v = 16), including the real [3, 10240] + [2048, 10240] shape:

| `CONCAT`, 2048 tokens | before | after | Vulkan |
|---|---|---|---|
| 27B, [2051,10240] | 10557 | **3158** | 20133 |
| MoE, [2051,8192] | 8252 | **2545** | 15912 |

Correctness: 104 of 104 `CONCAT` cases against the CPU, the four transposed ones among them
(`tbo-concat.log`). 3.3 times faster, and the one-token concat is unchanged (5.7 and 4.1 us).

## End to end

`llama-bench`, `-fa on`, three repetitions per figure, builds interleaved, two passes, the current build
against the experiment build with 8 lanes and the concat path (`log`):

| model | figure | current | + GDN 8 lanes, concat tile | |
|---|---|---|---|---|
| qwen3.8-27B | pp512 | 69.99 / 70.12 | **71.40 / 71.54** | **+2.0 %** |
| qwen3.8-27B | pp2048 | 63.96 (2.0) / 63.26 (3.7) | 62.90 (3.8) / 62.89 (2.8) | within its spread |
| qwen3.8-27B | tg64 | 12.44 (1.7) / 12.92 (1.2) | 13.49 (1.5) / 13.27 (1.8) | noisy after prefill, same kernels |
| qwen3.6-35B MoE | pp512 | 288.1 / 289.9 | **295.1 / 312.3** | **+2.5 to +7.7 %** |
| qwen3.6-35B MoE | pp2048 | 291.3 / 290.4 | **298.4 / 302.5** | **+2.5 to +4 %** |
| qwen3.6-35B MoE | tg64 | 67.5 / 66.9 | 66.8 / 67.4 | unchanged |

The 27B's pp2048 readings carry 2 to 4 percent spreads within a run on both builds, which is the board
warming through the three repetitions of a 30-second prefill; its pp512 gain is the clean figure. The
27B's greedy text is identical token for token between the builds (`text-27b-*.txt`).

## The final form

The lane count now follows the token count: 8 lanes above one token, the upstream geometry at one
(`chain2-log`, `tbo-gdn-final.log`, `ops-*-final-*.log`). Correctness 26 of 26; per op the one-token GDN is
back at 19.9 and 12.1 us and the 2048-token one at 7123 and 7082 us, the concat at 3199 and 2666. pp512
and tg64 in one invocation each, three repetitions, two passes:

| model | pp512 current | pp512 final | | tg64 current / final |
|---|---|---|---|---|
| qwen3.8-27B | 70.18 / 69.93 | **71.37 / 70.97** | **+1.5 %** | 14.39 / 14.37 and 14.39 / 14.33 |
| qwen3.6-35B MoE | 287.4 / 288.1 | **294.4 (31.9) / 297.8** | **+2.5 to +3.4 %** | 64.8 (4.3) / 65.8 (6.0) and 65.4 (3.9) / 64.9 (5.1) |

The MoE's perplexity over three 2048-token chunks (`wiki.test.raw`, `-c 2048`) is where the two changes
separate. The reference build reads 6.7545; the final build 6.7804; the final build with
`GGML_GDN_LANES=0`, that is the upstream lane geometry and only the concat path active, 6.7545 again, to
the digit. So the concat is the exact copy it should be and the 0.4 percent belongs to the gated delta
net's changed reduction order, four partial sums of sixteen instead of one of thirty-two, compounding
through a 2048-step recurrence. Vulkan, which runs the same op with 8 lanes, reads 6.8171 on the same
text, 0.9 percent above both HIP builds; the three sit within one percent of each other, which is where
the flash-attention patch's reorderings landed too. The tg64 readings of the MoE carry 4 to 6 percent
spreads here because each follows a pp512 in the same invocation, the artefact the campaign script
avoids; the campaign's own figure for this build is on the front page.
