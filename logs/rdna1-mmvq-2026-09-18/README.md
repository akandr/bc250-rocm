# The quantized matrix-vector kernels on RDNA1: the decode gap, 2026-09-18

Fedora 44, kernel 7.2.5 with the bc250 amdgpu module, native gfx1013 rocBLAS 7.1.1 and the corrected comgr
from `/opt/bc250-rocm`, clock policy 1500 MHz, `ollama` stopped, `hw-watcher.timer` and `crond` stopped from
08:08. Baseline is the final four-patch build (`build-hip-fatile`).

Decode on small models is the quantized matrix-vector kernel (MMVQ), and
[`logs/op-perf-hip-vs-vulkan-2026-09-17/`](../op-perf-hip-vs-vulkan-2026-09-17/) measured it at 1.5 to 1.9
times Vulkan's time per shape. Two things about it were checked at the instruction level first
(`mmvq-isa.log` in that directory): the q4_K matvec loop carries 84 vector instructions on gfx1013 against
44 on gfx1030, because every native `v_dot4` becomes four `v_mul_i32_i24` and two `v_add3_u32`; and the
launch parameters come from a table that RDNA1 is not in.

## Experiment A: the parameter table (`table-ab.log`)

`mmvq.cu` selects warps per block and rows per block from a per-architecture table: RDNA4, RDNA3, RDNA2
(also used for RDNA3.5), GCN/CDNA, Turing, and a GENERIC fallback. Both the device-side `#if` chain and the
host-side `cc` check leave RDNA1 out, so gfx1013 runs the GENERIC entry, the pre-Turing NVIDIA default:
four warps per row, 128 threads, a shared-memory exchange and a `__syncthreads()` before the final warp
reduction. The RDNA2 entry is one wave per row, no shared memory, no barrier, four times as many blocks.
Two lines route RDNA1 to the RDNA2 entry. [`scripts/mmvq_ab.sh`](../../scripts/mmvq_ab.sh): `tg128`, `-fa on`,
builds interleaved, two passes:

| model | GENERIC (baseline) | RDNA2 table | |
|---|---|---|---|
| qwen2.5-1.5B Q4_K_M | 112.75 / 112.83 | **142.48 / 142.33** | **+26 %** |
| qwen3-8B Q8_0 | 34.79 / 29.04 | 36.18 / 36.46 | +4 % |
| qwen3-14B Q4_K_M | 20.34 / 20.52 | **23.32 / 23.45** | **+14 %** |
| qwen3.6-35B MoE IQ2_M | 33.24 / 33.56 | **44.60 / 48.71** | **+34 to +45 %** |

The 8B baseline's second pass, 29.04 with a spread of 1.38, is a throttle reading; its first pass is the
one to compare. The MoE's new-build readings carry spreads of 9.3 and 6.7 inside each run, so its gain is
large and not yet precise. The 8B, whose decode already tied Vulkan, gains least.

The 1.5B's n=1 matmul shapes replayed through `test-backend-ops` (`ops-build-hip-*.log`), us per op,
baseline then RDNA2 table: `q4_K[1536,1536]` 18.19 to **14.52**; `q4_K[1536,8960]` (gate/up) 72.79 to
**47.02**; the `q6_K[1536,151936]` output head 1217.7 to **807.7**; `q4_K[8960,1536]` (down) 49.59 to
51.02 and `q6_K[8960,1536]` 57.26 to 60.64. One wave per row wins wherever K is 1536 and loses 3 to 6
percent where K is 8960, where four cooperating warps split the long reduction well; a row for long K is
the obvious follow-up and is not done here.

The 1.5B perplexity gate is **bit-identical**, 8.9498 on both builds, as it should be: the change moves
work between threads and changes no arithmetic. Decode text: `Paris`.

## Experiment B: the activation sums (`sad-ab.log`)

Half of the emulated dot products in the q4_K and q5_K matvec are not weight-times-activation at all:
`ggml_cuda_dp4a(0x01010101, u, acc)` sums four int8 activation values for the block-minimum term, and on
RDNA1 it costs the full six-instruction emulation. `v_sad_u8` sums four unsigned bytes in one instruction;
XOR with `0x80808080` turns two's-complement bytes into biased unsigned ones and the bias comes off once.
[`scripts/apply_rdna1_sad.py`](../../scripts/apply_rdna1_sad.py) puts that behind an RDNA1 guard at the
q4_K and q5_K matvec sites and the q2_K MMQ sums; it is exact integer arithmetic. The q4_K matvec loop
goes from 84 to 70 vector instructions, q5_K from 94 to 80, the two-column q4_K from 135 to 97
(`mmvq-isa.log` in the op-perf directory). [`scripts/mmvq_sad_chain.sh`](../../scripts/mmvq_sad_chain.sh)
built it on top of experiment A and ran the same A/B:

| model | RDNA2 table | RDNA2 table + `v_sad_u8` | |
|---|---|---|---|
| qwen2.5-1.5B Q4_K_M | 125.06 (spread 28.9) / 141.17 | **146.02 / 145.67** | +3 % |
| qwen3-8B Q8_0 | 33.39 (3.3) / 31.85 (5.3) | 36.26 / 36.22 | 8B has no q4_K sites; its baseline readings here are throttle noise, the table A/B read 36.2 to 36.5 |
| qwen3-14B Q4_K_M | 23.51 / 23.41 | **23.75 / 24.03** | +1.5 to 2.5 % |
| qwen3.6-35B MoE IQ2_M | 51.56 / 53.52 | 51.89 / 52.40 | unchanged, IQ2 has no such site |

Per-op on the 1.5B: `q4_K[1536,1536]` 14.54 to 14.09 us, gate/up 47.33 to 45.41, down 51.17 to 49.55. The
gates are bit-identical on both builds, 8.9498 and 9.1273, as exact arithmetic requires. A small gain
with no numerical cost, and it stays in the patch.

## What the two together are worth, and what is left

Against the four-patch build: decode 1.5B 112.8 to 145.8, qwen3-14B 20.4 to 23.9, MoE 33.4 to about 52,
8B 34.8 to 36.2. The remaining half of the emulation, the weight-times-activation dot itself, has no
cheaper instruction sequence on RDNA1 than the four `v_mul_i32_i24` it already uses; what remains after
these two changes is the 1.9x instruction count against a native `v_dot4`, and closing that means a
float-activation kernel, which is new work. The long-K shapes (`ffn_down`, K = 8960) lose 3 to 6 percent
to the one-wave-per-row geometry and would want a row of their own.

## Experiment C: the table made type-aware (`table2-ab.log`)

The campaign on the first form ([`logs/fedora44-campaign-five-patches-2026-09-18/`](../fedora44-campaign-five-patches-2026-09-18/))
showed the one-wave-per-row entry costing the Q8_0 8B 4 percent of decode. Its `vec_dot` is short and the
matvec is memory-bound, so the four cooperating warps were doing useful work there; the RDNA4 entry
upstream makes the same distinction per type. [`scripts/apply_rdna1_mmvq_table.py`](../../scripts/apply_rdna1_mmvq_table.py)
gives RDNA1 its own entry: one wave per row for the K-quants and IQ types, four warps for Q4_0, Q4_1,
Q5_0, Q5_1 and Q8_0. `tg128`, first form against final, builds interleaved:

| model | first form | type-aware | |
|---|---|---|---|
| qwen2.5-1.5B Q4_K_M | 145.21 / 145.21 | 145.75 / 146.31 | unchanged |
| qwen3-8B Q8_0 | 30.16 (spread 1.4) / 32.43 (2.5) | **36.35 / 36.80** | back to the four-warp figure |
| qwen3-14B Q4_K_M | 22.65 (2.1) / 23.86 | 24.04 / 22.41 (2.0) | unchanged, one throttled reading on each side |
| qwen3.6-35B MoE IQ2_M | 41.21 (4.1) / 52.10 | 52.19 / 52.04 | unchanged |

The 1.5B gate is bit-identical again (8.9498), decode text `Paris`. In the campaign that followed
([`logs/fedora44-campaign-patch5-2026-09-18/`](../fedora44-campaign-patch5-2026-09-18/)) the 8B reads 38.4
against the four-patch 39.0, inside its 2 to 3 percent decode spread, and every other model keeps its gain.

## Experiment D: a long-K row (`longk-ab.log`)

The one-wave geometry lost 3 to 6 percent on the K = 8960 `ffn_down` shapes. The host already dispatches a
`small_k` template variant; [`scripts/apply_rdna1_longk.py`](../../scripts/apply_rdna1_longk.py) adds its
mirror, `long_k`, set on the RDNA1 entry when K >= 8192, under which `calc_nwarps` returns four for the
K-quant and IQ types so the long row is split across warps again. Per op on the 1.5B, type-aware table
against it with `long_k`: `q4_K[8960,1536]` 49.59 to **47.36** us, `q6_K[8960,1536]` 60.68 to **57.29**,
below where the four-warp generic entry had them (49.5 and 57.3), and the short-K shapes unchanged. End
to end, `tg128`: 1.5B 145.2 / 146.1 to 146.6 / 146.8, qwen3-14B 24.0 to 23.9, the 8B and the MoE within
noise. Half a percent to one percent, exact, gate 8.9498 bit-identical, decode `Paris`. It stays, with
one correction from the campaign that followed
([`logs/fedora44-campaign-f32mv-2026-09-18/`](../fedora44-campaign-f32mv-2026-09-18/)): the row had
applied to every non-simple type, IQ included, and the qwen3.8-27B IQ3_XXS lost 4 percent of decode to
it. [`scripts/apply_rdna1_longk_kquants.py`](../../scripts/apply_rdna1_longk_kquants.py) restricts it to
the K-quants; the 27B reads 11.39 again against 11.0 with the row and 11.4 before it.

## Experiment E: float activations instead of the emulated dot (`f32mv-fused-ab.log`)

What remained after the four changes above was the emulated `v_dot4` itself. Vulkan's matvec on this GPU
never quantizes the activations: it unpacks the nibbles to float and runs `fma` chains against float
activations, and it is 1.5 to 1.9 times faster per shape. [`patches/llamacpp/rdna1-f32-matvec/`](../../patches/llamacpp/rdna1-f32-matvec/)
is that formulation in HIP for q4_K, laid out like `mul_mat_vec_q4_k.comp`: sixteen threads share a
256-value superblock, each owning sixteen values, one 32-lane wave covers two superblocks per iteration
and two rows per block; 65 VGPRs, occupancy 14, no spill, 99 vector instructions per 32 multiply-adds
against MMVQ's 4.4 per multiply-add. [`scripts/apply_rdna1_f32mv.py`](../../scripts/apply_rdna1_f32mv.py)
routes single-column q4_K matvecs to it on RDNA1 ahead of the q8_1 quantization, fused or not: the
kernel carries MMVQ's bias, gate and GLU epilogue, because in Qwen's graph the Q, K and V projections
have biases and gate/up carry the SwiGLU, and a first version without fusion reached only the `o` and
`down` projections and moved nothing end to end (`f32mv-ab.log`).

Correctness: `test-backend-ops test -o MUL_MAT` on the single-column q4_K cases, 14 of 14 against the CPU
(`f32mv-fused-correctness.log`); the 1.5B gate unchanged at 8.9498 (perplexity is batched prefill and
does not enter this kernel); and 64 greedy tokens from the 1.5B identical token for token between the
MMVQ build and this one (`greedy-mmvq.txt`, `greedy-f32mv.txt`).

`tg128`, `-fa on`, builds interleaved, two passes, after a five-minute cool-down:

| model | MMVQ, final table | float activations | |
|---|---|---|---|
| qwen2.5-1.5B Q4_K_M | 145.45 / 145.65 | **168.21 / 168.57** | **+16 %** |
| qwen3-14B Q4_K_M | 19.56 (throttled) / 23.85 | **28.90 / 28.68** | **+21 %** |
| qwen3-8B Q8_0 | 30.5 / 32.3 (throttled) | 35.4 / 36.3 | not covered, Q8_0 |
| qwen3.6-35B MoE IQ2_M | 53.36 / 53.70 | 48.9 (spread 6.8) / 53.28 | not covered, IQ2 |

Per op on the 1.5B: `q4_K[1536,256]` 9.52 to **5.06** us, `q4_K[1536,1536]` 13.99 to **8.49**,
`q4_K[8960,1536]` 47.31 to **34.98**, `q4_K[1536,8960]` 45.43 to **31.86**; the q6_K shapes are untouched
and are the next candidates. Against Vulkan's per-op figures the q4_K matvec is now within 10 to 15
percent instead of 1.5 to 1.9 times slower.

All five changes are [`patches/llamacpp/0005-rdna1-mmvq-table-and-sums.patch`](../../patches/llamacpp/0005-rdna1-mmvq-table-and-sums.patch),
which now also adds the two kernel files.

## Experiment F: the float kernel extended to q6_K and q8_0 (`f32all-ab.log`)

The q4_K kernel left the 1.5B's two q6_K tensors, its output head and its `ffn_down`, on the emulated
path, and they are the two largest single-column matmuls in its graph. The kernel in
[`patches/llamacpp/rdna1-f32-matvec/`](../../patches/llamacpp/rdna1-f32-matvec/) is now a template over
the type with the same layout for all three: sixteen threads per 256-value span, sixteen values each, one
wave two rows. For q6_K each thread unpacks its low nibbles and high bit pairs to a 0 to 63 float, runs
the `fma` chain against the activations and subtracts 32 times the activations' sum once, with the
eight-bit scales applied per sixteen-value slice, which is how Vulkan's `mul_mat_vec_q6_k.comp` does it;
for q8_0 a span is eight blocks and each thread converts its bytes and multiplies by the block scale.
The host predicate accepts Q4_K, Q6_K and Q8_0 with K a multiple of 256, single column, no expert ids,
fused or not.

Correctness, `test-backend-ops test -o MUL_MAT` on the single-column cases against the CPU: q4_K 14 of
14, q6_K 2 of 2, q8_0 16 of 16 (`tbo-*.log`). The 8B gate is unchanged at 9.1273 and the 1.5B's at
8.9498, as before: perplexity is batched prefill and never enters a single-column kernel. Greedy text
over 64 tokens: the Q8_0 8B is identical token for token to the int8 build (`greedy-8b-mmvq3.txt`,
`greedy-8b-f32all.txt`); the 1.5B is identical through "The three primary colours are red, blue, and
yellow. These colours are" and then continues "considered primary because they cannot be created by mixing
other colours" on the int8 build and "often used as the basis for mixing other colours to create a wider
range of hues" on this one (`greedy-1.5b-mmvq3.txt`, `greedy-1.5b-f32all.txt`). The q4_K-only version had
been token-identical because the output head, which decides the argmax, was still on the int8 path; a float
kernel with a different summation order shifts logits by rounding noise and a near tie between two
plausible tokens goes the other way. Both continuations are correct English and correct on the facts, and
the gate says the distribution is unchanged; this is the same kind of difference the flash-attention
patch produced, and it is the expected one.

Per op, the 1.5B graph's single-column shapes, int8 build, this build, and Vulkan from
[`logs/op-perf-hip-vs-vulkan-2026-09-17/`](../op-perf-hip-vs-vulkan-2026-09-17/) (`vk-graph-p1.log`):

| shape | int8 MMVQ | float q4_K only | float q4_K, q6_K, q8_0 | Vulkan |
|---|---|---|---|---|
| `q6_K[1536,256]` (V projection) | 10.24 | 10.24 | **4.95** | 6.43 |
| `q6_K[8960,1536]` (`ffn_down`) | 57.41 | 57.41 | **40.07** | 40.31 |
| `q6_K[1536,151936]` (output head) | 745.71 | 745.71 | **543.69** | 582.61 |
| `q4_K[1536,8960]` (`ffn_gate`) | 44.89 | 31.64 | 31.64 | 41.91 |

The q6_K shapes land at or ahead of Vulkan's, as the q4_K ones had. `tg128`, `-fa on`, builds
interleaved, two passes, the int8 build (final table, long-K row, `v_sad_u8`) against this one:

| model | int8 MMVQ | float q4_K only (experiment E) | float q4_K, q6_K, q8_0 | |
|---|---|---|---|---|
| qwen2.5-1.5B Q4_K_M | 145.74 / 145.79 | 168.2 / 168.6 | **180.82 / 181.94** | **+24 %** over int8, +8 over q4_K only |
| qwen3-14B Q4_K_M | 21.80 (1.5) / 22.35 (2.0) | 28.9 / 28.7 | **30.38 / 30.41** | +5 % over q4_K only |
| qwen3-8B Q8_0 | 31.77 (5.9) / 29.02 (throttled) | 35.4 / 36.3 | 36.80 / 36.66 | the campaign below says flat |
| qwen3.6-35B MoE IQ2_M | 52.86 / 52.43 | 53.4 / 53.3 | **55.09 / 55.71** | +5 %, the mix's few q6_K tensors |

The campaign on this build ([`logs/fedora44-campaign-float-all-2026-09-18/`](../fedora44-campaign-float-all-2026-09-18/))
puts it at nine samples per model: tg64 1.5B 176.8 to **194.6**, deepseek-r1-14B 29.2 to **31.0**, qwen3-14B
29.8 to **31.3**, MoE 57.0 to 58.6, the 8B 37.6 to 37.6, prefill unchanged everywhere. The Q8_0 8B's
matvec was already reading memory at about 310 GB/s in the int8 form. Per op, on the 8B's own graph
replayed later on the int8 build, the final float build and Vulkan (`ops-8b/`, medians of two passes):

| op | type | weights | int8 MMVQ | float | Vulkan | |
|---|---|---|---|---|---|---|
| `Kcur` | q8_0 | [4096,1024] | 19.0 | **14.2** | 24.6 | 1.34x |
| `Qcur` | q8_0 | [4096,4096] | 56.8 | 53.1 | 53.0 | 1.07x |
| `ffn_down` | q8_0 | [12288,4096] | 155.3 | 147.5 | 157.8 | 1.05x |
| `ffn_gate` | q8_0 | [4096,12288] | 151.2 | 154.4 | 161.9 | 0.98x |
| output head | q8_0 | [4096,151936] | 2147.9 | **1881.9** | 1845.3 | 1.14x |
| `Vcur` | f16 | [4096,1024] | 39.6 | 39.3 | 39.6 | not a quantized matvec |

The large shapes are within 2 percent of Vulkan's either way and the two projections that gain are a
few percent of the token, so the campaign reads flat. The kernel stays for q8_0: it is exact to
the same tests and it is the Vulkan figure per op. What the decode gap consisted of after this experiment
was the IQ2 and IQ3 types, which the MoE and the 27B are made of; experiment G is those.

## Experiment G: q5_K, the IQ types, and the experts (`f32iq/`)

After experiment F the decode gap sat in the two models whose weights the float kernel did not cover:
the qwen3.6-35B MoE, 81 percent IQ2_XXS and IQ3_XXS in its experts (`MUL_MAT_ID`) and 10 percent Q5_K in
attention, and the qwen3.8-27B, 53 percent IQ3_S, 23 percent IQ3_XXS, 13 percent IQ4_XS. The kernel gained
q5_K (q4_K's layout plus the fifth-bit bytes), IQ2_XXS, IQ3_XXS, IQ3_S and IQ4_XS (each thread takes 16
consecutive values of the superblock and unpacks them through the same codebook grids and sign tables
`vecdotq.cuh` uses), and the expert-id path: with `ids` each block's channel is one used expert, the
weights come from `ids[channel]`, and the activation vector is either per expert or, as the gate and up
projections have it, one vector broadcast to all of them. The predicate accepts one token only, which is
what decode is. [`scripts/apply_rdna1_f32iq.py`](../../scripts/apply_rdna1_f32iq.py) passes the ids
tensor through the hook in `mmvq.cu`.

Correctness on the first build, `test-backend-ops test` against the CPU (`f32iq/tbo-mm-v1.log`,
`f32iq/tbo-mmid-v1.log`): every single-column `MUL_MAT` case of the eight types, 13 of 13, and every
`MUL_MAT_ID` case of the eight types, 131 of 131, both broadcast and per-expert activations included.

Per op on the two graphs, the q4/q6/q8 float build against the first IQ build, Vulkan from the same
replay (`f32iq/ops-*-f32mv.log`, `ops-*-f32iq-v1.log`, `ops-*-vk.log`, us per op):

| model, op | type | weights | float q4/q6/q8 | + q5_K, IQ, experts | Vulkan | |
|---|---|---|---|---|---|---|
| 27B `attn_output` | iq3_s | [6144,5120] | 116.3 | **84.1** | 99.2 | 1.38x |
| 27B `ffn_down` | iq3_s | [17408,5120] | 305.5 | **224.7** | 179.7 | 1.36x |
| 27B `ffn_up` | iq3_s | [5120,17408] | 304.4 | **208.8** | 178.0 | 1.46x |
| 27B `node_13` | iq3_s | [5120,10240] | 187.2 | **131.9** | 117.3 | 1.42x |
| 27B `ffn_gate` | iq3_xxs | [5120,17408] | 295.8 | **196.5** | 157.3 | 1.51x |
| 27B `z` | iq3_xxs | [5120,6144] | 112.8 | **77.0** | 88.1 | 1.46x |
| 27B output head | q5_K | [5120,248320] | 3004.2 | **2546.7** | 2538.3 | 1.18x |
| 27B `Qcur_full` | iq4_xs | [5120,12288] | 100.8 | 172.6 | 153.8 | **0.58x** |
| 27B `linear_attn_out` | iq4_xs | [6144,5120] | 56.3 | 89.8 | 80.5 | **0.63x** |
| 27B `Vcur` | iq4_xs | [5120,1024] | 15.7 | 25.3 | 22.2 | **0.62x** |
| MoE `ffn_moe_gate`, 8 experts | iq2_xxs | [2048,512] x256 | 35.4 | **22.3** | 20.8 | 1.58x |
| MoE `ffn_moe_down`, 8 experts | iq3_xxs | [512,2048] x256 | 67.1 | **31.1** | 25.9 | 2.16x |
| MoE `ffn_moe_down`, 8 experts | iq4_xs | [512,2048] x256 | 34.6 | 31.8 | 32.3 | 1.09x |
| MoE `attn_output` | q5_K | [4096,2048] | 34.7 | **25.6** | 21.2 | 1.36x |
| MoE `node_13` | q5_K | [2048,8192] | 53.2 | **41.4** | 54.6 | 1.29x |
| MoE `z` | q5_K | [2048,4096] | 32.2 | **23.1** | 20.9 | 1.39x |
| MoE `ffn_gate` | q5_K | [2048,512] | 11.2 | **6.6** | 7.0 | 1.70x |

IQ3_S, IQ3_XXS, IQ2_XXS and q5_K land where the earlier types did, and the experts' `down` projection
halves. (Corrected 24 September: this sentence and the one below said the covered shapes land between
Vulkan's figure and 30 percent above it. Recomputed over all 18 covered rows in the tables here, the
range is 0.58 to 1.25 times Vulkan's time. Several shapes beat Vulkan, which the original wording ruled
out, and the upper end is 25 percent, not 30.) IQ4_XS is the exception: on the 27B's dense shapes
the float kernel is 0.6 times the int8 one, and Vulkan's own iq4_xs matvec is as slow as the float
kernel, so this is the formulation and not the port.

Averaging the per-shape rows by type makes that sharper, because the kernel is the same code for all
of them and only the decode differs:

| type | shapes | float kernel against int8 |
|---|---|---|
| IQ3_XXS | 3 | 1.71x |
| IQ2_XXS | 1 | 1.59x |
| IQ3_S | 4 | 1.40x |
| q5_K | 5 | 1.38x |
| q8_0 | 5 | 1.12x |
| **IQ4_XS** | 4 | **0.73x** |

Every type the float kernel covers gains except IQ4_XS, and it is the only one that loses. One kernel,
one difference: what it costs to turn that type's bytes into floats against what it costs to turn them
into int8 operands. For IQ4_XS the int8 side is a byte-permute lookup, which is as cheap as unpacking
gets, so the float path has nothing to win back. Whether some third formulation would beat both is a
question about kernel design instead of one this measurement can answer. MMVQ's iq4_xs `vec_dot` is
unusually cheap, a
byte-permute table lookup that turns four nibbles into eight int8 values and two dot products per
four bytes, and even emulated that does not lose enough to make float unpacking worth it; the mixed-in `MUL_MAT_ID` case gains 9 percent because the int8 expert path was
slow, not because the float one is fast. So IQ4_XS came back out for the second build; the 27B's
`Qcur_full`, `linear_attn_out` and `Vcur` return to the int8 kernel, about 115 us a layer. `tg128`,
`-fa on`, builds interleaved, two passes, the first IQ build:

| model | float q4/q6/q8 | + q5_K, IQ, experts (with iq4_xs) | |
|---|---|---|---|
| qwen3.8-27B IQ3_XXS | 10.88 / 11.30 | **13.59 / 13.48** | **+22 %** |
| qwen3.6-35B MoE IQ2_M | 56.36 / 55.13 | **65.88 / 66.27** | **+18 %** |
| qwen2.5-1.5B Q4_K_M (control) | 180.65 / 180.77 | 180.03 / 180.40 | unchanged, no new type |

Greedy text over 64 tokens: the MoE identical token for token to the previous build
(`f32iq/text-qwen3.6-35b-a3b-iq2m-build-hip-f32mv.txt`, `-f32iq-v1.txt`; the chain's own comparison flagged
it because it included the throughput line); the 27B diverges inside its thinking block, "(or cyan, magenta, yellow)" against "(or
CMY)", both continuations sound (`f32iq/text-qwen3.8-27b-iq3xxs-build-hip-f32mv.txt`, `-f32iq-v1.txt`), the same near-tie effect as the
1.5B's in experiment F.

The second build, without IQ4_XS (`f32iq/chain2-log`): correctness 14 of 14 `MUL_MAT` and 147 of 147
`MUL_MAT_ID` cases against the CPU (`tbo-mm-v2.log`, `tbo-mmid-v2.log`); the 27B's three iq4_xs shapes
back at their int8 figures, 9.3, 15.6, 56.6 and 101.2 us against 9.3, 15.7, 56.3 and 100.8, and the
covered shapes where the first build had them (`ops-qwen3.8-27b-iq3xxs-build-hip-f32iq-v2.log`). `tg128`,
same A/B:

| model | float q4/q6/q8 | + q5_K, IQ2_XXS, IQ3_XXS, IQ3_S, experts | |
|---|---|---|---|
| qwen3.8-27B IQ3_XXS | 10.92 / 11.26 | **14.45 / 13.43** | **+26 %** |
| qwen3.6-35B MoE IQ2_M | 55.93 / 56.44 | 51.98 (spread 5.9, throttled) / **65.93** | **+17 %** on the clean pass |
| qwen2.5-1.5B Q4_K_M (control) | 180.01 / 180.73 | 180.66 / 180.87 | unchanged |

The MoE's greedy text is again identical to the previous build's (`text-qwen3.6-35b-a3b-iq2m-build-hip-f32iq-v2.txt`),
the 27B's diverges at the same near-tie token as before (`text-qwen3.8-27b-iq3xxs-build-hip-f32iq-v2.txt`).
The campaign on this build ([`logs/fedora44-campaign-iq-float-2026-09-18/`](../fedora44-campaign-iq-float-2026-09-18/))
puts it at nine samples per model: tg64 27B 11.4 to **14.9**, MoE 58.6 to **70.1**, the other four within their
spreads, prefill unchanged everywhere. Against the three-patch build the MoE's decode has doubled (34.3
to 70.1) and the 27B's is 1.9 times (7.9 to 14.9); against Vulkan they now stand at 0.81 and 0.85 instead
of 0.39 and 0.45.

What is left of the decode gap is no longer a kernel type: per op the covered matvecs sit between
Vulkan's figure and 25 percent above it, see the correction above, and the IQ4_XS and IQ1_M shapes are already at their best on the
int8 path. The 27B's largest remaining single op is its q5_K output head at 2547 us, at Vulkan's 2538.

## Where the two hybrid models stand after G

Both the MoE and the 27B are Qwen3.5-family hybrids, three linear-attention layers (`GATED_DELTA_NET`
after a short causal conv) to every full-attention layer: 30 + 10 and 48 + 17. Their exported graphs
carry both a one-token and a 2048-token instance of every op, so the same replay says where the rest is
(`f32iq/ops-*`, us):

| op | shape | 27B ROCm | 27B Vulkan | MoE ROCm | MoE Vulkan |
|---|---|---|---|---|---|
| `GATED_DELTA_NET`, one token | [6144,129] / [4096,129] | 20.3 | 24.0 | 11.4 | 18.6 |
| `GATED_DELTA_NET`, 2048 tokens | [6144,2176] / [4096,2176] | **19929** | 4342 | **10087** | 2916 |
| `CONCAT` conv input, 2048 tokens | [2051,10240] / [2051,8192] | 10563 | 20115 | 8292 | 15914 |
| `SSM_CONV`, 2048 tokens | [10240,2048] / [8192,2048] | 888 | 778 | 741 | 605 |
| `RMS_NORM` on heads, 2048 tokens | [128,48,2048] / [128,32,2048] | 793 | 2079 | 530 | 1388 |

At one token the linear-attention op is at or ahead of Vulkan, so it is not where the decode gap
is; what remains of that gap on these two models is spread over the matvecs at Vulkan-plus-a-little and
the graph's small ops. At 2048 tokens the same op is 4.6 and 3.4 times slower on ROCm, about 20 ms a
layer on the 27B, which over its 48 linear layers is three quarters of a second of a 29-second pp2048
pass, 2 to 3 percent; the causal-conv `CONCAT` is slow on both backends, a 164 MB copy at 8 to 15 GB/s.
Neither is measured further here.
