# The RDNA1 flash-attention kernel spills, and fixing it is worth 37 percent of prefill, 2026-09-17

Fedora 44, kernel 7.2.5 with the bc250 amdgpu module, native gfx1013 rocBLAS 7.1.1 and the corrected
comgr from `/opt/bc250-rocm`, GPU clock policy at 1500 MHz with `oberon-governor` active, `ollama`
stopped. llama.cpp at `7ba604f` with this repository's three patches.

[`logs/op-perf-hip-vs-vulkan-2026-09-17/`](../op-perf-hip-vs-vulkan-2026-09-17/) found that ROCm's
`FLASH_ATTN_EXT` at pp2048 takes 114.7 ms against Vulkan's 16.3 ms, that the gap is flat in batch
size, and that it is not the tile table. This is the cause and a fix.

## The cause

`V_DOT2_F32_F16_AVAILABLE` in `ggml/src/ggml-cuda/common.cuh` is defined for RDNA2, RDNA3, RDNA4,
gfx906 and CDNA. **RDNA1 is not on that list**, correctly: gfx1010 and gfx1013 have no
`v_dot2_f32_f16`, which LLVM's `FeatureISAVersion10_1_3` confirms and `hipcc` enforces
([`logs/gfx1013-dot-isa-2026-09-17/`](../gfx1013-dot-isa-2026-09-17/)). Without it,
`ggml_cuda_mad(float &, half2, half2)` takes the fallback that unpacks each `half2` into two floats
and issues two separate FMAs.

In the tile flash-attention kernel that fallback sits in the innermost KQ accumulation loop, and it
doubles the live values there. The D=128 kernel then overruns the 256-VGPR budget badly
(`kernel-resource-usage.log`):

| arch | VGPRs | VGPRs spilled | scratch bytes/lane | occupancy waves/SIMD |
|---|---|---|---|---|
| gfx1010 (RDNA1) | 256 | **569** | **2280** | 4 |
| gfx1013 (RDNA1) | 256 | **569** | **2280** | 4 |
| gfx1030 (RDNA2) | 215 | 0 | 0 | 4 |
| gfx1100 (RDNA3) | 125 | 0 | 0 | 6 |

Every inner iteration goes through scratch memory. That is why the deficit is uniform in batch size,
not a cliff, and why swapping the tile table changed almost nothing: the generic table
spills too.

**gfx1010 spills identically, so this is RDNA1-wide and not a BC-250 quirk.** Anyone with an RX 5000
card can reproduce the resource numbers with `hipcc` alone, no BC-250 required.

## The fix

[`patches/llamacpp/0004-rdna1-fattn-tile-no-spill.patch`](../../patches/llamacpp/0004-rdna1-fattn-tile-no-spill.patch):
an RDNA1 tile-configuration table, selected ahead of the shared RDNA one on both the host and the
device side, that doubles `nthreads` to 512 and cuts `nbatch_fa` to 32 for the two D=128 rows
prefill uses. Everything else falls through to the existing RDNA table. That halves the columns each
warp carries and brings the kernel to 211 VGPRs with **no spill and no scratch**.

The change is scoped to RDNA1 deliberately. Applied to the shared RDNA table it would also retune
RDNA2, RDNA3 and RDNA4, which do not spill and were presumably tuned as they stand; nothing here
tested those parts. The patch grew on 18 September, with rows for the wider heads, rows for the
small-batch instances, a wider `nbatch_K` on row 64 and one dispatch check; each addition is measured
in its own section below and all of them are in the one patch file.

`nbatch_fa = 32` cannot be applied to the smaller `ncols` rows: it makes
`KQ_acc[nbatch_fa/(np*warp_size) * cpw]` a zero-length array and the build fails. Those rows are
left alone, so a few smaller instances still spill (284 to 968 bytes/lane). Only the two rows
prefill uses are fixed.

### The wider heads, added on 18 September

Three of the six benchmark models have wider heads than 128: the qwen3.6-35B MoE and qwen3.8-27B at
D=256, gemma4 at D=512, and their tile kernels spill worse than D=128 did (700 to 1200 registers at
D=256, 900 to 1040 at D=512, nothing on gfx1030 at either). Compile-time sweeps with
[`scripts/fa_row_sweep.sh`](../../scripts/fa_row_sweep.sh), in `kernel-resource-usage.log`, settled
on rows added to the same RDNA1 table:

| D | rows | nthreads, occupancy, nbatch_fa, nbatch_K | prefill instances, spilled VGPRs |
|---|---|---|---|
| 256 | 32, 16 | 512, 3, 32, 128 | 700 to 1200 down to 51 to 68 |
| 256 | 8, 4, 2 | 256/128/64, 4, 32, 64 | 254 to 392 down to 0 to 69 |
| 512 | 32, 16 | 512, 2, 32, 128 | 940 down to 63 to 67 |

The wider `nbatch_K` was the lever at both sizes (64 left D=512 at about 600, 32 made it worse), and
`nbatch_fa` below 32 does not compile. Neither size reaches zero the way D=128 did: D=256 and D=512
carry two and four times the per-column state, and a few dozen registers of spill remain on the
prefill instances, with the `oob_check` variants and the no-GQA `ncols2 = 1` variant still at 200 to
380. [`scripts/apply_rdna1_fattn_rows.py`](../../scripts/apply_rdna1_fattn_rows.py) writes the rows
into a tree that already carries patch 4, and the regenerated patch file includes them. What the
residual costs is a measurement, not a compile result:
[`scripts/fa_bighead_ab.sh`](../../scripts/fa_bighead_ab.sh) runs the three models against both
builds, and its results follow below.

## Result

**Prefill up 37 percent, decode unchanged, and the kernel goes from 7.05x Vulkan to 1.62x** with the
first row 64; with the final one (`nbatch_K = 128`, below) the op reads 16854 and 16869 us in two passes
(`depth-final.log`), **parity with Vulkan's 16260**, from 114673 three days earlier. `nbatch_K = 256` does
not compile, so 128 is where this row ends.

| | before | after | |
|---|---|---|---|
| `FLASH_ATTN_EXT` ne=[128,12,2048,1] | 114673 us | 26277 us, then 16862 with the final row | **6.8x faster** (Vulkan: 16260) |
| qwen2.5-1.5B pp2048, `-fa on` | 634.18 | 869.87 | **+37.2 %** |
| qwen3-8B pp2048, `-fa on` | 185.60 | 256.30 | **+38.1 %** |
| qwen2.5-1.5B tg128, `-fa on` | 113.44 | 113.61 | unchanged |

Both prefill figures reproduce across two passes to two decimals. The decode figures are medians of
three passes of five repeats each; the first `build-hip-f44` pass read 111.79 with a spread of 4.40
and the other five passes all sit between 113.43 and 113.64, so decode is a tie.

### What the wider-head rows are worth (`bighead-ab.log`)

[`scripts/fa_bighead_ab.sh`](../../scripts/fa_bighead_ab.sh), two passes, builds interleaved per point,
each figure its own invocation. gemma4 dropped out: its file is a truncated blob (`wrong number of
tensors; expected 2131, got 720`) that neither build can load, as the August notes already recorded,
so the D=512 rows are compile-verified only and no model here exercises them.

| | before | after | |
|---|---|---|---|
| qwen3.6-35B MoE, `FLASH_ATTN_EXT` prefill op | 338.6 ms | 97.0 ms | 3.5x faster |
| qwen3.6-35B MoE, pp2048 `-fa on` | 259.7 / 265.0 | **287.8 / 290.0** | **+10 %** |
| qwen3.6-35B MoE, pp2048 `-fa off` (control) | 294.5 / 297.6 | 291.8 / 300.5 | unchanged |
| qwen3.6-35B MoE, tg64 | 32.1 / 32.7 | 32.6 / 32.6 | unchanged |
| qwen3.8-27B, `FLASH_ATTN_EXT` prefill op | 507.3 ms | 145.4 / 153.3 ms | 3.4x faster |
| qwen3.8-27B, pp2048 `-fa on` | 61.1 / 61.5 | **64.8 / 65.2** | **+6 %** |
| qwen3.8-27B, pp2048 `-fa off` (control) | 65.7 / 63.3 | 66.5 / 65.0 | unchanged |
| qwen3.8-27B, tg64 | 7.61 / 7.62 | 7.60 / 7.63 | unchanged |

The kernel gains are of the same order as at D=128 (3.4 to 3.5x against 4.4x), but the end-to-end
gains are smaller because attention is a smaller share of these models' prefill: the MoE's time is in
its expert matmuls and the 27B's in dense ones. With the rows in place `-fa on` prefills within 1 to
3 percent of `-fa off` on both, where before it trailed by 10 percent on the MoE. The 27B's pp2048
readings carry a 4 to 6 point spread inside every run on both builds; it is the one model here that
fills memory, and the spread does not depend on the build.

Gates: MoE 6.4064 before, 6.3752 after (0.5 percent lower); 27B 6.2509 before, 6.2513 after. The 27B
is as close to unchanged as reordered accumulation gets. The MoE shift is three times the size of the
D=128 shifts, so the same command was run on Vulkan: **6.3991**. The unpatched build sits 0.1 percent
above Vulkan and the patched one 0.4 percent below it. That is a larger move than the D=128 rows
produced, still an order of magnitude below anything a wrong kernel has ever produced on this board
(the garbled decode of August read as a perplexity of 8.9425 against a correct 8.9442, which is why
decode text is checked separately), and it lands on the other side of the reference instead of
further from it in the same direction. It is reported as what it is: a 0.5 percent shift from
reordered accumulation on a model whose attention runs 3.5 times faster, not a bit-identical gate.
The 27B on Vulkan reads 6.2857, so both HIP builds sit 0.55 percent below it and the patch did not
move that model at all. Decode text on the MoE with the new build, whose decode-shaped D=256 rows
also changed: the model answers `The capital of France is` by opening a reasoning block, `Here's a
thinking process: 2. Identify Key Information: The question is asking for the capital city of
France.`, coherent and on topic, which is what a garbled attention kernel does not produce.

### The models without a power-of-two head ratio (`ab14b.log`)

[`scripts/fa_rebuild_and_ab14b.sh`](../../scripts/fa_rebuild_and_ab14b.sh) applies the rows and the
dispatch check, rebuilds, re-checks the 1.5B and 8B, then runs [`scripts/fa_ab14b.sh`](../../scripts/fa_ab14b.sh).

The four-patch campaign ([`logs/fedora44-campaign-fa4-2026-09-18/`](../fedora44-campaign-fa4-2026-09-18/))
moved every model except deepseek-r1-14B and qwen3-14B, both 40 query heads over 8 KV heads. That ratio
has no power-of-two divisor, so llama.cpp dispatches them with `ncols2 = 1`, and with the committed row
64 the `<128,128,64,1>` instance still spilled 137 registers while every GQA-sharing instance of the
same row was clean. A second sweep of row 64, this time replacing the row in place (the first pass had
inserted candidates after it and, the table being first-match-wins, changed nothing), found no
compiling configuration that brings that instance under about 115 spilled registers: each of its 64
columns carries its own K and V. The 32-column `<32,1>` instance, by contrast, is clean under every
candidate.

So the fix for these models is dispatch, not tuning. `launch_fattn_tile_switch_ncols1` picks 64 columns
on HIP whenever `Q->ne[1] > 32/ncols2`; the patch adds, in that host-side branch, a check gated on
`GGML_CUDA_CC_IS_RDNA1(cc) && ncols2 == 1` that keeps such models at 32 columns. The same rebuild moved
row 64 to `nbatch_K = 128`, which leaves `<64,1>` where it was but halves the register use of the GQA
variants (`<8,8>` from 211 VGPRs at occupancy 4 to 96 at occupancy 10), and added rows 16 and 8 that
clear the small-batch instances. All three changes are in the regenerated patch. Measured, builds
interleaved, two passes:

| | three patches | four patches, final | |
|---|---|---|---|
| deepseek-r1-14B pp512 | 92.78 / 92.61 | **96.55 / 96.49** | +4.1 % |
| deepseek-r1-14B pp2048 | 83.11 / 84.14 | **92.42 / 92.73** | +10.7 % |
| deepseek-r1-14B gate | 5.9756 | 5.9765 | |
| qwen3-14B pp512 | 94.65 / 94.75 | 80.01 / **98.33** | +3.8 % (see below) |
| qwen3-14B pp2048 | 87.66 / 87.56 | **95.25 / 95.05** | +8.6 % |
| qwen3-14B gate | 7.7645 | 7.7536 | |
| qwen2.5-1.5B pp2048 | 634 | 869.9 with the earlier row 64, **888.8** now | |
| qwen2.5-1.5B gate | 8.9442 | 8.9498 (Vulkan 8.9734) | |
| qwen3-8B pp2048 | 186 | 257 with the earlier row 64, **264.1** now | |
| qwen3-8B gate | 9.1117 | 9.1273 | |

Pooled over the two passes, the two 14B rows read 83.6 and 87.6 before and 92.6 and 95.2 after,
which is how the front page's flash-attention table quotes them.

The same rows at depth, qwen3-8B pp2048 `-fa on`, two passes (`depth-final.log`): 269.3 / 263.6 at
depth 0, 219.9 / 219.4 at 4096, 182.5 (spread 9.3) / 186.6 at 8192, against 194 / 97 / 65 with three
patches and 257 / 196 / 157 with the first row 64. Pooled over the two passes those are 266.4, 219.7 and
184.5, which is the four-patch column of the front page's prefill-at-depth table. At 4096 and 8192 that is ahead of Vulkan's default
flash-attention path (217.0 and 143.5) and at 78 percent of Vulkan with llama.cpp #28507 (293.6 and
236.1).

qwen3-14B's first-pass pp512 on the new build, 80.01 with a spread of 1.18, came directly after that
model's pp2048 run on the other build and is the throttling signature seen before; the second pass reads
98.33 with a spread of 0.34 and matches deepseek's gain. The gains are smaller than the GQA models' 37
percent because a 32-column tile does half the work per K and V load of a 64-column one; they are gains
where there were none. The `nbatch_K = 128` row is worth a further 2 to 3 percent on the D=128 GQA
models on top of the earlier measurements.

## Against depth, and the flash-attention trade inverts

[`scripts/fa_spill_depth.sh`](../../scripts/fa_spill_depth.sh), qwen3-8B Q8_0, pp2048 at three depths,
both builds and both `-fa` settings at every point, three passes, medians
(`depth-sweep.log`, `depth-medians.txt` from
[`scripts/fa_spill_depth_medians.py`](../../scripts/fa_spill_depth_medians.py)):

| context already present | `-fa off` (either build) | `-fa on`, unpatched | `-fa on`, patched |
|---|---|---|---|
| 0 | 243.5 | 194.3 | **256.7** |
| 4096 | 165.4 | 97.3 | **195.5** |
| 8192 | 117.3 | 64.9 | **157.4** |

The `-fa on` figures are tight: the unpatched arm repeats to 0.1 percent at every depth and the patched
arm to 0.3, 2.1 and 1.5 percent. So with the patch, flash attention is the faster prefill setting at
every depth, by 5, 18 and 34 percent, where before the patch it was the slower one by 20, 40 and 45.
The trade this repository documented in
[`logs/flash-attention-tradeoff-2026-09-17/`](../flash-attention-tradeoff-2026-09-17/) is gone on
ROCm: `-fa on` is now the right setting for prefill as well as decode.

**A control that only half passed.** The `-fa off` arms run identical code in both builds, so they
should agree, and at depth 0 they do (within 0.8 percent in every pass). At 4096 and 8192 they
disagree by 7 to 10 percent in passes 1 and 2 and agree in pass 3. Pass 1 overlapped compiler jobs I
was running on the board, which heat the package the GPU throttles on (the edge sensor fell from 81
to 59 C once they were killed); pass 2 did not, and the pattern is that the arm which runs first
after a heavy `-fa on` point reads low. That is an ordering effect the harness did not randomise.
The `-fa off` column above is therefore the pooled median of all six readings per depth, 243.47, 165.44
and 117.34, and its spread is 1.5, 15 and 10 percent at the three depths. Nothing in the `-fa on` comparison depends on
it: the patched and unpatched `-fa on` arms differ by factors the order effect cannot produce.

## Correctness

| gate | before | after | Vulkan |
|---|---|---|---|
| qwen2.5-1.5B, `-fa on`, c 4096, 8 chunks | 8.9442 | 8.9306 | 8.9734 |
| qwen3-8B Q8_0, `-fa on`, c 2048, 2 chunks | 9.1117 | 9.1248 | not run |

The gates move by 0.15 and 0.14 percent, in opposite directions, and the patched build sits closer
to the unpatched HIP build than either sits to Vulkan. Changing `nthreads` and `nbatch_fa` changes
the order of floating-point accumulation in the softmax and the V aggregation, so a shift of this
size is expected; a systematic bias would not change sign between models. Decode was checked
separately, because perplexity is computed over batched prefill and misses the decode kernel that
garbled for months on this board: the patched build answers `The capital of France is` with `Paris`.

This is a smaller check than the repository's usual bit-identical gate, and it is the honest one to
report: the patch changes arithmetic order by construction, so bit-identical results were never
available.

## Files

| | |
|---|---|
| `kernel-resource-usage.log` | register, scratch and occupancy figures across four architectures |
| `bench.log` | every benchmark and gate quoted above |

The numbers above were first measured with a two-line edit to the shared RDNA table, then the patch
was rewritten as a separate RDNA1 table so it could not retune RDNA2 and later, rebuilt and
re-measured: 26261 us on the kernel, 869.91 pp2048 and the same 8.9306 gate, so the rewrite is
equivalent.
