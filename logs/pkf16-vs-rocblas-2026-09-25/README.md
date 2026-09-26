# Where rocBLAS HGEMM's missing 3x is: the kernel, not the hardware, 2026-09-25

## The question this closes

rocBLAS HGEMM on this board reaches about a third of the measured packed-fp16 ceiling and is only
1.7 percent faster than its own SGEMM
([`logs/torch-rocblas-bench-2026-09-24/`](../torch-rocblas-bench-2026-09-24/)). The first explanation
offered, that Tensile does not emit `v_pk_fma_f16` for gfx1013, was wrong and was retracted the day
it was published ([`logs/hgemm-isa-2026-09-24/`](../hgemm-isa-2026-09-24/)): the code object HGEMM
loads carries 720 of them. That left the gap real and its cause open.

The question a retraction does not answer is whether the gap belongs to the library or to the
machine. This answers it: a straightforward hand-written packed-fp16 GEMM, on the same board at the
same clock, is **1.9 times rocBLAS HGEMM at N=8192**, at the same accuracy.

## Result

Both arms in one boot, nothing else using the GPU, oberon policy 1000 to 1500 MHz, the same data
(uniform in [-0.5, 0.5) from the same generator) and the same accuracy check (two output elements
against a double-precision host dot product). rocBLAS ran first, so the order does not favour it.

rocBLAS is from `pkf16_vs_rocblas.txt`; the kernel column is `pk_run3.txt`, the run that carries the
accuracy check.

| N | rocBLAS HGEMM | rel err | hand-written kernel | rel err | ratio |
|---|---|---|---|---|---|
| 512 | 1724.6 GFLOP/s | 3.8e-03 | 1640 GFLOP/s | 6.8e-03 | 0.95 |
| 1024 | 3381.3 | 5.4e-03 | 5140 | 4.9e-03 | **1.52** |
| 2048 | 2867.7 | 5.7e-03 | 4980 | 1.2e-02 | **1.74** |
| 4096 | 4514.6 | 3.3e-02 | 7660 | 1.8e-02 | **1.70** |
| 8192 | 4671.8 | 2.5e-02 | 8910 | 3.5e-03 | **1.91** |

The hand-written kernel reaches **8.9 TFLOP/s, 62 percent of the 14.4 TFLOP/s packed-fp16 ceiling**,
where rocBLAS reaches 32 percent. Three runs of it at N=8192 read 8.94, 8.95 and 8.91 TFLOP/s, so the
figure reproduces to better than half a percent. The first two are in `pkf16_vs_rocblas.txt`, before
the accuracy check was added; the third is `pk_run3.txt` and is the column above.

Accuracy is the same order on both arms and neither is good: both accumulate in fp16, and the
relative error runs from 3e-03 to 3e-02. Comparing them on speed is therefore like for like. Two
probe elements per size is a weak accuracy test and is quoted only to show the arms are comparable,
not to characterise either kernel.

## What the kernel is

[`scripts/pk_gemm_square.cpp`](../../scripts/pk_gemm_square.cpp), the prototype from
[`logs/pk-gemm-prototype-2026-09-19/`](../pk-gemm-prototype-2026-09-19/) run on rocBLAS's square
shapes. An LDS-staged tile GEMM, tiles into shared memory, accumulators held as `__half2`, one
`v_pk_fma_f16` per two outputs. No double buffering, no assembly, no per-shape tuning: one tile
geometry (BM 128, BN 128, BK 32, TM 8, TN 16) for every size. It is not a better GEMM than Tensile
could write; it is the least a packed-fp16 kernel can be.

## A correction to how the earlier disassembly was done

The 24 September disassembly was taken from `build-restored-2026-08-20`, which is the ROCm 6.4.2-era
rocBLAS. That build no longer loads at all on this system: it needs `libamdhip64.so.6` and the board
carries `.so.7`. The `strace` that identified the loaded object captured basenames only, so the file
that was disassembled was never shown to be the file in use.

Redone here with full paths (`tensile_objects_opened.txt`). Two installs are in play and they agree:
the timed run above loaded `/opt/bc250-rocm/lib64/librocblas.so.5`, which is what the recipe installs,
and the traced run loaded `/home/akandr/rb711/install/lib64` through `LD_LIBRARY_PATH`. Their gfx1013
HH objects are byte-identical, md5 `cc7443b90911`. Its gfx1013 HH object carries the same 720 `v_pk_fma_f16` and no
`v_fma_mix_f32` (`hh_object_across_builds.txt`), so the retraction stands. The evidence behind it
did not, until now.

## What this does and does not settle

Settled: the 3x is in rocBLAS's kernel for this target. The instruction is emitted, the hardware
sustains 8.9 TFLOP/s through it, and an untuned kernel gets most of the way there.

The library ships no tuned solutions for this target at all: all 54 of its gfx1013 Tensile files are
`fallback` (`tensile_gfx1013_files.txt`), and both precisions land on small macro tiles at workgroup
16x16.

**It is the tile, and it is not occupancy.** Reading the resource usage out of both kernels
(`kernel_resources.txt`) rules the obvious alternative out, because the faster kernel is the one with
the worse occupancy:

| | rocBLAS `MT64x128x8` | hand-written |
|---|---|---|
| VGPRs | 70 | **113** |
| LDS per workgroup | 3072 B | **17408 B** |
| threads per workgroup | 256 | 128 |
| outputs per thread | 4x8 = 32 | **8x16 = 128** |
| staged bytes per K step | 3072 | 16384 |
| MACs per staged byte | 21.3 | **32.0** |
| register spills | none | none |

The hand-written kernel asks for 1.6 times the registers and 5.7 times the shared memory, so fewer of
its waves fit on a SIMD, and it is still 1.9 times faster. What it buys with those registers is a
thread tile of 128 outputs against 32, so every value read from shared memory is reused four times as
often, and a K step that does 32 MACs per staged byte against 21.3.

That is what an untuned fallback tile costs on this part: Tensile's default has to be safe on
architectures it has never seen, and a tile that small leaves the arithmetic units waiting on LDS.
Neither kernel spills.

**The distance left to the ceiling is not tile choice either.** Sweeping the hand-written kernel's
geometry at N=8192 puts the shipped configuration at the top of nine tried, and every perturbation
loses (`tile_sweep.txt`, `tilesweep.sh`):

| BM BN BK TM TN | LDS | VGPR | TFLOP/s | of 13.02 |
|---|---|---|---|---|
| **128 128 32 8 16** | 17408 | 113 | **8.90** | **68 %** |
| 256 128 32 8 16 | 25600 | 142 | 6.88 | 53 % |
| 128 256 32 8 16 | 25600 | 146 | 6.68 | 51 % |
| 64 128 32 8 16 | 13312 | 127 | 5.89 | 45 % |
| 128 128 32 8 8 | 17408 | 162 | 5.74 | 44 % |
| 128 128 32 4 16 | 17408 | 169 | 4.32 | 33 % |
| 128 128 64 8 16 | 34816 | 113 | 4.22 | 32 % |
| 128 128 32 16 16 | 17408 | 182 | 4.07 | 31 % |
| 128 128 16 8 16 | 8704 | 208 | 2.38 | 18 % |

The two geometries with a larger macro tile, 256x128 and 128x256, stage more work per byte than the
winner and are still a third slower, because 25.6 KB of shared memory per workgroup costs more
occupancy than the extra reuse returns. Halving the depth to BK 16 is worst of all and, unexpectedly,
needs 208 registers, not 113.

So the tile explains the gap between rocBLAS and this kernel, and does not explain the gap between
this kernel and the ceiling. What is left is what this kernel does not do: no double buffering, no
attention to LDS bank behaviour, no instruction scheduling beyond what the compiler chooses. Two
caveats on that. Nine configurations around one point is a local search instead of a proof that no
better tile exists. And each timed run in the sweep followed two compilations on the same machine,
which heats the CPU and depresses every figure in the table equally, so the ranking is sound and the
absolute values are the pessimistic end; the 8.90 for the winner matches the clean runs elsewhere on
this page.

Also not addressed: fp16 accumulation is too inaccurate for most real work at these sizes, on either
arm. A kernel anyone would ship needs periodic promotion to fp32, which costs some of the rate. The
1.9x is the gap between two fp16-accumulating kernels, not a claim about a usable fp16 GEMM.

## Reproducing

    hipcc -O3 --offload-arch=gfx1013 -DBM=128 -DBN=128 -DBK=32 -DTM=8 -DTN=16 \
      -o pk_gemm_square scripts/pk_gemm_square.cpp -L/usr/lib64 -lamdhip64
    hipcc -O3 --offload-arch=gfx1013 -I<rocblas>/include \
      -o rocblas_bench scripts/rocblas_bench.cpp -L<rocblas>/lib64 -lrocblas -L/usr/lib64 -lamdhip64

Stop anything else using the GPU, check `/etc/oberon-config.yaml` caps at 1500 MHz, and let the board
cool between arms. Compiling on the board during a measurement heats the CPU enough to throttle the
GPU, so build first and measure afterwards.

**The script did not reproduce this page, 25 September 2026.** `scripts/pk_gemm_square.cpp` shipped
with `BK` defaulting to 16 while the comment at the top of that same file, and `tile_sweep.txt` below,
both give 32. Anyone running it got 2.31 TFLOP/s at N=8192, not the 8.90 on this page. Rebuilt
with `BK` 32 it returns 123.557 ms and 8.90 TFLOP/s, matching what is published here, so the figures
stand; the default is now 32. `tile_sweep.txt` also headed its last column `TFLOPs` while holding
milliseconds, which is corrected. Hardware counters on both builds
([`logs/hw-counters-2026-09-25/`](../hw-counters-2026-09-25/)) show why the two differ: they issue the
identical 671,088,640 LDS instructions and BK 32 issues 22 percent *more* arithmetic, but it uses 136
VGPRs against 208, which keeps 274 and not 154 wave-cycles resident per busy cycle.
