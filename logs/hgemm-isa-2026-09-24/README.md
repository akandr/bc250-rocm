# Does Tensile emit `v_pk_fma_f16` for gfx1013? Yes. 24 September 2026


**Ceiling correction, 25 September.** The ceilings this page divides by, 4.74 TFLOP/s fp32 and 14.40
packed fp16, were measured below the clock cap. Re-measured with the clock verified at 1500 MHz they
are 6.52 and 13.02, and packed fp16 is twice fp32, not three times
([`logs/alu-rates-recheck-2026-09-25/`](../alu-rates-recheck-2026-09-25/)). Every percentage of peak on
this page is therefore against the wrong denominator; the measured rates themselves are unaffected.

## Why this was measured

An earlier revision of README.md explained rocBLAS HGEMM running at roughly SGEMM speed
(4698 against 4593 GFLOP/s at N=8192, against a measured packed-fp16 ceiling of 14400) by
saying that Tensile does not emit `v_pk_fma_f16` for this target. That was an inference from
two timings. Nobody had looked at the ISA. This directory is the look, and it refutes the
explanation.

## What was done

`rocblas_bench` was run under `strace -f -e trace=openat` against the native gfx1013 rocBLAS
in `build-restored-2026-08-20`, to record which Tensile code objects the library actually
opens instead of which ones exist. Those objects were then disassembled with
`llvm-objdump -d --mcpu=gfx1013` and the vector-instruction mix counted.

## Result

`selected_objects.txt`: the fp16 path opens exactly one code object,
`TensileLibrary_Type_HH_Contraction_l_Ailk_Bljk_Cijk_Dijk_fallback_gfx1013.hsaco`.

`instruction_mix.txt`: that object contains **720 `v_pk_fma_f16`** and no `v_fma_mix_f32`.
The packed instruction is emitted, and it is in the object the library loads. An excerpt of
the surrounding code is in `pk_fma_f16_excerpt.txt`.

The four `HH_HPA` objects, the fp32-accumulate variants, use `v_fma_mix_f32` instead and
carry no `v_pk_fma_f16` at all. Those are not the objects HGEMM opens, so they do not explain
the timing either.

| code object | `v_pk_fma_f16` | `v_fma_mix_f32` | `v_fmac_f32` |
|---|---|---|---|
| `HH_Contraction_l_Ailk_Bjlk` | 3856 | 0 | 0 |
| `HH_Contraction_l_Ailk_Bljk` (**loaded by HGEMM**) | 720 | 0 | 0 |
| `HH_Contraction_l_Alik_Bjlk` | 1488 | 0 | 0 |
| `HH_Contraction_l_Alik_Bljk` | 720 | 0 | 0 |
| `HH_HPA_Contraction_l_Alik_Bljk` | 0 | 3040 | 0 |
| `SS_Contraction_l_Alik_Bljk` | 0 | 0 | 1664 |

## What this does and does not settle

Settled: the published explanation was wrong. The compiler emits the packed instruction, the
library ships it, and the runtime loads it.

Not settled: why HGEMM is then only 2.3 percent faster than SGEMM. One observation that is
not an explanation, recorded because it is what the disassembly shows: neither precision has
a tuned tile set for gfx1013. Both fall back to small macro tiles, 64x128x8, 64x128x4 and
64x64x8 for fp16 against 128x64x8 for fp32, all at workgroup 16x16 (`kernel_names.txt`). An
attempt to isolate the K loop and count its body did not separate the main loop from the edge
cases, so no per-iteration arithmetic-to-address ratio is quoted here. The cause is open.

## Reproducing

    strace -f -e trace=openat -o open.txt ./rocblas_bench
    grep -oE 'TensileLibrary[A-Za-z0-9_]*\.hsaco' open.txt | sort | uniq -c
    /usr/lib64/rocm/llvm/bin/llvm-objdump -d --mcpu=gfx1013 <object>.hsaco \
      | grep -oE '\bv_[a-z0-9_]+' | sort | uniq -c | sort -rn

Tool versions in `tool_versions.txt`. The benchmark itself is
[`scripts/rocblas_bench.cpp`](../../scripts/rocblas_bench.cpp); its timings are in
[`logs/torch-rocblas-bench-2026-09-24/`](../torch-rocblas-bench-2026-09-24/).

**The open question closed, 25 September 2026.** This page settled that Tensile emits
`v_pk_fma_f16` and left open why HGEMM is then no faster than SGEMM. Hardware counters answer it
([`logs/hw-counters-2026-09-25/`](../hw-counters-2026-09-25/)). At N=8192 HGEMM retires 9,294,512,128
VALU instructions against SGEMM's 18,288,607,232, almost exactly half and within 8 percent of the
minimum its precision allows, so the packed instruction is not only shipped and loaded but actually
issued. Its shared-memory traffic does not halve with it: LDS instructions fall only from 3,422,748,672
to 1,946,484,736, so arithmetic per LDS access goes *down*, 5.34 to 4.78, and the wall-clock rate does
not move. Packed math removes arithmetic from a kernel whose limit is not arithmetic. The small
fallback tiles noted below are the mechanism, and the hand-written kernel is the control: it issues
more instructions than HGEMM and keeps fewer waves resident, and is 1.9 times faster, on 17.34 VALU
instructions per LDS instruction.
