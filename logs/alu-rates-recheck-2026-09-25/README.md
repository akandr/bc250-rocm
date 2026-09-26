# The ALU ceilings were measured below the clock cap, and packed fp16 is 2x fp32, not 3x, 2026-09-25

## Why this was re-run

[`logs/alu-rates-2026-09-19/`](../alu-rates-2026-09-19/) established the three numbers this repository
quotes everywhere: 4.74 TFLOP/s for `v_fma_f32`, 14.40 for `v_pk_fma_f16`, 2.08 Tmac/s for the
emulated int8 dot product. They set the roofline, every percentage of peak, and the argument that
fp16 is worth three times fp32 here.

Two things prompted a re-run. A double-precision arm was being added, and nothing in this repository
had ever measured fp64. And the table in that page does not match the log beside it: the log reads
4.37 and 9.92 at its widest occupancy where the table says 4.74 and 14.40, while other cells match
exactly. No log for the table's numbers exists anywhere under `logs/`.

## Result

Same probe, with the clock sampled from `pp_dpm_sclk` four times a second throughout and a sustained
GPU load before the timed kernels. **The clock held 1500 MHz for 142 of 147 samples**
(`clock_trace_head.txt`, `alu_clock_verified.log`).

| instruction | cycles per instruction | rate | against fp32 |
|---|---|---|---|
| `v_fma_f32` | 5.323 | **6.52 TFLOP/s** | 1.00 |
| `v_fma_f16` scalar | 5.328 | **6.51 TFLOP/s** | 1.00 |
| `v_pk_fma_f16` | 5.323 | **13.02 TFLOP/s** | **2.00** |
| int8 dot4, emulated | 8.318 | 2.08 Tmac/s | |
| `v_fma_f64` | 80.778 | **0.43 TFLOP/s** | 1/15.2 |
| `v_mul_f64` | 78.494 | 0.21 TFLOP/s | |
| `v_add_f64` | 78.465 | 0.21 TFLOP/s | |

Cycles per instruction are the reproducible quantity: across four runs today and the September one
they agree to three decimals (fp32 5.318, 5.323, 5.324, 5.329). The rates are computed from wall
time. The probe also prints a "shader clock during run" of about 0.85 GHz, which is miscomputed and
should be ignored; the sampled clock is the one to trust.

## What changes

**Packed fp16 is exactly twice fp32, not three times.** `v_pk_fma_f16` and `v_fma_f32` cost the same
5.323 cycles and the packed one does two FMAs against one, so the ratio is 2.00 by construction, and
the measured rates give 1.997. The earlier 3.04 came from dividing 14.40 by 4.74, two figures taken in
different runs at different clocks. Scalar fp16 equals fp32 instead of being 1.34 times it, for the
same reason.

**The fp32 ceiling is 6.52 TFLOP/s, not 4.74.** The old figure is what this probe reports when the
governor is still at 1000 MHz: rate tracks clock, and September's log shows 4.37 TFLOP/s with its
clock estimate at 0.57 GHz against today's 6.52 at 0.85, a ratio of 1.49 in both. The page for that
measurement already warns that "the first kernel of a process runs before the clock has ramped, which
made fp32 look like half rate until a warm-up was added". The warm-up that was added was not enough.

**So rocBLAS SGEMM is not at the machine's limit.** 4.61 TFLOP/s against 6.52 is 71 percent, where the
old pair made it 97 percent of 4.74. The claim that "the fp32 figure is the board's real ceiling, and
rocBLAS is already at it" does not hold up.

**fp64 has a measured rate for the first time**, 0.43 TFLOP/s, and it costs 15.2 times an fp32 FMA per
instruction, consistent with the one-sixteenth rate RDNA1 is documented to have. rocBLAS DGEMM reaches
0.457 TFLOP/s, slightly above the probe's fp64 figure; the probe reaches 85 percent of the fp32
clock-and-lane figure, and applying the same efficiency to fp64 puts its peak near 0.51 TFLOP/s, which
would put DGEMM at about 90 percent. That last step is an inference, not a measurement.

## What this does not change

The gap between rocBLAS HGEMM and what the hardware can do. 4.67 TFLOP/s against a 13.02 ceiling is 36
percent, where the old numbers gave 32 percent of 14.40. The hand-written packed-fp16 kernel
([`logs/pkf16-vs-rocblas-2026-09-25/`](../pkf16-vs-rocblas-2026-09-25/)) reaches 8.91, which is 68
percent of the corrected ceiling, not 62 percent of the old one, and is still 1.9 times
rocBLAS. That comparison was measured directly between the two kernels and does not depend on the
ceiling at all.

The int8 figure is unchanged at 2.08 Tmac/s, and its cycle count is identical across all runs.

## Method note

Neither the September run nor the first two runs today were wrong about cycles; they were wrong about
what clock the board was at when the rate was computed. This is the third time a clock policy has put
a wrong number on this page, after the Fedora 44 comparison and the qwen3-14B prefill spread. The
check that catches it is cheap: sample `pp_dpm_sclk` during the measurement and report the residency
beside the result, which this log does.

## Reproducing

    hipcc -O3 --offload-arch=gfx1013 -o alu_cycles_f64 alu_cycles_f64.cpp -L/usr/lib64 -lamdhip64

Run a sustained GPU load first, sample the clock during the probe, and discard any run whose residency
is not at the policy cap. The probe's own clock estimate is not a substitute.

**The int8 row was measuring the scalar unit, 25 September 2026.** `scripts/alu_cycles.cpp` seeded its
accumulators from the kernel argument, which is uniform across the wave. Floating point survives that,
since RDNA1 has no scalar float ALU, but the integer chain does not: the compiler moved the whole int8
dot product onto the scalar unit, emitting `s_mul_i32`, `s_bfe_i32` and `s_sext_i32_i8`. Hardware
counters read 5 VALU and 397,342 SALU instructions per wave for that kernel, against 32,791 VALU for
the fp32 control ([`logs/hw-counters-2026-09-25/`](../hw-counters-2026-09-25/)). Seeding from the lane
index puts it back in vector registers, where it compiles to `v_mul_i32_i24_sdwa` and `v_add3_u32` and
costs 1.50 vector instructions per multiply-accumulate. Every VALU instruction on this chip costs the
same, so that puts emulated int8 near 2.2 Tmac/s, close to the 2.08 quoted here; the conclusion the row
supported is unchanged, but the agreement is a coincidence of two different quantities and the figure
should be taken from the instruction count, not from that kernel's timing. The fp32, fp16 and
packed-fp16 rows are unaffected and were re-measured to confirm it.
