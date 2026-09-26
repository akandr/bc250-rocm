# What the arithmetic units actually deliver on gfx1013, and what that says about prefill, 2026-09-19

Prefill on this board runs at half Vulkan's rate, and both ROCm paths, MMQ's emulated int8 tiles and
rocBLAS's HGEMM ([`logs/round5-2026-09-19/`](../round5-2026-09-19/)), land within a few percent of each
other. Two unrelated implementations at the same speed is not a kernel-quality story, so this measures
the machine underneath them: [`alu_cycles.cpp`](alu_cycles.cpp), independent accumulator chains in
registers, one kernel per instruction kind, occupancy swept, timed with HIP events
(`alu_cycles.log`). The clock policy was the usual 1500 MHz and `pp_dpm_sclk` held its step throughout.

**Corrected 25 September. The rates in this table were taken below the clock cap and two of them are
wrong.** Re-run with `pp_dpm_sclk` sampled throughout and the clock verified at 1500 MHz for 142 of
147 samples, `v_fma_f32` reads 6.52 TFLOP/s and `v_pk_fma_f16` reads 13.02, so packed fp16 is 2.00
times fp32, not 3.04, and scalar fp16 equals fp32 instead of exceeding it. The instruction
costs were right all along and are identical across every run: the three share 5.32 cycles, and a
packed FMA does two of what a scalar one does, which is where the factor of two comes from. The int8
figure is unchanged. See [`logs/alu-rates-recheck-2026-09-25/`](../alu-rates-recheck-2026-09-25/).
The table below is left as it stood.


| instruction | 0.5 waves/SIMD | 2 waves/SIMD | 8 waves/SIMD |
|---|---|---|---|
| `v_fma_f32` | 1.36 | 3.03 | **4.74 TFLOP/s** |
| `v_fma_f16` (scalar) | 1.55 | 3.02 | **6.36 TFLOP/s** |
| `v_pk_fma_f16` (packed) | 3.10 | 6.04 | **14.40 TFLOP/s** |
| int8 dot4, emulated as ggml does it | 0.42 | 1.01 | **2.08 Tmac/s** |

Three things follow, and the first is the one that matters.

**Packed fp16 is three times the fp32 rate and seven times the emulated int8 rate.** gfx1013 has
`v_pk_fma_f16` at full rate; it has no `v_dot4_i32_i8` at all (the compiler rejects
`__builtin_amdgcn_sdot4` for this target: *needs target feature dot1-insts*), so every int8 dot product
in MMQ and MMVQ costs four multiplies and four adds.

**The fp32 figure is the board's real ceiling, and rocBLAS is already at it.** 4.74 TFLOP/s against the
7.68 the clock and lane count suggest, and the repository's own SGEMM measurement, 4.56 to 4.66 TFLOP/s
([the GPGPU section](../../README.md#gpgpu)), is 97 percent of it. So rocBLAS SGEMM is not leaving
anything on the table; the fp32 pipeline does not sustain one FMA per lane per cycle here.

**Where each prefill path sits.** Per-op, the 1.5B's prefill GEMMs
([`logs/op-perf-hip-vs-vulkan-2026-09-17/`](../op-perf-hip-vs-vulkan-2026-09-17/)):

| path | achieved | ceiling for its arithmetic | at the ceiling? |
|---|---|---|---|
| ROCm MMQ (emulated int8) | 2.5 to 2.9 TFLOP/s | 2.08 Tmac/s | yes, it is the arithmetic |
| ROCm dequantise + rocBLAS HGEMM | about 3.0 | 14.40 (packed fp16) | no, 21 percent |
| Vulkan (RADV) | 4.9 to 5.4 | 14.40 | no, 37 percent |

MMQ is not badly written: it is running the slowest arithmetic the chip has, and it is close to that
limit. Vulkan is ahead because it is doing the multiplication in fp16, and it is itself only at a third
of what packed fp16 can do. The prefill gap is therefore not RDNA1 tile parameters (MMQ's kernels do not
even spill here, checked with `-Rpass-analysis=kernel-resource-usage`) and not rocBLAS: it is the
arithmetic format. The same reasoning fixed decode, where a float matvec replaced the emulated int8 one
and took the 1.5B from 146 to 195 tokens per second
([`logs/rdna1-mmvq-2026-09-18/`](../rdna1-mmvq-2026-09-18/)).

`alu_peak.cpp` is the first version of this measurement, kept because two of its traps are worth
remembering: a store guarded by a compile-time-false condition lets the compiler delete the whole
arithmetic chain, and the first kernel of a process runs before the clock has ramped, which made fp32
look like half rate until a warm-up was added.

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
