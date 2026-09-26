# Hardware performance counters on gfx1013, 2026-09-25

Until now nothing in this repository has read a hardware performance counter. Kernel timings came
from `scripts/kerntrace.cpp` over roctracer, arithmetic rates from `clock64()` inside the kernel, and
the front page recorded that `rocprof` is not packaged for Fedora and was therefore not tried. This
directory is the attempt, what it took to make it work, and what it measured.

Two separate things had to be fixed, and each fails silently.

## 1. Fedora's ROCm has no profiler handshake at all

`libhsa-runtime64.so` on this board has no `rocprofiler_register_*` symbols, and neither does
`libamdhip64.so`. ROCR-Runtime's build calls `find_package(rocprofiler-register)` and turns the
handshake off when the package is absent; Fedora does not list it as a build dependency. With no
handshake a profiler attaches, runs, and reports `Number of services generating output: 0` without
an error. This is not specific to gfx1013 or to this board: on a stock Fedora ROCm install, no ROCm
profiler can see anything on any GPU.

The fix is to rebuild ROCR-Runtime with `rocprofiler-register-devel` installed. It goes in `/opt`
and is selected with `LD_LIBRARY_PATH`; the packaged runtime is left alone, and `rpm -V rocm-runtime`
stays clean afterwards.

## 2. gfx1013 is absent from the counter definitions

`rocprofiler-sdk`'s `counter_defs.yaml` lists `gfx10`, `gfx1010`, `gfx1030`, `gfx1031` and `gfx1032`
for 89 counters. `metrics.cpp` looks the agent name up in a map keyed by those literals:

    get_val(mets->arch_to_metric, std::string(agent->name))

so the bare `gfx10` entry is a key like any other, not a wildcard, and `gfx1013` matches
nothing and gets no counters. gfx1011 and gfx1012 are missing in the same way. This is the same
shape of defect as the Tensile fallback, the RDNA1 macro, the comgr VGPR table and the MMVQ table:
a per-architecture table with the sibling entries present and this one absent.

Adding `gfx1013` beside every `gfx1010` gives all 89 counters. `aqlprofile`, which actually programs
them, needs no change: it dispatches on a name prefix, `{"gfx10", GFX10_GPU_ID}` matched with
`rfind(name, 0) == 0`, so gfx1013 already selects its generic gfx10 command builder.

`scripts/build_rocprof_gfx1013.sh` does both, along with two incidental fixes: Fedora ships no
`hsakmt-config.cmake` (libhsakmt is folded into rocm-runtime), and gcc 16 no longer pulls
`<cstdint>` in transitively, which the vendored elfio and yaml-cpp copies rely on.

## Whether the counters can be believed

Adding gfx1013 to the gfx1010 lists asserts that the two share a hardware block layout. That is an
assumption, so `scripts/counter_validate.cpp` tests it: a kernel whose VALU count is fixed by its
ISA at 8192 `v_fma_f32` per wave, run at four sizes (`validate_64` through `validate_2560`).

| waves | SQ_WAVES | SQ_INSTS_VALU | per wave |
|---|---|---|---|
| 64 | 64 | 524,992 | 8203.0 |
| 256 | 256 | 2,099,968 | 8203.0 |
| 1024 | 1024 | 8,399,872 | 8203.0 |
| 2560 | 2560 | 20,999,680 | 8203.0 |

Exactly 8203 per wave at every size: the 8192 FMAs the compiler emitted, plus the eleven instructions
that set the accumulators up and reduce them. `SQ_INSTS_SALU` reads 259 per wave against the 128 loop
trips the ISA shows. For the SQ block the gfx1010 mapping is correct.

**It is not correct for every block.** `GL2C_HIT_sum` reads exactly 0 on a tiled GEMM, an impossible
0 percent L2 hit rate, while `GL2C_MISS_sum` returns a plausible number; `FETCH_SIZE`, which is built
out of the `GL2C_EA_RDREQ_*` counters, reports 5266 GiB fetched in 440 ms, which would be 11.9 TB/s
on a board that streams 432 GB/s (`pkgemm_l2/`, `pkgemm_fetch/`). Those are in this directory as the
evidence that they are wrong. Nothing in this repository uses an L2 or fetch-size counter on gfx1013,
and nobody else should without validating it first.

Two further limits. The hardware refuses more counters than it has slots, with
`Request exceeds the capabilities of the hardware to collect`, and rocprofv3 then aborts on the first
dispatch instead of multiplexing, so counters have to be grouped by hand. And counter collection
costs kernels very unequal amounts of wall time, so no rate in this directory comes from a profiled
run; the rates are from `plain_rocblas2.txt` and `plain_pk_bk32.txt`, measured with the profiler
detached on a freshly booted board with ollama stopped.

## What the counters then showed

### A published script did not reproduce its own published figure

`scripts/pk_gemm_square.cpp` reached 2.31 TFLOP/s at N=8192, against the 8.91 this repository
published for it. The cause was in the script: `BK` defaulted to 16 while the comment at the top of
the same file, and the tile sweep in
[`logs/pkf16-vs-rocblas-2026-09-25/tile_sweep.txt`](../pkf16-vs-rocblas-2026-09-25/tile_sweep.txt),
both give 32. Rebuilt with `-DBK=32` it returns 123.557 ms and 8.90 TFLOP/s, matching the published
figure. The default is now 32. That sweep's last column was also headed `TFLOPs` while holding
milliseconds, which is fixed.

### Packed fp16 halves the arithmetic without halving the data movement

Square 8192, from `i_rocblas/`, `w_rocblas/`, `pk_bk32/`, `pk_bk16/`, `w_bk32/` and `w_bk16/`.
A `v_pk_fma_f16` retires 4 flops across 32 lanes, so 2N³ flops need at least 8.590e9 of them; a
`v_fma_f32` retires 2, so fp32 needs at least 1.718e10.

| kernel | VGPR | VALU instructions | of the minimum | LDS instructions | VALU per LDS | waves resident | TFLOP/s |
|---|---|---|---|---|---|---|---|
| rocBLAS SGEMM | 72 | 18,288,607,232 | 1.06 | 3,422,748,672 | 5.34 | 555.3 | 4.59 |
| rocBLAS HGEMM | 72 | 9,294,512,128 | 1.08 | 1,946,484,736 | 4.78 | 523.5 | 4.65 |
| this kernel, BK 32 | 136 | 11,634,278,400 | 1.35 | 671,088,640 | 17.34 | 274.2 | 8.90 |
| this kernel, BK 16 | 208 | 9,506,357,248 | 1.11 | 671,088,640 | 14.17 | 154.2 | 2.31 |

"Waves resident" is `SQ_WAVE_CYCLES / SQ_BUSY_CYCLES`, a ratio of two counters in the same clock
domain, so it survives both the clock and the profiler's serialization.

[`logs/hgemm-isa-2026-09-24/`](../hgemm-isa-2026-09-24/) established from the disassembly that
Tensile does emit `v_pk_fma_f16` for this target, and left open why HGEMM is then no faster than
SGEMM. The counters show the instructions are not merely present in the binary but actually issued:
HGEMM retires 9.29e9 VALU instructions where SGEMM retires 18.29e9, almost exactly half, both within
8 percent of their precision's minimum. What does not halve with them is the traffic through shared
memory, which falls only from 3.42e9 to 1.95e9 instructions, so arithmetic per LDS access drops from
5.34 to 4.78 and the wall-clock rate does not move. Packed math removes arithmetic from a kernel
whose limit is not arithmetic.

That is consistent with the hand-written kernel, which issues *more* instructions than rocBLAS HGEMM
(11.63e9 against 9.29e9, 35 percent above its own minimum) and keeps *half* as many waves resident,
and is still 1.9 times faster. It does 17.34 VALU instructions per LDS instruction where rocBLAS does
4.78, from staging deeper tiles into shared memory and holding more accumulators per thread.

The two tile depths of the same kernel isolate the other half of the story. BK 32 and BK 16 issue the
identical 671,088,640 LDS instructions, and BK 32 issues 22 percent *more* arithmetic, yet BK 32 is
3.8 times faster. The difference is register pressure: 136 VGPRs against 208, which keeps 274
instead of 154 wave-cycles resident per busy cycle, and cuts waiting from 12.47 to 2.73 million
cycles per
wave. Neither build spills to scratch.

Bank conflicts are not what separates them either, and the direction is the wrong way round for
them to be. `SQC_LDS_BANK_CONFLICT / SQC_LDS_IDX_ACTIVE` is 805306368 over 2684354560 for BK 32, and
268435456 over 2147483648 for BK 16: exactly 30.00 and 12.50 percent, fixed fractions, not noise.
The faster build conflicts nearly two and a half times as often. Waiting on LDS is 0.0155
cycles per LDS instruction on BK 16, which is where `SQ_WAIT_INST_LDS` was collected
(`sqc_bk32/`, `pkgemm_sqc/`).

### llama.cpp, and where the prefill kernel this repository added now sits

qwen2.5-1.5B q4_K_M, flash attention on, 16 decoded tokens and a 512-token prefill, on the
thirteen-patch build (`llama_decode/`, `llama_prefill/`). Counter collection serializes dispatches,
so the timings of these runs mean nothing; the instruction counts do.

| | dispatches | VALU instructions | dominant kernel | its share | its VALU per LDS |
|---|---|---|---|---|---|
| decode | 6559 | 5,201,223,434 | `mul_mat_vec_f32_rdna1` | 99.2 % | 64.87 |
| prefill | 1426 | 35,574,732,520 | `mmf16_rdna1_gemm` | 85.6 % | 13.89 |

Decode is one kernel doing almost all of the arithmetic at 64.87 VALU instructions per LDS
instruction, which is to say it barely touches shared memory at all. Nothing about decode is going to
be improved by the tiling arguments above; it is a bandwidth problem, which is what this repository
has said.

Prefill is the interesting one. `mmf16_rdna1_gemm` is the packed-fp16 GEMM this repository added
(`scripts/apply_rdna1_pkf16.py`), and it reaches 13.89 VALU instructions per LDS instruction. That is
most of the way from rocBLAS HGEMM's 4.78 to the 17.34 of the standalone prototype at BK 32, and the
remaining distance is a measurement of how much is still on the table in the shipped kernel rather
than an argument that it is there. `mul_mat_q`, the int8 path, reads 31.26, so its arithmetic per LDS
access is not what limits it.

### An int8 benchmark in this repository was measuring the scalar unit

The int8 row of `scripts/alu_cycles.cpp` seeded its accumulators from the kernel argument, which is
uniform across the wave. Floating point survives that, because RDNA1 has no scalar float ALU, but the
integer chain does not: the compiler moved the whole dot product to the scalar unit, emitting
`s_mul_i32`, `s_bfe_i32` and `s_sext_i32_i8`. The counters are unambiguous (`alu_orig_c/`,
`alu_fixed_c/`), at 640 waves:

| kernel | VALU per wave | SALU per wave |
|---|---|---|
| int8 dot product, seeded from the argument | **5** | **397,342** |
| int8 dot product, seeded from the lane index | 196,630 | 4,111 |
| `v_fma_f32`, control, before | 32,791 | 523 |
| `v_fma_f32`, control, after | 32,808 | 523 |

Seeding every chain from the lane index puts the emulation back in vector registers, where it
compiles to `v_mul_i32_i24_sdwa` and `v_add3_u32`. It then costs 196,630 VALU instructions for the
131,072 int8 multiply-accumulates a wave performs, **1.50 vector instructions per mac**.

The conclusion that row supported does not change. Every VALU instruction on this chip costs the same
(below), so 1.50 instructions per mac puts emulated int8 at about two thirds of the fp32 FMA rate in
macs per second, far below packed fp16, which is what the repository has said. But the number was
reached by timing the wrong execution unit, and `scripts/alu_cycles.cpp` is corrected.

### The ALU ceilings hold, measured a second way

`SQ_BUSY_CYCLES / SQ_INSTS_VALU` at 640 waves reads 0.0274 for `v_fma_f32`, 0.0275 for `v_fma_f16`
and 0.0275 for `v_pk_fma_f16` (`alu/`). The corrected sweep run separately with `clock64()` gives
5.292, 5.352 and 5.353 cycles per instruction (`alu_fixed.txt`). Two instruments that share nothing
agree that the three cost the same, which is what makes packed fp16 exactly twice fp32 and not three
times. The TFLOP/s column of that same run reads 14.65 for packed fp16, above the 13.02 ceiling, only
because its kernel happened to run at 0.96 GHz where the fp32 kernel ran at 0.85; the cycle counts
are the clock-independent statement and 13.02 stands.

## Files

| | |
|---|---|
| `validate_64/`, `validate_256/`, `validate_1024/`, `validate_2560/` | the validation sweep, the gate everything else rests on |
| `alu/`, `alu_orig_c/`, `alu_fixed_c/` | the ALU sweep, before and after the scalarisation fix |
| `alu_orig.txt`, `alu_fixed.txt` | the same sweep's own `clock64()` output |
| `i_rocblas/`, `w_rocblas/` | rocBLAS instruction counts and residency |
| `pk_bk32/`, `pk_bk16/`, `w_bk32/`, `w_bk16/` | the hand-written kernel at both tile depths |
| `pkgemm_sqc/`, `sqc_bk32/`, `pkgemm_lds/` | LDS bank conflicts and LDS waiting |
| `pkgemm_l2/`, `pkgemm_fetch/` | kept as the evidence that the L2 counters are wrong here |
| `llama_decode/`, `llama_prefill/` | llama.cpp under counters, qwen2.5-1.5B q4_K_M |
| `plain_rocblas2.txt`, `plain_pk_bk32.txt` | the rates, profiler detached, fresh boot |
| `pkgemm/`, `rocblas/`, `rocblas_sqc/`, `validate/` | the first pass, before the BK default was found |

## Reproducing

    scripts/build_rocprof_gfx1013.sh
    export PATH=/opt/rocprof-gfx1013/bin:$PATH
    export LD_LIBRARY_PATH=/opt/rocr-profreg/lib64:$LD_LIBRARY_PATH
    hipcc -O3 --offload-arch=gfx1013 scripts/counter_validate.cpp -o counter_validate -lamdhip64
    rocprofv3 --pmc SQ_WAVES SQ_INSTS_VALU -- ./counter_validate 2560

`SQ_WAVES` must read 2560 and `SQ_INSTS_VALU` 8203 per wave before anything else here is worth
reading. `scripts/counter_summary.py` prints the tables above from the CSVs, and
`scripts/make_counter_figures.py` draws `figures/counter-validation.png`, `figures/counter-gemm.png`
and `figures/counter-scalarised.png` from them.
