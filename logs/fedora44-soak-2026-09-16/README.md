# Three-hour soak on the default Fedora 44 configuration, 2026-09-16

[`scripts/soak_f44.sh`](../../scripts/soak_f44.sh) on the default boot: Fedora 44, kernel 7.1.8 with the
bc250 amdgpu module, SELinux enforcing, native gfx1013 rocBLAS 7.1.1 and the corrected comgr from
`/opt/bc250-rocm`, no environment variables but `HSA_ENABLE_SDMA=0`. Each round: qwen2.5-1.5B pp2048,
the 1.5B gate (reference 8.9442), the qwen3-8B gate at the default fp16 compute type (9.1117, which
goes through rocBLAS fp16 GEMMs), the MUL_MAT allocation-churn sweep, and every third round the
PyTorch training loop on the gfx1013 build.

## A caveat on the throughput figures

This soak ran before the GPU clock policy was corrected on 16 September: the governor was oscillating
between 1000 and 2000 MHz instead of holding 1500
([`../fedora44-benchmarks-2026-09-15/clock-corrected/`](../fedora44-benchmarks-2026-09-15/clock-corrected/)).
The prefill figures below are therefore not comparable with the benchmark pages. The correctness results,
the churn sweep and the fault count are unaffected, and the clock was oscillating in every round, so the
0.7 percent spread across 16 rounds also says the oscillation was stable in aggregate.

## Result

16 rounds, 23:16 to 02:22 (`log`). Every round: 1.5B gate 8.9442, churn exit 0, no kernel fault lines,
GPU edge temperature 69 to 79 C. Prefill held between 764.36 and 769.70 t/s, a spread of 0.7 percent.
The five PyTorch rounds each returned final loss 0.00048 and a maximum loss difference of 1.799e-05,
the same values as on Fedora 43.

The 8B gate returned its value in 13 of the 16 rounds and nothing in rounds 9, 10 and 11.

## The three failed rounds were another process, not the board

The journal shows an `ollama` service loading Qwen3 14B Q4_K_M through Vulkan at 00:51:11
(`ollama_interference.txt`), inside the window of rounds 9 to 11 (00:50 to 01:23). It held GPU memory
for about half an hour; the 8B gate loads 8.24 GiB with `--no-mmap` and could not fit beside it. The
1.5B gate in the same rounds passed, which fits a memory limit, not a fault. There were no
kernel fault lines in the whole boot, no core dumps in that window, and the same gate run three times
immediately afterwards returned 9.1117 each time (`run1.log` to `run3.log`).

Two harness faults this exposed, both fixed in the script: the gate output was not saved, so the
failures left no evidence of their own, and nothing recorded what else was using the GPU. The script now
writes `gate15_N.log` and `gate8_N.log` per round and logs other GPU users at the start.

Only two ollama model loads happened in this boot, 18:53 and 00:51. The benchmark campaigns ran 16:16 to
17:03 and 22:02 to 22:45, so no benchmark figure on the Fedora 44 pages is affected.
