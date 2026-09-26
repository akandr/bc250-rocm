# Fedora 44 validation: speed probes, Vulkan, allocation churn, PyTorch, 2026-09-15

Same board, kernel 7.1.8-100.fc43 with the bc250 amdgpu module on both systems. Fedora 44 runs used the
native gfx1013 rocBLAS 7.1.1 and the corrected comgr through `LD_LIBRARY_PATH`; Fedora 43 runs the
production build with the repaired rocBLAS 6.4.2.

## Speed probes (`probes/`)

[`scripts/os_speed_probe.sh`](../../scripts/os_speed_probe.sh), qwen2.5-1.5B, the same probes on each OS,
`llama-bench -mmp 0 -ngl 99 -r 3`:

| probe | Fedora 43 | Fedora 44 |
|---|---|---|
| pp512 / tg64, default | 808.08 / 113.75 | 993.57 / 147.17 |
| `GGML_CUDA_DISABLE_GRAPHS=1` | 811.79 / 119.70 | 997.62 / 150.80 |
| `-fa off` | 895.47 / 109.59 | 1174.06 / 144.21 |
| pp2048, ubatch and batch 2048 | 746.13 | 867.93 |
| whole `-p 0 -n 256` invocation, `/usr/bin/time` | 7.65 s wall, 5.06 user, 4.50 sys (117.91 t/s) | 2.63 s wall, 2.27 user, 0.36 sys (150.62 t/s) |

`gpu_busy_percent` is not supported on this device, so no GPU utilisation was recorded.
`f44_strace_tg64.txt` is `strace -f -c` over a Fedora 44 `-p 0 -n 64` invocation: 1905 ioctl calls, 5009
reads. No Fedora 43 counterpart was captured.

## Vulkan (`vulkan/`)

[`scripts/vulkan_os_bench.sh`](../../scripts/vulkan_os_bench.sh), llama.cpp Vulkan backend, `-fa on -p 512
-n 64 -r 5`. Fedora 43 (`build-vk`, Mesa 25.3.4): qwen2.5-1.5B 1843.61 / 210.77, qwen3-8B 400.50 / 39.06.
Fedora 44 (`build-vk-f44`, Mesa 26.1.8): qwen2.5-1.5B 2416.71 / 240.76 and 2417.27 / 241.60; qwen3-8B
486.75 +/- 64.87 / 38.28 +/- 2.32 and 363.74 +/- 112.47 / 37.99 +/- 2.31. A ten-repetition JSON run of the
8B on Fedora 44 recorded prefill samples 372.8, 267.0, 267.0, 384.9, 267.1, 267.2, 317.5, 267.1, 267.0,
317.3 and decode 35.86, 35.86, 37.36, 40.45, 40.46, 40.46, 40.46, 40.46, 40.46, 40.46 (console output,
not kept as a file).

## Allocation churn A/B/A (`churn/`)

[`scripts/churn_aba_f44.sh`](../../scripts/churn_aba_f44.sh), Fedora 44: each run is `test-backend-ops perf
-o MUL_MAT` plus the sequence reproducer [`patches/seq_probe.c`](../../patches/seq_probe.c) rebuilt for HIP
7. Kernel fault lines are counted from the journal per phase.

| phase | `bc250_flush_by_runlist` | MUL_MAT sweep | sequence reproducer | kernel fault lines |
|---|---|---|---|---|
| A | 3 | 3 of 3 exit 0, no runtime fault | 3 of 3 `bad_gens=0` | +0 |
| B | 1 | 2 of 2 exit 134, `Memory access fault by GPU` | 2 of 2 `bad_gens=0` | +10 |
| A2 | 3 | 3 of 3 exit 0 | 3 of 3 `bad_gens=0` | +0 |

The perplexity gate after A2 read 8.9442. The sequence reproducer did not detect corruption in phase B at
these sizes; the sweep is the discriminating workload here.

## SDMA (`sdma/gates.txt`)

With the navi12 microcode in place on Fedora 44, the 1.5B and 8B gates read 8.9442 and 9.1117 with
`HSA_ENABLE_SDMA=1` and with `=0`, and the kernel journal of that boot, which also covers the whole
benchmark campaign, holds no page fault, preemption or runlist-flush failure lines.

## PyTorch built from source (`torch/*source_build*`, `torch/build_times.txt`)

PyTorch 2.9.1 built from the same patched source tree as on Fedora 43, against Fedora 44's ROCm 7.1.1 with
`PYTORCH_ROCM_ARCH=gfx1013` (the options of
[`scripts/build_pytorch_gfx1013.sh`](../../scripts/build_pytorch_gfx1013.sh), five jobs), 17:28 to 19:12,
exit 0. Run with the default configuration's rocBLAS from `/opt/bc250-rocm`:

- [`patches/torch_opprobe.py`](../../patches/torch_opprobe.py): 11 of 11 cases, architecture list `['gfx1013']`.
- [`patches/pytorch/torch_train.py`](../../patches/pytorch/torch_train.py): final loss 0.00048 on CPU and GPU,
  maximum loss difference across the 50 steps 1.799e-05, parameter difference after 50 steps 9.312e-03. All
  three equal the Fedora 43 run in [`../torch-train-2026-08-19/`](../torch-train-2026-08-19/) to the printed
  digits. Timings from this one run: CPU 8.27 s, GPU 0.34 s, against 20.38 and 0.26 on Fedora 43.

## PyTorch, Fedora package (`torch/`)

Fedora 44's `python3-torch-2.9.1-10.fc44`, installed in the Fedora 44 copy. Its architecture list omits
gfx1013. `fedora_torch_narrow_test.txt` records one process doing a host-to-device copy, a copy back, a
matmul and an add: with the native rocBLAS the matmul matches the CPU (7.6e-06) and the add crashes with a
general protection fault in `libamdhip64`; with the system rocBLAS the matmul fails with
`HIPBLAS_STATUS_INTERNAL_ERROR`. The op probe (`opprobe_fedora_torch.log`) crashed before printing results.

## Mesa 26.1.8 against Mesa 25.3.4, both on Fedora 44 (`mesa-ab/`)

[`scripts/mesa_ab_f44.sh`](../../scripts/mesa_ab_f44.sh). Fedora 43's `mesa-vulkan-drivers-25.3.4-7.fc43`
rpm was extracted to `~/mesa2534-f44` without installing anything, the two libraries Fedora 44 no longer
ships (`libLLVM.so.21.1`, `libdisplay-info.so.2`) were copied from the Fedora 43 subvolume, and the driver
was selected with `VK_ICD_FILENAMES`. `vulkaninfo` then reports Mesa 25.3.4. Same llama.cpp binary, same
boot, drivers alternated, one test per invocation, three rounds of `-r 3`; medians of nine samples with
the range:

| model | 26.1.8 pp512 | 25.3.4 pp512 | 26.1.8 tg64 | 25.3.4 tg64 |
|---|---|---|---|---|
| qwen2.5-1.5B | 2415.7 (2415 to 2418) | 2408.6 (2408 to 2409) | 240.9 | 239.7 |
| qwen3-8B | 267.1 (267 to 362) | 271.2 (271 to 403) | 40.4 | 40.4 |
| qwen3-14B | 143.4 (138 to 169) | 153.8 (137 to 167) | 37.6 | 37.1 |
| qwen3.8-27B | 73.9 (73 to 82) | 75.0 (74 to 77) | 21.8 | 21.7 |

The qwen3-14B gate is 6.4547 on 26.1.8 and 6.4548 on 25.3.4.

**This overturns the Mesa reading on the benchmark page.** The two drivers perform the same here, within a
few percent and inside each other's spread, and the bimodal prefill on the 8B and 14B happens under both.
So the Vulkan differences between the Fedora 43 and Fedora 44 pages, the 1.5B rising from 1844 to 2415
t/s and qwen3-14B prefill falling from about 199 to about 148, are not caused by the Mesa version.

The llama.cpp binary is not the cause either: the Fedora 43 Vulkan build and the Fedora 44 one, run on
Fedora 44 against both drivers, give the same figures in all four combinations
(`cross_*.jsonl`): 1.5B 2407 to 2418, qwen3-14B 137 to 155. Mesa's driver-configuration files are
identical between the systems apart from game-specific entries, and the kernel and amdgpu parameters are
the same by construction. What in the Fedora 44 userspace changes Vulkan this way is not identified.
