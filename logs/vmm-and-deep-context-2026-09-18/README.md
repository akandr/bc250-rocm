# Virtual memory management does not work on gfx1013, and the deep-context ceiling has not moved, 2026-09-18

The recipe builds llama.cpp with `GGML_HIP_NO_VMM=ON`. Whether that was still necessary on ROCm 7.1.1
and kernel 7.2.5 had not been checked, and a working VMM pool would change how the backend allocates and
possibly where deep contexts stop. So: a second build of the same tree with `GGML_HIP_NO_VMM=OFF`
(`build-hip-vmm`, `vmm-configure.log`), measured against the four-patch production build on the default
boot, Fedora 44, kernel 7.2.5, clock 1500 MHz, `ollama` stopped. The whole sequence is `log`.

## VMM: the runtime says yes, the first allocation says no

`llama-bench` on the VMM build reports `VMM: yes` in its device line and then aborts on the first pool
allocation (`vmm-bench.err`):

    ggml/src/ggml-cuda/ggml-cuda.cu:669: HipVMM Failure: invalid argument
    ggml_cuda_pool_vmm::alloc(unsigned long, unsigned long*)

Every measurement with that build is therefore empty in `log`: pp2048, tg128, the gate and the deep-context
probes. `GGML_HIP_NO_VMM=ON` is required on this board, not a leftover, and the recipe stays as it is.
Why the HIP runtime advertises virtual memory management it cannot then service on this device is not
pursued here; the failing call is the reserve-and-map path that `hipMemCreate` and `hipMemMap` implement.

The production build alongside, as the control that the board and the session were sound: 1.5B pp2048
869.25 and 869.40, tg128 113.54 and 113.47, the same figures as every run of that build today.

## Deep context, four-patch build

qwen3-8B Q8_0, `tg32` after a pre-filled context, one invocation per depth:

| depth | result |
|---|---|
| 16384 | 11.32 tokens/s |
| 24576 | no result; the kernel logs `amdgpu: SVM mapping failed, exceeds resident system memory limit` three times |

That is the ceiling already established in
[`logs/fedora44-ceilings-2026-09-16/`](../fedora44-ceilings-2026-09-16/): the KFD system-memory limit,
13422 MiB on this board, which the patch does not touch and VMM could not have changed either.

## Also in `log`

The MoE decode-text check on the four-patch build (`moe-decode-fatile.txt`): coherent reasoning text in
answer to `The capital of France is`, discussed in
[`logs/rdna1-fattn-spill-2026-09-17/`](../rdna1-fattn-spill-2026-09-17/). The first attempt at the
fp16 GEMM harness failed to link (`-lamdhip64` was missing from the command); its results are in the
op-perf directory once run.
