# Index of logs

Every measurement in [README.md](../README.md) and [INVESTIGATION.md](../INVESTIGATION.md)
has a directory here holding the raw output and a README naming the harness that produced it.
There are 214 of them.

| directory | what it holds |
|---|---|
| [`alu-rates-2026-09-19`](alu-rates-2026-09-19/) | What the arithmetic units actually deliver on gfx1013, and what that says about prefill |
| [`alu-rates-recheck-2026-09-25`](alu-rates-recheck-2026-09-25/) | The ALU ceilings were measured below the clock cap, and packed fp16 is 2x fp32, not 3x |
| [`aslr-2026-08-23`](aslr-2026-08-23/) | Address space randomisation does not explain decode variance |
| [`bench-2026-08`](bench-2026-08/) | Working-configuration benchmark campaign, 2026-08-08 and -09 |
| [`bench-fixed-2026-08`](bench-fixed-2026-08/) | Benchmark logs, fixed llama.cpp stack, 2026-08-12 and |
| [`boot-entry-2026-09-21`](boot-entry-2026-09-21/) | A plain reboot came up on the wrong boot entry |
| [`campaign-current-2026-08-22`](campaign-current-2026-08-22/) | The headline campaign, re-run on the current configuration |
| [`campaign-graphopt-2026-09-24`](campaign-graphopt-2026-09-24/) | The campaign with ggml-cuda's multi-stream optimisation turned on |
| [`campaign-rerun-2026-08-18`](campaign-rerun-2026-08-18/) | (see its README) |
| [`clean-build-2026-08-18`](clean-build-2026-08-18/) | (see its README) |
| [`context-2026-08-14`](context-2026-08-14/) | Context ladder |
| [`context-ceilings-2026-08-17`](context-ceilings-2026-08-17/) | Context ceilings per model |
| [`corpus-instrument-2026-08-18`](corpus-instrument-2026-08-18/) | (see its README) |
| [`correctness-thirteen-2026-09-21`](correctness-thirteen-2026-09-21/) | The correctness table, re-measured on the thirteen-patch build |
| [`counterbalanced-2026-08-18`](counterbalanced-2026-08-18/) | (see its README) |
| [`decode-depth-2026-08-23`](decode-depth-2026-08-23/) | Does decode variance grow with depth? Not demonstrably |
| [`decode-queue-2026-08-22`](decode-queue-2026-08-22/) | Does decode rate track the queue the process is given? No |
| [`decode-residue-2026-09-25`](decode-residue-2026-09-25/) | What the part of a decode token outside the kernels is |
| [`decode-variance-clock-2026-09-22`](decode-variance-clock-2026-09-22/) | The decode-at-depth variance, a fifth elimination |
| [`decode-variance-state-2026-08-21`](decode-variance-state-2026-08-21/) | What makes decode at depth vary between processes |
| [`deep-decode-faults-2026-08-20`](deep-decode-faults-2026-08-20/) | (see its README) |
| [`deep-dive-2026-07-28`](deep-dive-2026-07-28/) | Deep-dive round |
| [`defects-recheck-2026-08-22`](defects-recheck-2026-08-22/) | Do the documented defects still behave as documented |
| [`depth-graphopt-2026-09-24`](depth-graphopt-2026-09-24/) | The decode ladder with the multi-stream option |
| [`depth-soak-2026-09-22`](depth-soak-2026-09-22/) | Is deep context the trigger? Three hours says no, and says what the dropouts were |
| [`depth-thirteen-2026-09-22`](depth-thirteen-2026-09-22/) | Depth on the thirteen-patch build, and a measurement trap |
| [`depth8192-comparison-2026-08-18`](depth8192-comparison-2026-08-18/) | (see its README) |
| [`dispatch-bimodal-2026-09-23`](dispatch-bimodal-2026-09-23/) | A process either gets the fast dispatch path or it does not |
| [`dispatch-floor-2026-09-22`](dispatch-floor-2026-09-22/) | What a kernel dispatch costs when the kernel costs nothing |
| [`env-audit-2026-08-26`](env-audit-2026-08-26/) | Every environment variable, checked against the library that would read it |
| [`env-versions-2026-08-27`](env-versions-2026-08-27/) | The environment line, checked against the machine |
| [`f32-cost-2026-08-21`](f32-cost-2026-08-21/) | What the f32 compute-type workaround costs |
| [`factorial`](factorial/) | The flush-by-scheduler factorial |
| [`fault-caught-2026-09-22`](fault-caught-2026-09-22/) | The rare fault, caught with the mitigation in place |
| [`fault-hunt-2026-08-22`](fault-hunt-2026-08-22/) | The fault hunt, and what it did not find |
| [`fault-probe-2026-08-22`](fault-probe-2026-08-22/) | Provoking the fault deliberately |
| [`fault-repro-2026-09-22`](fault-repro-2026-09-22/) | Trying to reproduce the fault, and finding out what the dropouts were |
| [`fault-usability-2026-08-24`](fault-usability-2026-08-24/) | What a fault under `gpu_recovery=0` leaves behind |
| [`fedora44-benchmarks-2026-09-15`](fedora44-benchmarks-2026-09-15/) | Benchmarks on the default Fedora 44 configuration |
| [`fedora44-campaign-experts-2026-09-21`](fedora44-campaign-experts-2026-09-21/) | The campaign on the twelve-patch build |
| [`fedora44-campaign-f32mv-2026-09-18`](fedora44-campaign-f32mv-2026-09-18/) | The split throughput campaign with the float-activation matvec |
| [`fedora44-campaign-fa4-2026-09-18`](fedora44-campaign-fa4-2026-09-18/) | The split throughput campaign on the four-patch build |
| [`fedora44-campaign-final-2026-09-18`](fedora44-campaign-final-2026-09-18/) | The split throughput campaign on the final four-patch build |
| [`fedora44-campaign-final-defaults-2026-09-20`](fedora44-campaign-final-defaults-2026-09-20/) | Confirmation campaign on the shipped defaults |
| [`fedora44-campaign-five-patches-2026-09-18`](fedora44-campaign-five-patches-2026-09-18/) | The split throughput campaign with the first form of patch 5 |
| [`fedora44-campaign-five-patches-final-2026-09-18`](fedora44-campaign-five-patches-final-2026-09-18/) | The split throughput campaign on the final five-patch build |
| [`fedora44-campaign-float-all-2026-09-18`](fedora44-campaign-float-all-2026-09-18/) | The split throughput campaign with the float matvec on q4_K, q6_K and q8_0 |
| [`fedora44-campaign-iq-float-2026-09-18`](fedora44-campaign-iq-float-2026-09-18/) | The split throughput campaign with the float matvec on the IQ types and the experts |
| [`fedora44-campaign-iq-shmem-2026-09-20`](fedora44-campaign-iq-shmem-2026-09-20/) | The campaign on the nine-patch build |
| [`fedora44-campaign-patch5-2026-09-18`](fedora44-campaign-patch5-2026-09-18/) | The split throughput campaign on the five-patch build |
| [`fedora44-campaign-pkf16-2026-09-20`](fedora44-campaign-pkf16-2026-09-20/) | The split throughput campaign with the packed-fp16 prefill GEMM |
| [`fedora44-campaign-pkf16-all-2026-09-20`](fedora44-campaign-pkf16-all-2026-09-20/) | The split throughput campaign with every supported type in the prefill GEMM |
| [`fedora44-campaign-q8-2026-09-20`](fedora44-campaign-q8-2026-09-20/) | The campaign on the eleven-patch build |
| [`fedora44-campaign-seven-patches-2026-09-19`](fedora44-campaign-seven-patches-2026-09-19/) | The split throughput campaign on the seven-patch build |
| [`fedora44-campaign-tile-2026-09-20`](fedora44-campaign-tile-2026-09-20/) | The campaign on the ten-patch build |
| [`fedora44-ceilings-2026-09-16`](fedora44-ceilings-2026-09-16/) | Context ceilings on Fedora 44, and two depths that no longer work |
| [`fedora44-hostoverhead-2026-09-16`](fedora44-hostoverhead-2026-09-16/) | Where Fedora 44's lower host CPU time comes from |
| [`fedora44-rocm711-2026-09-15`](fedora44-rocm711-2026-09-15/) | Fedora 44 userspace with ROCm 7.1.1 on the production kernel |
| [`fedora44-soak-2026-09-16`](fedora44-soak-2026-09-16/) | Three-hour soak on the default Fedora 44 configuration |
| [`fedora44-soak8-2026-09-16`](fedora44-soak8-2026-09-16/) | Eight-hour soak on the default Fedora 44 configuration |
| [`fedora44-thermal-2026-09-16`](fedora44-thermal-2026-09-16/) | Does the board hold its clock under sustained load? |
| [`fedora44-validation-2026-09-15`](fedora44-validation-2026-09-15/) | Fedora 44 validation: speed probes, Vulkan, allocation churn, PyTorch |
| [`fedora44-working-2026-09-15`](fedora44-working-2026-09-15/) | Fedora 44 with ROCm 7.1.1: two fixes make it work |
| [`fedora45-rocm722-2026-09-16`](fedora45-rocm722-2026-09-16/) | Fedora 45 with ROCm 7.2.2: one custom piece instead of two |
| [`first-rep-graphs-2026-09-25`](first-rep-graphs-2026-09-25/) | The slow first repetition is HIP graph instantiation |
| [`flash-attention-tradeoff-2026-09-17`](flash-attention-tradeoff-2026-09-17/) | Flash attention is a trade at depth, not a default |
| [`floor-vs-vulkan-2026-09-23`](floor-vs-vulkan-2026-09-23/) | Vulkan's dispatch floor, and what it does to the explanation of the MoE's decode gap |
| [`flush-cost-2026-08-18`](flush-cost-2026-08-18/) | (see its README) |
| [`fp16-arch-2026-08-20`](fp16-arch-2026-08-20/) | (see its README) |
| [`fp16-arch-2026-08-22`](fp16-arch-2026-08-22/) | The fp16 defect across presented architectures |
| [`fp16-batch-boundary-2026-08-23`](fp16-batch-boundary-2026-08-23/) | The zeroed GEMMs are a batch boundary, not a graph replay |
| [`fp16-boundary-scope-2026-08-23`](fp16-boundary-scope-2026-08-23/) | The deterministic baseline is only deterministic at one chunk |
| [`fp16-cascade-2026-08-23`](fp16-cascade-2026-08-23/) | The defect escalates from one call to whole batches |
| [`fp16-dispatch-2026-08-19`](fp16-dispatch-2026-08-19/) | (see its README) |
| [`fp16-escalation-2026-08-23`](fp16-escalation-2026-08-23/) | Escalation is not data corruption |
| [`fp16-flashattn-2026-08-20`](fp16-flashattn-2026-08-20/) | (see its README) |
| [`fp16-kernel-isolation-2026-08-20`](fp16-kernel-isolation-2026-08-20/) | (see its README) |
| [`fp16-library-2026-08-20`](fp16-library-2026-08-20/) | (see its README) |
| [`fp16-mechanism-2026-08-23`](fp16-mechanism-2026-08-23/) | The fp16 mechanism, re-verified directly |
| [`fp16-mf16c-2026-09-15`](fp16-mf16c-2026-09-15/) | llama.cpp built with -mf16c, and which libraries carry the broken helpers |
| [`fp16-operands-2026-08-23`](fp16-operands-2026-08-23/) | The zeroed GEMM has good operands and identical arguments |
| [`fp16-pool-2026-08-19`](fp16-pool-2026-08-19/) | (see its README) |
| [`fp16-pool-2026-08-23`](fp16-pool-2026-08-23/) | The memory pool causes the non-determinism, not the defect |
| [`fp16-recheck-2026-08-25`](fp16-recheck-2026-08-25/) | The fp16 defect still reproduces, and still never the same way twice |
| [`fp16-root-cause-2026-09-15`](fp16-root-cause-2026-09-15/) | The zeroed fp16 GEMM: root cause and fix |
| [`fp16-scalar-2026-09-15`](fp16-scalar-2026-09-15/) | The zeroed fp16 GEMM: scalars, launch errors, and a trace that makes it disappear |
| [`fp16-scope-2026-08-20`](fp16-scope-2026-08-20/) | (see its README) |
| [`fp16-secondmodel-2026-08-23`](fp16-secondmodel-2026-08-23/) | The batch-boundary pattern is not one model's pattern |
| [`fp16-sentinel-2026-08-23`](fp16-sentinel-2026-08-23/) | The GEMM runs and writes zeros, and the scaling factor is not the cause |
| [`fp16-solutions-2026-08-20`](fp16-solutions-2026-08-20/) | (see its README) |
| [`fp16-survival-2026-08-23`](fp16-survival-2026-08-23/) | The defect as a survival curve |
| [`fp16-ubatch-2026-08-23`](fp16-ubatch-2026-08-23/) | The zeroed fp16 GEMM is one per batch boundary, and a single batch avoids it |
| [`frontpage-verify-2026-08-22`](frontpage-verify-2026-08-22/) | The remaining published claims, verified in one powered window |
| [`ftrace`](ftrace/) | Function-tracer captures of the wedge |
| [`fusion-value-2026-09-23`](fusion-value-2026-09-23/) | What kernel fusion is already worth here, and where the rest of the floor is |
| [`gate-verify-2026-08-25`](gate-verify-2026-08-25/) | The correctness gate, run again |
| [`gates-2026-08-14`](gates-2026-08-14/) | Perplexity gates against Vulkan |
| [`gfx1013-dot-isa-2026-09-17`](gfx1013-dot-isa-2026-09-17/) | What the RDNA1 macro actually buys the integer dot product |
| [`gpu-recovery-cost-2026-08-21`](gpu-recovery-cost-2026-08-21/) | What `amdgpu.gpu_recovery=0` costs |
| [`gpu-reset-fatal-2026-08-21`](gpu-reset-fatal-2026-08-21/) | A GPU reset kills this board, with or without a fault |
| [`graph-opt-2026-09-24`](graph-opt-2026-09-24/) | ggml-cuda can overlap independent work too, and the switch is off |
| [`hgemm-isa-2026-09-24`](hgemm-isa-2026-09-24/) | Does Tensile emit `v_pk_fma_f16` for gfx1013? Yes. |
| [`hipify-cuda-2026-09-24`](hipify-cuda-2026-09-24/) | Porting a CUDA program to this board, and what it costs |
| [`historical-sources`](historical-sources/) | (see its README) |
| [`hw-counters-2026-09-25`](hw-counters-2026-09-25/) | Hardware performance counters on gfx1013 |
| [`inference`](inference/) | Inference logs from the July investigation |
| [`integrated-remeasure-2026-08-18`](integrated-remeasure-2026-08-18/) | (see its README) |
| [`journal-retro-2026-08-20`](journal-retro-2026-08-20/) | What the persistent journal shows that dmesg could not |
| [`kernel-7.1.5`](kernel-7.1.5/) | First test on kernel 7.1.5, July 2026 |
| [`kernel-718-2026-08-19`](kernel-718-2026-08-19/) | (see its README) |
| [`kernel-725-2026-09-17`](kernel-725-2026-09-17/) | Kernel 7.2.5 with the board patches |
| [`kernel-equivalence-2026-08-17`](kernel-equivalence-2026-08-17/) | Kernel equivalence and the CU-by-scheduler factorial |
| [`kerntrace-2026-09-19`](kerntrace-2026-09-19/) | Where a decoded token's time goes: kernel traces of real runs |
| [`kfd-reset-probe-2026-08-22`](kfd-reset-probe-2026-08-22/) | The reset mitigation, confirmed without waiting for a fault |
| [`knobs-2026-09-21`](knobs-2026-09-21/) | The runtime knobs, measured again on the thirteen-patch build |
| [`kqv-dispatch-2026-08-21`](kqv-dispatch-2026-08-21/) | Does the KQV corruption correspond to a different GEMM being dispatched? No |
| [`kqv-divergence-2026-08-21`](kqv-divergence-2026-08-21/) | Where does the KQV divergence begin, and what makes it disappear |
| [`kqv-ladder-2026-08-21`](kqv-ladder-2026-08-21/) | The KQV context ladder, reproduced |
| [`kqv-periodicity-2026-08-21`](kqv-periodicity-2026-08-21/) | Is the KQV corruption periodic across processes? No |
| [`kqv-remeasure-2026-08-18`](kqv-remeasure-2026-08-18/) | (see its README) |
| [`ladder-2026-08-13`](ladder-2026-08-13/) | Kernel ladder, first pass |
| [`ladder-churn-2026-08-16`](ladder-churn-2026-08-16/) | Cross-kernel churn evidence |
| [`llamacpp-master-recheck-2026-09-14`](llamacpp-master-recheck-2026-09-14/) | llama.cpp master against the three patches |
| [`loose-ends-2026-08-18`](loose-ends-2026-08-18/) | Closing the remaining loose ends |
| [`macro-remeasure-2026-08-18`](macro-remeasure-2026-08-18/) | (see its README) |
| [`membw-2026-08-19`](membw-2026-08-19/) | (see its README) |
| [`mmq-2026-08-14`](mmq-2026-08-14/) | The zeroed fp16 GEMM: per-tensor instrumentation |
| [`moe-decode-2026-09-20`](moe-decode-2026-09-20/) | Where the MoE's token goes, decode and prefill |
| [`moe-kernels-reweighted-2026-09-23`](moe-kernels-reweighted-2026-09-23/) | The MoE per-op replay, reweighted, and the artifact that reversed it |
| [`moe-stream-aliasing-2026-09-25`](moe-stream-aliasing-2026-09-25/) | Why the MoE and the 27B launch no concurrent streams: K and V share one buffer |
| [`nommap-ceiling-2026-08-21`](nommap-ceiling-2026-08-21/) | Loading without mmap lowers the usable context |
| [`nommap-ceiling-2026-09-22`](nommap-ceiling-2026-09-22/) | What `--no-mmap` costs in usable context |
| [`once-checked-2026-08-17`](once-checked-2026-08-17/) | Re-measuring the once-checked capability claims |
| [`op-perf-hip-vs-vulkan-2026-09-17`](op-perf-hip-vs-vulkan-2026-09-17/) | Where the ROCm prefill deficit lives, operation by operation |
| [`opreplay-getrows-fault-2026-09-21`](opreplay-getrows-fault-2026-09-21/) | The per-op replay faults the GPU, and it is the harness |
| [`override-trap-2026-08-26`](override-trap-2026-08-26/) | What `HSA_OVERRIDE_GFX_VERSION` does to a kernel you built yourself |
| [`pasid-diagnosis-2026-08-19`](pasid-diagnosis-2026-08-19/) | (see its README) |
| [`patch-currency-2026-08-19`](patch-currency-2026-08-19/) | (see its README) |
| [`patched`](patched/) | Patched-module compute probe, July 2026 |
| [`pk-gemm-prototype-2026-09-19`](pk-gemm-prototype-2026-09-19/) | What a packed-fp16 GEMM is worth on this board: a prototype |
| [`pkf16-vs-rocblas-2026-09-25`](pkf16-vs-rocblas-2026-09-25/) | Where rocBLAS HGEMM's missing 3x is: the kernel, not the hardware |
| [`qwen38-2026-08-17`](qwen38-2026-08-17/) | Qwen3.8-27B on both backends |
| [`radv-patches-2026-09-17`](radv-patches-2026-09-17/) | Two community RADV patches measured |
| [`rdna1-fattn-remainder-2026-09-19`](rdna1-fattn-remainder-2026-09-19/) | Flash attention on the final build, every model, against Vulkan |
| [`rdna1-fattn-spill-2026-09-17`](rdna1-fattn-spill-2026-09-17/) | The RDNA1 flash-attention kernel spills, and fixing it is worth 37 percent of prefill |
| [`rdna1-gdn-concat-2026-09-19`](rdna1-gdn-concat-2026-09-19/) | The hybrid models' linear-attention layers at prefill: the gated delta net's lane count and a transposed concat |
| [`rdna1-iq-matvec-2026-09-20`](rdna1-iq-matvec-2026-09-20/) | Six attempts on the codebook matvec, one of which worked |
| [`rdna1-mmvq-2026-09-18`](rdna1-mmvq-2026-09-18/) | The quantized matrix-vector kernels on RDNA1: the decode gap |
| [`rdna1-pkf16-experts-2026-09-21`](rdna1-pkf16-experts-2026-09-21/) | The mixture-of-experts prefill, off MMQ at last |
| [`rdna1-pkf16-thresholds-2026-09-21`](rdna1-pkf16-thresholds-2026-09-21/) | The GEMM's admission threshold, measured again at the new tile |
| [`rdna1-pkf16-tile-2026-09-20`](rdna1-pkf16-tile-2026-09-20/) | The prefill GEMM's tile was tuned on the wrong thing |
| [`rebind-recovery-2026-09-15`](rebind-recovery-2026-09-15/) | Rebinding amdgpu as a recovery path |
| [`recipe-e2e-2026-08-17`](recipe-e2e-2026-08-17/) | End-to-end verification of the documented patch recipe |
| [`recipe-retest-2026-08-17`](recipe-retest-2026-08-17/) | Retesting two recipe restrictions |
| [`reproduce-verify-2026-08-19`](reproduce-verify-2026-08-19/) | (see its README) |
| [`reproduce-verify-2026-08-22`](reproduce-verify-2026-08-22/) | The shipped reproducer, re-run on the current configuration |
| [`reproduce-verify-2026-08-25`](reproduce-verify-2026-08-25/) | The shipped reproducer, run again after a week of module rebuilds |
| [`reset-ccwrite-2026-08-23`](reset-ccwrite-2026-08-23/) | The 40-CU unlock does not cause the host stall |
| [`reset-cp-bisect-2026-08-23`](reset-cp-bisect-2026-08-23/) | The reset hangs in the KIQ resume |
| [`reset-cp-resume-2026-08-23`](reset-cp-resume-2026-08-23/) | The reset stall is in the Command Processor resume |
| [`reset-dyndbg-2026-08-23`](reset-dyndbg-2026-08-23/) | Dynamic debug does not reach past the stall |
| [`reset-fault-addresses-2026-08-21`](reset-fault-addresses-2026-08-21/) | Where the faults that kill the board land |
| [`reset-honest-2026-09-15`](reset-honest-2026-09-15/) | Honest resets, the recovery default, and a gfx10 per-queue reset |
| [`reset-kiq-bisect-2026-08-23`](reset-kiq-bisect-2026-08-23/) | Inside the KIQ resume, and the stall moves when you add prints |
| [`reset-kiq-gap-2026-08-24`](reset-kiq-gap-2026-08-24/) | The gap probe answers a different question than the one it was built for |
| [`reset-kiq-repeat-2026-08-24`](reset-kiq-repeat-2026-08-24/) | The reset stall is a race, and three earlier readings were single runs of it |
| [`reset-netconsole-2026-08-23`](reset-netconsole-2026-08-23/) | The reset does not fail. The host hangs afterwards |
| [`reset-path-2026-08-24`](reset-path-2026-08-24/) | It is not one register: the first GC access after a reset is what hangs |
| [`reset-resume-bisect-2026-08-23`](reset-resume-bisect-2026-08-23/) | The stall is inside the GFX block's resume, which never returns |
| [`reset-settle-delay-2026-08-24`](reset-settle-delay-2026-08-24/) | A settle delay does not fix the reset stall |
| [`reset-smu-gc-2026-09-14`](reset-smu-gc-2026-09-14/) | The reset that is not a reset, and two attempts at a real one |
| [`reset-stall-detectors-2026-08-23`](reset-stall-detectors-2026-08-23/) | Trying to get a backtrace out of the stall |
| [`rocblas-rebuild-attempt-2026-08-20`](rocblas-rebuild-attempt-2026-08-20/) | (see its README) |
| [`rocblas-recovery-2026-08-20`](rocblas-recovery-2026-08-20/) | (see its README) |
| [`rocblas`](rocblas/) | Native rocBLAS SGEMM sweeps, July 2026 |
| [`rocm-only-verify-2026-08-25`](rocm-only-verify-2026-08-25/) | The ROCm-only claims, run, not read |
| [`rocr-queue-scratch-2026-09-18`](rocr-queue-scratch-2026-09-18/) | The segfault at the memory limit: ROCr's queue scratch guard, and what is behind it |
| [`round10-2026-09-20`](round10-2026-09-20/) | The IQ types and q5_K in the prefill GEMM |
| [`round11-2026-09-20`](round11-2026-09-20/) | q8_0 in the prefill GEMM, measured properly this time |
| [`round12-2026-09-20`](round12-2026-09-20/) | What prefill is made of now |
| [`round13-2026-09-20`](round13-2026-09-20/) | How often the f16 accumulators need promoting, and what that says about the kernel |
| [`round14-2026-09-20`](round14-2026-09-20/) | Double buffering the tiles: a wave of occupancy for nothing |
| [`round2-2026-09-19`](round2-2026-09-19/) | Round two after the seven patches: runtime knobs, the 8B's q8_0 kernel, the launch gaps |
| [`round3-2026-09-19`](round3-2026-09-19/) | Round three: the hardware queue count, the 8B settled, one more D=256 row |
| [`round4-2026-09-19`](round4-2026-09-19/) | What prefill actually runs, and a null result about switching it |
| [`round5-2026-09-19`](round5-2026-09-19/) | Dequantise and multiply with rocBLAS instead of MMQ |
| [`round6-2026-09-19`](round6-2026-09-19/) | The packed-fp16 GEMM inside llama.cpp: correct, faster per kernel, slower per model |
| [`round7-2026-09-20`](round7-2026-09-20/) | The packed-fp16 GEMM, dequantising inside the kernel: 36 percent of the 1.5B's prefill |
| [`sdma-copy-inventory-2026-09-25`](sdma-copy-inventory-2026-09-25/) | What a decode actually copies, and why that does not explain the SDMA cost |
| [`sdma-decode-cost-2026-09-22`](sdma-decode-cost-2026-09-22/) | Enabling SDMA costs the small model 5 percent of decode |
| [`sdma-depth-2026-08-20`](sdma-depth-2026-08-20/) | (see its README) |
| [`sdma-firmware-2026-08-19`](sdma-firmware-2026-08-19/) | (see its README) |
| [`sdma-interrupt-2026-08-17`](sdma-interrupt-2026-08-17/) | Does the SDMA completion interrupt arrive? |
| [`sdma-onebyte-2026-08-18`](sdma-onebyte-2026-08-18/) | (see its README) |
| [`sdma-sizes-2026-08-19`](sdma-sizes-2026-08-19/) | (see its README) |
| [`soak-2026-08-13`](soak-2026-08-13/) | Eight-hour soak, small model |
| [`soak-crash-2026-08-20`](soak-crash-2026-08-20/) | The alternating-SDMA soak, and the crash that ended it |
| [`soak-large-2026-08-14`](soak-large-2026-08-14/) | Large-model endurance soak |
| [`soak-sdma-2026-09-22`](soak-sdma-2026-09-22/) | The same soak with SDMA enabled |
| [`soak-thirteen-2026-09-22`](soak-thirteen-2026-09-22/) | Eight hours on the thirteen-patch build, 2026-09-21 to |
| [`stock`](stock/) | Stock-module compute probe, July 2026 |
| [`suspend-recovery-2026-09-15`](suspend-recovery-2026-09-15/) | Suspend as a hardware reset |
| [`svm-flush-2026-08`](svm-flush-2026-08/) | The SVM map-side flush: ftrace evidence and A/B batteries, 2026-08 |
| [`svm-flush-bits-2026-08-13`](svm-flush-bits-2026-08-13/) | Which half of the runlist flush the sequence reproducer needs |
| [`tlb-alt-2026-09-15`](tlb-alt-2026-09-15/) | Lighter replacements for the runlist-rebuild flush, 2026-09-14/15 |
| [`torch-pristine-2026-08-20`](torch-pristine-2026-08-20/) | (see its README) |
| [`torch-probe-2026-08-19`](torch-probe-2026-08-19/) | (see its README) |
| [`torch-recheck-2026-08-22`](torch-recheck-2026-08-22/) | PyTorch training, re-checked on the current configuration |
| [`torch-rocblas-bench-2026-09-24`](torch-rocblas-bench-2026-09-24/) | What PyTorch and rocBLAS actually deliver here, against the board's own ceilings |
| [`torch-train-2026-08-19`](torch-train-2026-08-19/) | (see its README) |
| [`umr`](umr/) | umr captures of a stalled board, July 2026 |
| [`vmid-flush-2026-08-20`](vmid-flush-2026-08-20/) | (see its README) |
| [`vmm-and-deep-context-2026-09-18`](vmm-and-deep-context-2026-09-18/) | Virtual memory management does not work on gfx1013, and the deep-context ceiling has not moved |
| [`vulkan-fa-staging-2026-09-17`](vulkan-fa-staging-2026-09-17/) | Two upstream Vulkan patches tested on this board |
