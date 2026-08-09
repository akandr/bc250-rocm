#!/usr/bin/env bash
# reproduce.sh - build and run the probes from this repository on a BC-250.
#
# Scheduler guidance depends on the module, see the README:
# - Stock flush (flush_pasid_uses_kiq true, e.g. an unpatched module): boot with
#   amdgpu.sched_policy=2, or a HIP process can freeze the board on exit.
# - The working configuration (patched module, flush_pasid_uses_kiq false, kernel
#   7.1.5): do NOT set sched_policy; the default hardware scheduling is required,
#   and sched_policy=2 is what makes sustained compute wedge there.
# The gate below warns when the combination looks wrong (set FORCE=1 to override).
#
# Requires: ROCm clang++ (/usr/lib64/rocm/llvm/bin/clang++), OpenCL headers
# (ocl-icd-devel, opencl-headers), and the OpenCL ICD loader plus RustiCL.
set -u
CLXX=/usr/lib64/rocm/llvm/bin/clang++
HERE="$(cd "$(dirname "$0")" && pwd)/patches"

SCHED=$(cat /sys/module/amdgpu/parameters/sched_policy 2>/dev/null || echo '?')
FLUSHPARAM=/sys/module/amdgpu/parameters/bc250_flush_pasid_kiq
if [ -e "$FLUSHPARAM" ] && [ "$(cat $FLUSHPARAM)" = "0" ]; then
    # working configuration: corrected flush, hardware scheduling expected
    if [ "$SCHED" = "2" ] && [ "${FORCE:-0}" != "1" ]; then
        echo "Corrected flush is active but amdgpu.sched_policy=2 is set; on this"
        echo "combination sustained compute wedges. Remove the sched_policy argument"
        echo "and reboot, or re-run with FORCE=1 to proceed anyway."
        exit 1
    fi
else
    # stock flush: the exit freeze applies, policy 2 protects against it
    if [ "$SCHED" != "2" ] && [ "${FORCE:-0}" != "1" ]; then
        echo "Stock flush with amdgpu.sched_policy '$SCHED': a HIP process can freeze"
        echo "this board on exit. Boot with amdgpu.sched_policy=2 for the historical"
        echo "reproduction, or use the working configuration from the README, or"
        echo "re-run with FORCE=1 to proceed anyway."
        exit 1
    fi
fi

echo "== versions =="
echo "kernel:  $(uname -r)"
echo "rocblas: $(rpm -q rocblas 2>/dev/null || echo '?')"
echo "mesa:    $(rpm -q mesa-vulkan-drivers 2>/dev/null || echo '?')"
echo

echo "== 1. rocBLAS gfx1013 code objects =="
echo "librocblas embedded ISAs:"
strings /usr/lib64/librocblas.so.4.* 2>/dev/null | grep -oE 'gfx10[0-9]+' | sort -u | tr '\n' ' '; echo
echo "(gfx1013 absent is the reason GEMM aborts; see README Problem 2)"
echo

echo "== 2. rocBLAS SGEMM: native gfx1013 vs override =="
g++ -O2 -D__HIP_PLATFORM_AMD__ "$HERE/rocblas_probe.c" -o /tmp/rocblas_probe -I/usr/include -lrocblas -lamdhip64 \
  || { echo "rocblas_probe build failed"; exit 1; }
echo "native gfx1013:"; HSA_ENABLE_SDMA=0 /tmp/rocblas_probe 512 2>&1 | grep -aiE 'ROCBLAS|CORRECT|status' | head -3
echo "with HSA_OVERRIDE_GFX_VERSION=10.1.0:"; HSA_ENABLE_SDMA=0 HSA_OVERRIDE_GFX_VERSION=10.1.0 /tmp/rocblas_probe 512 2>&1 | grep -aiE 'CORRECT|PROBE_OK|status' | head -2
echo

echo "== 3. compute correctness: graphics queue (RustiCL, OpenCL) =="
cc "$HERE/ocl_vecadd.c" -o /tmp/ocl_vecadd -lOpenCL
RUSTICL_ENABLE=radeonsi /tmp/ocl_vecadd 2>&1 | grep -aiE 'device|OK|WRONG'
echo

echo "== 4. compute correctness: MEC compute queue (HIP), small vs large =="
$CLXX -x hip --offload-arch=gfx1013 -I/usr/include "$HERE/compute_probe.c" -o /tmp/compute_probe -L/usr/lib64 -lamdhip64
echo "small kernel (1M threads):"; timeout 30 env HSA_ENABLE_SDMA=0 /tmp/compute_probe 4096 500 5 2>&1 | grep -aE 'RESULT'
echo "large kernel (16M threads), may hang or produce wrong results:"; timeout 60 env HSA_ENABLE_SDMA=0 /tmp/compute_probe 65536 800 5 2>&1 | grep -aE 'RESULT|MISMATCH' | head -3
echo "(large dispatches are non-deterministic; rerun a few times. See README.)"
