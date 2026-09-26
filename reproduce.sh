#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Artur Andrzejczak <andrzejczak.artur@gmail.com>
# Assisted-by: Claude (Anthropic)
# reproduce.sh - build and run the probes from this repository on a BC-250.
#
# Scheduler guidance depends on the module, see the README:
# - Stock flush (flush_pasid_uses_kiq true, e.g. an unpatched module): boot with
#   amdgpu.sched_policy=2, or a HIP process can freeze the board on exit.
# - The working configuration (patched module, flush_pasid_uses_kiq false): do
#   NOT set sched_policy; the default hardware scheduling is required, and
#   sched_policy=2 is what makes sustained compute wedge. The kernel version is
#   not part of the recipe: six kernels from 6.18.9 to 7.2.5 measure identically
#   with the same patch set.
# The gate below warns when the combination looks wrong (set FORCE=1 to override).
#
# Requires: ROCm clang++ (/usr/lib64/rocm/llvm/bin/clang++), OpenCL headers
# (ocl-icd-devel, opencl-headers), and the OpenCL ICD loader plus RustiCL.
#
# Works on both configurations this repository has measured: Fedora 43 with ROCm 6.4.2 (rocBLAS soname 4,
# native build under ~/rocBLAS) and Fedora 44 with ROCm 7.1.1 (soname 5, overrides installed in
# /opt/bc250-rocm by scripts/install_rocm711_overrides.sh). It reads which of the two it is running on
# instead of assuming.
set -u
CLXX=/usr/lib64/rocm/llvm/bin/clang++
HERE="$(cd "$(dirname "$0")" && pwd)/patches"

SCHED=$(cat /sys/module/amdgpu/parameters/sched_policy 2>/dev/null || echo '?')
FLUSHPARAM=/sys/module/amdgpu/parameters/bc250_flush_pasid_kiq

# The runlist-rebuild flush is a bitmask: bit 1 on unmap, bit 2 on map. Only the
# map side closes the allocation-reuse fault that the churn workloads hit, so a
# module carrying just the original unmap hook will still fault under them.
RUNLIST=/sys/module/amdgpu/parameters/bc250_flush_by_runlist
if [ -e "$RUNLIST" ]; then
    RL=$(cat $RUNLIST)
    if [ $(( RL & 2 )) -eq 0 ] && [ "${FORCE:-0}" != "1" ]; then
        echo "amdgpu.bc250_flush_by_runlist=$RL: the map-side bit (2) is not set."
        echo "Allocation-churn workloads fault without it; see the allocation-reuse"
        echo "section of the README. Boot with =3, or re-run with FORCE=1."
        exit 1
    fi
fi
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
echo "distro:  $(cat /etc/fedora-release 2>/dev/null || echo '?')"
echo "kernel:  $(uname -r)"
echo "rocblas: $(rpm -q rocblas 2>/dev/null || echo '?')"
echo "mesa:    $(rpm -q mesa-vulkan-drivers 2>/dev/null || echo '?')"
SIMD=$(grep -h simd_count /sys/class/kfd/kfd/topology/nodes/*/properties 2>/dev/null | \
       awk '{print $2}' | sort -rn | head -1)
case "${SIMD:-0}" in
    80) echo "CUs:     40 (simd_count 80), the community unlock is active" ;;
    48) echo "CUs:     24 (simd_count 48). The unlock is NOT active: amdgpu.bc250_cc_write_mode=3"
        echo "         is missing, or the running module is not the one you installed. The"
        echo "         module comes from the initramfs, so 'dracut -f' after installing it." ;;
    *)  echo "CUs:     could not read simd_count (got '${SIMD:-}'); is the KFD topology present?" ;;
esac
echo

# The GPU clock policy decides whether any timing here means anything: with the governor's package
# default this board overheats and oscillates between clock steps. This only reports it; see the README.
echo "clock:   $(grep '\*' /sys/class/drm/card*/device/pp_dpm_sclk 2>/dev/null | grep -oE '[0-9]+Mhz' | tr '\n' ' ')(idle reading; check it again under load)"
if [ -r /etc/oberon-config.yaml ]; then
    echo "governor config: $(tr -d ' \n' < /etc/oberon-config.yaml | sed 's/opps://')"
    THR=$(journalctl -b 0 -u oberon-governor --no-pager 2>/dev/null | grep -ci throttl)
    [ "${THR:-0}" != "0" ] && echo "         the governor has logged $THR throttle events this boot"
fi
echo

echo "== 1. rocBLAS gfx1013 code objects =="
SYSROCBLAS=$(ls /usr/lib64/librocblas.so.[0-9].* 2>/dev/null | head -1)
echo "system library: ${SYSROCBLAS:-none found}"
echo "embedded ISAs:"
strings "${SYSROCBLAS:-/dev/null}" 2>/dev/null | grep -oE 'gfx10[0-9]+' | sort -u | tr '\n' ' '; echo
echo "(gfx1013 absent is why GEMM aborts against the system library; the native build is the fix)"
echo

echo "== 2. rocBLAS SGEMM: system library, override, and native build =="
g++ -O2 -D__HIP_PLATFORM_AMD__ "$HERE/rocblas_probe.c" -o /tmp/rocblas_probe -I/usr/include -lrocblas -lamdhip64 \
  || { echo "rocblas_probe build failed"; exit 1; }
# Three arms: the system library as installed (expected to fail, it has no
# gfx1013 code objects), the same library under the gfx1010 override, and the
# natively built one. The override arm passing here is not an endorsement: it
# works for a prebuilt library carrying real gfx1010 kernels, and silently
# stops running anything you compiled for gfx1013. See the README.
echo "system rocBLAS, no override:"; HSA_ENABLE_SDMA=0 /tmp/rocblas_probe 512 2>&1 | grep -aiE 'ROCBLAS|CORRECT|status' | head -3
echo "system rocBLAS with HSA_OVERRIDE_GFX_VERSION=10.1.0:"; HSA_ENABLE_SDMA=0 HSA_OVERRIDE_GFX_VERSION=10.1.0 /tmp/rocblas_probe 512 2>&1 | grep -aiE 'CORRECT|PROBE_OK|status' | head -2

# The two arms above are the problem, not the solution: the system library has
# no gfx1013 code objects, and the override is a dead end for real work. What
# actually fixes it is a native gfx1013 rocBLAS (scripts/build_rocblas_gfx1013.sh).
# Point ROCBLAS_NATIVE at its install prefix to see the same probe pass.
# Where the native library lives differs between the two configurations: installed system-wide in
# /opt/bc250-rocm on Fedora 44 (in which case the probe above already used it), or on LD_LIBRARY_PATH
# from a build tree on Fedora 43.
if [ -d /opt/bc250-rocm/lib64 ]; then
    echo "native gfx1013 rocBLAS is installed system-wide:"
    ldconfig -p | grep -E "librocblas.so.[0-9] " | head -2
    echo "gfx1013 Tensile files: $(ls /opt/bc250-rocm/lib64/rocblas/library 2>/dev/null | grep -c gfx1013)"
    if ldconfig -p | grep -q "libamd_comgr.so.3 => /opt/bc250-rocm"; then
        echo "comgr override in place (gfx10 VGPR fix, see README step 5)"
    else
        echo "no comgr override: on ROCm 7.0 to 7.1.1 flash attention asserts without it"
    fi
fi
for CAND in "${ROCBLAS_NATIVE:-}" "$HOME/rb711/install" "$HOME/rocBLAS/build/release/rocblas-install"; do
    [ -n "$CAND" ] || continue
    LIBDIR=""
    [ -d "$CAND/lib" ] && LIBDIR="$CAND/lib"
    [ -d "$CAND/lib64" ] && LIBDIR="$CAND/lib64"
    if [ -n "$LIBDIR" ]; then
        echo "with the native gfx1013 rocBLAS ($LIBDIR):"
        HSA_ENABLE_SDMA=0 LD_LIBRARY_PATH="$LIBDIR" /tmp/rocblas_probe 512 2>&1 \
          | grep -aiE 'CORRECT|PROBE_OK|status' | head -2
        break
    fi
done
echo

echo "== 3. compute correctness: graphics queue (RustiCL, OpenCL) =="
cc "$HERE/ocl_vecadd.c" -o /tmp/ocl_vecadd -lOpenCL
RUSTICL_ENABLE=radeonsi /tmp/ocl_vecadd 2>&1 | grep -aiE 'device|OK|WRONG'
echo

echo "== 4. compute correctness: MEC compute queue (HIP), small vs large =="
$CLXX -x hip --offload-arch=gfx1013 -I/usr/include "$HERE/compute_probe.c" -o /tmp/compute_probe -L/usr/lib64 -lamdhip64
echo "small kernel (1M threads):"; timeout 30 env HSA_ENABLE_SDMA=0 /tmp/compute_probe 4096 500 5 2>&1 | grep -aE 'RESULT'
echo "large kernel (16M threads), may hang or produce wrong results:"; timeout 60 env HSA_ENABLE_SDMA=0 /tmp/compute_probe 65536 800 5 2>&1 | grep -aE 'RESULT|MISMATCH' | head -3
echo "(under the working configuration these are correct; the non-determinism this line used to
warn about belonged to the stock flush and to sched_policy=2. See README.)"
