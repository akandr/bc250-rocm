#!/usr/bin/env bash
# Is the zeroed fp16 GEMM a property of the native rocBLAS build?
#
# Seven hypotheses are eliminated, and every one of them was tested against the
# same library: the gfx1013 rocBLAS built locally. That library has never been
# varied. The system rocBLAS with HSA_OVERRIDE_GFX_VERSION=10.1.0 services the
# same calls through different code objects, so if the defect follows the call
# rather than the build, it should appear there too. If it does not, the fault
# is in the native build and everything above is about the wrong suspect.
#
# Fault counts below come from dmesg, which sees only the current boot and
# only what is still in the ring buffer. If a run ends in a GPU reset the
# board goes down and the following check reads an empty buffer; and on a long
# run the buffer wraps, so early messages are gone. A zero here means "nothing
# dmesg can still see", not "nothing happened". Use scripts/fault_count.sh in
# new work.
set -u
D=~/inv87; mkdir -p "$D"
NAT=${ROCBLAS_LIB_DIR:-/home/akandr/rocBLAS/build/release/rocblas-install/lib}
HIP=~/llama-master/build-hip/bin
M8=/opt/models/qwen3-8b-q8_0.gguf
W=~/wiki.test.raw
log () { echo "[$(date +%H:%M:%S)] $*" | tee -a "$D/log"; sync; }

gate () { # gate <tag> <env assignments...>
  local tag=$1; shift
  local v
  v=$(env HSA_ENABLE_SDMA=0 "$@" timeout -k 30 1800 $HIP/llama-perplexity -m $M8 --no-mmap \
      -ngl 99 -fa on -c 2048 -f $W --chunks 2 2>&1 | grep -aoE "Final estimate: PPL = [0-9.]+" | grep -oE "[0-9.]+$")
  log "  $tag: ${v:-FAIL}"
}

log "=== native gfx1013 rocBLAS (the library everything so far was tested against)"
for r in 1 2; do
  gate "native f32 r$r" LD_LIBRARY_PATH=$NAT GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32
  gate "native f16 r$r" LD_LIBRARY_PATH=$NAT GGML_CUDA_CUBLAS_COMPUTE_TYPE=f16
done
log "=== system rocBLAS with HSA_OVERRIDE_GFX_VERSION=10.1.0 (different code objects)"
for r in 1 2; do
  gate "override f32 r$r" HSA_OVERRIDE_GFX_VERSION=10.1.0 GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32
  gate "override f16 r$r" HSA_OVERRIDE_GFX_VERSION=10.1.0 GGML_CUDA_CUBLAS_COMPUTE_TYPE=f16
done
log "=== dmesg faults: $(sudo dmesg | grep -ciE \"memory access fault|preemption time out\")"
touch "$D/DONE"; log done
