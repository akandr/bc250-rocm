#!/usr/bin/env bash
# Does the zeroed fp16 GEMM follow the call or the rocBLAS build?
#
# Every prior test of this defect used the locally built gfx1013 rocBLAS, which
# was never varied. A llama.cpp carrying both gfx1010 and gfx1013 code makes the
# comparison possible: with HSA_OVERRIDE_GFX_VERSION=10.1.0 the device presents
# as gfx1010, llama.cpp has kernels for that, and the system rocBLAS services the
# GEMMs through its real gfx1010 code objects instead of the native build.
#
# If the defect appears under both, it belongs to the fp16 path. If it appears
# only with the native build, seven eliminations were chasing the wrong suspect.
#
# Fault counts below come from dmesg, which sees only the current boot and
# only what is still in the ring buffer. If a run ends in a GPU reset the
# board goes down and the following check reads an empty buffer; and on a long
# run the buffer wraps, so early messages are gone. A zero here means "nothing
# dmesg can still see", not "nothing happened". Use scripts/fault_count.sh in
# new work.
set -u
D=~/inv88; mkdir -p "$D"
NAT=${ROCBLAS_LIB_DIR:-/home/akandr/rocBLAS/build/release/rocblas-install/lib}
BIN=~/llama-master/build-multi/bin/llama-perplexity
M8=/opt/models/qwen3-8b-q8_0.gguf
W=~/wiki.test.raw
log () { echo "[$(date +%H:%M:%S)] $*" | tee -a "$D/log"; sync; }

gate () { local tag=$1; shift
  local v=$(env HSA_ENABLE_SDMA=0 "$@" timeout -k 30 1800 $BIN -m $M8 --no-mmap \
      -ngl 99 -fa on -c 2048 -f $W --chunks 2 2>&1 | grep -aoE "Final estimate: PPL = [0-9.]+" | grep -oE "[0-9.]+$")
  log "  $tag: ${v:-FAIL}"
}

log "=== dual-arch build, native gfx1013 rocBLAS, as the reference"
for r in 1 2; do
  gate "gfx1013 native f32 r$r" LD_LIBRARY_PATH=$NAT GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32
  gate "gfx1013 native f16 r$r" LD_LIBRARY_PATH=$NAT GGML_CUDA_CUBLAS_COMPUTE_TYPE=f16
done
log "=== dual-arch build as gfx1010, system rocBLAS servicing the GEMMs"
for r in 1 2; do
  gate "gfx1010 system f32 r$r" HSA_OVERRIDE_GFX_VERSION=10.1.0 GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32
  gate "gfx1010 system f16 r$r" HSA_OVERRIDE_GFX_VERSION=10.1.0 GGML_CUDA_CUBLAS_COMPUTE_TYPE=f16
done
log "=== dmesg faults: $(sudo dmesg | grep -ciE \"memory access fault|preemption time out\")"
touch "$D/DONE"; log done
