#!/usr/bin/env bash
# Is the zeroed fp16 GEMM about the pool temporary it writes into?
#
# What is known: 36 parameter-identical calls per graph go to the same rocBLAS
# kernel with correct arguments, and only the first of them returns zeros. Kernel
# selection, graph capture, instrumentation and argument corruption are all ruled
# out. The trigger is positional, which points at state around the call rather
# than the call itself.
#
# With fp16 compute the result goes to a pool temporary rather than straight to
# the destination tensor. The tree already carries two hooks from the
# allocation-reuse work: GGML_CUDA_NO_POOL bypasses the pool and allocates
# directly, and GGML_CUDA_POOL_NOREUSE never returns buffers to the pool. If
# either changes the outcome, the defect belongs to buffer reuse, which would
# connect it to the allocation-reuse family rather than to rocBLAS.
#
# Arms alternated, since this board has already produced two opposite answers
# from blocked designs.
set -u
D=~/inv77; mkdir -p "$D"
L=/home/akandr/rocBLAS/build/release/rocblas-install/lib
HIP=~/llama-master/build-hip/bin
M8=/opt/models/qwen3-8b-q8_0.gguf
WIKI=~/wiki.test.raw
log () { echo "[$(date +%H:%M:%S)] $*" | tee -a "$D/log"; sync; }

gate () { # gate <tag> <extra env...>
  local tag=$1; shift
  env HSA_ENABLE_SDMA=0 GGML_CUDA_CUBLAS_COMPUTE_TYPE=f16 LD_LIBRARY_PATH=$L "$@" \
    timeout -k 30 1800 $HIP/llama-perplexity -m $M8 --no-mmap -ngl 99 -fa on \
    -c 2048 -f $WIKI --chunks 2 > "$D/$tag.log" 2>&1
  log "  $tag: $(grep -aoE "Final estimate: PPL = [0-9.]+" "$D/$tag.log" | grep -oE "[0-9.]+$" || echo FAIL)"
}

log "=== reference: f32 compute, the known-correct value is 9.0975"
env HSA_ENABLE_SDMA=0 GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32 LD_LIBRARY_PATH=$L \
  timeout -k 30 1800 $HIP/llama-perplexity -m $M8 --no-mmap -ngl 99 -fa on \
  -c 2048 -f $WIKI --chunks 2 > "$D/f32ref.log" 2>&1
log "  f32 reference: $(grep -aoE "Final estimate: PPL = [0-9.]+" "$D/f32ref.log" | grep -oE "[0-9.]+$")"

log "=== three arms, alternated over three rounds"
for r in 1 2 3; do
  gate "baseline_r$r"
  gate "nopool_r$r"   GGML_CUDA_NO_POOL=1
  gate "noreuse_r$r"  GGML_CUDA_POOL_NOREUSE=1
done
log "=== summary"
for a in baseline nopool noreuse; do
  log "  $a: $(grep -h "  $a" "$D/log" | grep -oE "[0-9]+\.[0-9]+$" | tr "\n" " ")"
done
log "=== dmesg faults: $(sudo dmesg | grep -ciE \"memory access fault|preemption time out\")"
touch "$D/DONE"; log done
