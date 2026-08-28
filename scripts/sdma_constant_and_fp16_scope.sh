#!/usr/bin/env bash
# Two open points in one run.
#
# 1. Every measurement in this repository was taken with HSA_ENABLE_SDMA=0,
#    which was forced rather than chosen. SDMA works now, so by this project own
#    standard that constant needs re-testing rather than assuming it stayed
#    neutral. Throughput, gates and Vulkan are already re-checked; the churn
#    sweep and the context ceilings are not.
#
# 2. The zeroed fp16 GEMM has only ever been characterised on five models, of
#    which two are affected. Three more are on the board and have never been
#    tested, which is cheap data on how wide the defect is.
#
# Fault counts below come from dmesg, which sees only the current boot and
# only what is still in the ring buffer. If a run ends in a GPU reset the
# board goes down and the following check reads an empty buffer; and on a long
# run the buffer wraps, so early messages are gone. A zero here means "nothing
# dmesg can still see", not "nothing happened". Use scripts/fault_count.sh in
# new work.
set -u
D=~/inv83; mkdir -p "$D"
L=${ROCBLAS_LIB_DIR:-/home/akandr/rocBLAS/build/release/rocblas-install/lib}
HIP=~/llama-master/build-hip/bin
W=~/wiki.test.raw
log () { echo "[$(date +%H:%M:%S)] $*" | tee -a "$D/log"; sync; }

log "=== kernel $(uname -r), SDMA firmware = navi12"

log "=== PART 1: allocation churn with SDMA enabled against disabled, alternated"
for r in 1 2; do
  for s in 1 0 0 1; do
    t0=$(date +%s)
    env HSA_ENABLE_SDMA=$s GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32 LD_LIBRARY_PATH=$L \
      timeout -k 20 1800 $HIP/test-backend-ops -o MUL_MAT > "$D/churn_r${r}_s${s}.log" 2>&1
    rc=$?; t1=$(date +%s)
    log "  round $r sdma=$s: rc=$rc wall=$((t1-t0))s fails=$(grep -ac FAIL "$D/churn_r${r}_s${s}.log")"
  done
done

log "=== PART 2: context ceiling spot check with SDMA enabled (8B at 16384, ref 16.2 to 18.7)"
for s in 1 0; do
  v=$(env HSA_ENABLE_SDMA=$s GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32 LD_LIBRARY_PATH=$L \
      timeout -k 20 1800 $HIP/llama-bench -m /opt/models/qwen3-8b-q8_0.gguf -ngl 99 -fa 1 -p 0 -n 8 -d 16128 -r 1 2>&1 \
      | grep -aoE "tg8 @ d[0-9]+ \| +[0-9]+\.[0-9]+" | grep -oE "[0-9.]+$")
  log "  sdma=$s d16128: ${v:-FAIL}"
done

log "=== PART 3: how wide is the fp16 defect? three models never tested"
for m in gemma4 qwen3.5-9b qwen3.8-27b-iq3xxs; do
  [ -f /opt/models/$m.gguf ] || { log "  $m: not present"; continue; }
  f32=$(env HSA_ENABLE_SDMA=0 GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32 LD_LIBRARY_PATH=$L \
        timeout -k 30 1800 $HIP/llama-perplexity -m /opt/models/$m.gguf --no-mmap -ngl 99 -fa on -c 2048 -f $W --chunks 2 2>&1 \
        | grep -aoE "Final estimate: PPL = [0-9.]+" | grep -oE "[0-9.]+$")
  f16=$(env HSA_ENABLE_SDMA=0 GGML_CUDA_CUBLAS_COMPUTE_TYPE=f16 LD_LIBRARY_PATH=$L \
        timeout -k 30 1800 $HIP/llama-perplexity -m /opt/models/$m.gguf --no-mmap -ngl 99 -fa on -c 2048 -f $W --chunks 2 2>&1 \
        | grep -aoE "Final estimate: PPL = [0-9.]+" | grep -oE "[0-9.]+$")
  ratio=$(echo "scale=3; ${f16:-0} / (${f32:-1})" | bc 2>/dev/null)
  log "  $m: f32=${f32:-FAIL} f16=${f16:-FAIL} ratio=${ratio}"
done
log "=== dmesg faults: $(sudo dmesg | grep -ciE \"memory access fault|preemption time out\")"
touch "$D/DONE"; log done
