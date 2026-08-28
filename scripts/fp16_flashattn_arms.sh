#!/usr/bin/env bash
# Part 2 of the previous run, which never executed because the board rebooted
# after a GPU memory access fault at run 28 of 30.
#
# The zeroed fp16 GEMM has never been tested with flash attention off. The
# defect lives in the cuBLAS dequant path and -fa off reaches it differently, so
# this either widens the description or narrows it.
#
# Fault counts below come from dmesg, which sees only the current boot and
# only what is still in the ring buffer. If a run ends in a GPU reset the
# board goes down and the following check reads an empty buffer; and on a long
# run the buffer wraps, so early messages are gone. A zero here means "nothing
# dmesg can still see", not "nothing happened". Use scripts/fault_count.sh in
# new work.
set -u
D=~/inv86; mkdir -p "$D"
L=${ROCBLAS_LIB_DIR:-/home/akandr/rocBLAS/build/release/rocblas-install/lib}
HIP=~/llama-master/build-hip/bin
M8=/opt/models/qwen3-8b-q8_0.gguf
W=~/wiki.test.raw
log () { echo "[$(date +%H:%M:%S)] $*" | tee -a "$D/log"; sync; }
log "=== fp16 defect against flash attention, all four cells, two rounds"
for r in 1 2; do
  for fa in on off; do
    for ct in f32 f16; do
      v=$(env HSA_ENABLE_SDMA=0 GGML_CUDA_CUBLAS_COMPUTE_TYPE=$ct LD_LIBRARY_PATH=$L \
          timeout -k 30 1800 $HIP/llama-perplexity -m $M8 --no-mmap -ngl 99 -fa $fa \
          -c 2048 -f $W --chunks 2 2>&1 | grep -aoE "Final estimate: PPL = [0-9.]+" | grep -oE "[0-9.]+$")
      log "  round $r fa=$fa compute=$ct: ${v:-FAIL}"
    done
  done
done
log "=== dmesg faults: $(sudo dmesg | grep -ciE \"memory access fault|preemption time out\")"
touch "$D/DONE"; log done
