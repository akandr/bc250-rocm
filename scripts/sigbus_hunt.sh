#!/usr/bin/env bash
# Two more open points.
#
# 1. The SIGBUS in ROCr AqlQueue::StoreRelaxed has been seen once in roughly
#    twelve deep-decode runs and never reproduced deliberately. At that rate a
#    run of thirty should yield two or three if the estimate is right, and zero
#    would say the rate is lower than believed or the condition has changed.
#    Every crash is captured with its backtrace so a second instance can be
#    compared against the first.
#
# 2. The zeroed fp16 GEMM has never been tested with flash attention off. The
#    defect lives in the cuBLAS dequant path, which -fa off reaches differently,
#    so this either widens the description or narrows it.
#
# Fault counts below come from dmesg, which sees only the current boot and
# only what is still in the ring buffer. If a run ends in a GPU reset the
# board goes down and the following check reads an empty buffer; and on a long
# run the buffer wraps, so early messages are gone. A zero here means "nothing
# dmesg can still see", not "nothing happened". Use scripts/fault_count.sh in
# new work.
set -u
D=~/inv85; mkdir -p "$D"
L=${ROCBLAS_LIB_DIR:-/home/akandr/rocBLAS/build/release/rocblas-install/lib}
HIP=~/llama-master/build-hip/bin
M8=/opt/models/qwen3-8b-q8_0.gguf
W=~/wiki.test.raw
log () { echo "[$(date +%H:%M:%S)] $*" | tee -a "$D/log"; sync; }

log "waiting for the previous job"
while [ ! -f ~/inv84/DONE ]; do sleep 60; done

log "=== PART 1: thirty deep-decode runs, counting SIGBUS"
before=$(coredumpctl list --no-pager 2>/dev/null | grep -c llama-bench)
crashes=0
for i in $(seq 1 30); do
  env HSA_ENABLE_SDMA=0 GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32 LD_LIBRARY_PATH=$L \
    timeout -k 20 900 $HIP/llama-bench -m $M8 -ngl 99 -fa 1 -p 0 -n 8 -d 16128 -r 1 \
    > "$D/deep_$i.log" 2>&1
  rc=$?
  v=$(grep -aoE "tg8 @ d[0-9]+ \| +[0-9]+\.[0-9]+" "$D/deep_$i.log" | grep -oE "[0-9.]+$")
  if [ -z "$v" ]; then
    crashes=$((crashes+1))
    log "  run $i: NO RESULT rc=$rc $(grep -aoE "zrzut pamięci|core dumped|Aborted|Bus error" "$D/deep_$i.log" | tail -1)"
  else
    [ $((i % 10)) -eq 0 ] && log "  run $i: $v (clean so far: $((i-crashes))/$i)"
  fi
done
after=$(coredumpctl list --no-pager 2>/dev/null | grep -c llama-bench)
log "  no-result runs: $crashes of 30; new core dumps: $((after-before))"
if [ $((after-before)) -gt 0 ]; then
  coredumpctl info -1 2>/dev/null | grep -aE "Signal|^ *#[0-9]" | head -12 | tee -a "$D/log"
fi

log "=== PART 2: fp16 defect with flash attention off (never tested)"
for fa in on off; do
  for ct in f32 f16; do
    v=$(env HSA_ENABLE_SDMA=0 GGML_CUDA_CUBLAS_COMPUTE_TYPE=$ct LD_LIBRARY_PATH=$L \
        timeout -k 30 1800 $HIP/llama-perplexity -m $M8 --no-mmap -ngl 99 -fa $fa \
        -c 2048 -f $W --chunks 2 2>&1 | grep -aoE "Final estimate: PPL = [0-9.]+" | grep -oE "[0-9.]+$")
    log "  fa=$fa compute=$ct: ${v:-FAIL}"
  done
done
log "=== dmesg faults: $(sudo dmesg | grep -ciE \"memory access fault|preemption time out\")"
touch "$D/DONE"; log done
