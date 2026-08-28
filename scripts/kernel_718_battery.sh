#!/usr/bin/env bash
# Does the patch set behave the same on kernel 7.1.8?
#
# GabriWar reports the flush patch working on newer kernels without much time to
# test. This repository has measured 6.18.9, 6.18.16, 6.19.14 and 7.1.5 as
# indistinguishable and concluded the kernel version is not an ingredient. 7.1.8
# extends that by one release and gives the claim independent data.
#
# All four patches were ported: the 40-CU unlock, the flush-pasid-kiq parameter,
# the runlist flush and the SVM map-side flush. Same battery as the ladder.
#
# Fault counts below come from dmesg, which sees only the current boot and
# only what is still in the ring buffer. If a run ends in a GPU reset the
# board goes down and the following check reads an empty buffer; and on a long
# run the buffer wraps, so early messages are gone. A zero here means "nothing
# dmesg can still see", not "nothing happened". Use scripts/fault_count.sh in
# new work.
set -u
D=~/inv81; mkdir -p "$D"
L=${ROCBLAS_LIB_DIR:-/home/akandr/rocBLAS/build/release/rocblas-install/lib}
HIP=~/llama-master/build-hip/bin
E=(env HSA_ENABLE_SDMA=0 GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32 LD_LIBRARY_PATH=$L)
log () { echo "[$(date +%H:%M:%S)] $*" | tee -a "$D/log"; sync; }

log "=== kernel $(uname -r), CU $(grep -h simd_count /sys/class/kfd/kfd/topology/nodes/*/properties | sort -u | tail -1)"
log "=== compute probe, the size that used to fault"
timeout -k 20 300 ~/compute_probe 65536 800 5 2>&1 | grep -aE "RESULT" | tee -a "$D/log"
log "=== native rocBLAS SGEMM sweep"
for N in 512 1024 2048 4096; do
  r=$(HSA_ENABLE_SDMA=0 LD_LIBRARY_PATH=$L timeout -k 20 600 ~/sgemm_probe $N 20 2>&1 | grep -aoE "GFLOP/s[: ]+[0-9.]+" | grep -oE "[0-9.]+$" | tail -1)
  log "  N=$N: ${r:-see log} GFLOP/s"
done
log "=== perplexity gates (7.1.5 references: 8.9442 and 9.0975)"
for m in qwen2.5-1.5b-q4km:4096:8:8.9442 qwen3-8b-q8_0:2048:2:9.0975; do
  f=${m%%:*}; rest=${m#*:}; c=${rest%%:*}; rest=${rest#*:}; ch=${rest%%:*}; ref=${rest##*:}
  v=$("${E[@]}" timeout -k 30 1800 $HIP/llama-perplexity -m /opt/models/$f.gguf --no-mmap -ngl 99 -fa on -c $c -f ~/wiki.test.raw --chunks $ch 2>&1 | grep -aoE "Final estimate: PPL = [0-9.]+" | grep -oE "[0-9.]+$")
  log "  $f: ${v:-FAIL}  (7.1.5 gave $ref)"
done
log "=== throughput (7.1.5: pp512 805.53, tg64 113.50)"
"${E[@]}" timeout -k 20 900 $HIP/llama-bench -m /opt/models/qwen2.5-1.5b-q4km.gguf -mmp 0 -ngl 99 -fa on -p 512 -n 64 -r 3 > "$D/bench.log" 2>&1
log "  pp512 $(grep -aoE "pp512 \| +[0-9.]+" "$D/bench.log" | grep -oE "[0-9.]+$")  tg64 $(grep -aoE "tg64 \| +[0-9.]+" "$D/bench.log" | grep -oE "[0-9.]+$")"
log "=== allocation churn"
"${E[@]}" timeout -k 20 1800 $HIP/test-backend-ops -o MUL_MAT > "$D/churn.log" 2>&1
log "  churn rc=$?"
log "=== SDMA (navi12 firmware carried over)"
HSA_ENABLE_SDMA=1 timeout -k 5 20 ~/sdma_one 16385 4 2>&1 | tail -1 | tee -a "$D/log"
log "=== dmesg faults: $(sudo dmesg | grep -ciE \"memory access fault|preemption time out\")"
touch "$D/DONE"; log done
