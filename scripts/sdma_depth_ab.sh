#!/usr/bin/env bash
# Decode at depth with SDMA enabled against disabled, alternated.
#
# The first pass at this compared one run of each and got 15.45 against 19.33,
# which looks decisive and means nothing: this measurement has a documented
# range of 10.03 to 19.18 on a single configuration. Single samples cannot
# separate the arms. ABBA over three rounds can.
#
# Fault counts below come from dmesg, which sees only the current boot and
# only what is still in the ring buffer. If a run ends in a GPU reset the
# board goes down and the following check reads an empty buffer; and on a long
# run the buffer wraps, so early messages are gone. A zero here means "nothing
# dmesg can still see", not "nothing happened". Use scripts/fault_count.sh in
# new work.
set -u
D=~/inv84; mkdir -p "$D"
L=${ROCBLAS_LIB_DIR:-/home/akandr/rocBLAS/build/release/rocblas-install/lib}
HIP=~/llama-master/build-hip/bin
log () { echo "[$(date +%H:%M:%S)] $*" | tee -a "$D/log"; sync; }
run () {
  local s=$1 tag=$2
  local v=$(env HSA_ENABLE_SDMA=$s GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32 LD_LIBRARY_PATH=$L \
    timeout -k 20 1800 $HIP/llama-bench -m /opt/models/qwen3-8b-q8_0.gguf -ngl 99 -fa 1 -p 0 -n 8 -d 16128 -r 1 2>&1 \
    | grep -aoE "tg8 @ d[0-9]+ \| +[0-9]+\.[0-9]+" | grep -oE "[0-9.]+$")
  log "  $tag sdma=$s: ${v:-FAIL}"
}
log "=== 8B decode at depth 16128, SDMA on against off, ABBA"
for r in 1 2 3; do
  if [ $((r % 2)) -eq 1 ]; then o="1 0 0 1"; else o="0 1 1 0"; fi
  i=0; for s in $o; do i=$((i+1)); run $s "r${r}_${i}"; done
done
log "=== dmesg faults: $(sudo dmesg | grep -ciE \"memory access fault|preemption time out\")"
touch "$D/DONE"; log done
