#!/usr/bin/env bash
# Endurance soak alternating SDMA on and off, the last item of the SDMA
# constant re-test.
#
# Every measurement in this repository was taken with HSA_ENABLE_SDMA=0 because
# SDMA could not complete a transfer. It can now, and throughput, the gates,
# Vulkan, allocation churn and decode at depth have all been re-checked with it
# enabled and are unchanged. The soaks have not, and they are what the stability
# claims rest on.
#
# Alternating per round rather than running two blocks, because this board has
# repeatedly shown blocked designs producing differences that are not there.
#
# Fault counts below come from dmesg, which sees only the current boot and
# only what is still in the ring buffer. If a run ends in a GPU reset the
# board goes down and the following check reads an empty buffer; and on a long
# run the buffer wraps, so early messages are gone. A zero here means "nothing
# dmesg can still see", not "nothing happened". Use scripts/fault_count.sh in
# new work.
set -u
D=~/inv89; mkdir -p "$D"
L=${ROCBLAS_LIB_DIR:-/home/akandr/rocBLAS/build/release/rocblas-install/lib}
HIP=~/llama-master/build-hip/bin
Q15=/opt/models/qwen2.5-1.5b-q4km.gguf
WIKI=~/wiki.test.raw
HOURS=${1:-8}
REF=8.9442
log () { echo "[$(date +%H:%M:%S)] $*" | tee -a "$D/log"; sync; }
temp () { sensors 2>/dev/null | grep -m1 edge | grep -oE "[0-9]+\.[0-9]" | head -1; }

end=$(( $(date +%s) + HOURS*3600 ))
round=0
log "soak start, ${HOURS}h, alternating SDMA, reference ppl $REF, kernel $(uname -r)"
while [ "$(date +%s)" -lt "$end" ]; do
  round=$((round+1))
  s=$(( (round + 1) % 2 ))   # alternate 1,0,1,0...
  E=(env HSA_ENABLE_SDMA=$s GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32 LD_LIBRARY_PATH=$L)

  "${E[@]}" timeout -k 20 900 "$HIP/llama-bench" -m "$Q15" -ngl 99 -fa 1 -p 2048 -n 0 -r 3 > "$D/pp_$round.log" 2>&1
  pp=$(grep -aoE "pp2048 \| +[0-9]+\.[0-9]+" "$D/pp_$round.log" | grep -oE "[0-9]+\.[0-9]+$")

  "${E[@]}" timeout -k 30 1800 "$HIP/llama-perplexity" -m "$Q15" --no-mmap -ngl 99 -fa on \
    -c 4096 -f "$WIKI" --chunks 8 > "$D/ppl_$round.log" 2>&1
  ppl=$(grep -aoE "Final estimate: PPL = [0-9.]+" "$D/ppl_$round.log" | grep -oE "[0-9.]+$")

  "${E[@]}" timeout -k 20 1800 "$HIP/test-backend-ops" -o MUL_MAT > "$D/churn_$round.log" 2>&1
  crc=$?

  # count runtime-reported faults too, not only dmesg: dmesg misses this class
  rtf=$(grep -lac "Memory access fault" "$D"/*_$round.log 2>/dev/null | wc -l)
  dmf=$(sudo dmesg | grep -ciE "memory access fault|preemption time out")
  log "round=$round sdma=$s pp2048=${pp:-FAIL} ppl=${ppl:-FAIL} churn_rc=$crc runtime_faults=$rtf dmesg_faults=$dmf tmax=$(temp)"
done
log "=== summary over $round rounds"
log "  distinct ppl values: $(grep -oE "ppl=[0-9.]+" "$D/log" | sort -u | tr "\n" " ")"
log "  sdma=1 rounds: $(grep -c "sdma=1" "$D/log"), sdma=0 rounds: $(grep -c "sdma=0" "$D/log")"
log "  churn failures: $(grep -c "churn_rc=[^0]" "$D/log")"
log "  rounds with runtime-reported faults: $(grep -vc "runtime_faults=0" "$D/log")"
touch "$D/DONE"; log done
