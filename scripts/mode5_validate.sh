#!/usr/bin/env bash
# Validate the PASID-filtered runlist rebuild (amdgpu.bc250_tlb_alt=5) against production.
#
# Round 2 of logs/tlb-alt-2026-09-15/ showed it surviving the 150 s churn discriminator
# three times of three. Before it can be called a fix it has to match production on
# everything production is trusted for, and it has to be better at the one thing it
# exists for: a rebuild filtered to one PASID should not preempt other processes.
#
#   1. churn discriminator, 600 s cap (the full August length)
#   2. perplexity gate, 1.5B, ctx 4096, 8 chunks: 8.9442 on 7ba604f
#   3. 8B pp512/tg64, ABBA: production, mode 5, mode 5, production
#   4. interference: 1.5B decode (tg512) running while the churn runs, ABBA
#   5. last, because it faults: SDMA matched-VMID invalidation with the ACK register
#      read back after each job
#
# Runs on the board, fresh boot, services stopped. Every line is synced.
#
# Fault counting here reads dmesg on a boot that is still up, and was left that way once the run
# was logged, since changing the counter would change what the log means. dmesg cannot see a
# fault from a run that ended by taking the board down; new work should use
# scripts/fault_count.sh, which reads the persistent journal.
set -u
D=~/s0915/mode5; mkdir -p "$D"
P=/sys/module/amdgpu/parameters
H=~/llama-master/build-hip/bin
L=/home/akandr/rocBLAS/build/release/rocblas-install/lib
Q15=/opt/models/qwen2.5-1.5b-q4km.gguf
M8=/opt/models/qwen3-8b-q8_0.gguf
export LD_LIBRARY_PATH=$L GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32
log () { echo "[$(date +%T)] $*" | tee -a "$D/log"; sync; }
arm () { # arm prod|m5
  echo 3 | sudo tee $P/bc250_flush_by_runlist >/dev/null
  if [ "$1" = m5 ]; then echo 5 | sudo tee $P/bc250_tlb_alt >/dev/null; else echo 0 | sudo tee $P/bc250_tlb_alt >/dev/null; fi
}
events () { sudo dmesg | grep -ciE 'page fault|preemption|create queue .* failed|ring .* timeout|runlist'; }

log "=== $(uname -r), uptime $(uptime -p), gpu_recovery=$(cat $P/gpu_recovery)"

log "=== 1. churn, mode 5, 600 s cap"
arm m5; e0=$(events)
env HSA_ENABLE_SDMA=0 timeout -k 10 600 $H/test-backend-ops perf -o MUL_MAT -b ROCm0 > $D/churn600.log 2>&1
log "  rc=$? faults=$(grep -ac 'Memory access fault' $D/churn600.log) lines=$(wc -l < $D/churn600.log) new_kernel_events=$(( $(events) - e0 ))"

log "=== 2. gate, mode 5"
timeout -k 30 1500 $H/llama-perplexity -m $Q15 --no-mmap -ngl 99 -fa on -c 4096 -f ~/wiki.test.raw --chunks 8 > $D/gate_m5.log 2>&1
log "  $(grep -aoE 'Final estimate: PPL = [0-9.]+' $D/gate_m5.log || echo FAIL)"

log "=== 3. 8B throughput ABBA"
for a in prod m5 m5 prod; do
  arm $a
  timeout -k 20 600 $H/llama-bench -m $M8 -mmp 0 -ngl 99 -fa on -p 512 -n 64 -r 2 > $D/bench8b_$a.$RANDOM.log 2>&1
  f=$(ls -t $D/bench8b_$a.*.log | head -1)
  log "  $a: pp512 $(grep -aoE 'pp512 \| +[0-9.]+' $f | grep -oE '[0-9.]+$') tg64 $(grep -aoE 'tg64 \| +[0-9.]+' $f | grep -oE '[0-9.]+$')"
done

log "=== 4. interference: 1.5B decode while churn runs, ABBA"
for a in prod m5 m5 prod; do
  arm $a
  env HSA_ENABLE_SDMA=0 timeout -k 10 200 $H/test-backend-ops perf -o MUL_MAT -b ROCm0 > $D/interf_churn_$a.log 2>&1 &
  cp=$!
  sleep 5
  timeout -k 20 180 $H/llama-bench -m $Q15 -ngl 99 -fa 1 -p 0 -n 512 -r 2 > $D/interf_tg_$a.$RANDOM.log 2>&1
  f=$(ls -t $D/interf_tg_$a.*.log | head -1)
  wait $cp; crc=$?
  log "  $a: tg512 during churn $(grep -aoE 'tg512 \| +[0-9.]+ ± [0-9.]+' $f | grep -oE '[0-9.]+ ± [0-9.]+$') churn_rc=$crc churn_faults=$(grep -ac 'Memory access fault' $D/interf_churn_$a.log)"
done
arm prod
timeout -k 20 180 $H/llama-bench -m $Q15 -ngl 99 -fa 1 -p 0 -n 512 -r 2 > $D/interf_tg_alone.log 2>&1
log "  alone (prod, no churn): tg512 $(grep -aoE 'tg512 \| +[0-9.]+ ± [0-9.]+' $D/interf_tg_alone.log | grep -oE '[0-9.]+ ± [0-9.]+$')"

log "=== 5. SDMA matched-VMID invalidation, ACK read back (expected to fault)"
echo 3 | sudo tee $P/bc250_flush_by_runlist >/dev/null; echo 4 | sudo tee $P/bc250_tlb_alt >/dev/null
echo 12 | sudo tee $P/bc250_tlb_dump >/dev/null
env HSA_ENABLE_SDMA=0 timeout -k 10 60 $H/test-backend-ops perf -o MUL_MAT -b ROCm0 > $D/sdma_ack.log 2>&1
log "  rc=$? faults=$(grep -ac 'Memory access fault' $D/sdma_ack.log)"
sudo dmesg | grep 'BC250TLBALT mode=4' | tail -12 > $D/sdma_ack_lines.txt
log "  ack lines: $(awk '{for(i=1;i<=NF;i++) if ($i ~ /^acked=/) print $i}' $D/sdma_ack_lines.txt | sort | uniq -c | tr '\n' ' ')"
arm prod; echo 0 | sudo tee $P/bc250_tlb_dump >/dev/null
log "=== done"; touch $D/DONE
