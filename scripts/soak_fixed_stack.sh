#!/usr/bin/env bash
# soak_fixed_stack.sh - endurance soak on the fixed stack.
# Alternates prefill-heavy llama-bench rounds with perplexity gates for HOURS
# hours (default 8), logging temperature and sclk every 20 s. Each round runs
# prefill, a perplexity gate, and an allocation-churn sweep (the workload the
# SVM-side flush fixes), so it tests durability of that fix too. The 94C thermal
# ceiling in the README was measured at pre-fix speeds (124 t/s prefill);
# 892 t/s works the chip much harder.
set -u
HOURS=${1:-8}
D=~/soak-2026-08-12; mkdir -p $D
HIP=~/llama-master/build-hip/bin
TBO=~/llama-master/build-hip/bin/test-backend-ops
Q15=/opt/models/qwen2.5-1.5b-q4km.gguf
WIKI=~/wiki.test.raw
export HSA_ENABLE_SDMA=0
export LD_LIBRARY_PATH=$HOME/rocBLAS/build/release/rocblas-install/lib
export GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32
END=$(( $(date +%s) + HOURS*3600 ))

( while true; do
    t=$(sensors 2>/dev/null | grep -m1 edge | grep -oE '[0-9]+\.[0-9]' | head -1)
    c=$(cat /sys/class/drm/card*/device/pp_dpm_sclk 2>/dev/null | grep '\*' | head -1)
    echo -e "$(date +%s)\t${t:-NA}\t${c:-NA}" >> $D/thermals.tsv
    sleep 20
  done ) & THERM=$!
trap 'kill $THERM 2>/dev/null' EXIT

round=0
while [ $(date +%s) -lt $END ]; do
  round=$((round+1))
  # 1. prefill hammer: ~10 min of sustained pp2048 (fa on)
  timeout -k 20 900 $HIP/llama-bench -m $Q15 -mmp 0 -ngl 99 -fa on \
    -p 2048 -n 0 -r 20 > $D/r${round}_pp.log 2>&1
  rc1=$?
  pp=$(grep -oE 'pp2048 \|[^|]+' $D/r${round}_pp.log | grep -oE '[0-9]+\.[0-9]+' | head -1)
  # 2. correctness gate: ppl chunks 2 ctx 4096 (expect 11.06; drift = corruption)
  timeout -k 20 600 $HIP/llama-perplexity -m $Q15 --no-mmap -ngl 99 -fa on \
    -c 4096 -f $WIKI --chunks 2 > $D/r${round}_ppl.log 2>&1
  rc2=$?
  # NB: the log line is prefixed with a dotted timestamp, so anchor on the
  #     "PPL = " text or the parse silently returns the timestamp instead.
  ppl=$(grep -oE 'Final estimate: PPL = [0-9]+\.[0-9]+' $D/r${round}_ppl.log | grep -oE '[0-9]+\.[0-9]+$')
  # 3. allocation-churn round: the workload the SVM-side flush fixes, so the
  #    soak tests durability of that fix and not only sustained prefill
  timeout -k 10 900 $TBO perf -o MUL_MAT -b ROCm0 > $D/r${round}_churn.log 2>&1
  churn_done=$(grep -c "backends passed" $D/r${round}_churn.log)
  churn_fault=$(grep -ac "Memory access fault" $D/r${round}_churn.log)
  tmax=$(sort -t$'\t' -k2 -g $D/thermals.tsv | tail -1 | cut -f2)
  echo "[$(date +%H:%M:%S)] round=$round pp2048=${pp:-FAIL} ppl=${ppl:-FAIL} churn_complete=$churn_done churn_faults=$churn_fault tmax=$tmax" \
    | tee -a $D/soak.log
  if [ "$churn_done" != "1" ] || [ "$churn_fault" != "0" ]; then
    echo "CHURN REGRESSED at round $round" | tee -a $D/soak.log; break
  fi
  sync
  # abort the soak if the gate breaks badly (don't hammer a corrupted state)
  if [ -n "${ppl:-}" ]; then
    bad=$(echo "$ppl > 11.5 || $ppl < 10.5" | bc -l 2>/dev/null || echo 0)
    [ "$bad" = "1" ] && { echo "PPL GATE BROKE at round $round" | tee -a $D/soak.log; break; }
  fi
done
echo done > $D/DONE
echo "[$(date +%H:%M:%S)] SOAK DONE after $round rounds" | tee -a $D/soak.log
