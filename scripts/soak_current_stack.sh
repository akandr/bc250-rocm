#!/usr/bin/env bash
# Endurance soak on the CURRENT stack.
#
# The two soaks in this repository predate several things now in the recipe:
# the corrected patch scripts, the native PyTorch build, the map-side flush as
# shipped, and the llama.cpp patch set as it now stands. They remain valid for
# what they measured, but nothing has run for hours on the configuration a
# reader would actually build today.
#
# Each round: a prefill benchmark, a perplexity gate against the reference, an
# allocation-churn sweep, and (every third round) a PyTorch training loop, so
# the soak covers the paths that were added since the earlier ones.
set -u
D=~/soak-current; mkdir -p "$D"
HIP=~/llama-master/build-hip/bin
E=(env HSA_ENABLE_SDMA=0 GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32
   LD_LIBRARY_PATH=/home/akandr/rocBLAS/build/release/rocblas-install/lib)
Q15=/opt/models/qwen2.5-1.5b-q4km.gguf
WIKI=~/wiki.test.raw
HOURS=${1:-8}
REF=8.9442

log () { echo "[$(date +%H:%M:%S)] $*" | tee -a "$D/log"; sync; }
temp () { sensors 2>/dev/null | grep -oE 'edge:.*\+[0-9.]+' | grep -oE '[0-9.]+$' | head -1; }

end=$(( $(date +%s) + HOURS*3600 ))
round=0
log "soak start, target ${HOURS}h, reference ppl $REF"

while [ "$(date +%s)" -lt "$end" ]; do
  round=$((round+1))
  # 1. prefill
  "${E[@]}" timeout -k 20 900 "$HIP/llama-bench" -m "$Q15" -ngl 99 -fa 1 -p 2048 -n 0 -r 3 \
    > "$D/pp_$round.log" 2>&1
  pp=$(grep -aoE "pp2048 \| +[0-9]+\.[0-9]+" "$D/pp_$round.log" | grep -oE "[0-9]+\.[0-9]+$")

  # 2. perplexity gate
  "${E[@]}" timeout -k 30 1800 "$HIP/llama-perplexity" -m "$Q15" --no-mmap -ngl 99 -fa on \
    -c 4096 -f "$WIKI" --chunks 8 > "$D/ppl_$round.log" 2>&1
  ppl=$(grep -aoE "Final estimate: PPL = [0-9.]+" "$D/ppl_$round.log" | grep -oE "[0-9.]+$")

  # 3. allocation churn
  "${E[@]}" timeout -k 20 1800 "$HIP/test-backend-ops" perf -o MUL_MAT -b ROCm0 > "$D/churn_$round.log" 2>&1
  crc=$?

  # 4. PyTorch training every third round, the path the older soaks never covered
  tr="skipped"
  if [ $((round % 3)) -eq 0 ]; then
    tr=$(env HSA_ENABLE_SDMA=0 LD_LIBRARY_PATH=/home/akandr/rocBLAS/build/release/rocblas-install/lib \
         ~/torchnative/bin/python ~/torch_train.py 2>/dev/null | grep -oE "last loss [0-9.]+" | head -1)
    tr="${tr:-FAILED}"
  fi

  faults=$(sudo dmesg | grep -ciE "memory access fault|preemption time out")
  log "round=$round pp2048=${pp:-FAIL} ppl=${ppl:-FAIL} churn_rc=$crc torch='${tr}' tmax=$(temp) faults=$faults"
done

log "=== summary over $round rounds"
log "  distinct ppl values: $(grep -oE 'ppl=[0-9.]+' "$D/log" | sort -u | tr '\n' ' ')"
log "  prefill range: $(grep -oE 'pp2048=[0-9.]+' "$D/log" | cut -d= -f2 | sort -n | sed -n '1p;$p' | tr '\n' ' ')"
log "  churn failures: $(grep -c 'churn_rc=[^0]' "$D/log")"
touch "$D/DONE"; log done
