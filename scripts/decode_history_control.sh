#!/usr/bin/env bash
# Does the 8B decode rate depend on what ran before it?
#
# Within one boot the rate varies by about 10 percent (16.2 to 17.9 over ten
# runs). Across boots it has been seen from 12.9 to 18.3, a 42 percent spread.
# That gap says the variable is not run-to-run noise but something that differs
# between boots or accumulates during one.
#
# The obvious candidate is memory state: this is a 16 GiB shared board, the
# model is 8.24 GiB, and the historical low readings were all taken in
# campaigns that had just run other large models. This measures the same
# workload in three conditions within one boot.
set -u
D=~/decode-history; mkdir -p "$D"
HIP=~/llama-master/build-hip/bin
E=(env HSA_ENABLE_SDMA=0 GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32
   LD_LIBRARY_PATH=/home/akandr/rocBLAS/build/release/rocblas-install/lib)
M8=/opt/models/qwen3-8b-q8_0.gguf
MOE=/opt/models/qwen3.6-35b-a3b-iq2m.gguf

log () { echo "[$(date +%H:%M:%S)] $*" | tee -a "$D/log"; sync; }
measure () { # measure <tag>
  local tag=$1
  local fm; fm=$(free -m | awk '/^Mem:/{print $7}')
  "${E[@]}" timeout -k 20 1800 "$HIP/llama-bench" -m "$M8" -ngl 99 -fa 1 -p 0 -n 8 -d 16128 -r 2 \
    > "$D/$tag.log" 2>&1
  local r; r=$(grep -aoE "tg8 @ d[0-9]+ \| +[0-9]+\.[0-9]+" "$D/$tag.log" | grep -oE "[0-9]+\.[0-9]+$")
  log "  $tag: rate=${r:-FAIL} free_before=${fm}MiB"
}

log "=== A: clean board, nothing large has run this boot"
measure A_clean_1
measure A_clean_2

log "=== B: immediately after loading and running a 10.7 GiB MoE"
"${E[@]}" timeout -k 20 1800 "$HIP/llama-bench" -m "$MOE" -ngl 99 -fa 1 -p 0 -n 8 -r 1 \
  > "$D/moe_warmup.log" 2>&1
log "  (MoE run: $(grep -aoE 'tg8 \| +[0-9.]+' "$D/moe_warmup.log" | tail -1))"
measure B_after_moe_1
measure B_after_moe_2

log "=== C: after dropping caches and compacting memory"
sync; echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null 2>&1
echo 1 | sudo tee /proc/sys/vm/compact_memory > /dev/null 2>&1
sleep 5
measure C_after_drop_1
measure C_after_drop_2

log "=== summary"
grep -oE "rate=[0-9.]+" "$D/log" | cut -d= -f2 | sort -n | \
  awk '{a[NR]=$1} END {printf "  min=%.2f max=%.2f spread=%.0f%%\n", a[1], a[NR], (a[NR]-a[1])/a[1]*100}' | tee -a "$D/log"
touch "$D/DONE"; log done
