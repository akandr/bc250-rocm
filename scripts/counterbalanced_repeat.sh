#!/usr/bin/env bash
# Counterbalanced replacements for two comparisons run earlier today.
#
# Both were run as blocks: arm A ten times, then arm B ten times; and model one
# ten times, then model two, then model three. Anything that drifts over the
# ninety minutes a block takes is therefore indistinguishable from the variable
# under test, and in the model comparison the 14B, whose high variance carried
# the whole conclusion, ran last. That is the same failure as carrying a
# workaround into an experiment untested, so it gets the same treatment.
#
# Part A alternates the graph-capture arms in ABBA order, which cancels linear
# drift. Part B rotates model order every round. Clock and temperature are
# sampled per run so drift is visible rather than assumed absent.
set -u
D=~/inv69; mkdir -p "$D"
HIP=~/llama-master/build-hip/bin
L=/home/akandr/rocBLAS/build/release/rocblas-install/lib
Q15=/opt/models/qwen2.5-1.5b-q4km.gguf
M8=/opt/models/qwen3-8b-q8_0.gguf
Q14=/opt/models/qwen3-14b.gguf
E=(env HSA_ENABLE_SDMA=0 GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32 LD_LIBRARY_PATH=$L)
log () { echo "[$(date +%H:%M:%S)] $*" | tee -a "$D/log"; sync; }
temp () { sensors 2>/dev/null | grep -m1 edge | grep -oE "[0-9]+\.[0-9]" | head -1; }
clk  () { grep "\*" /sys/class/drm/card*/device/pp_dpm_sclk 2>/dev/null | head -1 | grep -oE "[0-9]+Mhz" | head -1; }

log "waiting for the previous job"
while [ ! -f ~/inv68/DONE ]; do sleep 60; done

one () { # one <tag> <model> <depth> <extra env...>
  local tag=$1 m=$2 depth=$3; shift 3
  local t0; t0=$(temp)
  "${E[@]}" "$@" timeout -k 20 1800 $HIP/llama-bench -m "$m" -ngl 99 -fa 1 -p 0 -n 8 -d "$depth" -r 1 \
    > "$D/${tag}.log" 2>&1
  local v; v=$(grep -aoE "tg8( @ d[0-9]+)? \| +[0-9]+\.[0-9]+" "$D/${tag}.log" | grep -oE "[0-9.]+$")
  log "  $tag: ${v:-FAIL}  temp_before=${t0}C clk=$(clk)"
}

log "=== PART A: graph capture on and off, ABBA order, depth 16128"
for r in 1 2 3; do
  if [ $((r % 2)) -eq 1 ]; then order="off on on off"; else order="on off off on"; fi
  i=0
  for a in $order; do
    i=$((i+1))
    if [ "$a" = off ]; then one "A_r${r}_${i}_graphsoff" "$M8" 16128 GGML_CUDA_DISABLE_GRAPHS=1
    else one "A_r${r}_${i}_graphson" "$M8" 16128; fi
  done
done

log "=== PART B: three models at depth 8192, graph capture on, model order rotated"
for r in 1 2 3 4 5 6; do
  case $((r % 3)) in
    1) seq_="q15 q8b q14" ;;
    2) seq_="q8b q14 q15" ;;
    0) seq_="q14 q15 q8b" ;;
  esac
  for m in $seq_; do
    case $m in q15) f=$Q15 ;; q8b) f=$M8 ;; q14) f=$Q14 ;; esac
    one "B_r${r}_${m}" "$f" 8192
  done
done
log "=== dmesg faults: $(sudo dmesg | grep -ciE \"memory access fault|preemption time out\")"
touch "$D/DONE"; log done
