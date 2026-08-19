#!/usr/bin/env bash
# Why does the 8B decode rate vary by 40 percent at a fixed depth?
#
# Observed across boots: 12.9, 16.6, 18.3 t/s at d16128 and 22.6, 22.8, 23.9 at
# d7936. Nothing else in the campaign moves like that. The shader clock is the
# first suspect: this board idles at 1000 MHz and the governor is supposed to
# take it to 1500 under load, so a run served at the lower state would land
# about a third slow, which is the size of the effect.
#
# Each run records the clock sampled *during* it, not just before and after.
set -u
D=~/decode-variance; mkdir -p "$D"
HIP=~/llama-master/build-hip/bin
M=/opt/models/qwen3-8b-q8_0.gguf
DEPTH=${1:-16128}
N=${2:-10}

temp () {
  local t
  t=$(sensors 2>/dev/null | grep -oE 'edge:.*\+[0-9.]+' | grep -oE '[0-9.]+$' | head -1)
  echo "${t:-NA}"
}
clk () { cat /sys/class/drm/card*/device/pp_dpm_sclk 2>/dev/null | grep '\*' | grep -oE '[0-9]+Mhz' | tr -d 'Mhz'; }

{
  echo "depth=$DEPTH runs=$N model=$(basename $M) governor=$(systemctl is-active oberon-governor 2>/dev/null)"
  printf "%3s %6s %6s %8s %8s %14s\n" run temp0 clk0 freeMiB rate clk_during
} | tee "$D/log"

for i in $(seq 1 "$N"); do
  t0=$(temp); c0=$(clk); fm=$(free -m | awk '/^Mem:/{print $7}')
  ( while :; do clk; sleep 1; done > "$D/clk$i.txt" ) &
  SAMP=$!
  env HSA_ENABLE_SDMA=0 GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32 \
      LD_LIBRARY_PATH=/home/akandr/rocBLAS/build/release/rocblas-install/lib \
      timeout -k 20 1800 "$HIP/llama-bench" -m "$M" -ngl 99 -fa 1 -p 0 -n 8 -d "$DEPTH" -r 1 \
      > "$D/run$i.log" 2>&1
  kill "$SAMP" 2>/dev/null; wait "$SAMP" 2>/dev/null
  rate=$(grep -aoE "tg8 @ d[0-9]+ \| +[0-9]+\.[0-9]+" "$D/run$i.log" | grep -oE "[0-9]+\.[0-9]+$")
  lo=$(sort -n "$D/clk$i.txt" 2>/dev/null | head -1)
  hi=$(sort -n "$D/clk$i.txt" 2>/dev/null | tail -1)
  mode=$(sort "$D/clk$i.txt" 2>/dev/null | uniq -c | sort -rn | head -1 | awk '{print $2}')
  printf "%3d %6s %6s %8s %8s %14s\n" "$i" "$t0" "$c0" "$fm" "${rate:-FAIL}" "${lo:-NA}-${hi:-NA}(${mode:-NA})" | tee -a "$D/log"
  sleep 5
done

echo "--- rates:" | tee -a "$D/log"
awk 'NR>2 && $5 ~ /^[0-9]/ {print $5}' "$D/log" | sort -n | \
  awk '{a[NR]=$1; s+=$1} END {printf "  n=%d min=%.2f max=%.2f mean=%.2f spread=%.0f%%\n", NR, a[1], a[NR], s/NR, (a[NR]-a[1])/a[1]*100}' | tee -a "$D/log"
touch "$D/DONE"
