#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 Artur Andrzejczak <andrzejczak.artur@gmail.com>
# Assisted-by: Claude (Anthropic)
# Phase 2 of ladders_8b_15b.sh: the 8B decode points that throttled, 16384 and 24576, once more with -r 1 and
# three minutes of idle before each arm, from a cold board, to separate the fill's own heat from accumulation.
# Original header follows.
# The two ladders the front page lacked: decode on qwen3-8B Q8_0 against depth (the depth tables had it
# for the 1.5B only) and prefill on qwen2.5-1.5B Q4_K_M against depth (the prefill ladder had the 8B only).
# One llama-bench invocation per point, arms interleaved within each depth, three passes, 25 s of cooling
# between points, the shader clock sampled during every run. Decode arms: ROCm build-hip-pkf16 as shipped,
# the same patches with GGML_CUDA_GRAPH_OPT=1 and its fix (llama-graphopt build), and Vulkan build-vk-f44.
# Prefill arms: ROCm, Vulkan, and Vulkan with PR #28507 (llama-vk28507/build-vk-patched, no -mmp there).
# The decode ladder runs the llama-bench of llama-bench-no-depth-cache.patch (LLAMA_BENCH_NO_DEPTH_CACHE=1),
# linked to each build's own libraries: stock llama-bench pins a copy of the KV cache after the fill and the
# 8B then runs out of ROCm memory at 16384 (ggml-cuda.cu:111, ROCm error), as logs/context-ceilings-2026-09-30
# describes. The prefill ladder keeps stock llama-bench, like the 8B prefill ladder it pairs with.
set -u
O=${1:-$HOME/ladders-2026-10-15-phase2}; mkdir -p "$O"
L=/opt/bc250-rocm/lib64
HIP=$HOME/llama-master/build-hip-pkf16/bin
FIX=$HOME/llama-graphopt/build-hip-pkf16/bin
VK=$HOME/llama-master/build-vk-f44/bin
VKPR=$HOME/llama-vk28507/build-vk-patched/bin
NS_HIP=$HOME/llama-bench-nosnap/hip
NS_FIX=$HOME/llama-bench-nosnap-graphopt/hip
NS_VK=$HOME/llama-bench-nosnap/vk
M8=/opt/models/qwen3-8b-q8_0.gguf
M15=/opt/models/qwen2.5-1.5b-q4km.gguf
export HSA_ENABLE_SDMA=0
PAT="page fault \(src_id|memory access fault|preemption time out|Queue preemption failed|runlist rebuild flush failed|GPU reset begin"
faults () { sudo journalctl -b 0 -k --no-pager 2>/dev/null | grep -ciE "$PAT"; }
thr () { sudo journalctl -b 0 -u oberon-governor --no-pager 2>/dev/null | grep -ci overheat; }
log () { echo "[$(date +%H:%M:%S)] $*" | tee -a "$O/log"; sync; }
temp () { sensors 2>/dev/null | awk '/edge/{print $2}'; }
cool () { sleep 25; }
one () {  # $1 bin dir, $2 env prefix ("X=1" for none), $3 model, $4 tag, rest llama-bench args
  ( while :; do grep -h '\*' /sys/class/drm/card*/device/pp_dpm_sclk | tr -d ' *'; sleep 2; done ) > "$O/clk_$4" 2>/dev/null &
  local sp=$!
  local mm="-mmp 0"; case "$1" in "$VKPR") mm="";; esac
  env $2 LD_LIBRARY_PATH=$L timeout -k 30 3600 "$1/llama-bench" -m "$3" $mm -ngl 99 -fa on "${@:5}" \
    > "$O/raw_$4" 2>"$O/err_$4"
  kill $sp 2>/dev/null; wait $sp 2>/dev/null
  local v; v=$(grep -aE "pp2048|tg64" "$O/raw_$4" | awk -F'|' '{v=$(NF-1); gsub(/ /,"",v); print v}' | head -1)
  echo "${v:-FAIL} clk=$(sort "$O/clk_$4" | uniq -c | awk '{printf "%s:%s ", $2, $1}')"
}
STOPPED=""
for u in ollama hw-watcher.timer crond; do
  if [ "$(systemctl is-active $u 2>/dev/null)" = active ]; then sudo systemctl stop $u; STOPPED="$STOPPED $u"; fi
done
restore () { for u in $STOPPED; do sudo systemctl start $u; done; log "restored:$STOPPED"; }
trap restore EXIT
sleep 10
B=$(faults); THR0=$(thr)
log "start: stopped$STOPPED; faults $B; overheat lines $THR0; edge $(temp); avail $(free -m | awk '/^Mem:/{print $7}') MiB; kfd holders: $(sudo fuser /dev/kfd 2>&1 | tail -1)"
log "cmdline: $(cat /proc/cmdline)"
log "oberon: $(systemctl is-active oberon-governor), $(grep -E 'max' /etc/oberon-config.yaml | tr -d '\n' | tr -s ' ')"
check () { n=$(faults); [ "$n" -gt "$B" ] && { log "FAULTED at $1, stopping"; touch "$O/DONE"; exit 1; }
           t=$(thr); [ "$t" -gt "$THR0" ] && { log "GOVERNOR OVERHEAT LINE at $1 ($THR0 -> $t)"; THR0=$t; }; return 0; }
for p in 1 2 3; do
  for d in 16384 24576; do
    for a in hip fix vk; do
      case $a in hip) b=$NS_HIP; e="LLAMA_BENCH_NO_DEPTH_CACHE=1 GGML_CUDA_GRAPH_OPT=0";;
                 fix) b=$NS_FIX; e="LLAMA_BENCH_NO_DEPTH_CACHE=1 GGML_CUDA_GRAPH_OPT=1 GGML_CUDA_GRAPH_OPT_ALLOC_DEPS=1";;
                 vk)  b=$NS_VK;  e="LLAMA_BENCH_NO_DEPTH_CACHE=1";; esac
      sleep 180
      log "A2 p$p d=$d r1 $a start edge=$(temp)"
      log "A2 p$p d=$d r1 $a tg64=$(one "$b" "$e" "$M8" A2_p${p}_d${d}_$a -p 0 -n 64 -d $d -r 1) edge=$(temp)"
      check "A2 p$p d=$d $a"
    done
  done
done
log "faults at end: $(faults)"
touch "$O/DONE"; log DONE
