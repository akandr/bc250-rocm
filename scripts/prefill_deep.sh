#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Artur Andrzejczak <andrzejczak.artur@gmail.com>
# Assisted-by: Claude (Anthropic)
# qwen3-8B Q8_0 prefill (pp2048) behind 16384 and 24576 cached tokens, with 8192 as the control point
# that logs/depth-thirteen-2026-09-22 and logs/vulkan-fa-staging-2026-09-17 already have. Three arms,
# interleaved within each depth: ROCm (build-hip-pkf16), Vulkan (build-vk-f44) and Vulkan with llama.cpp
# PR #28507 (llama-vk28507/build-vk-patched, the staging build). One invocation per point, -r 3, three
# passes, 25 s cool-down between points, the shader clock sampled during every run.
set -u
O=${1:-$HOME/prefill-deep}; mkdir -p "$O"
L=/opt/bc250-rocm/lib64
HIP=$HOME/llama-master/build-hip-pkf16/bin
VK=$HOME/llama-master/build-vk-f44/bin
VKPR=$HOME/llama-vk28507/build-vk-patched/bin
M=/opt/models/qwen3-8b-q8_0.gguf
export HSA_ENABLE_SDMA=0
PAT="page fault \(src_id|memory access fault|preemption time out|Queue preemption failed|runlist rebuild flush failed|GPU reset begin"
faults () { sudo journalctl -b 0 -k --no-pager 2>/dev/null | grep -ciE "$PAT"; }
thr () { sudo journalctl -b 0 -u oberon-governor --no-pager 2>/dev/null | grep -ci overheat; }
log () { echo "[$(date +%H:%M:%S)] $*" | tee -a "$O/log"; sync; }
temp () { sensors 2>/dev/null | awk '/edge/{print $2}'; }
cool () { sleep 25; }
# The PR build is master bfdc321, newer than 7ba604f, and its llama-bench no longer takes -mmp, so that
# arm loads the model the default way, as the staging run did.
one () {  # $1 bin dir, $2 tag, rest llama-bench args; prints the pp2048 figure and the clocks seen
  ( while :; do grep -h '\*' /sys/class/drm/card*/device/pp_dpm_sclk | tr -d ' *'; sleep 2; done ) > "$O/clk_$2" 2>/dev/null &
  local sp=$!
  local mm="-mmp 0"; case "$1" in "$VKPR") mm="";; esac
  LD_LIBRARY_PATH=$L timeout -k 30 3600 "$1/llama-bench" -m "$M" $mm -ngl 99 -fa on -p 2048 -n 0 -r 3 "${@:3}" \
    > "$O/raw_$2" 2>"$O/err_$2"
  kill $sp 2>/dev/null; wait $sp 2>/dev/null
  local v; v=$(grep -a pp2048 "$O/raw_$2" | awk -F'|' '{v=$(NF-1); gsub(/ /,"",v); print v}')
  echo "pp2048=${v:-FAIL} clk=$(sort "$O/clk_$2" | uniq -c | awk '{printf "%s:%s ", $2, $1}')"
}
# quiet the board for the window: ollama (GPU), the hardware watcher timer and cron; restored on exit
STOPPED=""
for u in ollama hw-watcher.timer crond; do
  if [ "$(systemctl is-active $u 2>/dev/null)" = active ]; then sudo systemctl stop $u; STOPPED="$STOPPED $u"; fi
done
restore () { for u in $STOPPED; do sudo systemctl start $u; done; log "restored:$STOPPED"; }
trap restore EXIT
sleep 10
B=$(faults); THR0=$(thr)
log "start: stopped$STOPPED; faults $B; edge $(temp); avail $(free -m | awk '/^Mem:/{print $7}') MiB; kfd holders: $(sudo fuser /dev/kfd 2>&1 | tail -1)"
log "cmdline: $(cat /proc/cmdline)"
log "oberon: $(systemctl is-active oberon-governor), $(grep -E 'max' /etc/oberon-config.yaml | tr -d '\n' | tr -s ' ')"
for p in 1 2 3; do
  for d in 8192 16384 24576; do
    for a in hip vk vkpr; do
      case $a in hip) b=$HIP;; vk) b=$VK;; vkpr) b=$VKPR;; esac
      log "p$p d=$d $a $(one $b p${p}_d${d}_$a -d $d) edge=$(temp)"
      cool
    done
    n=$(faults); [ "$n" -gt "$B" ] && { log "FAULTED at p$p d=$d, stopping"; touch "$O/DONE"; exit 1; }
    t=$(thr); [ "$t" -gt "$THR0" ] && log "GOVERNOR THROTTLED at p$p d=$d, points from here are suspect"
  done
  log "pass $p done, edge $(temp)"
done
log "faults at end: $(faults)"
touch "$O/DONE"; log DONE
