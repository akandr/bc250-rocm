#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 Artur Andrzejczak <andrzejczak.artur@gmail.com>
# Assisted-by: Claude (Anthropic)
# Part D: the 8B's decode ladder carried past 24576, to the rungs its 32768 f16-cache limit on ROCm still allows:
# tg64 behind 28672, 30720 and 32704 cached tokens (the last one 32768 in context exactly). The same three arms as
# logs/ladders-2026-10-15 (ROCm as shipped, ROCm with GGML_CUDA_GRAPH_OPT=1 and its fix, Vulkan), the no-depth-cache
# llama-bench linked to each build, -r 1 after IDLE seconds (default 180, as in that run's phase 2), arms interleaved
# within each point, three passes. 32704 comes last, after every pass of the other two, because a run that does not
# fit may abort and an abort leaks the mappings until the next reboot; the script stops at the first failure.
set -u
O=${1:-$HOME/ladders-rest-2026-10-18-d}
IDLE=${IDLE:-180}
NS_FIX=$HOME/llama-bench-nosnap-graphopt/hip; mkdir -p "$O"
L=/opt/bc250-rocm/lib64
HIP=$HOME/llama-master/build-hip-pkf16/bin
VK=$HOME/llama-master/build-vk-f44/bin
VKPR=$HOME/llama-vk28507/build-vk-patched/bin
NS_HIP=$HOME/llama-bench-nosnap/hip
NS_VK=$HOME/llama-bench-nosnap/vk
M8=/opt/models/qwen3-8b-q8_0.gguf
declare -A MODEL=( [ds14]=/opt/models/deepseek-r1-14b.gguf [q14]=/opt/models/qwen3-14b.gguf
                   [moe]=/opt/models/qwen3.6-35b-a3b-iq2m.gguf [b27]=/opt/models/qwen3.8-27b-iq3xxs.gguf )
export HSA_ENABLE_SDMA=0
PAT="page fault \(src_id|memory access fault|preemption time out|Queue preemption failed|runlist rebuild flush failed|GPU reset begin"
faults () { sudo journalctl -b 0 -k --no-pager 2>/dev/null | grep -ciE "$PAT"; }
thr () { sudo journalctl -b 0 -u oberon-governor --no-pager 2>/dev/null | grep -ci overheat; }
log () { echo "[$(date +%H:%M:%S)] $*" | tee -a "$O/log"; sync; }
temp () { sensors 2>/dev/null | awk '/edge/{print $2}'; }
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
for u in ollama hw-watcher.timer crond dnf-makecache.timer sysstat-collect.timer plocate-updatedb.timer; do
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


run () {  # $1 pass, $2 depth, $3 arm
  local b e
  case $3 in hip) b=$NS_HIP; e="LLAMA_BENCH_NO_DEPTH_CACHE=1 GGML_CUDA_GRAPH_OPT=0";;
             fix) b=$NS_FIX; e="LLAMA_BENCH_NO_DEPTH_CACHE=1 GGML_CUDA_GRAPH_OPT=1 GGML_CUDA_GRAPH_OPT_ALLOC_DEPS=1";;
             vk)  b=$NS_VK;  e="LLAMA_BENCH_NO_DEPTH_CACHE=1";; esac
  sleep $IDLE
  log "D p$1 d=$2 r1 $3 start edge=$(temp) avail=$(free -m | awk '/^Mem:/{print $7}')"
  local r; r=$(one "$b" "$e" "$M8" D_p$1_d$2_$3 -p 0 -n 64 -d $2 -r 1)
  log "D p$1 d=$2 r1 $3 tg64=$r edge=$(temp)"
  case "$r" in FAIL*)
    log "D p$1 $3 err: $(grep -a -v -i 'terminal\|load-mode' "$O/err_D_p$1_d$2_$3" | tail -c 300 | tr '\n' ' ')"
    log "FAILED RUN, stopping (avail now $(free -m | awk '/^Mem:/{print $7}') MiB)"
    touch "$O/DONE"; exit 1;;
  esac
  check "D p$1 d=$2 $3"
}
log "idle $IDLE s"
for p in 1 2 3; do
  for d in 28672 30720; do
    for a in hip fix vk; do run $p $d $a; done
  done
done
log "D1 done; faults $(faults)"
for p in 1 2 3; do
  for a in hip fix vk; do run $p 32704 $a; done
done
log "faults at end: $(faults)"
touch "$O/DONE"; log DONE
