#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 Artur Andrzejczak <andrzejczak.artur@gmail.com>
# Assisted-by: Claude (Anthropic)
# The two gaps left by the October ladders. First, the 8B's prefill one rung past 24576: qwen3-8B Q8_0 pp2048
# behind 28672 cached tokens (30720 in context, the deepest pp2048 under its 32768 f16-cache limit on ROCm),
# the same three arms as logs/prefill-deep-2026-10-12: ROCm build-hip-pkf16, Vulkan build-vk-f44 and Vulkan
# with PR #28507 (llama-vk28507/build-vk-patched, no -mmp there), stock llama-bench, -r 1 after three idle
# minutes, three passes. If an arm's stock llama-bench fails at that depth (it pins a copy of the KV cache
# after the fill), the arm is run again with the llama-bench of llama-bench-no-depth-cache.patch and tagged ns.
# Second, decode against depth on the four models the decode ladders did not have: deepseek-r1-14B Q4_K_M,
# qwen3-14B Q4_K_M, qwen3.6-35B-A3B MoE IQ2_M and qwen3.8-27B UD-IQ3_XXS, tg64 behind 4096, 8192, 16384 and
# 24576 cached tokens, ROCm as shipped against Vulkan, the no-depth-cache llama-bench for both (the stock one
# would not fit the 27B's 25600-token ROCm limit with its pinned copy), -r 1 after 90 idle seconds, arms
# interleaved within each point, three passes, the shader clock sampled during every run.
set -u
O=${1:-$HOME/ladders-rest-2026-10-17}; mkdir -p "$O"
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

# A: the 8B's prefill behind 28672 cached tokens
for p in 1 2 3; do
  for a in hip vk vkpr; do
    case $a in hip) b=$HIP;; vk) b=$VK;; vkpr) b=$VKPR;; esac
    sleep 180
    log "A p$p d=28672 r1 $a start edge=$(temp)"
    r=$(one "$b" "X=1" "$M8" A_p${p}_d28672_$a -p 2048 -n 0 -d 28672 -r 1)
    log "A p$p d=28672 r1 $a pp2048=$r edge=$(temp)"
    check "A p$p $a"
    case "$r" in FAIL*)
      log "A p$p $a err: $(tail -c 300 "$O/err_A_p${p}_d28672_$a" | tr '\n' ' ')"
      nb=""; case $a in hip) nb=$NS_HIP;; vk) nb=$NS_VK;; esac
      if [ -n "$nb" ]; then
        sleep 180
        log "A p$p d=28672 r1 ${a}ns pp2048=$(one "$nb" "LLAMA_BENCH_NO_DEPTH_CACHE=1" "$M8" A_p${p}_d28672_${a}ns -p 2048 -n 0 -d 28672 -r 1) edge=$(temp)"
        check "A p$p ${a}ns"
      fi;;
    esac
  done
done
log "A done; faults $(faults)"

# B: decode against depth on the other four models
for p in 1 2 3; do
  for m in ds14 q14 moe b27; do
    for d in 4096 8192 16384 24576; do
      for a in hip vk; do
        case $a in hip) b=$NS_HIP; e="LLAMA_BENCH_NO_DEPTH_CACHE=1 GGML_CUDA_GRAPH_OPT=0";;
                   vk)  b=$NS_VK;  e="LLAMA_BENCH_NO_DEPTH_CACHE=1";; esac
        sleep 90
        log "B p$p $m d=$d r1 $a start edge=$(temp)"
        r=$(one "$b" "$e" "${MODEL[$m]}" B_p${p}_${m}_d${d}_$a -p 0 -n 64 -d $d -r 1)
        log "B p$p $m d=$d r1 $a tg64=$r edge=$(temp)"
        case "$r" in FAIL*) log "B p$p $m $a err: $(tail -c 300 "$O/err_B_p${p}_${m}_d${d}_$a" | tr '\n' ' ')";; esac
        check "B p$p $m d=$d $a"
      done
    done
  done
done
log "faults at end: $(faults)"
touch "$O/DONE"; log DONE
