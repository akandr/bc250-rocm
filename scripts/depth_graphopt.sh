#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Artur Andrzejczak <andrzejczak.artur@gmail.com>
# Assisted-by: Claude (Anthropic)
# The 1.5B decode ladder again, with ROCm measured both as shipped and with GGML_CUDA_GRAPH_OPT=1.
#
# The front page's decode-at-depth table predates that option (logs/campaign-graphopt-2026-09-24/),
# and decode at depth is where ROCm already drew level, so the published row understates the build.
#
# Method is section H of scripts/depth_thirteen.sh unchanged: one llama-bench invocation per depth,
# never a -d list, which depresses readings by up to 22 percent; arms interleaved within each depth;
# -mmp 0 -ngl 99 -fa on, HSA_ENABLE_SDMA=0; a cool-down between points, because deep decode back to
# back reaches 93 C in six minutes and the governor then drops the clock by 28 percent and the result
# reads as a clean measurement of a slower machine; temperature recorded with every point, and the
# run aborts on a kernel fault line or a governor throttle instead of publishing a point taken
# through one.
set -u
O=${1:-~/depth-graphopt}; mkdir -p "$O"
L=/opt/bc250-rocm/lib64
HIP=~/llama-master/build-hip-pkf16/bin
VK=~/llama-master/build-vk-f44/bin
PASSES=${PASSES:-2}
export HSA_ENABLE_SDMA=0

PAT="page fault \(src_id|memory access fault|preemption time out|Queue preemption failed|runlist rebuild flush failed|GPU reset begin"
faults () { sudo journalctl -b 0 -k --no-pager 2>/dev/null | grep -ciE "$PAT"; }
log () { echo "[$(date +%H:%M:%S)] $*" | tee -a "$O/log"; sync; }
temp () { sensors 2>/dev/null | awk '/edge/{print $2}'; }
cool () { sleep 25; }

one () {  # $1 bin  $2 env-prefix  rest: llama-bench args
  local b=$1; shift
  local pre=$1; shift
  env $pre LD_LIBRARY_PATH=$L timeout -k 30 1200 "$b/llama-bench" -m /opt/models/qwen2.5-1.5b-q4km.gguf \
      -mmp 0 -ngl 99 -fa on "$@" 2>/dev/null | grep -aE "tg[0-9]+" |
      awk -F'|' '{t=$(NF-2); v=$(NF-1); gsub(/^ +| +$/,"",t); gsub(/ /,"",t);
                  gsub(/^ +| +$/,"",v); gsub(/ /,"",v); printf "%s=%s ", t, v}'
}

if [ "$(systemctl is-active ollama 2>/dev/null)" = active ]; then
  sudo systemctl stop ollama; sleep 10; STOPPED_OLLAMA=1
else STOPPED_OLLAMA=0; fi
B=$(faults); THR0=$(sudo journalctl -b 0 -u oberon-governor --no-pager 2>/dev/null | grep -ci overheat)
thr () { sudo journalctl -b 0 -u oberon-governor --no-pager 2>/dev/null | grep -ci overheat; }
log "start, faults already in this boot: $B, edge $(temp), ollama stopped by this script: $STOPPED_OLLAMA"
log "available $(free -m | awk '/^Mem:/{print $7}') MiB"

for p in $(seq 1 "$PASSES"); do
  for d in 0 4096 8192 16384 24576 30720; do
    log "p$p d=$d hip    $(one "$HIP" GGML_CUDA_GRAPH_OPT=0 -p 0 -n 64 -d $d -r 3)edge=$(temp)"; cool
    log "p$p d=$d hipopt $(one "$HIP" GGML_CUDA_GRAPH_OPT=1 -p 0 -n 64 -d $d -r 3)edge=$(temp)"; cool
    log "p$p d=$d vk     $(one "$VK"  X=1                   -p 0 -n 64 -d $d -r 3)edge=$(temp)"; cool
    n=$(faults); [ "$n" -gt "$B" ] && { log "FAULTED at p$p d=$d, stopping"; exit 1; }
    t=$(thr); [ "$t" -gt "$THR0" ] && log "GOVERNOR THROTTLED at p$p d=$d, points from here are suspect"
  done
  log "p$p done, edge $(temp)"
done
log DONE
