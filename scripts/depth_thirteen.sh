#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Artur Andrzejczak <andrzejczak.artur@gmail.com>
# Assisted-by: Claude (Anthropic)
# The depth tables again, with one llama-bench invocation per point.
#
# The first sweep passed depths as a list, -d 0,4096,8192,..., and that turns out to depress readings
# by up to 22 percent, unpredictably, on either backend: the 1.5B's Vulkan decode at 4096 reads 140 in
# a list of six depths and 179.5 in its own invocation, six passes each, and the 27B's ROCm decode at
# 4096 reads 12.1 in a list of two and 14.5 alone. The September tables gave each depth its own
# invocation, so they reproduce and the sweep did not. So does this.
set -u
O=${1:-~/depth13c}; mkdir -p "$O"
L=/opt/bc250-rocm/lib64
HIP=~/llama-master/build-hip-pkf16/bin
VK=~/llama-master/build-vk-f44/bin
export HSA_ENABLE_SDMA=0
PAT="page fault \(src_id|memory access fault|preemption time out|Queue preemption failed|runlist rebuild flush failed|GPU reset begin"
faults () { sudo journalctl -b 0 -k --no-pager 2>/dev/null | grep -ciE "$PAT"; }
log () { echo "[$(date +%H:%M:%S)] $*" | tee -a "$O/log"; sync; }
temp () { sensors 2>/dev/null | awk '/edge/{print $2}'; }
# Deep decode run back to back reaches 93 C in six minutes and the governor then drops the shader
# clock to 1000 MHz, which costs 28 percent and reads as a clean measurement of a slower machine
# (logs/fault-repro-2026-09-22). Cool between points and record the temperature with each one, so a
# throttled point can be identified afterwards instead of silently averaged in.
cool () { sleep 25; }
one () {
  LD_LIBRARY_PATH=$L timeout -k 30 1200 "$1/llama-bench" -m "/opt/models/$2.gguf" -mmp 0 -ngl 99 \
      -fa on "${@:3}" 2>/dev/null | grep -aE "pp[0-9]+|tg[0-9]+" |
      awk -F'|' '{t=$(NF-2); v=$(NF-1); gsub(/^ +| +$/,"",t); gsub(/ /,"",t);
                  gsub(/^ +| +$/,"",v); gsub(/ /,"",v); printf "%s=%s ", t, v}'
}
# Stop other GPU users first. A reboot restarts the ollama service, and with a model loaded behind it
# the board has 9.6 GiB free where the 27B needs 11: the first attempt at this sweep returned empty for
# every model except the 1.5B, because they could not be loaded at all. The front page has warned about
# this since September and it still caught me out, so the script now does it instead of relying on the
# operator remembering after a reboot.
if [ "$(systemctl is-active ollama 2>/dev/null)" = active ]; then
  sudo systemctl stop ollama; sleep 10; STOPPED_OLLAMA=1
else STOPPED_OLLAMA=0; fi
B=$(faults); log "start, faults already in this boot: $B, edge $(temp)"
log "ollama $(systemctl is-active ollama 2>/dev/null) (stopped by this script: $STOPPED_OLLAMA), \
available $(free -m | awk '/^Mem:/{print $7}') MiB, kfd holders: $(sudo fuser /dev/kfd 2>&1 | tail -1)"
THR0=$(sudo journalctl -b 0 -u oberon-governor --no-pager 2>/dev/null | grep -ci overheat)
thr () { sudo journalctl -b 0 -u oberon-governor --no-pager 2>/dev/null | grep -ci overheat; }
log "cmdline: $(cat /proc/cmdline)"

# H. the 1.5B decode ladder, the six depths the front page's table uses, backends interleaved per point
for p in 1 2 3; do
  for d in 0 4096 8192 16384 24576 30720; do
    for be in hip vk; do
      b=$HIP; [ $be = vk ] && b=$VK
      log "H p$p d=$d $be $(one $b qwen2.5-1.5b-q4km -p 0 -n 64 -d $d -r 3)edge=$(temp)"
      cool
    done
    n=$(faults); [ "$n" -gt "$B" ] && { log "FAULTED at H p$p d=$d, stopping"; touch "$O/DONE"; exit 1; }
    t=$(thr); [ "$t" -gt "$THR0" ] && log "GOVERNOR THROTTLED at H p$p d=$d, points from here are suspect"
  done
  log "H p$p edge $(sensors 2>/dev/null | awk '/edge/{print $2}')"
done

# I. decode at a 4096-token prefix, all six models
for p in 1 2; do
  for m in qwen2.5-1.5b-q4km qwen3-8b-q8_0 deepseek-r1-14b qwen3-14b qwen3.6-35b-a3b-iq2m qwen3.8-27b-iq3xxs; do
    for be in hip vk; do
      b=$HIP; [ $be = vk ] && b=$VK
      log "I p$p $m $be $(one $b $m -p 0 -n 32 -d 4096 -r 3)edge=$(temp)"
      cool
    done
    n=$(faults); [ "$n" -gt "$B" ] && { log "FAULTED at I p$p $m, stopping"; touch "$O/DONE"; exit 1; }
    t=$(thr); [ "$t" -gt "$THR0" ] && log "GOVERNOR THROTTLED at I p$p $m, points from here are suspect"
  done
  log "I p$p edge $(sensors 2>/dev/null | awk '/edge/{print $2}')"
done

log "faults at end: $(faults)"
touch "$O/DONE"; log DONE
