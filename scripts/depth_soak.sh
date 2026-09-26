#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Artur Andrzejczak <andrzejczak.artur@gmail.com>
# Assisted-by: Claude (Anthropic)
# Does deep-context decode fault at a measurably higher rate than shallow work?
#
# The eight-hour soak of 21 September ran 152 rounds at depth 0 and logged no fault line at all. A few
# hours later, a sweep that repeated qwen2.5-1.5B decode at depths 16384 and 24576 faulted on its fifth
# invocation. That is one event against one clean soak, which is not a rate, and the whole reason this
# fault has been hard to study is that nobody has had a rate for it.
#
# This alternates two arms inside one run so they share a boot, a clock and a thermal state: the same
# model decoding at depth 0, and the same model decoding at 16384 and 24576. If deep context is the
# trigger, the fault lines appear in one arm and not the other. If it is not, they appear in neither or
# in both, and the deep-context lead is dead.
#
# It does not stop at the first fault: once the compute queue wedges every later ROCm run fails, so it
# records the arm it happened in and keeps going only long enough to confirm the wedge, then exits.
# Usage: depth_soak.sh [hours]
set -u
HOURS=${1:-2}
O=~/depthsoak; mkdir -p "$O"
L=/opt/bc250-rocm/lib64
HIP=~/llama-master/build-hip-pkf16/bin
M=/opt/models/qwen2.5-1.5b-q4km.gguf
export HSA_ENABLE_SDMA=0
PAT="page fault \(src_id|memory access fault|preemption time out|Queue preemption failed|runlist rebuild flush failed|GPU reset begin"
faults () { sudo journalctl -b 0 -k --no-pager 2>/dev/null | grep -ciE "$PAT"; }
log () { echo "[$(date +%H:%M:%S)] $*" | tee -a "$O/log"; sync; }
temp () { sensors 2>/dev/null | awk '/edge/{print $2}'; }
# Both arms get the same pause, so the comparison is depth and not duty cycle. Without it the deep arm
# takes twice as long per round and heats the board past the governor's limit, which would confound
# exactly the thing being tested: back-to-back deep decode reaches 93 C in six minutes and the
# throttling that follows is worth 28 percent (logs/fault-repro-2026-09-22).
cool () { sleep 30; }
thr () { sudo journalctl -b 0 -u oberon-governor --no-pager 2>/dev/null | grep -ci overheat; }

B=$(faults)
if [ "$(systemctl is-active ollama 2>/dev/null)" = active ]; then sudo systemctl stop ollama; sleep 10; fi
log "start, ${HOURS}h, faults already in this boot: $B, boot $(cat /proc/sys/kernel/random/boot_id)"
log "ollama $(systemctl is-active ollama 2>/dev/null), available $(free -m | awk '/^Mem:/{print $7}') MiB, edge $(temp), throttles so far $(thr)"
end=$(( $(date +%s) + HOURS*3600 )); n=0
while [ "$(date +%s)" -lt "$end" ]; do
  for arm in shallow deep; do
    [ "$(date +%s)" -ge "$end" ] && break
    n=$((n+1))
    if [ "$arm" = shallow ]; then depths="0 0"; else depths="16384 24576"; fi
    for d in $depths; do
      r=$(LD_LIBRARY_PATH=$L timeout -k 20 600 "$HIP/llama-bench" -m "$M" -mmp 0 -ngl 99 -fa on \
            -p 0 -n 64 -d $d -r 3 2>/dev/null | grep -aE "tg64" |
          awk -F'|' '{v=$(NF-1); gsub(/ /,"",v); print v}')
      f=$(faults)
      log "n=$n arm=$arm d=$d tg64=${r:-FAIL} faults=$f edge=$(temp) throttles=$(thr)"
      cool
      if [ "$f" -gt "$B" ]; then
        log "FAULTED in the $arm arm at depth $d, invocation $n"
        sudo journalctl -b 0 -k --no-pager --since "-5 min" > "$O/journal.txt" 2>/dev/null
        touch "$O/DONE"; exit 0
      fi
    done
  done
done
log "no fault in $n rounds of both arms"
touch "$O/DONE"; log DONE
