#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Artur Andrzejczak <andrzejczak.artur@gmail.com>
# Assisted-by: Claude (Anthropic)
# The same soak as soak13.sh, with SDMA left enabled.
#
# Every measurement in this repository was taken with HSA_ENABLE_SDMA=0, which was forced by a defect
#, not chosen, and the defect is fixed: substituting the navi12 microcode makes the user-mode
# SDMA path work. The headline throughput, the correctness gates, the allocation-churn sweep and decode
# at depth have all been re-checked with SDMA on and none of them moved. The endurance runs never were,
# and an hours-long run is where a copy engine that is quietly wrong would show up, so this is the last
# place that constant is still untested.
#
# Same four models, same gate references, same fault counting as soak13.sh. The only difference is the
# environment: if a figure here disagrees with that run, the copy engine is the difference.
# Usage: soak13sdma.sh [hours] [dir]
set -u
HOURS=${1:-3}; D=${2:-~/soak-sdma}; mkdir -p "$D"
HIP=~/llama-master/build-hip-pkf16/bin
WIKI=~/wiki.test.raw
unset HSA_ENABLE_SDMA   # the point of this run
export LD_LIBRARY_PATH=/opt/bc250-rocm/lib64
PAT="page fault \(src_id|memory access fault|preemption time out|Queue preemption failed|runlist rebuild flush failed|GPU reset begin"
faults () { sudo journalctl -b 0 -k --no-pager 2>/dev/null | grep -ciE "$PAT"; }
log () { echo "[$(date +%H:%M:%S)] $*" | tee -a "$D/log"; sync; }
temp () { sensors 2>/dev/null | grep -oE 'edge:.*\+[0-9.]+' | grep -oE '[0-9.]+$' | head -1; }

# gate references, six chunks of wiki.test.raw at context 512, as measured when each patch landed
declare -A REF=( [1.5b]=10.2088 [27b]=6.2737 [moe]=6.2265 [8b]=9.4017 )
declare -A MOD=( [1.5b]=/opt/models/qwen2.5-1.5b-q4km.gguf [27b]=/opt/models/qwen3.8-27b-iq3xxs.gguf
                 [moe]=/opt/models/qwen3.6-35b-a3b-iq2m.gguf [8b]=/opt/models/qwen3-8b-q8_0.gguf )

end=$(( $(date +%s) + HOURS*3600 )); round=0; bad=0
log "other GPU users at start: ollama $(systemctl is-active ollama 2>/dev/null), processes $(sudo fuser -v /dev/kfd 2>&1 | tail -1)"
log "soak start (SDMA ENABLED), ${HOURS}h, kernel $(uname -r), boot $(cat /proc/sys/kernel/random/boot_id)"
log "gpu_recovery=$(cat /sys/module/amdgpu/parameters/gpu_recovery), faults this boot $(faults), previous boot $(sudo journalctl -b -1 -k --no-pager 2>/dev/null | grep -ciE "$PAT")"

while [ "$(date +%s)" -lt "$end" ]; do
  for tag in 1.5b 27b moe 8b; do
    [ "$(date +%s)" -ge "$end" ] && break
    round=$((round+1)); m=${MOD[$tag]}
    timeout -k 20 1800 $HIP/llama-bench -m "$m" -mmp 0 -ngl 99 -fa on -p 512 -n 64 -r 2 > "$D/r${round}_${tag}_bench.log" 2>&1
    pp=$(grep -aoE "pp512 +\| +[0-9.]+" "$D/r${round}_${tag}_bench.log" | grep -oE "[0-9.]+$" | tail -1)
    tg=$(grep -aoE "tg64 +\| +[0-9.]+" "$D/r${round}_${tag}_bench.log" | grep -oE "[0-9.]+$" | tail -1)
    timeout -k 30 2400 $HIP/llama-perplexity -m "$m" --no-mmap -ngl 99 -fa on -c 512 -f "$WIKI" --chunks 6 > "$D/r${round}_${tag}_ppl.log" 2>&1
    g=$(grep -aoE "Final estimate: PPL = [0-9.]+" "$D/r${round}_${tag}_ppl.log" | grep -oE "[0-9.]+$")
    ok=ok; [ "${g:-none}" = "${REF[$tag]}" ] || { ok=MISMATCH; bad=$((bad+1)); }
    log "round=$round model=$tag pp512=${pp:-FAIL} tg64=${tg:-FAIL} gate=${g:-FAIL} ref=${REF[$tag]} $ok edge=$(temp) faults=$(faults)"
  done
  # allocation churn once per rotation: model loads and frees are the historically fragile part
  timeout -k 20 2400 $HIP/test-backend-ops perf -o MUL_MAT -b ROCm0 > "$D/churn_$round.log" 2>&1
  log "  churn rc=$? faults=$(faults)"
done
log "summary: rounds=$round gate mismatches=$bad"
for tag in 1.5b 27b moe 8b; do
  log "  $tag distinct gates: $(grep -aoE "model=$tag .*gate=[0-9.]+" "$D/log" | grep -oE 'gate=[0-9.]+' | sort -u | tr '\n' ' ')"
done
log "  fault lines now: $(faults)"
touch "$D/DONE"; log done
