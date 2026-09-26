#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Artur Andrzejczak <andrzejczak.artur@gmail.com>
# Assisted-by: Claude (Anthropic)
# Is the decode-at-depth variance thermal?
#
# logs/decode-variance-state-2026-08-21 measured qwen3-8B Q8_0 generating 8 tokens at a primed depth of
# 16128 and found a coefficient of variation of 7.7 to 8.8 percent between invocations, against much
# less inside one. Four candidates were then eliminated one at a time: CPU pinning, page cache and
# compaction, the hardware queue the runtime is given, and address-space randomisation. Thermal was
# ruled out on the grounds that temperature was flat.
#
# On 22 September the governor turned out to explain a different set of dropouts, at depth and under
# sustained load, dropping the shader clock from 1500 to 1000 MHz at 93 C (logs/fault-repro-2026-09-22).
# Every invocation here prefills 16128 tokens before generating its eight, which is real heat, and the
# original design ran them back to back.
#
# Two arms, interleaved, identical except for the pause between invocations:
#   hot   back to back, as the original was run
#   cool  ninety seconds of idle first
# If the variance is thermal, the hot arm keeps its 8 percent and the cool arm loses it. Temperature and
# the governor's throttle count are recorded with every reading either way.
# Usage: variance_thermal.sh [pairs]
set -u
N=${1:-12}
O=~/variance-thermal; mkdir -p "$O"
L=/opt/bc250-rocm/lib64
HIP=~/llama-master/build-hip-pkf16/bin
M=/opt/models/qwen3-8b-q8_0.gguf
export HSA_ENABLE_SDMA=0
log () { echo "[$(date +%H:%M:%S)] $*" | tee -a "$O/log"; sync; }
temp () { sensors 2>/dev/null | awk '/edge/{print $2}'; }
thr () { sudo journalctl -b 0 -u oberon-governor --no-pager 2>/dev/null | grep -ci overheat; }

if [ "$(systemctl is-active ollama 2>/dev/null)" = active ]; then sudo systemctl stop ollama; sleep 10; fi
log "start, $N pairs, ollama $(systemctl is-active ollama 2>/dev/null), available $(free -m | awk '/^Mem:/{print $7}') MiB"
log "edge $(temp), throttles so far $(thr), clock $(cat /sys/class/drm/card*/device/pp_dpm_sclk 2>/dev/null | tr -d '\n')"

for p in $(seq 1 "$N"); do
  for arm in hot cool; do
    [ "$arm" = cool ] && { log "  cooling 90s from $(temp)"; sleep 90; }
    t0=$(temp)
    # mmap is left at llama-bench's default, as the August measurement had it. Passing -mmp 0, which
    # every throughput campaign here does, costs enough usable context that this model aborts at this
    # depth: the front page has said --no-mmap costs context since September and this is what that
    # looks like. Keep stderr: the first attempt hid it and a failed load read as a failed measurement.
    out=$(LD_LIBRARY_PATH=$L timeout -k 20 900 "$HIP/llama-bench" -m "$M" -ngl 99 -fa on \
            -p 0 -n 8 -d 16128 -r 3 2>&1)
    r=$(echo "$out" | grep -aE "tg8" | awk -F'|' '{v=$(NF-1); gsub(/^ +| +$/,"",v); gsub(/ /,"",v); print v}')
    e=$(echo "$out" | grep -aoE "ROCm error|failed to load model|out of memory" | head -1)
    log "p$p arm=$arm tg8=${r:-FAIL} ${e:+[$e]} edge_before=$t0 edge_after=$(temp) throttles=$(thr)"
  done
done
log "done, throttles total $(thr)"
touch "$O/DONE"
