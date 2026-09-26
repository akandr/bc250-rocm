#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Artur Andrzejczak <andrzejczak.artur@gmail.com>
# Assisted-by: Claude (Anthropic)
# Does enabling SDMA cost the small model decode speed?
#
# The eight-hour soak ran with HSA_ENABLE_SDMA=0 and the three-hour one with it enabled, same boot,
# same build, back to back. Three of the four models read identically. The qwen2.5-1.5B read 196.99
# against 183.61, a 7 percent difference, against spreads of about 1.4 percent within each soak.
#
# That is two blocked runs, not an interleaved comparison, which is the design this board has
# punished repeatedly, so it is not a result yet. This interleaves the two settings inside one session,
# alternating, which is what the repository's own throughput comparisons do.
#
# The existing claim it would revise: "it buys no measurable speed for inference, 0.1 percent on 8B
# decode measured ABBA and nothing on Vulkan". The 8B is carried here too, since that is the model
# that claim rests on.
# Usage: sdma_ab.sh [pairs]
set -u
N=${1:-10}
O=~/sdma-ab; mkdir -p "$O"
L=/opt/bc250-rocm/lib64
HIP=~/llama-master/build-hip-pkf16/bin
log () { echo "[$(date +%H:%M:%S)] $*" | tee -a "$O/log"; sync; }
temp () { sensors 2>/dev/null | awk '/edge/{print $2}'; }

if [ "$(systemctl is-active ollama 2>/dev/null)" = active ]; then sudo systemctl stop ollama; sleep 10; fi
log "start, $N pairs, edge $(temp), available $(free -m | awk '/^Mem:/{print $7}') MiB"
for p in $(seq 1 "$N"); do
  for m in qwen2.5-1.5b-q4km qwen3-8b-q8_0; do
    for sd in off on; do
      if [ $sd = off ]; then export HSA_ENABLE_SDMA=0; else unset HSA_ENABLE_SDMA; fi
      r=$(LD_LIBRARY_PATH=$L timeout -k 20 900 "$HIP/llama-bench" -m /opt/models/$m.gguf -mmp 0 \
            -ngl 99 -fa on -p 0 -n 64 -r 3 2>/dev/null | grep -aE "tg64" |
          awk -F'|' '{v=$(NF-1); gsub(/^ +| +$/,"",v); gsub(/ /,"",v); print v}')
      log "p$p $m sdma=$sd tg64=${r:-FAIL} edge=$(temp)"
      sleep 15
    done
  done
done
log done; touch "$O/DONE"
