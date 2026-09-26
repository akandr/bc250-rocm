#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Artur Andrzejczak <andrzejczak.artur@gmail.com>
# Assisted-by: Claude (Anthropic)
# Context ceilings on Fedora 44: how deep each model can still generate, and at what rate.
# One depth per llama-bench invocation (several depths in one invocation distort the rate), mmap on, which
# is llama-bench's default and what the Fedora 43 ceilings used. A model that fails at a depth prints
# nothing; the harness records that as "fails".
set -u
D=${1:-~/ceilings-f44}; mkdir -p "$D"
export HSA_ENABLE_SDMA=0
B=~/llama-master/build-hip-f44/bin
log () { echo "[$(date +%T)] $*" | tee -a "$D/log"; sync; }
log "clock during this run should stay at one step: $(grep '\*' /sys/class/drm/card*/device/pp_dpm_sclk)"
for entry in "qwen2.5-1.5b-q4km:8192,16384,32768,131072" "qwen3-8b-q8_0:8192,16384" "qwen3-14b:8192,16384" "qwen3.8-27b-iq3xxs:8192,16384"; do
  m=${entry%%:*}; depths=${entry#*:}
  for d in ${depths//,/ }; do
    timeout -k 30 2400 $B/llama-bench -m /opt/models/$m.gguf -ngl 99 -fa on -p 0 -n 64 -d $d -r 2 > $D/${m}_d$d.log 2>&1
    v=$(grep -aoE "tg64 @ d$d +\| +[0-9.]+" $D/${m}_d$d.log | grep -oE "[0-9.]+$")
    log "  $m d$d: ${v:-fails} $(grep -aoiE "out of memory|failed to allocate|error" $D/${m}_d$d.log | head -1)"
  done
done
log DONE
