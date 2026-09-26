#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Artur Andrzejczak <andrzejczak.artur@gmail.com>
# Assisted-by: Claude (Anthropic)
# Prefill against context depth, with and without the RDNA1 flash-attention spill fix
# (patches/llamacpp/0004-rdna1-fattn-tile-no-spill.patch), flash attention on and off.
#
# The two builds are run back to back at each point, because a first attempt measured
# them in separate sweeps and the -fa off arms, which the patch cannot touch, still
# differed by up to 11 percent at depth 8192. That is thermal drift across the sweep,
# so the arms have to be interleaved and the sweep repeated.
#
# One llama-bench invocation per point, three internal repeats, median of the passes
# taken afterwards.
set -u
M=${M:-/opt/models/qwen3-8b-q8_0.gguf}
OUT=${OUT:-$HOME/fa-spill-depth}
L=/opt/bc250-rocm/lib64
B=$HOME/llama-master
PASSES=${1:-3}
mkdir -p "$OUT"
log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$OUT/log"; }

log "model $M"
log "clock $(cat /sys/class/drm/card*/device/pp_dpm_sclk | tr -d '\n')"
log "governor $(systemctl is-active oberon-governor)"

for p in $(seq 1 "$PASSES"); do
  for d in 0 4096 8192; do
    for fa in 0 1; do
      for v in base:build-hip-f44 fixed:build-hip-fatile; do
        t=${v%%:*}; b=${v##*:}
        r=$(LD_LIBRARY_PATH=$L timeout -k 20 1200 "$B/$b/bin/llama-bench" -m "$M" \
              -ngl 99 -fa "$fa" -p 2048 -n 0 -d "$d" -r 3 2>/dev/null \
            | grep -a pp2048 | awk -F'|' '{print $(NF-1)}' | tr -d ' ')
        log "pass $p d=$d fa=$fa $t ${r:-FAIL}"
      done
    done
  done
  log "pass $p edge $(sensors 2>/dev/null | awk '/edge/{print $2}')"
done
log "clock after $(cat /sys/class/drm/card*/device/pp_dpm_sclk | tr -d '\n')"
log DONE
