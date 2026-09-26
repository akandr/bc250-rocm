#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Artur Andrzejczak <andrzejczak.artur@gmail.com>
# Assisted-by: Claude (Anthropic)
# Where the ROCm prefill deficit lives, operation by operation.
#
# End-to-end the 1.5B prefills at 197 tokens/s under ROCm and 367 under Vulkan, and
# nothing said which operation carried that. This exports the real graph of a
# pp2048 prefill with test-export-graph-ops, then replays exactly those shapes
# through test-backend-ops perf on each backend in turn, so the two are compared
# case by case on shapes the model actually runs, not on a synthetic grid.
#
# test-export-graph-ops needs no weights, so the export is seconds. The synthetic
# MUL_MAT grid is 1544 cases at roughly 10 a minute; the exported graph is 45.
#
# Both backends are run inside one pass and the pass is repeated, so thermal drift
# shows up as pass-to-pass spread and not as a difference between backends.
# Times are us/run as test-backend-ops reports them; the comparison is a ratio, so
# the clock only has to hold still within a pass.
#
# Usage: op_perf_hip_vs_vulkan.sh [passes]   (default 3)
set -u
PASSES=${1:-3}
M=${M:-$HOME/llama-master}
MODEL=${MODEL:-/opt/models/qwen2.5-1.5b-q4km.gguf}
OUT=${OUT:-$HOME/opgraph}
mkdir -p "$OUT"; cd "$OUT" || exit 1
log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$OUT/log"; }

log "model $MODEL"
log "clock $(cat /sys/class/drm/card*/device/pp_dpm_sclk | tr -d '\n')"
log "governor $(systemctl is-active oberon-governor)"
log "other GPU users: $(pgrep -a -f 'ollama|llama-cli|llama-bench' | tr '\n' ';')"

OPS=ops-1.5b-pp2048.txt
if [ ! -s "$OPS" ]; then
  "$M/build-hip-f44/bin/test-export-graph-ops" -m "$MODEL" -ngl 99 -c 4096 \
    -b 2048 -ub 2048 -fa off -o "$OPS" > export.log 2>&1
  log "export rc=$? ops=$(wc -l < "$OPS")"
fi

for p in $(seq 1 "$PASSES"); do
  for be in hip:ROCm0 vk:Vulkan0; do
    t=${be%%:*}; d=${be##*:}
    timeout -k 30 1800 "$M/build-$t-f44/bin/test-backend-ops" perf \
      --test-file "$OPS" -b "$d" > "$t-graph-p$p.log" 2>&1
    log "pass $p $t rc=$? cases=$(grep -c 'us/run' "$t-graph-p$p.log")"
  done
done

log "clock after $(cat /sys/class/drm/card*/device/pp_dpm_sclk | tr -d '\n')"
log DONE
