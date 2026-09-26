#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Artur Andrzejczak <andrzejczak.artur@gmail.com>
# Assisted-by: Claude (Anthropic)
# The three benchmark models whose heads are wider than 128: qwen3.6-35B MoE (D=256),
# qwen3.8-27B (D=256), gemma4 (D=512). Compares the production build against a build
# whose RDNA1 flash-attention table also covers those head sizes.
#
# Per model: the prefill flash-attention op alone (test-export-graph-ops + test-backend-ops
# perf, so the kernel change is visible in isolation), pp2048 end to end, tg64, and a
# perplexity gate, each in its own invocation, builds interleaved per point.
#
# gemma4's graph export needs the model's own context settings; if the export fails the
# script says so and skips that op-level row, not the model.
set -u
OUT=${OUT:-$HOME/bighead-ab}
L=/opt/bc250-rocm/lib64
M=$HOME/llama-master
BASE=${BASE:-build-hip-f44}
NEW=${NEW:-build-hip-fatile}
mkdir -p "$OUT"; cd "$OUT" || exit 1
log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$OUT/log"; }
log "clock $(cat /sys/class/drm/card*/device/pp_dpm_sclk | tr -d '\n')  governor $(systemctl is-active oberon-governor)"
log "other GPU users: $(pgrep -a -f 'ollama|llama-cli|llama-bench' | grep -v bighead | tr '\n' ';')"

bench() { # tag build args...
  local tag=$1 b=$2; shift 2
  LD_LIBRARY_PATH=$L timeout -k 20 1800 "$M/$b/bin/llama-bench" -ngl 99 -r 3 "$@" 2>/dev/null \
    | grep -aE 'pp2048|tg64' | awk -F'|' '{print $(NF-1)}' | tr -d ' ' | tr '\n' ' '
}

for m in qwen3.6-35b-a3b-iq2m qwen3.8-27b-iq3xxs gemma4; do
  model=/opt/models/$m.gguf
  ops=ops-$m.txt
  if [ ! -s "$ops" ]; then
    LD_LIBRARY_PATH=$L "$M/$BASE/bin/test-export-graph-ops" -m "$model" -ngl 99 -c 4096 -b 2048 -ub 2048 -fa on -o "$ops" > "export-$m.log" 2>&1 \
      && log "$m export ok ($(wc -l < "$ops") ops)" || log "$m export FAILED (see export-$m.log)"
  fi
  for pass in 1 2; do
    for b in $BASE $NEW; do
      if [ -s "$ops" ]; then
        fa=$(LD_LIBRARY_PATH=$L timeout -k 20 1800 "$M/$b/bin/test-backend-ops" perf --test-file "$ops" -b ROCm0 2>&1 \
             | grep -a -A1 'FLASH_ATTN_EXT.*,2048,1\]' | grep -ao '[0-9.]* us/run' | head -1)
        log "$m p$pass $b FLASH_ATTN_EXT(prefill) ${fa:-n/a}"
      fi
      log "$m p$pass $b pp2048 fa=1: $(bench pp $b -m "$model" -fa 1 -p 2048 -n 0)"
      log "$m p$pass $b pp2048 fa=0: $(bench pp $b -m "$model" -fa 0 -p 2048 -n 0)"
      log "$m p$pass $b tg64   fa=1: $(bench tg $b -m "$model" -fa 1 -p 0 -n 64)"
    done
  done
  for b in $BASE $NEW; do
    ppl=$(LD_LIBRARY_PATH=$L timeout -k 30 3600 "$M/$b/bin/llama-perplexity" -m "$model" --no-mmap -ngl 99 -fa on -c 2048 -f ~/wiki.test.raw --chunks 2 2>&1 | grep -ao 'Final estimate: PPL = [0-9.]*')
    log "$m gate $b: ${ppl:-FAIL}"
  done
done
log "clock after $(cat /sys/class/drm/card*/device/pp_dpm_sclk | tr -d '\n')"
log DONE
